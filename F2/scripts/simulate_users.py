"""
F2 用户行为多轮模拟脚本（重写版）
=====================================
功能：
    1. 冷启动初始化每个用户的兴趣画像（primary_field + gamma + 初始观看/点赞）
    2. 多轮（1 轮 = 1 天）模拟用户观看与点赞行为
         - 每轮为每个用户构造个性化 Top-100 候选池（三槽融合）
             槽 A：兴趣领域槽  — 来自用户 interest_fields 的视频
             槽 B：打分排序槽  — 全库随机取样后按 Aff + γ·pop 打分
             槽 C：冷门探索槽  — 低热度长尾视频，保证多样性
         - 每轮每人不放回地观看 WATCH_PER_ROUND 个视频
         - 从当轮观看中按兴趣匹配度点赞 LIKE_PER_ROUND 个
         - 点赞结果在本轮结束后更新用户兴趣画像，用于下一轮
    3. 事件日志增量写文件，避免内存积压
    4. 输出用户终态表与事件日志

运行方式:
    python simulate_users.py
    python simulate_users.py --n-users 200 --n-rounds 10 --seed 42   # 快速测试

输出文件:
    F2/data/users_sim.csv       用户终态（兴趣画像、gamma、累计观看/点赞数）
    F2/data/watch_events.csv    事件日志（round, user_id, video_id, event_type）

接口说明（与 F5 对接）:
    watch_events.csv 中 round 即"天"，F5 按 (video_id, round) 聚合 watch/like 数
    即可得到每天每视频的 V_t^obs 和 L_t^obs。
"""

import argparse
import csv
import io
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── 路径 ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent.parent
VIDEO_DB    = ROOT_DIR / "F1" / "data" / "tiktok_meta_100k.csv"
POP_FILE    = ROOT_DIR / "F1" / "data" / "video_popularity_init.csv"
OUT_DIR     = Path(__file__).parent.parent / "data"
OUT_USERS   = OUT_DIR / "users_sim.csv"
OUT_EVENTS  = OUT_DIR / "watch_events.csv"

# ── 模拟参数 ─────────────────────────────────────────────────────────────────
N_ROUNDS        = 30    # 模拟轮数（1 轮 = 1 天）
WATCH_PER_ROUND = 20    # 每轮每人观看数（不放回）
LIKE_PER_ROUND  = 3     # 每轮每人点赞数（从当轮观看中选）
TOP_K           = 100   # 每轮个性化候选池大小

# Top-100 三槽配额
SLOT_A          = 40    # 槽 A：兴趣领域槽
SLOT_B_CANDS    = 400   # 槽 B：从全库随机取样用于打分的候选数
SLOT_B          = 40    # 槽 B：打分后保留数
SLOT_C          = 20    # 槽 C：冷门/探索槽（低 H0_norm 视频）
LOW_POP_PCTILE  = 0.30  # 冷门池：H0_norm 最低的 30%

# 打分权重
W_AFF           = 1.0   # 兴趣匹配权重（领域命中 + 0.5×标签命中）
W_POP           = 1.0   # 热度权重（与用户 gamma 相乘）
SCORE_NOISE     = 0.15  # 分数加性高斯噪声标准差

# 用户热度敏感性 gamma ~ Uniform(GAMMA_LOW, GAMMA_HIGH)，每用户固定
GAMMA_LOW       = 0.3
GAMMA_HIGH      = 1.2

# 冷启动参数（初始化阶段，不记入事件日志）
INIT_WATCH      = 30
INIT_LIKE       = 10
INIT_PRIMARY_W  = 10    # 冷启动中强制来自主领域的视频数

# 输出兴趣画像限制
MAX_FIELDS_OUT  = 3
MAX_TAGS_OUT    = 5

# 20 个预定义兴趣领域（与 F1 的 tag_pipeline 保持一致）
ALL_FIELDS = [
    "Movement & Performance",    "Physical Wellness & Fitness",
    "Beauty & Personal Style",   "Mental Health & Mindfulness",
    "Food & Culinary Arts",      "Knowledge & Education",
    "Literature & Storytelling", "Finance & Wealth Building",
    "Entrepreneurship & Business","Home & Domestic Life",
    "Digital Gaming & Technology","Visual Arts & Design",
    "Travel & Exploration",      "Relationships & Social Life",
    "Music & Sound Culture",     "Comedy & Entertainment",
    "Nature & Sustainability",   "Family & Parenting",
    "Pop Culture & Commentary",  "DIY & Maker Culture",
]

USER_COLUMNS = [
    "user_id", "primary_field", "gamma",
    "interest_fields", "interest_tags",
    "total_watched", "total_liked",
]
EVENT_COLUMNS = ["round", "user_id", "video_id", "event_type"]


# ── 工具函数 ──────────────────────────────────────────────────────────────────
import re as _re
_TAG_VALID_RE = _re.compile(r"^[A-Za-z0-9 ]+$")


def _is_valid_tag(tag: str) -> bool:
    """只保留由字母、数字、空格构成的 tag；含标点视为无效。"""
    return bool(tag) and _TAG_VALID_RE.match(tag.strip()) is not None


def _parse_json_list(val: str) -> list:
    """安全地将 JSON 字符串解析为 list，失败返回空 list。"""
    if not val:
        return []
    try:
        r = json.loads(val)
        return r if isinstance(r, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _derive_fields(ctr: Counter, primary_field: str) -> list[str]:
    """
    从点赞计数器中提炼兴趣领域列表：primary_field 置顶，
    再按频率补足至最多 MAX_FIELDS_OUT 个。
    """
    extras = [f for f, _ in ctr.most_common(MAX_FIELDS_OUT + 1) if f != primary_field]
    return ([primary_field] + extras)[: MAX_FIELDS_OUT]


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_video_library(
    path: Path,
) -> tuple[list[dict], dict[str, list[int]]]:
    """
    读取 F1 视频库 CSV，解析 fields / preference_tags，
    建立领域倒排索引，并预计算每条视频的字段集合（用于快速打分）。
    """
    videos: list[dict] = []
    field_index: dict[str, list[int]] = defaultdict(list)

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            row["fields"]          = _parse_json_list(row.get("fields", ""))
            row["preference_tags"] = _parse_json_list(row.get("preference_tags", ""))
            # 预计算 frozenset 供交集运算，避免每次打分重复创建 set
            row["fields_set"]      = frozenset(row["fields"])
            row["tags_set"]        = frozenset(row["preference_tags"])
            videos.append(row)
            for field in row["fields"]:
                if field:
                    field_index[field].append(idx)

    print(f"[加载] 视频库: {len(videos):,} 条，{len(field_index)} 个领域")
    return videos, field_index


def load_popularity(path: Path) -> dict[str, float]:
    """读取 F1/data/video_popularity_init.csv，返回 {video_id: H0_norm}。"""
    pop: dict[str, float] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pop[row["id"]] = float(row["H0_norm"])
    print(f"[加载] 热度表: {len(pop):,} 条")
    return pop


def build_low_pop_pool(
    videos: list[dict], pop_norm: dict[str, float]
) -> list[int]:
    """
    按 H0_norm 升序排列后取最低 LOW_POP_PCTILE 比例的视频，
    作为「冷门/探索槽」的候选池。
    """
    scored = sorted(range(len(videos)), key=lambda i: pop_norm.get(videos[i]["id"], 0.0))
    cutoff = int(len(scored) * LOW_POP_PCTILE)
    pool = scored[:cutoff]
    print(f"[预处理] 冷门池: {len(pool):,} 条 (H0_norm 最低 {LOW_POP_PCTILE*100:.0f}%)")
    return pool


# ── 用户初始化（冷启动）────────────────────────────────────────────────────────
def init_user(
    user_id: int,
    videos: list[dict],
    field_index: dict[str, list[int]],
    rng: random.Random,
) -> dict:
    """
    冷启动：随机分配主领域和 gamma（热度敏感性）；
    模拟一次初始观看/点赞来建立第一轮前的兴趣画像。
    冷启动行为不计入事件日志。

    行为逻辑（与原 F2 保持一致）：
        观看 = INIT_PRIMARY_W 个主领域视频 + 随机补足至 INIT_WATCH
        点赞 = 主领域优先，不足则随机补至 INIT_LIKE
    """
    primary_field = rng.choice(ALL_FIELDS)
    gamma         = rng.uniform(GAMMA_LOW, GAMMA_HIGH)
    n_videos      = len(videos)
    primary_pool  = field_index.get(primary_field, [])

    # 冷启动观看（用过采样替代 O(N) 列表过滤）
    n_p = min(INIT_PRIMARY_W, len(primary_pool))
    primary_sample = rng.sample(primary_pool, n_p) if n_p > 0 else []
    primary_set    = set(primary_sample)
    n_rand         = INIT_WATCH - len(primary_sample)
    # 过采样 5× 后取前 n_rand 个不与 primary 重叠的，几乎不需要重试
    rand_sample: list[int] = []
    for i in rng.sample(range(n_videos), min(n_rand * 5, n_videos)):
        if i not in primary_set:
            rand_sample.append(i)
            if len(rand_sample) >= n_rand:
                break
    watched_init = primary_sample + rand_sample
    rng.shuffle(watched_init)

    # 冷启动点赞（主领域优先）
    primary_w = [i for i in watched_init if primary_field in videos[i]["fields"]]
    other_w   = [i for i in watched_init if i not in set(primary_w)]
    if len(primary_w) >= INIT_LIKE:
        liked_init = rng.sample(primary_w, INIT_LIKE)
    else:
        extra = rng.sample(other_w, min(INIT_LIKE - len(primary_w), len(other_w)))
        liked_init = primary_w + extra

    # 从点赞中提炼初始兴趣
    liked_fields_ctr: Counter = Counter()
    liked_tags_ctr:   Counter = Counter()
    for idx in liked_init:
        for f in videos[idx]["fields"]:
            if f:
                liked_fields_ctr[f] += 1
        for t in videos[idx]["preference_tags"]:
            if _is_valid_tag(t):
                liked_tags_ctr[t] += 1

    interest_fields = _derive_fields(liked_fields_ctr, primary_field)
    interest_tags   = [t for t, _ in liked_tags_ctr.most_common(MAX_TAGS_OUT)]

    return {
        "user_id":            user_id,
        "primary_field":      primary_field,
        "gamma":              gamma,
        "interest_fields":    interest_fields,
        "interest_tags":      interest_tags,
        "interest_fields_set":frozenset(interest_fields),
        "interest_tags_set":  frozenset(interest_tags),
        "liked_fields_ctr":   liked_fields_ctr,
        "liked_tags_ctr":     liked_tags_ctr,
        "total_watched":      len(watched_init),
        "total_liked":        len(liked_init),
    }


# ── 评分函数 ──────────────────────────────────────────────────────────────────
def _score(
    video_idx: int,
    user: dict,
    videos: list[dict],
    pop_norm: dict[str, float],
) -> float:
    """
    计算视频对用户的确定性评分（不含噪声），噪声由调用方添加：
        s = W_AFF · Aff(u, v) + gamma_u · W_POP · pop_norm(v)
    其中 Aff = 领域命中数 + 0.5 × 标签命中数
    gamma_u：用户热度敏感性，越高则越倾向高热度视频。
    """
    v   = videos[video_idx]
    aff = (
        len(v["fields_set"] & user["interest_fields_set"])
        + 0.5 * len(v["tags_set"] & user["interest_tags_set"])
    )
    pop = pop_norm.get(v["id"], 0.0)
    return W_AFF * aff + user["gamma"] * W_POP * pop


# ── Top-100 候选池构造 ────────────────────────────────────────────────────────
def build_top100(
    user: dict,
    videos: list[dict],
    field_index: dict[str, list[int]],
    low_pop_pool: list[int],
    pop_norm: dict[str, float],
    rng: random.Random,
) -> list[int]:
    """
    三槽融合构造个性化 Top-100：

    槽 A (SLOT_A = 40)
        从用户当前 interest_fields 对应的视频池中均匀采样，
        确保推荐内容与兴趣直接相关。

    槽 B (SLOT_B = 40)
        从全库随机取 SLOT_B_CANDS 个候选，对每个计算
        s + noise，取高分前 SLOT_B 个。
        gamma_u 越高，热度项拉力越大。

    槽 C (SLOT_C = 20)
        从低热度冷门池随机采样，模拟「刷到意外长尾内容」，
        防止推荐同质化。

    不足 TOP_K 时用全库随机补足。
    """
    seen: set[int]   = set()
    result: list[int] = []
    n_videos = len(videos)

    # ── 槽 A：兴趣领域 ─────────────────────────────────────────────────── #
    interest_fields = user["interest_fields"]
    per_field = max(1, SLOT_A // len(interest_fields)) if interest_fields else SLOT_A
    for field in interest_fields:
        pool = field_index.get(field, [])
        # 过采样（×3）后用计数器提前退出，避免 O(pool_size) 过滤
        n_draw = min(per_field * 3, len(pool))
        added  = 0
        if n_draw > 0:
            for i in rng.sample(pool, n_draw):
                if added >= per_field:
                    break
                if i not in seen:
                    seen.add(i)
                    result.append(i)
                    added += 1

    # ── 槽 B：打分排序 ─────────────────────────────────────────────────── #
    # 用 range() 替代全量 list，rng.sample 对 range 有专用优化路径
    b_raw   = rng.sample(range(n_videos), min(SLOT_B_CANDS, n_videos))
    b_cands = [i for i in b_raw if i not in seen]
    b_scored = [
        (_score(i, user, videos, pop_norm) + rng.gauss(0, SCORE_NOISE), i)
        for i in b_cands
    ]
    b_scored.sort(reverse=True)
    for _, i in b_scored[:SLOT_B]:
        seen.add(i)
        result.append(i)

    # ── 槽 C：冷门探索 ─────────────────────────────────────────────────── #
    c_draw = min(SLOT_C * 4, len(low_pop_pool))
    c_added = 0
    if c_draw > 0:
        for i in rng.sample(low_pop_pool, c_draw):
            if c_added >= SLOT_C:
                break
            if i not in seen:
                seen.add(i)
                result.append(i)
                c_added += 1

    # ── 补足至 TOP_K ───────────────────────────────────────────────────── #
    if len(result) < TOP_K:
        n_need = TOP_K - len(result)
        n_draw = min(n_need * 6, n_videos)
        for i in rng.sample(range(n_videos), n_draw):
            if len(result) >= TOP_K:
                break
            if i not in seen:
                seen.add(i)
                result.append(i)

    return result


# ── 观看采样 ──────────────────────────────────────────────────────────────────
def sample_watched(
    top100: list[int],
    user: dict,
    videos: list[dict],
    pop_norm: dict[str, float],
    rng: random.Random,
) -> list[int]:
    """
    对 Top-100 中每个视频加噪声打分，按分数降序不放回地选取
    WATCH_PER_ROUND 个，模拟「越感兴趣越容易点进去」。
    """
    scored = [
        (_score(i, user, videos, pop_norm) + rng.gauss(0, SCORE_NOISE), i)
        for i in top100
    ]
    scored.sort(reverse=True)
    n = min(WATCH_PER_ROUND, len(scored))
    return [i for _, i in scored[:n]]


# ── 点赞采样 ──────────────────────────────────────────────────────────────────
def sample_liked(
    watched_indices: list[int],
    user: dict,
    videos: list[dict],
    rng: random.Random,
) -> list[int]:
    """
    从当轮 WATCH_PER_ROUND 个观看视频中，按兴趣匹配度 + 随机噪声
    不放回地选取 LIKE_PER_ROUND 个点赞。

    点赞分 = primary_field/interest_fields 命中数
           + 0.5 × interest_tags 命中数
           + 高斯噪声
    点赞倾向主领域/兴趣，但不完全排除其他内容（含随机因素）。
    """
    primary_set = {user["primary_field"]} | user["interest_fields_set"]
    tag_set     = user["interest_tags_set"]

    scored = [
        (
            len(videos[i]["fields_set"] & primary_set)
            + 0.5 * len(videos[i]["tags_set"] & tag_set)
            + rng.gauss(0, SCORE_NOISE),
            i,
        )
        for i in watched_indices
    ]
    scored.sort(reverse=True)
    n = min(LIKE_PER_ROUND, len(scored))
    return [i for _, i in scored[:n]]


# ── 更新用户兴趣（本轮结束后调用）────────────────────────────────────────────
def update_user_interests(
    user: dict, liked_indices: list[int], videos: list[dict]
) -> None:
    """
    将本轮点赞视频的 fields / preference_tags 累加到计数器，
    重新派生 interest_fields 和 interest_tags（体现「越用越贴近自己」）。
    严格在本轮采样结束后调用，保证时间因果性。
    """
    for idx in liked_indices:
        v = videos[idx]
        for f in v["fields"]:
            if f:
                user["liked_fields_ctr"][f] += 1
        for t in v["preference_tags"]:
            if _is_valid_tag(t):
                user["liked_tags_ctr"][t] += 1

    interest_fields = _derive_fields(user["liked_fields_ctr"], user["primary_field"])
    interest_tags   = [t for t, _ in user["liked_tags_ctr"].most_common(MAX_TAGS_OUT)]

    user["interest_fields"]     = interest_fields
    user["interest_tags"]       = interest_tags
    user["interest_fields_set"] = frozenset(interest_fields)
    user["interest_tags_set"]   = frozenset(interest_tags)


# ── 主模拟循环 ────────────────────────────────────────────────────────────────
def simulate(
    videos: list[dict],
    field_index: dict[str, list[int]],
    pop_norm: dict[str, float],
    low_pop_pool: list[int],
    n_users: int,
    n_rounds: int,
    base_seed: int,
    event_writer: "csv.writer",
) -> list[dict]:
    """
    完整的多轮模拟：
        1. 冷启动初始化所有用户
        2. 按轮次循环，每轮为每个用户独立采样并记录事件
        3. 每轮结束后更新兴趣画像（供下一轮使用）

    随机种子方案：
        冷启动: f"{base_seed}_init_{uid}"
        第 r 轮: f"{base_seed}_r{r}_{uid}"
    使用字符串种子避免不同阶段的种子碰撞，每用户每轮完全独立可复现。
    """
    # ── 冷启动初始化 ──────────────────────────────────────────────────── #
    print(f"\n[初始化] 冷启动 {n_users:,} 个用户 ...")
    users: list[dict] = []
    report = max(1, n_users // 10)
    for uid in range(n_users):
        rng  = random.Random(f"{base_seed}_init_{uid}")
        user = init_user(uid, videos, field_index, rng)
        users.append(user)
        if (uid + 1) % report == 0:
            print(f"  已初始化 {uid + 1:,} / {n_users:,}")

    # ── 多轮模拟 ──────────────────────────────────────────────────────── #
    print(f"\n[模拟] {n_rounds} 轮 × {n_users:,} 用户（每轮看 {WATCH_PER_ROUND} 赞 {LIKE_PER_ROUND}）...")
    for rnd in range(1, n_rounds + 1):
        total_watch = 0
        total_like  = 0

        for user in users:
            uid = user["user_id"]
            rng = random.Random(f"{base_seed}_r{rnd}_{uid}")

            # 构造个性化 Top-100
            top100 = build_top100(
                user, videos, field_index,
                low_pop_pool, pop_norm, rng,
            )

            # 观看
            watched = sample_watched(top100, user, videos, pop_norm, rng)

            # 点赞
            liked = sample_liked(watched, user, videos, rng)

            # 增量写入事件日志
            for idx in watched:
                event_writer.writerow([rnd, uid, videos[idx]["id"], "watch"])
            for idx in liked:
                event_writer.writerow([rnd, uid, videos[idx]["id"], "like"])

            # 本轮结束后更新兴趣（不影响本轮采样，保证因果性）
            update_user_interests(user, liked, videos)
            user["total_watched"] += len(watched)
            user["total_liked"]   += len(liked)

            total_watch += len(watched)
            total_like  += len(liked)

        print(f"  轮次 {rnd:>2}/{n_rounds}  观看 {total_watch:,}  点赞 {total_like:,}")

    return users


# ── 保存用户终态 ──────────────────────────────────────────────────────────────
def save_users(users: list[dict], path: Path) -> None:
    """将用户终态表写入 CSV，JSON 字段与 F3/F4 格式保持一致。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=USER_COLUMNS)
        writer.writeheader()
        for u in users:
            writer.writerow({
                "user_id":       u["user_id"],
                "primary_field": u["primary_field"],
                "gamma":         round(u["gamma"], 4),
                "interest_fields": json.dumps(u["interest_fields"], ensure_ascii=False),
                "interest_tags":   json.dumps(u["interest_tags"],   ensure_ascii=False),
                "total_watched": u["total_watched"],
                "total_liked":   u["total_liked"],
            })
    print(f"[保存] 用户终态 → {path}  ({len(users):,} 行)")


# ── CLI 参数 ──────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F2 多轮用户行为模拟")
    parser.add_argument("--n-users",    type=int,  default=10_000,  help="模拟用户数（默认 10000）")
    parser.add_argument("--n-rounds",   type=int,  default=N_ROUNDS, help=f"模拟轮数（默认 {N_ROUNDS}）")
    parser.add_argument("--seed",       type=int,  default=2025,    help="随机基础种子（默认 2025）")
    parser.add_argument("--video-db",   type=Path, default=VIDEO_DB, help="视频库 CSV 路径")
    parser.add_argument("--pop-file",   type=Path, default=POP_FILE, help="初始热度 CSV 路径")
    parser.add_argument("--out-users",  type=Path, default=OUT_USERS,  help="用户终态输出路径")
    parser.add_argument("--out-events", type=Path, default=OUT_EVENTS, help="事件日志输出路径")
    return parser.parse_args()


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    args = parse_args()

    print("=" * 60)
    print(f"F2 多轮用户行为模拟")
    print(f"  用户数={args.n_users}  轮数={args.n_rounds}  seed={args.seed}")
    print("=" * 60)

    for p, label in [(args.video_db, "视频库"), (args.pop_file, "热度表")]:
        if not p.exists():
            print(f"[错误] {label}文件不存在: {p}")
            sys.exit(1)

    videos, field_index = load_video_library(args.video_db)
    pop_norm            = load_popularity(args.pop_file)
    low_pop_pool        = build_low_pop_pool(videos, pop_norm)

    args.out_events.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_events, "w", encoding="utf-8", newline="") as ef:
        event_writer = csv.writer(ef)
        event_writer.writerow(EVENT_COLUMNS)

        users = simulate(
            videos, field_index, pop_norm, low_pop_pool,
            args.n_users, args.n_rounds, args.seed,
            event_writer,
        )

    save_users(users, args.out_users)

    n_events = args.n_users * args.n_rounds * (WATCH_PER_ROUND + LIKE_PER_ROUND)
    print(f"\n完成。预计事件行数 ≈ {n_events:,}")
    print(f"  事件日志 → {args.out_events}")
    print(f"  用户终态 → {args.out_users}")


if __name__ == "__main__":
    main()

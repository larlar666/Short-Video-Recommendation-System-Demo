"""
F3 相似用户群体画像分析
=======================
针对某目标用户，从 Qdrant「users」集合中检索相似用户，然后：
  1. 手写哈希表统计兴趣标签频率          O(N×M)
  2. 手写最小堆取 Top-K 标签             O(N log K)
  3. 统计 primary_field 领域分布
  4. 用 matplotlib 绘制柱状图 + 饼图，以及词云（可选）

运行前提：
  已运行 F3/scripts/build_user_index.py，Qdrant「users」collection 已存在。

  如果没有外部 Qdrant，可以加 --memory 参数：脚本会自动从 CSV 构建内存索引再分析。
  （内存模式下需要 sentence-transformers，首次运行需下载约 420MB 模型）

运行方式：
  # 连接本地 Qdrant（推荐，索引已存在时最快）
  python analyze_similar_users.py --user-id 42

  # 内存模式（无需 Docker，自动从 CSV 建索引，慢约 1~2 分钟）
  python analyze_similar_users.py --memory --max-rows 500 --user-id 0

  # 随机选一个目标用户
  python analyze_similar_users.py --random-user

  # 保存图表到文件（不弹窗）
  python analyze_similar_users.py --user-id 0 --save-figs
"""

# ── 依赖检查 ──────────────────────────────────────────────────────────────────
import importlib
import importlib.util
import sys

def _check_pkg(pkg: str, install_hint: str) -> bool:
    if importlib.util.find_spec(pkg) is None:
        print(f"[提示] 缺少 {pkg}，请安装：{install_hint}")
        return False
    return True

_HAS_WORDCLOUD = _check_pkg(
    "wordcloud",
    "pip install wordcloud -i http://vivo-private-pip.vivo.lan:9090 "
    "--extra-index-url https://mirrors.aliyun.com/pypi/simple "
    "--trusted-host vivo-private-pip.vivo.lan",
)

import argparse
import csv
import heapq
import io
import json
import math
import re
import random as _random
from collections import defaultdict
from pathlib import Path
from typing import Optional

_TAG_VALID_RE = re.compile(r"^[A-Za-z0-9 ]+$")


def _is_valid_tag(tag: str) -> bool:
    """只保留由字母、数字、空格构成的 tag；含标点视为无效。"""
    return bool(tag) and _TAG_VALID_RE.match(tag.strip()) is not None

import matplotlib
matplotlib.use("Agg")          # 非交互后端，保证无显示器环境也能运行
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, Filter, HasIdCondition, PointStruct, VectorParams,
)

# ── 路径与配置 ────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent.parent
USERS_CSV  = ROOT_DIR / "F2" / "data" / "users_sim.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "data"

COLLECTION_NAME = "users"
MODEL_NAME      = "all-mpnet-base-v2"
VECTOR_DIM      = 768
DEFAULT_TOP_K   = 50
TOP_TAGS        = 10   # 词云 / 柱状图展示几个标签

# ── 字体（Windows 下用微软雅黑，否则 matplotlib 显示中文乱码）────────────────
def _find_cjk_font() -> Optional[str]:
    """
    尝试找一个支持 CJK 的字体文件路径。
    tags 都是英文，但标题/轴标签可能含中文，找到就用，找不到降级到默认字体。
    """
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simkai.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    # 在已安装字体里搜索
    for f in fm.findSystemFonts():
        name = Path(f).stem.lower()
        if any(k in name for k in ("msyh", "simhei", "simkai", "wqy")):
            return f
    return None

_CJK_FONT = _find_cjk_font()

# 把中文字体注册进 matplotlib 全局 rc，覆盖 DejaVu Sans
if _CJK_FONT:
    _cjk_name = fm.FontProperties(fname=_CJK_FONT).get_name()
    plt.rcParams["font.sans-serif"] = [_cjk_name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False   # 避免负号变方框

def _apply_cjk(ax=None):
    """返回 CJK FontProperties 对象（用于单独设置 legend/title 字体）。"""
    if _CJK_FONT:
        return fm.FontProperties(fname=_CJK_FONT)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤 1  连接 Qdrant 并验证数据
# ═══════════════════════════════════════════════════════════════════════════════

def connect_and_verify(qdrant_url: str) -> QdrantClient:
    """
    创建 QdrantClient 并检验「users」collection 是否存在。
    打印集合基本信息（向量维度、点总数）。

    参数
    ----
    qdrant_url : ":memory:" 表示内存模式，其余视为 HTTP URL。
    """
    if qdrant_url == ":memory:":
        client = QdrantClient(":memory:")
        print("[Qdrant] 使用内存模式（数据随进程退出丢失）")
    else:
        client = QdrantClient(url=qdrant_url)
        print(f"[Qdrant] 已连接 {qdrant_url}")

    if not client.collection_exists(COLLECTION_NAME):
        print(f"[错误] collection「{COLLECTION_NAME}」不存在。")
        print("       请先运行 F3/scripts/build_user_index.py 构建索引，")
        print("       或使用 --memory 参数在本脚本内自动构建内存索引。")
        sys.exit(1)

    info   = client.get_collection(COLLECTION_NAME)
    n_pts  = info.points_count
    v_size = info.config.params.vectors.size
    print(f"[Qdrant] collection「{COLLECTION_NAME}」: {n_pts:,} 个用户，向量维度={v_size}")
    return client


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤 2  相似用户检索
# ═══════════════════════════════════════════════════════════════════════════════

def find_similar_users(
    target_id: int,
    top_k: int,
    client: QdrantClient,
) -> list[dict]:
    """
    从 Qdrant 检索与目标用户兴趣最相近的 top_k 个用户。

    实现逻辑
    --------
    1. 用 retrieve() 从集合里直接取目标用户已存好的向量，避免重新编码。
    2. 用 query_points() 做 ANN 相似度搜索（余弦距离）。
    3. 通过 HasIdCondition must_not 过滤掉目标用户自身。

    参数
    ----
    target_id : 目标用户 ID（必须已存入 Qdrant）
    top_k     : 要检索的相似用户数量
    client    : QdrantClient 实例

    返回
    ----
    list[dict]  每个元素包含 user_id、similarity、primary_field、
                interest_fields、interest_tags
    """
    # 1. 获取目标用户向量
    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[target_id],
        with_vectors=True,
        with_payload=True,
    )
    if not points:
        raise ValueError(
            f"user_id={target_id} 不存在于「{COLLECTION_NAME}」collection。\n"
            "       请确认已运行 build_user_index.py，或在 --user-id 参数中传入有效 ID。"
        )

    target_vec     = points[0].vector
    target_payload = points[0].payload

    print(f"\n目标用户 user_id={target_id}")
    print(f"  primary_field  : {target_payload.get('primary_field', '-')}")
    print(f"  interest_fields: {target_payload.get('interest_fields', [])}")
    print(f"  interest_tags  : {target_payload.get('interest_tags', [])}")

    # 2. 相似度搜索（排除自身）
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=target_vec,
        limit=top_k + 1,              # 多取 1 防过滤后不够
        query_filter=Filter(
            must_not=[HasIdCondition(has_id=[target_id])]
        ),
        with_payload=True,
        score_threshold=0.0,
    )

    # 3. 提取 payload 列表
    payloads = []
    for hit in result.points[:top_k]:
        payloads.append({
            "user_id":        hit.payload.get("user_id", hit.id),
            "similarity":     round(hit.score, 4),
            "primary_field":  hit.payload.get("primary_field", ""),
            "interest_fields": hit.payload.get("interest_fields", []),
            "interest_tags":   hit.payload.get("interest_tags", []),
        })

    return payloads


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤 3  手写哈希表统计标签频率
# ═══════════════════════════════════════════════════════════════════════════════

def hash_tag_frequency(similar_users_payloads: list[dict]) -> dict[str, int]:
    """
    手写哈希表（Python dict）统计相似用户群体中各兴趣标签出现的次数。

    时间复杂度：O(N × M)
      N = 相似用户数，M = 每人平均标签数。
    空间复杂度：O(T)，T 为不同标签数。

    不使用 collections.Counter，手动维护 dict。

    参数
    ----
    similar_users_payloads : find_similar_users() 的返回值

    返回
    ----
    dict  key=标签名(str)，value=出现次数(int)
    """
    freq: dict[str, int] = {}   # 手写哈希表

    for user in similar_users_payloads:         # 遍历每个相似用户：O(N)
        tags = user.get("interest_tags", [])
        for tag in tags:                        # 遍历该用户每个标签：O(M)
            if not _is_valid_tag(tag):
                continue
            tag = tag.strip()
            if tag in freq:
                freq[tag] += 1
            else:
                freq[tag] = 1   # 新 key 插入哈希表：均摊 O(1)

    return freq


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤 4  手写最小堆取 Top-K 标签
# ═══════════════════════════════════════════════════════════════════════════════

def top_k_tags(freq_dict: dict[str, int], k: int = TOP_TAGS) -> list[tuple[int, str]]:
    """
    使用 heapq 实现大小为 k 的最小堆，在单次遍历中取出频率最高的 k 个标签。

    时间复杂度：O(N log K)
      N = 不同标签总数，K = 堆大小。
      优于全排序 O(N log N)（当 K << N 时收益显著）。
    空间复杂度：O(K)，堆只保存 k 个元素。

    堆里存 (count, tag)；Python heapq 是最小堆，堆顶始终是当前 k 个里最小的。
    当新元素 count > 堆顶 count 时，弹出堆顶并压入新元素。

    参数
    ----
    freq_dict : hash_tag_frequency() 的返回值
    k         : 取前 k 个高频标签

    返回
    ----
    list[(count, tag)]，按 count 从高到低排序
    """
    heap: list[tuple[int, str]] = []   # 大小 ≤ k 的最小堆

    for tag, count in freq_dict.items():         # 遍历所有标签：O(N)
        if len(heap) < k:
            heapq.heappush(heap, (count, tag))   # 堆未满：直接压入 O(log K)
        elif count > heap[0][0]:
            # 新频率超过堆顶（当前 k 个里最小的）→ 替换
            heapq.heapreplace(heap, (count, tag))  # O(log K)

    # 按频率从高到低排序后返回（堆本身不保证有序）
    heap.sort(reverse=True)
    return heap


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤 5  领域分布统计
# ═══════════════════════════════════════════════════════════════════════════════

def field_distribution(similar_users_payloads: list[dict]) -> dict[str, int]:
    """
    统计相似用户群体中各 primary_field（主兴趣领域）的人数分布。

    统计口径：每个用户只算其 primary_field（第一兴趣领域）。

    时间复杂度：O(N)，N = 相似用户数。

    参数
    ----
    similar_users_payloads : find_similar_users() 的返回值

    返回
    ----
    dict  key=领域名，value=该领域用户数
    """
    dist: dict[str, int] = {}

    for user in similar_users_payloads:
        field = user.get("primary_field", "Unknown") or "Unknown"
        if field in dist:
            dist[field] += 1
        else:
            dist[field] = 1

    return dist


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤 6  可视化
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_results(
    freq_dict: dict[str, int],
    top_k_list: list[tuple[int, str]],
    field_dict: dict[str, int],
    target_id: int,
    save_figs: bool = False,
) -> None:
    """
    生成三张图表：
      1. 水平柱状图：Top-10 标签出现次数
      2. 饼图：primary_field 领域分布
      3. 词云：全部标签（需 wordcloud 库）

    参数
    ----
    freq_dict  : 全量标签频率字典（词云用）
    top_k_list : top_k_tags() 返回的 [(count, tag), ...]（柱状图用）
    field_dict : field_distribution() 返回值（饼图用）
    target_id  : 目标用户 ID（用于图表标题）
    save_figs  : True 则保存 PNG 而非弹窗显示
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fp = _apply_cjk()   # 中文字体属性（可能为 None）

    def _title_kw(text: str) -> dict:
        """生成含中文字体的 title 关键字参数。"""
        if fp:
            return {"label": text, "fontproperties": fp}
        return {"label": text}

    # ── 图1：水平柱状图 ────────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    if top_k_list:
        counts = [c for c, _ in top_k_list]
        tags   = [t for _, t in top_k_list]
        bars   = ax1.barh(range(len(tags)), counts, color="#4C9BE8", edgecolor="white")
        ax1.set_yticks(range(len(tags)))
        ax1.set_yticklabels(tags, fontsize=10)
        ax1.invert_yaxis()
        ax1.set_xlabel("出现次数", fontproperties=fp)
        ax1.set_title(**_title_kw(f"Top-{len(top_k_list)} 兴趣标签（user_id={target_id} 的相似群体）"))
        for bar, cnt in zip(bars, counts):
            ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     str(cnt), va="center", fontsize=9)
    else:
        ax1.text(0.5, 0.5, "无标签数据", ha="center", va="center",
                 transform=ax1.transAxes)

    fig1.tight_layout()
    if save_figs:
        p = OUTPUT_DIR / f"tag_bar_user{target_id}.png"
        fig1.savefig(p, dpi=120)
        print(f"[图1] 柱状图已保存：{p}")
    else:
        plt.show(block=False)

    # ── 图2：饼图 ─────────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    if field_dict:
        labels = list(field_dict.keys())
        sizes  = list(field_dict.values())
        colors = plt.cm.Set3.colors[:len(labels)]
        wedges, texts, autotexts = ax2.pie(
            sizes,
            labels=None,
            autopct="%1.1f%%",
            colors=colors,
            startangle=140,
            pctdistance=0.82,
        )
        ax2.legend(
            wedges, labels,
            title="兴趣领域",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            prop=fp,
        )
        ax2.set_title(**_title_kw(f"领域分布（user_id={target_id} 的相似群体）"))
    else:
        ax2.text(0.5, 0.5, "无领域数据", ha="center", va="center",
                 transform=ax2.transAxes)

    fig2.tight_layout()
    if save_figs:
        p = OUTPUT_DIR / f"field_pie_user{target_id}.png"
        fig2.savefig(p, dpi=120, bbox_inches="tight")
        print(f"[图2] 饼图已保存：{p}")
    else:
        plt.show(block=False)

    # ── 图3：词云 ─────────────────────────────────────────────────────────────
    if _HAS_WORDCLOUD and freq_dict:
        from wordcloud import WordCloud   # noqa: PLC0415
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        wc_font = _CJK_FONT or None
        wc = WordCloud(
            width=900, height=420,
            background_color="white",
            colormap="RdYlBu",
            max_words=200,
            font_path=wc_font,
        ).generate_from_frequencies(freq_dict)
        ax3.imshow(wc, interpolation="bilinear")
        ax3.axis("off")
        ax3.set_title(**_title_kw(f"兴趣标签词云（user_id={target_id} 的相似群体）"))
        fig3.tight_layout()
        if save_figs:
            p = OUTPUT_DIR / f"tag_wordcloud_user{target_id}.png"
            fig3.savefig(p, dpi=120)
            print(f"[图3] 词云已保存：{p}")
        else:
            plt.show(block=False)
    elif not _HAS_WORDCLOUD:
        print("[跳过] 词云未生成：wordcloud 库未安装（见脚本开头的安装提示）")

    if not save_figs:
        plt.pause(0.1)
        input("\n按 Enter 关闭图表窗口...")
        plt.close("all")


# ═══════════════════════════════════════════════════════════════════════════════
# 内存模式辅助：从 CSV 快速构建内存 Qdrant 索引
# ═══════════════════════════════════════════════════════════════════════════════

def _build_memory_index(
    client: QdrantClient,
    users_csv: Path,
    max_rows: Optional[int],
) -> list[int]:
    """
    （仅内存模式使用）从 CSV 读取用户数据，编码后写入内存 Qdrant。
    返回所有写入的 user_id 列表。
    """
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    print(f"[内存模式] 读取 CSV：{users_csv}")
    with open(users_csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if max_rows:
        rows = rows[:max_rows]
    print(f"[内存模式] 共 {len(rows):,} 条用户数据，开始编码 ...")

    def _user_to_text(row: dict) -> str:
        fields = json.loads(row.get("interest_fields") or "[]")
        tags   = json.loads(row.get("interest_tags")   or "[]")
        if not fields:
            return " ".join(tags)
        primary = fields[0]
        others  = fields[1:]
        return " ".join([primary] * 3 + [f for f in others for _ in range(2)] + tags)

    model   = SentenceTransformer(MODEL_NAME)
    texts   = [_user_to_text(r) for r in rows]
    vectors = model.encode(texts, batch_size=128, show_progress_bar=True,
                           normalize_embeddings=True)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=int(r["user_id"]),
            vector=vectors[i].tolist(),
            payload={
                "user_id":        int(r["user_id"]),
                "primary_field":  r.get("primary_field", ""),
                "interest_fields": json.loads(r.get("interest_fields") or "[]"),
                "interest_tags":   json.loads(r.get("interest_tags")   or "[]"),
            },
        )
        for i, r in enumerate(rows)
    ]
    BATCH = 500
    for start in range(0, len(points), BATCH):
        client.upsert(COLLECTION_NAME, points=points[start:start + BATCH])
    print(f"[内存模式] 已写入 {len(points):,} 条向量\n")
    return [int(r["user_id"]) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤 7  主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="F3 相似用户群体画像分析")
    parser.add_argument("--qdrant-url",   default="http://localhost:6333",
                        help="Qdrant 服务地址（默认 http://localhost:6333）")
    parser.add_argument("--memory",       action="store_true",
                        help="使用内存 Qdrant，自动从 CSV 构建索引（无需 Docker）")
    parser.add_argument("--users-csv",    type=Path, default=USERS_CSV,
                        help=f"用户数据 CSV 路径（默认 {USERS_CSV}）")
    parser.add_argument("--max-rows",     type=int, default=None,
                        help="内存模式：只索引前 N 条（默认全部）")
    parser.add_argument("--user-id",      type=int, default=None,
                        help="目标用户 ID（与 --random-user 二选一）")
    parser.add_argument("--random-user",  action="store_true",
                        help="随机选取一个目标用户")
    parser.add_argument("--top-k",        type=int, default=DEFAULT_TOP_K,
                        help=f"检索相似用户数量（默认 {DEFAULT_TOP_K}）")
    parser.add_argument("--save-figs",    action="store_true",
                        help="将图表保存为 PNG 文件（不弹窗）")
    args = parser.parse_args()

    print("=" * 60)
    print("F3  相似用户群体画像分析")
    print("=" * 60)

    # ── 1. 连接 Qdrant ────────────────────────────────────────────────────────
    qdrant_url = ":memory:" if args.memory else args.qdrant_url
    client     = QdrantClient(":memory:") if args.memory else QdrantClient(url=qdrant_url)

    all_ids: Optional[list[int]] = None

    if args.memory:
        print("[内存模式] 启动，先从 CSV 构建索引 ...")
        if not args.users_csv.exists():
            print(f"[错误] 找不到 CSV 文件：{args.users_csv}")
            sys.exit(1)
        all_ids = _build_memory_index(client, args.users_csv, args.max_rows)
    else:
        # 普通模式：检验集合是否存在
        if not client.collection_exists(COLLECTION_NAME):
            print(f"[错误] collection「{COLLECTION_NAME}」不存在。")
            print("       请先运行 F3/scripts/build_user_index.py，")
            print("       或使用 --memory 参数在内存中自动构建。")
            sys.exit(1)

        info  = client.get_collection(COLLECTION_NAME)
        n_pts = info.points_count
        v_sz  = info.config.params.vectors.size
        print(f"[Qdrant] collection「{COLLECTION_NAME}」: {n_pts:,} 用户，维度={v_sz}")

    # ── 2. 确定目标用户 ID ────────────────────────────────────────────────────
    if args.random_user:
        # 从集合里随机取一个点
        scroll_result, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=False,
            with_vectors=False,
        )
        if not scroll_result:
            print("[错误] 集合为空")
            sys.exit(1)
        target_id = _random.choice([p.id for p in scroll_result])
        print(f"[随机] 选取 user_id={target_id} 作为目标用户")
    elif args.user_id is not None:
        target_id = args.user_id
    elif all_ids:
        target_id = all_ids[0]
        print(f"[默认] 使用第一个用户 user_id={target_id}")
    else:
        print("[错误] 请指定 --user-id <ID> 或 --random-user")
        sys.exit(1)

    print(f"\n正在分析用户：{target_id}")

    # ── 3. 检索相似用户 ───────────────────────────────────────────────────────
    try:
        similar_payloads = find_similar_users(target_id, args.top_k, client)
    except ValueError as e:
        print(f"[错误] {e}")
        sys.exit(1)

    print(f"找到 {len(similar_payloads)} 个相似用户")

    # ── 4. 手写哈希表统计标签频率 ─────────────────────────────────────────────
    freq_dict = hash_tag_frequency(similar_payloads)
    print(f"[统计] 共涉及 {len(freq_dict)} 个不同兴趣标签")

    # ── 5. 手写堆取 Top-K ────────────────────────────────────────────────────
    top_list = top_k_tags(freq_dict, k=TOP_TAGS)

    print(f"\nTop-{TOP_TAGS} 兴趣标签：")
    print(f"  {'排名':<5} {'标签':<30} {'出现次数'}")
    print("  " + "─" * 50)
    for rank, (cnt, tag) in enumerate(top_list, 1):
        bar = "█" * min(cnt, 30)
        print(f"  {rank:<5} {tag:<30} {cnt:>4}  {bar}")

    # ── 6. 领域分布 ───────────────────────────────────────────────────────────
    field_dist = field_distribution(similar_payloads)
    print(f"\n领域分布（共 {len(field_dist)} 个领域）：")
    for field, cnt in sorted(field_dist.items(), key=lambda x: -x[1]):
        pct = cnt / len(similar_payloads) * 100
        print(f"  {field:<40} {cnt:>3} 人  ({pct:.1f}%)")

    # ── 7. 可视化 ─────────────────────────────────────────────────────────────
    print("\n[可视化] 生成图表 ...")
    visualize_results(
        freq_dict, top_list, field_dist,
        target_id=target_id,
        save_figs=args.save_figs,
    )

    # ── 8. 分析结论 ───────────────────────────────────────────────────────────
    if top_list and field_dist:
        top1_tag  = top_list[0][1]
        top1_cnt  = top_list[0][0]
        top_field = max(field_dist, key=field_dist.__getitem__)
        top_field_cnt = field_dist[top_field]
        top_field_pct = top_field_cnt / len(similar_payloads) * 100

        print()
        print("=" * 60)
        print("分析结论")
        print("=" * 60)
        print(f"目标用户 user_id={target_id} 的相似群体画像：")
        print(f"  • 核心兴趣标签：「{top1_tag}」（{top1_cnt} 人提及）")
        print(f"  • 主要分布领域：「{top_field}」（{top_field_cnt} 人，占 {top_field_pct:.1f}%）")
        if len(top_list) >= 3:
            other_tags = "、".join(f"「{t}」" for _, t in top_list[1:4])
            print(f"  • 高频关联标签：{other_tags}")
        print()


if __name__ == "__main__":
    main()

"""
F4 视频推荐
===========
给定一个 user_id，从 Qdrant 的 "videos" collection 中找出最适合该用户的视频。

推荐流程：
  1. 读取用户的 interest_fields 和 interest_tags，按权重拼成查询文本
     （primary_field × 3，其余 fields × 2，tags × 1）
  2. 编码为查询向量，向 Qdrant 检索 Top-80 候选视频（CANDIDATE_POOL）
  3. 在 Python 端过滤掉用户已看过的视频（watched_video_ids）
  4. 多样性后处理：每个领域最多保留 MAX_PER_FIELD 条，凑足 top_k 条

运行前提：
  已运行 build_video_index.py 构建视频向量索引。

运行方式：
  python recommend_videos.py --user-id 42
  python recommend_videos.py --user-id 42 --top-k 20
  python recommend_videos.py --user-id 42 --out result.json
"""

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ── 路径与配置 ────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent.parent.parent
EVENTS_CSV    = ROOT_DIR / "F2" / "data" / "watch_events.csv"


def _default_users_csv() -> Path:
    p_sim = ROOT_DIR / "F2" / "data" / "users_sim.csv"
    p_old = ROOT_DIR / "F2" / "data" / "users_10k.csv"
    return p_sim if p_sim.exists() else p_old


USERS_CSV = _default_users_csv()

COLLECTION_NAME = "videos"
MODEL_NAME      = "all-mpnet-base-v2"
DEFAULT_TOP_K   = 20
CANDIDATE_POOL  = 80    # 先从 Qdrant 取这么多，过滤和多样性处理后再截断
MAX_PER_FIELD   = 5     # 多样性上限：每个领域最多保留几条


# ── 用户文本序列化（与 F3 保持一致）──────────────────────────────────────────

def user_to_text(row: dict) -> str:
    """
    将用户兴趣序列化为查询文本：
    primary_field × 3，其余 interest_fields × 2，interest_tags × 1。
    """
    interest_fields = json.loads(row.get("interest_fields") or "[]")
    interest_tags   = json.loads(row.get("interest_tags")   or "[]")

    if not interest_fields:
        return " ".join(interest_tags)

    primary = interest_fields[0]
    others  = interest_fields[1:]

    parts = (
        [primary] * 3
        + [f for f in others for _ in range(2)]
        + interest_tags
    )
    return " ".join(parts)


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_user(user_id: int, path: Path) -> Optional[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["user_id"]) == user_id:
                return row
    return None


def watched_ids_from_events(user_id: int, events_path: Path) -> set[str]:
    """users_sim.csv 无 watched_video_ids 时，从 watch_events 流式聚合（可能较慢）。"""
    out: set[str] = set()
    with open(events_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["user_id"]) != user_id:
                continue
            if row.get("event_type") == "watch":
                out.add(row["video_id"])
    return out


# ── 多样性后处理 ──────────────────────────────────────────────────────────────

def diversify(candidates: list[dict], top_k: int, max_per_field: int) -> list[dict]:
    """
    从候选视频（已按相似度降序排列）中选出 top_k 条，
    同时限制每个领域最多出现 max_per_field 条，保证结果多样性。

    使用视频 fields 的第一个领域作为分组 key。
    """
    field_counts: dict[str, int] = defaultdict(int)
    result: list[dict] = []

    for hit in candidates:
        if len(result) >= top_k:
            break
        primary_field = hit["fields"][0] if hit["fields"] else "Unknown"
        if field_counts[primary_field] < max_per_field:
            result.append(hit)
            field_counts[primary_field] += 1

    return result


# ── 核心推荐 ──────────────────────────────────────────────────────────────────

def recommend(
    user_id: int,
    top_k: int,
    client: QdrantClient,
    users_csv: Path,
    events_csv: Optional[Path] = None,
) -> Tuple[dict, list]:
    """
    返回 (user_row, recommendations)。
    recommendations 是长度 ≤ top_k 的推荐视频列表，按相似度降序排列。
    """
    user = load_user(user_id, users_csv)
    if user is None:
        raise ValueError(f"user_id={user_id} 在数据集中不存在")

    raw_w = user.get("watched_video_ids")
    if raw_w:
        watched_set = set(json.loads(raw_w))
    elif events_csv and events_csv.exists():
        print("[提示] 用户表无 watched_video_ids，正从 watch_events 聚合已看视频（首次较慢）...")
        watched_set = watched_ids_from_events(user_id, events_csv)
    else:
        watched_set = set()

    # 多取一些候选，以确保过滤后仍够 top_k 条
    fetch_limit = CANDIDATE_POOL + len(watched_set)

    model    = SentenceTransformer(MODEL_NAME)
    user_vec = model.encode(
        [user_to_text(user)], normalize_embeddings=True
    )[0]

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=user_vec.tolist(),
        limit=fetch_limit,
        with_payload=True,
        score_threshold=0.0,
    ).points

    # 过滤已看过的视频（payload 中的 video_id 字符串与 watched_set 对比）
    candidates = [
        {
            "video_id":        hit.payload["video_id"],
            "similarity":      round(hit.score, 4),
            "fields":          hit.payload["fields"],
            "preference_tags": hit.payload["preference_tags"],
            "play_count":      hit.payload["play_count"],
            "digg_count":      hit.payload["digg_count"],
            "desc":            hit.payload["desc"],
        }
        for hit in hits
        if hit.payload["video_id"] not in watched_set
    ]

    return user, diversify(candidates, top_k, MAX_PER_FIELD)


# ── 输出格式化 ────────────────────────────────────────────────────────────────

def print_results(user: dict, results: list[dict]) -> None:
    ifields = json.loads(user.get("interest_fields") or "[]")
    itags   = json.loads(user.get("interest_tags")   or "[]")

    print(f"用户 user_id={user['user_id']}")
    print(f"  primary_field  : {user['primary_field']}")
    print(f"  interest_fields: {ifields}")
    print(f"  interest_tags  : {itags}")
    print()
    print(f"推荐视频 Top-{len(results)}：")
    print("─" * 85)

    for rank, v in enumerate(results, 1):
        fields_str = " | ".join(v["fields"])[:40]
        tags_str   = ", ".join(v["preference_tags"][:5])
        print(f"  [{rank:>2}] 相似度={v['similarity']}  👁 {v['play_count']:>8,}  ♥ {v['digg_count']:>7,}")
        print(f"       领域: {fields_str}")
        print(f"       标签: {tags_str}")
        print(f"       简介: {v['desc'][:80]}")
        print()


# ── 入口 ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F4 视频推荐")
    p.add_argument("--user-id",    type=int, required=True, help="目标用户 ID")
    p.add_argument("--top-k",      type=int, default=DEFAULT_TOP_K, help="推荐几条视频")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--users-csv",  type=Path, default=USERS_CSV)
    p.add_argument("--events-csv", type=Path, default=EVENTS_CSV,
                   help="watch_events.csv，用于 users_sim 时解析已看视频")
    p.add_argument("--out",        type=Path, default=None,
                   help="可选：将推荐结果保存为 JSON 文件")
    return p.parse_args()


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    args = parse_args()

    print("=" * 55)
    print(f"F4  视频推荐  user_id={args.user_id}  Top-{args.top_k}")
    print("=" * 55 + "\n")

    client = QdrantClient(url=args.qdrant_url)
    user, results = recommend(
        args.user_id, args.top_k, client, args.users_csv, args.events_csv
    )
    print_results(user, results)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[保存] 推荐结果已写入 {args.out}")


if __name__ == "__main__":
    main()

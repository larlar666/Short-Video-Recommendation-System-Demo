"""
F4 冒烟测试（内存模式，无需 Docker）
======================================
在同一个 Python 进程内：
  1. 用 QdrantClient(":memory:") 启动内存 Qdrant（不需要 Docker、不占端口）
  2. 读取视频库前 MAX_ROWS 条，编码向量，写入内存 collection
  3. 对指定用户做推荐，打印结果

运行方式:
    python F4/scripts/smoke_test_f4.py
    python F4/scripts/smoke_test_f4.py --max-rows 200 --user-id 5
"""

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

ROOT_DIR   = Path(__file__).parent.parent.parent
VIDEOS_CSV = ROOT_DIR / "F1" / "data" / "tiktok_meta_100k.csv"
USERS_CSV  = ROOT_DIR / "F2" / "data" / "users_sim.csv"

COLLECTION = "videos"
MODEL_NAME = "all-mpnet-base-v2"
MAX_ROWS   = 100


# ── 工具 ─────────────────────────────────────────────────────────────────────

def _parse(val: str) -> list:
    try:
        r = json.loads(val or "[]")
        return r if isinstance(r, list) else []
    except Exception:
        return []

def _to_int(val) -> int:
    try:
        return int(float(val)) if val else 0
    except Exception:
        return 0

def video_to_text(row: dict) -> str:
    fields = _parse(row.get("fields", ""))
    tags   = _parse(row.get("preference_tags", ""))
    return " ".join(fields + fields + tags)

def user_to_text(row: dict) -> str:
    fields = _parse(row.get("interest_fields", ""))
    tags   = _parse(row.get("interest_tags", ""))
    if not fields:
        return " ".join(tags)
    primary = fields[0]
    others  = fields[1:]
    parts   = [primary] * 3 + [f for f in others for _ in range(2)] + tags
    return " ".join(parts)


# ── 建索引 ────────────────────────────────────────────────────────────────────

def build_index(client: QdrantClient, videos: list[dict], model: SentenceTransformer) -> None:
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print(f"[Qdrant] 内存 collection '{COLLECTION}' 已创建")

    texts   = [video_to_text(v) for v in videos]
    vectors = model.encode(texts, batch_size=64, show_progress_bar=True,
                           normalize_embeddings=True)

    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                "video_id":        v["id"],
                "fields":          _parse(v.get("fields", "")),
                "preference_tags": _parse(v.get("preference_tags", "")),
                "play_count":      _to_int(v.get("play_count")),
                "digg_count":      _to_int(v.get("digg_count")),
                "desc":            (v.get("desc") or "")[:120],
            },
        )
        for i, v in enumerate(videos)
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"[Qdrant] 已写入 {len(points)} 条视频向量\n")


# ── 推荐 ──────────────────────────────────────────────────────────────────────

def recommend(user_id: int, top_k: int,
              client: QdrantClient, model: SentenceTransformer) -> None:
    user = None
    with open(USERS_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["user_id"]) == user_id:
                user = row
                break
    if user is None:
        print(f"[错误] user_id={user_id} 不存在")
        return

    watched_set = set(_parse(user.get("watched_video_ids", "")))
    user_vec    = model.encode([user_to_text(user)], normalize_embeddings=True)[0]

    hits = client.query_points(
        collection_name=COLLECTION,
        query=user_vec.tolist(),
        limit=top_k + 20,
        with_payload=True,
        score_threshold=0.0,
    ).points

    # 过滤已看过，多样性限制
    field_cnt: dict[str, int] = defaultdict(int)
    results = []
    for hit in hits:
        if len(results) >= top_k:
            break
        vid = hit.payload["video_id"]
        if vid in watched_set:
            continue
        pf = (hit.payload["fields"] or ["Unknown"])[0]
        if field_cnt[pf] >= 5:
            continue
        field_cnt[pf] += 1
        results.append(hit)

    # 打印
    ifields = _parse(user.get("interest_fields", ""))
    itags   = _parse(user.get("interest_tags", ""))
    print(f"用户 user_id={user['user_id']}")
    print(f"  primary_field  : {user.get('primary_field','')}")
    print(f"  interest_fields: {ifields}")
    print(f"  interest_tags  : {itags}")
    print()
    print(f"推荐视频 Top-{len(results)}：")
    print("─" * 80)
    for rank, hit in enumerate(results, 1):
        pl = hit.payload
        print(f"  [{rank:>2}] 相似度={hit.score:.4f}  播放={pl['play_count']:>8,}  点赞={pl['digg_count']:>7,}")
        print(f"       领域: {' | '.join(pl['fields'])[:50]}")
        print(f"       标签: {', '.join(pl['preference_tags'][:5])}")
        print(f"       简介: {pl['desc'][:75]}")
        print()


# ── 入口 ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="F4 冒烟测试（内存 Qdrant）")
    p.add_argument("--max-rows", type=int, default=MAX_ROWS, help="索引前 N 条视频")
    p.add_argument("--user-id",  type=int, default=0,        help="目标用户 ID")
    p.add_argument("--top-k",    type=int, default=10,       help="推荐数量")
    return p.parse_args()


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    args = parse_args()

    print("=" * 60)
    print(f"F4 冒烟测试（内存 Qdrant）  max_rows={args.max_rows}  user_id={args.user_id}")
    print("=" * 60)

    # 加载视频
    print(f"\n[1/3] 读取视频库（前 {args.max_rows} 条）...")
    with open(VIDEOS_CSV, encoding="utf-8-sig", newline="") as f:
        videos = []
        for i, row in enumerate(csv.DictReader(f)):
            if i >= args.max_rows:
                break
            videos.append(row)
    print(f"      已读取 {len(videos)} 条视频")

    # 加载模型（只加载一次，建索引 + 推荐共用）
    print(f"\n[2/3] 加载编码模型 {MODEL_NAME} ...")
    print("      （首次运行会下载模型文件约 420MB，请耐心等待）")
    model = SentenceTransformer(MODEL_NAME)
    print("      模型加载完成")

    # 内存 Qdrant：建索引
    print(f"\n[2.5] 编码并写入内存 Qdrant ...")
    client = QdrantClient(":memory:")
    build_index(client, videos, model)

    # 推荐
    print(f"[3/3] 为用户 {args.user_id} 生成推荐（Top-{args.top_k}）...\n")
    recommend(args.user_id, args.top_k, client, model)

    print("冒烟测试完成。内存 Qdrant 数据随进程退出自动释放。")
    print("如需持久化，请启动 Docker 版 Qdrant 后用 build_video_index.py 全量建索引。")


if __name__ == "__main__":
    main()

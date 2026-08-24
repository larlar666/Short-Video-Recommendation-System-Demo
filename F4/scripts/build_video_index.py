"""
F4 构建视频向量索引
===================
从 F1 的 tiktok_meta_100k.csv 出发，将每条视频的领域和标签用 all-mpnet-base-v2
编码成向量，批量存入 Qdrant "videos" collection，为视频推荐做准备。

运行前提：
  1. 已运行 F1/scripts/build_video_db.py 生成 tiktok_meta_100k.csv
  2. Qdrant 已启动：docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

运行方式：
  python build_video_index.py
  python build_video_index.py --qdrant-url http://localhost:6333

耗时说明：
  CPU（无 GPU）：约 15-20 分钟
  有 CUDA GPU  ：约 2-3 分钟
  编码只需执行一次，结果永久保存在 Qdrant 本地存储中。
"""

import argparse
import csv
import io
import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── 路径 ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent.parent
VIDEOS_CSV = ROOT_DIR / "F1" / "data" / "tiktok_meta_100k.csv"

# ── 配置 ──────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "videos"
MODEL_NAME = Path(__file__).parent.parent.parent / "models" / "all-mpnet-base-v2"
VECTOR_DIM      = 768
ENCODE_BATCH    = 256
UPSERT_BATCH    = 500
DESC_MAX_LEN    = 120   # desc 截断长度（仅存于 payload，用于展示）


# ── 文本序列化 ─────────────────────────────────────────────────────────────────

def video_to_text(row: dict) -> str:
    """
    将视频内容序列化为编码文本。

    权重设计：
      fields（兴趣领域）× 2   ← 大方向，权重更高
      preference_tags   × 1   ← 具体内容词
    """
    fields = json.loads(row.get("fields")          or "[]")
    tags   = json.loads(row.get("preference_tags") or "[]")

    # fields 重复一次使其权重约为 tags 的两倍
    return " ".join(fields + fields + tags)


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_videos(path: Path) -> list[dict]:
    print(f"[加载] 读取视频库：{path}")
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"[加载] 共 {len(rows):,} 条视频")
    return rows


# ── 索引构建 ──────────────────────────────────────────────────────────────────

def build_index(videos: list[dict], client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        print(f"[Qdrant] collection '{COLLECTION_NAME}' 已存在，删除重建 ...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"[Qdrant] collection '{COLLECTION_NAME}' 创建完成（余弦相似度，{VECTOR_DIM} 维）")

    print(f"\n[编码] 加载模型 {MODEL_NAME} ...")
    model = SentenceTransformer(str(MODEL_NAME), trust_remote_code=False)

    texts = [video_to_text(v) for v in videos]
    print(f"[编码] 开始编码 {len(texts):,} 条视频，batch_size={ENCODE_BATCH}")
    print("       （CPU 约 15-20 分钟，有 GPU 约 2-3 分钟）\n")

    vectors = model.encode(
        texts,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print(f"\n[Qdrant] 写入 {len(videos):,} 条视频向量 ...")
    for start in range(0, len(videos), UPSERT_BATCH):
        batch_videos  = videos [start : start + UPSERT_BATCH]
        batch_vectors = vectors[start : start + UPSERT_BATCH]

        points = [
            PointStruct(
                id=start + i,   # 用连续整数作 point id（video_id 太大，存入 payload）
                vector=vec.tolist(),
                payload={
                    "video_id":        v["id"],
                    "fields":          json.loads(v.get("fields")          or "[]"),
                    "preference_tags": json.loads(v.get("preference_tags") or "[]"),
                    "play_count":      _to_int(v.get("play_count")),
                    "digg_count":      _to_int(v.get("digg_count")),
                    "desc":            (v.get("desc") or "")[:DESC_MAX_LEN],
                },
            )
            for i, (v, vec) in enumerate(zip(batch_videos, batch_vectors))
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  已写入 {min(start + UPSERT_BATCH, len(videos)):,} / {len(videos):,}")

    print(f"\n[完成] '{COLLECTION_NAME}' collection 共 {len(videos):,} 条点")
    print(f"       Qdrant Dashboard：http://localhost:6333/dashboard")


def _to_int(val) -> int:
    """安全地将数值字段转为整数，失败时返回 0。"""
    try:
        return int(float(val)) if val else 0
    except (ValueError, TypeError):
        return 0


# ── 入口 ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F4 构建视频向量索引")
    p.add_argument("--qdrant-url",  default="http://localhost:6333")
    p.add_argument("--videos-csv",  type=Path, default=VIDEOS_CSV)
    return p.parse_args()


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    args = parse_args()

    if not args.videos_csv.exists():
        print(f"[错误] 视频库文件不存在：{args.videos_csv}")
        print("       请先运行 F1/scripts/build_video_db.py")
        sys.exit(1)

    print("=" * 55)
    print("F4  构建视频向量索引")
    print("=" * 55)

    videos = load_videos(args.videos_csv)
    client = QdrantClient(url=args.qdrant_url)
    build_index(videos, client)


if __name__ == "__main__":
    main()

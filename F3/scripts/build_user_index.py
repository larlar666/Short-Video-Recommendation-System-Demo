"""
F3 构建用户向量索引
===================
从 F2 的用户画像 CSV（默认 users_sim.csv，或旧版 users_10k.csv）出发，
将每个用户的兴趣用 all-mpnet-base-v2 编码成向量，
批量存入 Qdrant "users" collection，为后续相似用户查询做准备。

运行前提：
  1. 已运行 F2/scripts/simulate_users.py 生成 users_sim.csv（或 users_10k.csv）
  2. Qdrant 已启动：docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

运行方式：
  python build_user_index.py
  python build_user_index.py --qdrant-url http://localhost:6333
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
ROOT_DIR  = Path(__file__).parent.parent.parent


def _default_users_csv() -> Path:
    """优先使用 F2 多轮模拟产物 users_sim.csv，否则兼容旧版 users_10k.csv。"""
    p_sim = ROOT_DIR / "F2" / "data" / "users_sim.csv"
    p_old = ROOT_DIR / "F2" / "data" / "users_10k.csv"
    return p_sim if p_sim.exists() else p_old


USERS_CSV = _default_users_csv()

# ── 配置 ──────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "users"
from pathlib import Path
MODEL_NAME = Path(__file__).parent.parent.parent / "models" / "all-mpnet-base-v2"
VECTOR_DIM      = 768
ENCODE_BATCH    = 256   # 每批编码多少条（显存/内存限制）
UPSERT_BATCH    = 500   # 每批写入 Qdrant 多少条


# ── 文本序列化 ─────────────────────────────────────────────────────────────────

def user_to_text(row: dict) -> str:
    """
    将用户兴趣序列化为一段文本供向量模型编码。

    权重设计：
      primary_field（interest_fields[0]）× 3   ← 核心兴趣，权重最高
      其余 interest_fields                × 2   ← 补充兴趣
      interest_tags                       × 1   ← 具体标签，权重最低
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

def load_users(path: Path) -> list[dict]:
    print(f"[加载] 读取用户数据：{path}")
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"[加载] 共 {len(rows):,} 个用户")
    return rows


# ── 索引构建 ──────────────────────────────────────────────────────────────────

def build_index(users: list[dict], client: QdrantClient) -> None:
    # 如已存在则重建（方便重新运行）
    if client.collection_exists(COLLECTION_NAME):
        print(f"[Qdrant] collection '{COLLECTION_NAME}' 已存在，删除重建 ...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"[Qdrant] collection '{COLLECTION_NAME}' 创建完成（余弦相似度，{VECTOR_DIM} 维）")

    # 批量编码
    print(f"\n[编码] 加载模型 {MODEL_NAME} ...")
    model = SentenceTransformer(str(MODEL_NAME), trust_remote_code=False)

    texts = [user_to_text(u) for u in users]
    print(f"[编码] 开始编码 {len(texts):,} 个用户，batch_size={ENCODE_BATCH}")
    print("       （首次运行需下载模型，约 400MB；此后从本地缓存加载）\n")

    vectors = model.encode(
        texts,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,   # 归一化后余弦相似度等价于点积，Qdrant 查询更快
    )

    # 分批写入 Qdrant
    print(f"\n[Qdrant] 写入 {len(users):,} 个用户向量 ...")
    for start in range(0, len(users), UPSERT_BATCH):
        batch_users   = users  [start : start + UPSERT_BATCH]
        batch_vectors = vectors[start : start + UPSERT_BATCH]

        points = [
            PointStruct(
                id=int(u["user_id"]),
                vector=vec.tolist(),
                payload={
                    "user_id":        int(u["user_id"]),
                    "primary_field":  u["primary_field"],
                    "interest_fields": json.loads(u.get("interest_fields") or "[]"),
                    "interest_tags":   json.loads(u.get("interest_tags")   or "[]"),
                },
            )
            for u, vec in zip(batch_users, batch_vectors)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  已写入 {min(start + UPSERT_BATCH, len(users)):,} / {len(users):,}")

    print(f"\n[完成] '{COLLECTION_NAME}' collection 共 {len(users):,} 条点")
    print(f"       Qdrant Dashboard：http://localhost:6333/dashboard")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F3 构建用户向量索引")
    p.add_argument("--qdrant-url", default="http://localhost:6333",
                   help="Qdrant 服务地址（默认 http://localhost:6333）")
    p.add_argument("--users-csv", type=Path, default=USERS_CSV,
                   help="用户数据 CSV 路径")
    return p.parse_args()


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    args = parse_args()

    if not args.users_csv.exists():
        print(f"[错误] 用户数据文件不存在：{args.users_csv}")
        print("       请先运行 F2/scripts/simulate_users.py")
        sys.exit(1)

    print("=" * 55)
    print("F3  构建用户向量索引")
    print("=" * 55)

    users  = load_users(args.users_csv)
    client = QdrantClient(url=args.qdrant_url)
    build_index(users, client)


if __name__ == "__main__":
    main()

"""
F3 查询相似用户
===============
给定一个 user_id，从 Qdrant 的 "users" collection 中找出兴趣最相近的 K 个用户，
打印结果并可选地将结果保存为 JSON。

运行前提：
  已运行 build_user_index.py 构建用户向量索引。

运行方式：
  python find_similar_users.py --user-id 42
  python find_similar_users.py --user-id 42 --top-k 5
  python find_similar_users.py --user-id 42 --out result.json
"""

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, HasIdCondition

# ── 路径与配置 ────────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).parent.parent.parent


def _default_users_csv() -> Path:
    p_sim = ROOT_DIR / "F2" / "data" / "users_sim.csv"
    p_old = ROOT_DIR / "F2" / "data" / "users_10k.csv"
    return p_sim if p_sim.exists() else p_old


USERS_CSV = _default_users_csv()

COLLECTION_NAME = "users"
MODEL_NAME      = "all-mpnet-base-v2"
DEFAULT_TOP_K   = 10


# ── 工具函数（与 build_user_index.py 保持一致）────────────────────────────────

def user_to_text(row: dict) -> str:
    """
    将用户兴趣序列化为编码文本：
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


def load_user(user_id: int, path: Path) -> Optional[dict]:
    """按 user_id 从 CSV 中读取单个用户，不存在返回 None。"""
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["user_id"]) == user_id:
                return row
    return None


# ── 核心查询 ──────────────────────────────────────────────────────────────────

def find_similar_users(
    user: dict,
    top_k: int,
    client: QdrantClient,
) -> list[dict]:
    """
    查找与给定用户兴趣最相近的 top_k 个用户，返回结构化结果列表。

    参数直接接收已加载的 user dict，避免重复读取 CSV。
    排除规则：用户自身通过 Qdrant must_not 过滤，不会出现在结果里。
    """
    user_id = int(user["user_id"])

    model    = SentenceTransformer(MODEL_NAME)
    user_vec = model.encode(
        [user_to_text(user)], normalize_embeddings=True
    )[0]

    # query_points 是 qdrant-client 1.7+ 推荐的新 API，替代已弃用的 search()
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=user_vec.tolist(),
        limit=top_k + 1,   # 多取 1 条，防止因过滤导致数量不足
        query_filter=Filter(
            must_not=[HasIdCondition(has_id=[user_id])]  # 排除自己
        ),
        with_payload=True,
        score_threshold=0.0,
    )
    hits = result.points

    return [
        {
            "rank":           rank,
            "user_id":        hit.payload["user_id"],
            "similarity":     round(hit.score, 4),
            "primary_field":  hit.payload["primary_field"],
            "interest_fields": hit.payload["interest_fields"],
            "interest_tags":   hit.payload["interest_tags"],
        }
        for rank, hit in enumerate(hits[:top_k], start=1)
    ]


# ── 输出格式化 ────────────────────────────────────────────────────────────────

def print_results(query_user: dict, results: list[dict]) -> None:
    ifields = json.loads(query_user.get("interest_fields") or "[]")
    itags   = json.loads(query_user.get("interest_tags")   or "[]")

    print(f"查询用户 user_id={query_user['user_id']}")
    print(f"  primary_field  : {query_user['primary_field']}")
    print(f"  interest_fields: {ifields}")
    print(f"  interest_tags  : {itags}")
    print()

    header = f"{'排名':<5} {'user_id':<10} {'相似度':<8} 兴趣领域"
    print(header)
    print("─" * 75)

    for r in results:
        fields_str = " | ".join(r["interest_fields"])
        print(f"  {r['rank']:<4} {r['user_id']:<10} {r['similarity']:<8} {fields_str}")
        print(f"       标签: {', '.join(r['interest_tags'])}")
        print()


# ── 入口 ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F3 查询相似用户")
    p.add_argument("--user-id",    type=int, required=True, help="目标用户 ID")
    p.add_argument("--top-k",      type=int, default=DEFAULT_TOP_K, help="返回几个相似用户")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--users-csv",  type=Path, default=USERS_CSV)
    p.add_argument("--out",        type=Path, default=None,
                   help="可选：将结果保存为 JSON 文件")
    return p.parse_args()


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    args = parse_args()

    client     = QdrantClient(url=args.qdrant_url)
    query_user = load_user(args.user_id, args.users_csv)

    if query_user is None:
        print(f"[错误] user_id={args.user_id} 不存在")
        sys.exit(1)

    print("=" * 55)
    print(f"F3  相似用户查询  (Top-{args.top_k})")
    print("=" * 55 + "\n")

    results = find_similar_users(query_user, args.top_k, client)
    print_results(query_user, results)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[保存] 结果已写入 {args.out}")


if __name__ == "__main__":
    main()

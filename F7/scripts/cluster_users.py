"""
F7 用户聚类分析（增强版）
==========================
将具有相似观看兴趣的用户聚成一团。

核心思想：
    "两个用户如果 interest_fields 和 interest_tags 相似，他们应该被归为同一类。"

实现流程：
    1. 从 Qdrant "users" collection 拉回所有用户的 768 维兴趣向量
       （由 F3/build_user_index.py 通过 SentenceTransformer 编码，已归一化）
    2. K-Means 聚类（默认 K=20 个簇，对应 20 个兴趣领域）
    3. 结果保存：
         F7/data/user_clusters.csv        —— user_id → cluster_id 标签表
         Qdrant "user_clusters" collection —— 向量 + 聚类标签
           （point ID = user_id 整数，便于后续直接按 user_id 检索）
         F7/data/cluster_visualization.png —— PCA 2D 聚类散点图
    4. 输出每簇的「代表性兴趣领域」统计

增强项（相比原版）：
    A. Qdrant point ID 改为 int(user_id)，而非存储序号
       → 可直接 client.retrieve("user_clusters", ids=[42]) 定位用户
    B. 新增 query_same_cluster()：给定 user_id，通过 Qdrant filter 检索同簇用户
    C. 新增 PCA 2D 可视化：保存聚类散点图 cluster_visualization.png
    D. 新增 --query-user CLI 参数：终端直接查看某用户的同簇邻居

运行前提：
    1. 已运行 F3/scripts/build_user_index.py（Qdrant "users" collection 已存在）
    2. Qdrant 已启动：docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

运行方式：
    python cluster_users.py                         # 聚类 + 写 Qdrant + 生成图
    python cluster_users.py --n-clusters 15         # 指定簇数
    python cluster_users.py --no-qdrant             # 不写回 Qdrant（仅 CSV）
    python cluster_users.py --query-user 42         # 查询用户 42 的同簇邻居
    python cluster_users.py --query-user 42 --top-k 10
"""

import argparse
import csv
import io
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── 路径 ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent.parent
OUT_DIR    = Path(__file__).parent.parent / "data"
OUT_CSV    = OUT_DIR / "user_clusters.csv"
OUT_VIZ    = OUT_DIR / "cluster_visualization.png"

# ── 默认参数 ──────────────────────────────────────────────────────────────────
N_CLUSTERS_DEFAULT  = 20
RANDOM_STATE        = 42
SRC_COLLECTION      = "users"           # F3 已建好的 collection
DST_COLLECTION      = "user_clusters"   # 本脚本写入的 collection
SCROLL_BATCH        = 1000
UPSERT_BATCH        = 500


# ── 步骤 1：从 Qdrant 拉回所有用户向量 ───────────────────────────────────────
def fetch_user_vectors(qdrant_url: str):
    """
    通过 Qdrant scroll API 分批拉取 "users" collection 中的所有点。
    返回 (向量矩阵 ndarray, payload 列表, client 实例)。

    使用 scroll 而非 search 的原因：
        search 需要提供查询向量（最近邻检索）；
        scroll 是全量扫描，适合需要所有数据的批处理场景（如聚类）。
    """
    from qdrant_client import QdrantClient

    print(f"[1/3] 从 Qdrant '{SRC_COLLECTION}' 拉取用户向量 ...")
    client = QdrantClient(url=qdrant_url)

    if not client.collection_exists(SRC_COLLECTION):
        print(f"[错误] Qdrant collection '{SRC_COLLECTION}' 不存在。")
        print("       请先运行 F3/scripts/build_user_index.py")
        sys.exit(1)

    all_vectors:  list[list[float]] = []
    all_payloads: list[dict]        = []
    offset = None

    while True:
        result, next_offset = client.scroll(
            collection_name=SRC_COLLECTION,
            limit=SCROLL_BATCH,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        if not result:
            break
        for pt in result:
            all_vectors.append(pt.vector)
            all_payloads.append(pt.payload or {})
        print(f"  已拉取 {len(all_vectors):,} 个用户 ...")
        if next_offset is None:
            break
        offset = next_offset

    vectors = np.array(all_vectors, dtype=np.float32)
    print(f"      共 {vectors.shape[0]:,} 个用户，向量维度 {vectors.shape[1]}")
    return vectors, all_payloads, client


# ── 步骤 2：K-Means 聚类 ─────────────────────────────────────────────────────
def kmeans_cluster(vectors: np.ndarray, n_clusters: int) -> tuple[np.ndarray, float]:
    """
    K-Means 聚类（init='k-means++' 智能初始化）。
    向量已由 F3 归一化（L2 norm=1），故欧氏距离等价于余弦距离。
    """
    print(f"[2/3] K-Means 聚类  K={n_clusters} ...")
    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    labels = km.fit_predict(vectors)

    sample_size = min(5000, len(vectors))
    idx = np.random.default_rng(RANDOM_STATE).choice(len(vectors), sample_size, replace=False)
    sil = silhouette_score(vectors[idx], labels[idx], metric="euclidean")
    print(f"      Silhouette Score（采样 {sample_size:,}）: {sil:.4f}")

    dist = np.bincount(labels, minlength=n_clusters)
    print(f"      簇大小  min={dist.min()}  max={dist.max()}  mean={dist.mean():.1f}")
    return labels, sil


# ── 步骤 3a：保存 CSV + 输出簇统计 ───────────────────────────────────────────
def save_csv(
    payloads: list[dict],
    labels: np.ndarray,
    n_clusters: int,
    out_path: Path,
) -> None:
    """保存 user_id → cluster_id 映射，并打印每簇最具代表性的 primary_field。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    cluster_fields: dict[int, list[str]] = {i: [] for i in range(n_clusters)}

    for payload, lbl in zip(payloads, labels.tolist()):
        uid = payload.get("user_id", "")
        pf  = payload.get("primary_field", "")
        rows.append({"user_id": uid, "cluster_id": lbl, "primary_field": pf})
        if pf:
            cluster_fields[lbl].append(pf)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "cluster_id", "primary_field"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[保存] 用户聚类标签 → {out_path}  ({len(rows):,} 行)")

    print("\n  簇代表性兴趣领域（top primary_field）：")
    for cid in range(n_clusters):
        if cluster_fields[cid]:
            top = Counter(cluster_fields[cid]).most_common(1)[0]
            print(f"    簇 {cid:>2}  ({len(cluster_fields[cid]):>4} 人)  → {top[0]}（{top[1]} 人）")
        else:
            print(f"    簇 {cid:>2}  ( 0 人)")


# ── 步骤 3b：写回 Qdrant（增强：用 user_id 作为 point ID）───────────────────
def save_to_qdrant(
    payloads: list[dict],
    labels: np.ndarray,
    vectors: np.ndarray,
    client,
) -> None:
    """
    将用户向量 + 聚类标签写入 Qdrant "user_clusters" collection。

    重要设计：point ID = int(user_id)（而非序号），使得：
        client.retrieve("user_clusters", ids=[42])  可直接定位用户 42
        query_same_cluster() 无需全量扫描即可按用户 ID 查询

    后续 Streamlit 界面、推荐引擎等可高效调用此 collection。
    """
    from qdrant_client.models import Distance, VectorParams, PointStruct

    print(f"\n[Qdrant] 写入 collection '{DST_COLLECTION}' ...")
    dim = vectors.shape[1]

    if client.collection_exists(DST_COLLECTION):
        client.delete_collection(DST_COLLECTION)
    client.create_collection(
        collection_name=DST_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"[Qdrant] collection '{DST_COLLECTION}' 创建完成（{dim} 维，point ID = user_id）")

    for start in range(0, len(payloads), UPSERT_BATCH):
        end    = min(start + UPSERT_BATCH, len(payloads))
        points = [
            PointStruct(
                id=int(payloads[start + i]["user_id"]),    # 用 user_id 作为 point ID
                vector=vectors[start + i].tolist(),
                payload={
                    **payloads[start + i],
                    "cluster_id": int(labels[start + i]),
                },
            )
            for i in range(end - start)
        ]
        client.upsert(collection_name=DST_COLLECTION, points=points)
        print(f"  已写入 {end:,} / {len(payloads):,}")

    print(f"[Qdrant] 完成。collection '{DST_COLLECTION}' 已更新。")


# ── 步骤 3c：PCA 2D 可视化 ────────────────────────────────────────────────────
def save_visualization(
    vectors: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    out_path: Path,
) -> None:
    """
    将 768 维用户兴趣向量 PCA 降至 2D，按聚类结果着色，保存散点图。

    为什么用 PCA 而不是 t-SNE / UMAP：
        PCA 是线性变换，速度快（万级数据秒级完成）；
        t-SNE/UMAP 能更好保留局部结构，但耗时且需额外依赖；
        此处目的是给聚类结果提供直观可视化，PCA 已足够。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # 非交互后端，适合服务器/无 GUI 环境
        import matplotlib.pyplot as plt
    except ImportError:
        print("[可视化] 未安装 matplotlib，跳过。")
        print("         pip install matplotlib 后重新运行即可生成聚类图。")
        return

    from sklearn.decomposition import PCA

    print("\n[可视化] PCA 降维 → 2D 聚类散点图 ...")
    sample_size = min(5000, len(vectors))
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(vectors), sample_size, replace=False)

    pca    = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(vectors[idx])
    clabels = labels[idx]

    explained = pca.explained_variance_ratio_.sum() * 100
    cmap      = plt.get_cmap("tab20", n_clusters)

    fig, ax = plt.subplots(figsize=(11, 9))
    for c in range(n_clusters):
        mask = clabels == c
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(c)], label=f"Cluster {c}",
            alpha=0.5, s=8,
        )

    ax.set_title(
        f"F7 User Cluster Visualization (PCA)\n"
        f"K={n_clusters}  Explained Variance={explained:.1f}%  "
        f"Sample={sample_size:,}",
        fontsize=12,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="upper right", fontsize=6, ncol=2, markerscale=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[可视化] 已保存 → {out_path}")


# ── 查询同簇用户（增强功能）─────────────────────────────────────────────────
def query_same_cluster(
    user_id: int,
    top_k: int,
    qdrant_url: str,
) -> list[dict]:
    """
    从 Qdrant "user_clusters" collection 中找与指定用户同簇的 Top-K 近邻。

    流程：
      1. 以 user_id 为 point ID 直接检索目标用户的向量和 cluster_id
         （这就是为什么写入时要用 user_id 作为 point ID 的原因）
      2. 在同簇内（filter: cluster_id == target_cluster_id）执行向量近邻搜索
      3. 排除用户自身，返回 Top-K 结果

    设计说明：
        使用 Qdrant filter + query 的组合查询，既保证「同簇」约束，
        又在簇内按向量相似度排序，比随机取同簇用户更有语义意义。
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = QdrantClient(url=qdrant_url)

    if not client.collection_exists(DST_COLLECTION):
        print(f"[错误] collection '{DST_COLLECTION}' 不存在。请先运行聚类脚本。")
        return []

    # 直接按 user_id 取目标用户的向量和聚类标签
    results = client.retrieve(
        collection_name=DST_COLLECTION,
        ids=[user_id],
        with_vectors=True,
        with_payload=True,
    )
    if not results:
        print(f"[错误] user_id={user_id} 不在 '{DST_COLLECTION}' 中，请先运行聚类。")
        return []

    target     = results[0]
    cluster_id = target.payload["cluster_id"]
    user_vec   = target.vector

    print(f"用户 user_id={user_id} 所在簇: cluster_{cluster_id}")
    print(f"  primary_field: {target.payload.get('primary_field', 'N/A')}")

    # 同簇内向量近邻搜索（filter 限定 cluster_id）
    hits = client.query_points(
        collection_name=DST_COLLECTION,
        query=user_vec,
        query_filter=Filter(
            must=[FieldCondition(key="cluster_id", match=MatchValue(value=cluster_id))]
        ),
        limit=top_k + 1,    # +1 避免自身占位
        with_payload=True,
    ).points

    neighbors = [
        {
            "user_id":       h.payload.get("user_id"),
            "cluster_id":    h.payload.get("cluster_id"),
            "primary_field": h.payload.get("primary_field"),
            "similarity":    round(h.score, 4),
        }
        for h in hits
        if int(h.payload.get("user_id", -1)) != user_id
    ][:top_k]

    print(f"\n  同簇近邻 Top-{len(neighbors)}：")
    print(f"  {'user_id':>8}  {'cluster':>7}  {'similarity':>10}  primary_field")
    print(f"  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*30}")
    for nb in neighbors:
        print(f"  {nb['user_id']:>8}  {nb['cluster_id']:>7}  "
              f"{nb['similarity']:>10.4f}  {nb['primary_field']}")

    return neighbors


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F7 用户聚类分析（增强版）")
    p.add_argument("--n-clusters",  type=int,  default=N_CLUSTERS_DEFAULT,
                   help=f"K-Means 簇数（默认 {N_CLUSTERS_DEFAULT}）")
    p.add_argument("--out-csv",     type=Path, default=OUT_CSV)
    p.add_argument("--no-qdrant",   action="store_true",
                   help="不写回 Qdrant（仅保存 CSV，也不生成可视化）")
    p.add_argument("--no-viz",      action="store_true",
                   help="不生成 PCA 可视化图")
    p.add_argument("--qdrant-url",  default="http://localhost:6333")
    p.add_argument("--query-user",  type=int, default=None,
                   help="查询指定 user_id 的同簇近邻（需先运行聚类写入 Qdrant）")
    p.add_argument("--top-k",       type=int, default=10,
                   help="--query-user 时返回的近邻数量（默认 10）")
    return p.parse_args()


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    args = parse_args()

    # ── 仅查询模式（跳过聚类，直接查 Qdrant） ────────────────────────────── #
    if args.query_user is not None:
        print("=" * 60)
        print(f"F7 同簇用户查询  user_id={args.query_user}  Top-{args.top_k}")
        print("=" * 60 + "\n")
        query_same_cluster(args.query_user, args.top_k, args.qdrant_url)
        return

    # ── 聚类模式 ──────────────────────────────────────────────────────────── #
    print("=" * 60)
    print(f"F7 用户聚类分析（增强版）  K={args.n_clusters}")
    print("=" * 60)

    # 1. 从 Qdrant 拉取用户向量
    vectors, payloads, client = fetch_user_vectors(args.qdrant_url)

    # 2. K-Means 聚类
    labels, sil = kmeans_cluster(vectors, args.n_clusters)

    # 3. 保存
    print(f"\n[3/3] 保存结果 ...")
    save_csv(payloads, labels, args.n_clusters, args.out_csv)

    if not args.no_qdrant:
        save_to_qdrant(payloads, labels, vectors, client)
        if not args.no_viz:
            save_visualization(vectors, labels, args.n_clusters, OUT_VIZ)
    else:
        print("[Qdrant] 已跳过（--no-qdrant）。PCA 可视化同样跳过。")

    print(f"\n完成。Silhouette Score = {sil:.4f}")
    print(f"  用户聚类标签 → {args.out_csv}")
    if not args.no_qdrant and not args.no_viz:
        print(f"  聚类可视化   → {OUT_VIZ}")
    print(f"\n提示：使用 --query-user <user_id> 可查询某用户的同簇近邻。")


if __name__ == "__main__":
    main()

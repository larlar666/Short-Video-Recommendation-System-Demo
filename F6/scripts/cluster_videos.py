"""
F6 视频聚类分析
================
将具有相似观看用户群体的视频聚成一团（协同过滤视角）。

核心思想：
    "两个视频如果经常被同一批用户观看，它们就应该被归为同一类。"

实现流程：
    1. 读取 F2/data/watch_events.csv，构造 视频 × 用户 共现矩阵（稀疏）
    2. TF-IDF 加权：高频被观看的视频降权（类比文档中的高频词），
       突出「特定用户群独有偏好」的视频
    3. TruncatedSVD 降维到 SVD_DIM 维（默认 64），提取潜在观众画像
    4. K-Means 聚类（默认 K=20 个簇，对应 20 个兴趣领域）
    5. 结果写入：
         F6/data/video_clusters.csv  —— video_id → cluster_id 标签表
         Qdrant "video_clusters" collection —— 64 维行为向量 + 聚类标签
           （Qdrant 存储便于后续查询「与某视频同簇的其他视频」）

运行方式：
    python cluster_videos.py
    python cluster_videos.py --n-clusters 30 --svd-dim 128
    python cluster_videos.py --no-qdrant    # 不写入 Qdrant（仅输出 CSV）

前置条件：
    已运行 F2/scripts/simulate_users.py 生成 watch_events.csv
    若使用 --qdrant：Qdrant 已启动
        docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
"""

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# ── 路径 ─────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent.parent.parent
EVENTS_FILE  = ROOT_DIR / "F2" / "data" / "watch_events.csv"
VIDEOS_CSV   = ROOT_DIR / "F1" / "data" / "tiktok_meta_100k.csv"
OUT_DIR      = Path(__file__).parent.parent / "data"
OUT_CSV      = OUT_DIR / "video_clusters.csv"

# ── 默认参数 ──────────────────────────────────────────────────────────────────
N_CLUSTERS_DEFAULT = 20   # 聚类数（与 20 个预定义领域对应，可调）
SVD_DIM_DEFAULT    = 64   # TruncatedSVD 降维后的维度
RANDOM_STATE       = 42
QDRANT_COLLECTION  = "video_clusters"
UPSERT_BATCH       = 500
CHUNK_SIZE         = 500_000  # 分块读取 watch_events.csv，控制内存峰值


# ── 步骤 1：构造视频-用户共现矩阵 ─────────────────────────────────────────────
def build_cooccurrence_matrix(
    events_path: Path,
) -> tuple[csr_matrix, list[str], list[int]]:
    """
    读取 watch_events.csv，只保留 event_type == 'watch'，
    构造 视频 × 用户 稀疏矩阵 M，其中 M[i,j] = 用户 j 观看视频 i 的次数。

    返回:
        matrix     — 稀疏矩阵 (n_videos, n_users)
        video_ids  — 行顺序对应的 video_id 列表
        user_ids   — 列顺序对应的 user_id 列表
    """
    print(f"[1/4] 读取事件表（chunksize={CHUNK_SIZE:,}）: {events_path}")
    dtype = {"round": "int16", "user_id": "int32", "video_id": "str", "event_type": "str"}

    vid_arrays: list[np.ndarray] = []
    uid_arrays: list[np.ndarray] = []
    n_total = 0
    n_watch = 0

    for chunk in pd.read_csv(events_path, dtype=dtype, chunksize=CHUNK_SIZE):
        n_total += len(chunk)
        w = chunk[chunk["event_type"] == "watch"]
        n_watch += len(w)
        vid_arrays.append(w["video_id"].values)
        uid_arrays.append(w["user_id"].values)

    all_vid = np.concatenate(vid_arrays)
    all_uid = np.concatenate(uid_arrays)
    del vid_arrays, uid_arrays

    video_ids = sorted(set(all_vid.tolist()))
    user_ids  = sorted(set(all_uid.tolist()))
    print(f"      事件总行数: {n_total:,}  watch 事件: {n_watch:,}  "
          f"视频数: {len(video_ids):,}  用户数: {len(user_ids):,}")

    vid2idx = {v: i for i, v in enumerate(video_ids)}
    uid2idx = {u: i for i, u in enumerate(user_ids)}

    row  = np.array([vid2idx[v] for v in all_vid], dtype=np.int32)
    col  = np.array([uid2idx[u] for u in all_uid], dtype=np.int32)
    data = np.ones(len(all_vid), dtype=np.float32)
    del all_vid, all_uid

    matrix = csr_matrix(
        (data, (row, col)),
        shape=(len(video_ids), len(user_ids)),
        dtype=np.float32,
    )
    print(f"      共现矩阵: {matrix.shape[0]:,} 视频 × {matrix.shape[1]:,} 用户  "
          f"非零元素: {matrix.nnz:,}  稀疏度: {matrix.nnz / (matrix.shape[0]*matrix.shape[1]):.4%}")
    return matrix, video_ids, user_ids


# ── 步骤 2：TF-IDF 加权 ───────────────────────────────────────────────────────
def tfidf_weight(matrix: csr_matrix) -> csr_matrix:
    """
    对共现矩阵做 TF-IDF 加权（视频为"文档"，用户为"词"）：
        TF(视频 i, 用户 j)  = 用户 j 观看视频 i 的次数（本实现中均为 1）
        IDF(用户 j)          = log(总视频数 / 有用户 j 的视频数 + 1)

    作用：降低「广泛出现的大众用户」的权重，
          放大「只出现在特定视频群体中的小众用户」的信号。
    这样聚类时，「小众用户群共有偏好」会成为更强的分组依据。
    """
    print("[1.5] TF-IDF 加权（突出特定用户群信号）...")
    n_videos = matrix.shape[0]

    # 每个用户出现在多少个视频里（列方向求和 → 转为行向量）
    doc_freq = np.diff(matrix.tocsc().indptr)   # shape (n_users,)
    idf      = np.log1p(n_videos / (doc_freq + 1)).astype(np.float32)

    # 原地乘以 IDF（列缩放）
    from scipy.sparse import diags
    weighted = matrix.dot(diags(idf))   # (n_videos, n_users) × diag(IDF)
    return weighted


# ── 步骤 3：TruncatedSVD 降维 ────────────────────────────────────────────────
def reduce_dimensions(matrix: csr_matrix, svd_dim: int) -> np.ndarray:
    """
    TruncatedSVD（截断奇异值分解）将高维稀疏向量降到 svd_dim 维稠密向量。

    选择 TruncatedSVD 而非 PCA 的原因：
        PCA 需要先中心化，会破坏稀疏结构（产生全量密集矩阵，内存爆炸）；
        TruncatedSVD 直接在稀疏矩阵上操作，等价于 LSA（潜在语义分析）。

    降维后做 L2 归一化，使后续 K-Means 等价于基于余弦相似度的聚类。
    """
    print(f"[2/4] TruncatedSVD 降维: {matrix.shape[1]} → {svd_dim} 维 ...")
    svd = TruncatedSVD(n_components=svd_dim, random_state=RANDOM_STATE)
    reduced = svd.fit_transform(matrix)
    explained = svd.explained_variance_ratio_.sum()
    print(f"      前 {svd_dim} 个奇异值解释方差: {explained:.3%}")

    # L2 归一化：使余弦距离等价于欧氏距离，K-Means 效果更稳定
    reduced = normalize(reduced, norm="l2")
    return reduced


# ── 步骤 4：K-Means 聚类 ─────────────────────────────────────────────────────
def kmeans_cluster(
    vectors: np.ndarray, n_clusters: int
) -> tuple[np.ndarray, float]:
    """
    K-Means 聚类，返回 (labels, silhouette_score)。

    Silhouette Score（轮廓系数）衡量聚类质量：
        越接近 1 表示簇内紧凑、簇间分离好；
        接近 0 表示边界模糊；负值表示点可能被分错了簇。
    """
    print(f"[3/4] K-Means 聚类  K={n_clusters} ...")
    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",   # 智能初始化，避免随机初始化的局部最优
        n_init=10,
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    labels = km.fit_predict(vectors)

    # 轮廓系数（大数据集用采样以控制耗时）
    sample_size = min(5000, len(vectors))
    idx = np.random.default_rng(RANDOM_STATE).choice(len(vectors), sample_size, replace=False)
    sil = silhouette_score(vectors[idx], labels[idx], metric="euclidean")
    print(f"      Silhouette Score（采样 {sample_size:,}）: {sil:.4f}")

    dist = np.bincount(labels, minlength=n_clusters)
    print(f"      簇大小  min={dist.min()}  max={dist.max()}  mean={dist.mean():.1f}")
    return labels, sil


# ── 步骤 5a：保存 CSV ─────────────────────────────────────────────────────────
def save_csv(
    video_ids: list[str],
    labels: np.ndarray,
    vectors: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "cluster_id"])
        for vid, lbl in zip(video_ids, labels.tolist()):
            writer.writerow([vid, lbl])
    print(f"[保存] 聚类标签 → {out_path}  ({len(video_ids):,} 行)")


# ── 步骤 5b：写入 Qdrant ──────────────────────────────────────────────────────
def save_to_qdrant(
    video_ids: list[str],
    labels: np.ndarray,
    vectors: np.ndarray,
    qdrant_url: str,
) -> None:
    """
    将视频行为向量（svd_dim 维）和聚类标签写入 Qdrant "video_clusters" collection。
    后续可用「给定 video_id，找出同簇或相邻簇视频」进行协同过滤推荐。
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
    except ImportError:
        print("[警告] qdrant_client 未安装，跳过 Qdrant 写入。")
        return

    print(f"\n[Qdrant] 写入 collection '{QDRANT_COLLECTION}' ...")
    client = QdrantClient(url=qdrant_url)
    dim    = vectors.shape[1]

    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"[Qdrant] collection '{QDRANT_COLLECTION}' 创建完成（{dim} 维行为向量）")

    # 建立 video_id → int point_id 的映射（Qdrant id 要求整数或 UUID）
    for start in range(0, len(video_ids), UPSERT_BATCH):
        end    = min(start + UPSERT_BATCH, len(video_ids))
        points = [
            PointStruct(
                id=start + i,
                vector=vectors[start + i].tolist(),
                payload={
                    "video_id":   video_ids[start + i],
                    "cluster_id": int(labels[start + i]),
                },
            )
            for i in range(end - start)
        ]
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"  已写入 {end:,} / {len(video_ids):,}")

    print(f"[Qdrant] 完成。可在 {qdrant_url}/dashboard 查看 '{QDRANT_COLLECTION}'")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F6 视频聚类分析")
    p.add_argument("--events",      type=Path, default=EVENTS_FILE)
    p.add_argument("--out-csv",     type=Path, default=OUT_CSV)
    p.add_argument("--n-clusters",  type=int,  default=N_CLUSTERS_DEFAULT,
                   help=f"K-Means 簇数（默认 {N_CLUSTERS_DEFAULT}）")
    p.add_argument("--svd-dim",     type=int,  default=SVD_DIM_DEFAULT,
                   help=f"SVD 降维维度（默认 {SVD_DIM_DEFAULT}）")
    p.add_argument("--no-qdrant",   action="store_true", help="不写入 Qdrant")
    p.add_argument("--qdrant-url",  default="http://localhost:6333")
    return p.parse_args()


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    args = parse_args()

    print("=" * 60)
    print(f"F6 视频聚类分析  K={args.n_clusters}  SVD={args.svd_dim}")
    print("=" * 60)

    if not args.events.exists():
        print(f"[错误] 事件表不存在: {args.events}")
        sys.exit(1)

    # 1. 共现矩阵
    matrix, video_ids, user_ids = build_cooccurrence_matrix(args.events)

    # 1.5 TF-IDF 加权
    matrix = tfidf_weight(matrix)

    # 2. 降维
    vectors = reduce_dimensions(matrix, args.svd_dim)

    # 3. 聚类
    labels, sil = kmeans_cluster(vectors, args.n_clusters)

    # 4. 保存
    print(f"\n[4/4] 保存结果 ...")
    save_csv(video_ids, labels, vectors, args.out_csv)

    if not args.no_qdrant:
        save_to_qdrant(video_ids, labels, vectors, args.qdrant_url)

    print(f"\n完成。Silhouette Score = {sil:.4f}")
    print(f"  聚类标签 → {args.out_csv}")


if __name__ == "__main__":
    main()

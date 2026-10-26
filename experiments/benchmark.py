"""Reproducible retrieval and clustering comparisons. See experiments/README.md."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from importlib.metadata import version

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def check_file(path):
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Missing data: {path}")
    with path.open("rb") as handle:
        if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Git LFS pointer instead of data: {path}. Run git lfs pull.")
    return path


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(Path(path).resolve()), "sha256": h.hexdigest()}


def as_list(value):
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    return parsed if isinstance(parsed, list) else []


def load_users(path):
    frame = pd.read_csv(check_file(path))
    frame = frame.sort_values("user_id").reset_index(drop=True)
    if frame.user_id.duplicated().any():
        raise ValueError("Duplicate user IDs")
    features = []
    for row in frame.to_dict("records"):
        weights = {}
        for token, weight in ([(str(row["primary_field"]), 3)]
                              + [(str(t), 2) for t in as_list(row["interest_fields"])]
                              + [(str(t), 1) for t in as_list(row["interest_tags"]) if str(t).isascii()
                                 and str(t).replace(" ", "").isalnum()]):
            weights[token] = weights.get(token, 0) + weight
        features.append(weights)
    matrix = DictVectorizer(dtype=np.float32).fit_transform(features)
    return normalize(matrix).toarray(), frame.primary_field.astype(str).to_numpy(), frame.user_id.to_numpy()


def fetch_qdrant(url, collection):
    from qdrant_client import QdrantClient
    client = QdrantClient(url=url, timeout=60)
    points, offset = [], None
    while True:
        batch, offset = client.scroll(collection, limit=1000, offset=offset,
                                      with_vectors=True, with_payload=True)
        points.extend(batch)
        if offset is None:
            break
    if not points or isinstance(points[0].vector, dict):
        raise ValueError("Expected a nonempty collection with one unnamed dense vector per point")
    # Keep identical vectors/IDs for exhaustive, cluster-pruned and remote searches.
    points.sort(key=lambda p: str(p.id))
    x = normalize(np.asarray([p.vector for p in points], dtype=np.float32))
    if not np.all(np.linalg.norm(x, axis=1) > 0):
        raise ValueError("Zero vectors cannot be evaluated with cosine similarity")
    fields = np.array([str((p.payload or {}).get("primary_field", "")) for p in points])
    ids = [p.id for p in points]
    digest = hashlib.sha256(x.tobytes() + json.dumps(ids).encode()).hexdigest()
    return x, fields, ids, client, digest


def topk(scores, candidates, k):
    """Stable tie policy: higher similarity, then smaller corpus row index."""
    candidates = np.asarray(candidates, dtype=np.int64)
    scores = np.asarray(scores)
    if len(candidates) == 0:
        return candidates
    # Partial selection, including all ties at the cutoff before deterministic sorting.
    count = min(k, len(candidates))
    cutoff = np.partition(scores, len(scores) - count)[len(scores) - count]
    keep = np.flatnonzero(scores >= cutoff)
    order = np.lexsort((candidates[keep], -scores[keep]))[:count]
    return candidates[keep[order]]


def exact_search(x, query, k):
    scores = x @ x[query]
    candidates = np.arange(len(x))
    mask = candidates != query
    return topk(scores[mask], candidates[mask], k)


def cluster_search(x, query, k, centers, inverted, probes):
    # Squared Euclidean distance matches ordinary K-Means assignments; centers are not unit vectors.
    distances = np.sum((centers - x[query]) ** 2, axis=1)
    selected = np.argsort(distances, kind="stable")[:probes]
    candidates = np.concatenate([inverted[c] for c in selected])
    candidates = candidates[candidates != query]
    return topk(x[candidates] @ x[query], candidates, k), len(candidates)


def recall_with_ties(x, query, result, truth, k):
    # Any neighbor tied with the exact K-th score is an equally valid result.
    threshold = float(x[truth[-1]] @ x[query])
    values = x[result] @ x[query]
    return float(np.count_nonzero(values >= threshold - 1e-6) / k)


def benchmark_search(x, fields, ids, args, client=None):
    rng = np.random.default_rng(args.seed)
    queries = rng.choice(len(x), size=min(args.queries, len(x)), replace=False)
    k = min(args.top_k, len(x) - 1)
    start = perf_counter()
    km = KMeans(n_clusters=args.clusters, n_init=10, random_state=args.seed).fit(x)
    inverted = [np.flatnonzero(km.labels_ == c) for c in range(args.clusters)]
    build_s = perf_counter() - start
    probes = sorted(set(min(args.clusters, p) for p in [1, 3, 5, args.clusters]))
    methods = ["exhaustive"] + [f"cluster_probe_{p}" for p in probes]
    if client is not None:
        methods.append("qdrant")
    raw = []
    id_rows = {str(v): i for i, v in enumerate(ids)}

    def search(method, query):
        if method == "exhaustive":
            return exact_search(x, query, k), len(x) - 1
        if method == "qdrant":
            from qdrant_client.models import Filter, HasIdCondition, SearchParams
            hits = client.query_points(
                args.collection, query=x[query].tolist(), limit=k, with_payload=False,
                query_filter=Filter(must_not=[HasIdCondition(has_id=[ids[query]])]),
                search_params=SearchParams(hnsw_ef=args.hnsw_ef, exact=False),
            ).points
            return np.array([id_rows[str(h.id)] for h in hits], dtype=np.int64), None
        return cluster_search(x, query, k, km.cluster_centers_, inverted,
                              int(method.rsplit("_", 1)[1]))

    # Warm up every path; no encoding, disk reads, network collection download or fitting in query timers.
    for method in methods:
        search(method, int(queries[0]))
    for query in queries:
        query = int(query)
        truth = exact_search(x, query, k)
        for repeat in range(args.repeats):
            for method in rng.permutation(methods):
                start = perf_counter()
                result, candidates = search(method, query)
                elapsed_ms = (perf_counter() - start) * 1000
                raw.append({"method": method, "query_row": query, "repeat": repeat,
                            "query_ms": elapsed_ms, "candidates": candidates,
                            "recall_at_k": len(set(result) & set(truth)) / k,
                            "tie_aware_recall_at_k": recall_with_ties(x, query, result, truth, k),
                            "same_field_precision_at_k": (float(np.count_nonzero(fields[result] == fields[query]) / k)
                                                          if fields[query] else None),
                            "returned": len(result)})
    detail = pd.DataFrame(raw)
    summary = detail.groupby("method", sort=False).agg(
        mean_ms=("query_ms", "mean"), p50_ms=("query_ms", "median"),
        p95_ms=("query_ms", lambda v: np.percentile(v, 95)),
        recall_at_k=("recall_at_k", "mean"), tie_aware_recall_at_k=("tie_aware_recall_at_k", "mean"),
        same_field_precision_at_k=("same_field_precision_at_k", "mean"),
        mean_candidates=("candidates", "mean"), mean_returned=("returned", "mean"),
    ).reset_index()
    baseline_ms = float(summary.loc[summary.method == "exhaustive", "mean_ms"].iloc[0])
    summary["speedup"] = baseline_ms / summary.mean_ms
    summary["index_build_s"] = [build_s if m.startswith("cluster_") else 0 if m == "exhaustive" else np.nan
                                for m in summary.method]
    summary["break_even_queries"] = [build_s / ((baseline_ms - row.mean_ms) / 1000)
                                      if row.method.startswith("cluster_") and row.mean_ms < baseline_ms else np.nan
                                      for row in summary.itertuples()]
    return summary, detail, {"n": len(x), "dimensions": x.shape[1], "queries": len(queries),
                             "top_k": k, "cluster_index_build_s": build_s,
                             "qdrant_query_config": {"hnsw_ef": args.hnsw_ef, "exact": False} if client else None}


def load_video_matrix(path, limit, seed):
    path = check_file(path)
    columns = ["video_id", "user_id", "event_type"]
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        batches = (batch.to_pandas() for batch in pq.ParquetFile(path).iter_batches(
            batch_size=500_000, columns=columns))
    else:
        batches = pd.read_csv(path, usecols=columns, dtype={"video_id": str}, chunksize=500_000)
    pairs = []
    for chunk in batches:
        pairs.append(chunk.loc[chunk.event_type == "watch", ["video_id", "user_id"]])
    frame = pd.concat(pairs, ignore_index=True)
    videos = np.sort(frame.video_id.astype(str).unique())
    if limit and len(videos) > limit:
        videos = np.sort(np.random.default_rng(seed).choice(videos, limit, replace=False))
    frame = frame.loc[frame.video_id.astype(str).isin(videos)]
    rows = pd.Categorical(frame.video_id.astype(str), categories=videos).codes
    cols, users = pd.factorize(frame.user_id, sort=True)
    matrix = sparse.csr_matrix((np.ones(len(frame), dtype=np.float32), (rows, cols)),
                               shape=(len(videos), len(users)))
    # Same weighting formula as F6; both pipelines receive this identical weighted input.
    df = np.diff(matrix.tocsc().indptr)
    weighted = matrix @ sparse.diags(np.log1p(len(videos) / (df + 1)).astype(np.float32))
    return normalize(weighted).tocsr(), weighted.tocsr(), {"n": len(videos), "users": len(users),
                                                         "nnz": matrix.nnz, "sample_limit": limit}


def benchmark_clusters(x, svd_input, fields, args, task):
    rng = np.random.default_rng(args.seed)
    eval_rows = rng.choice(x.shape[0], min(args.eval_samples, x.shape[0]), replace=False)
    eval_x = x[eval_rows]  # A fixed original space for ALL methods, never each method's own embedding.
    rows = []
    for repeat in range(args.cluster_repeats):
        seed = args.seed + repeat
        methods = ["kmeans_random", "kmeans_plus_plus", "svd_kmeans_plus_plus"]
        for method in rng.permutation(methods):
            start = perf_counter()
            if method.startswith("svd_"):
                dim = min(args.svd_dim, x.shape[0] - 1, x.shape[1] - 1)
                svd = TruncatedSVD(n_components=dim, random_state=seed)
                features = normalize(svd.fit_transform(svd_input))
                explained = float(svd.explained_variance_ratio_.sum())
            else:
                features, explained = x, None
            prep_s = perf_counter() - start
            start = perf_counter()
            km = KMeans(n_clusters=args.clusters, init="random" if method == "kmeans_random" else "k-means++",
                        n_init=10, max_iter=300, random_state=seed)
            labels = km.fit_predict(features)
            fit_s = perf_counter() - start
            eval_labels = labels[eval_rows]
            unique = len(np.unique(eval_labels))
            sil = float(silhouette_score(eval_x, eval_labels, metric="cosine")) if 1 < unique < len(eval_rows) else None
            memory_bytes = (features.data.nbytes + features.indices.nbytes + features.indptr.nbytes
                            if sparse.issparse(features) else features.nbytes)
            rows.append({"task": task, "method": method, "repeat": repeat, "seed": seed,
                         "n": x.shape[0], "dimensions": features.shape[1], "preprocess_s": prep_s,
                         "fit_s": fit_s, "total_s": prep_s + fit_s, "iterations": km.n_iter_,
                         "silhouette_original_cosine": sil, "feature_storage_mib": memory_bytes / 2**20,
                         "explained_variance": explained,
                         "ari_primary_field": float(adjusted_rand_score(fields, labels)) if fields is not None else None,
                         "nmi_primary_field": float(normalized_mutual_info_score(fields, labels)) if fields is not None else None})
            print(f"{task} {method} repeat={repeat}: {prep_s + fit_s:.3f}s silhouette={sil}", flush=True)
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["task", "method"], sort=False).agg(
        mean_total_s=("total_s", "mean"), std_total_s=("total_s", "std"), mean_fit_s=("fit_s", "mean"),
        mean_preprocess_s=("preprocess_s", "mean"), silhouette_original_cosine=("silhouette_original_cosine", "mean"),
        ari_primary_field=("ari_primary_field", "mean"), nmi_primary_field=("nmi_primary_field", "mean"),
        feature_storage_mib=("feature_storage_mib", "mean"), n=("n", "first"), dimensions=("dimensions", "first"),
    ).reset_index()
    base = float(summary.loc[summary.method == "kmeans_plus_plus", "mean_total_s"].iloc[0])
    summary["speedup_vs_plus_plus"] = base / summary.mean_total_s
    return summary, detail


def markdown_table(frame):
    # No optional tabulate dependency.
    def cell(v):
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else "—" if pd.isna(v) else str(v)
    return "| " + " | ".join(frame.columns) + " |\n| " + " | ".join(["---"] * len(frame.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["local", "qdrant"], default="local")
    p.add_argument("--users", type=Path, default=ROOT / "F2/data/users_sim.csv")
    p.add_argument("--events", type=Path, default=ROOT / "F2/data/watch_events.parquet")
    p.add_argument("--tasks", nargs="+", choices=["search", "users", "videos"], default=["search", "users", "videos"])
    p.add_argument("--out", type=Path, default=ROOT / "experiments/results/local")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--collection", default="users")
    p.add_argument("--hnsw-ef", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clusters", type=int, default=20)
    p.add_argument("--queries", type=int, default=100)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cluster-repeats", type=int, default=3)
    p.add_argument("--video-limit", type=int, default=10000, help="0 means all videos")
    p.add_argument("--eval-samples", type=int, default=1500)
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    for key in ["clusters", "queries", "top_k", "repeats", "cluster_repeats", "eval_samples", "svd_dim", "threads", "hnsw_ef"]:
        if getattr(args, key) < 1:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if args.video_limit < 0 or args.eval_samples < 3 or args.clusters < 2:
        p.error("Require video-limit >= 0, eval-samples >= 3, clusters >= 2")
    metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "source": args.source,
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "environment": {"python": sys.version, "platform": platform.platform(), "cpu": platform.processor(),
                                "packages": {m: version(m) for m in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]}},
                "inputs": [], "notes": ["Local vectors are weighted tags, not MPNet semantic embeddings.",
                    "Same-field precision and ARI/NMI are simulation label proxies, not real recommendation accuracy.",
                    "Silhouette uses fixed sampled rows in original normalized space with cosine distance.",
                    "Timings exclude data loading and evaluation; SVD time is included in cluster total.",
                    "Feature storage is array storage, NOT peak process RAM.",
                    "Qdrant timings include HTTP round trips; server may choose a full scan for small collections.",
                    "Qdrant collection must remain unchanged throughout the benchmark."]}
    outputs = {}
    client = None
    with threadpool_limits(limits=args.threads):
        if "search" in args.tasks or "users" in args.tasks:
            if args.source == "local":
                x, fields, ids = load_users(args.users)
                metadata["inputs"].append(fingerprint(args.users))
            else:
                x, fields, ids, client, digest = fetch_qdrant(args.qdrant_url, args.collection)
                metadata["qdrant_snapshot_sha256"] = digest
                # Capture actual server configuration; exact=False alone does not prove HNSW was used.
                metadata["qdrant_collection_info"] = client.get_collection(args.collection).model_dump(mode="json")
                metadata["environment"]["packages"]["qdrant-client"] = version("qdrant-client")
                vector_config = metadata["qdrant_collection_info"]["config"]["params"]["vectors"]
                if vector_config.get("distance") != "Cosine":
                    raise ValueError("Qdrant comparison requires an unnamed Cosine vector collection")
            if args.clusters >= len(x):
                raise ValueError("clusters must be smaller than the number of users")
            if "search" in args.tasks:
                print(f"Retrieval: {x.shape}", flush=True)
                summary, detail, info = benchmark_search(x, fields, ids, args, client)
                outputs.update(search_summary=summary, search_queries=detail)
                metadata["search"] = info
            if "users" in args.tasks:
                valid_fields = fields if all(fields) else None
                summary, detail = benchmark_clusters(x, x, valid_fields, args, "users")
                outputs.update(user_clustering_summary=summary, user_clustering_runs=detail)
        if "videos" in args.tasks:
            x, weighted, info = load_video_matrix(args.events, args.video_limit, args.seed)
            if args.clusters >= x.shape[0]:
                raise ValueError("clusters must be smaller than the number of videos")
            metadata["inputs"].append(fingerprint(args.events))
            metadata["videos"] = info
            summary, detail = benchmark_clusters(x, weighted, None, args, "videos")
            outputs.update(video_clustering_summary=summary, video_clustering_runs=detail)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(args.out / f"{name}.csv", index=False)
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    report = ["# 算法对照实验实测结果", "数据来源：" + args.source,
              "以下数值为本次实测，具体规模、参数、文件 SHA-256 和环境见 metadata.json。",
              "本地实验使用加权兴趣标签向量；只有 source=qdrant 且包含 qdrant 行时，才是服务器检索实测。",
              "同主兴趣领域比例、ARI/NMI 仅作模拟标签一致性参考，不等于真实用户推荐准确率。",
              "检索 Recall@K 对照全量精确搜索；并列得分会影响集合重合率，因此同时给出允许同分替代的召回率。",
              "聚类轮廓系数统一在原空间、同一采样集合计算；total 包含 SVD，排除评估和加载。",
              "检索速度不含离线建簇；index_build_s、break_even_queries 单列其代价。feature_storage_mib 不是峰值内存。"]
    for name, frame in outputs.items():
        if name.endswith("summary"):
            report.extend(["## " + name, markdown_table(frame)])
    (args.out / "report.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved results: {args.out}", flush=True)


if __name__ == "__main__":
    main()

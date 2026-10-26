import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from experiments import benchmark as b


class BenchmarkTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("qdrant_client"), "Qdrant client is optional for local runs")
    def test_qdrant_read_and_query_path_with_ephemeral_collection(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        client = QdrantClient(":memory:")
        x = normalize(np.random.default_rng(15).normal(size=(24, 6)).astype(np.float32))
        client.create_collection("users", vectors_config=VectorParams(size=6, distance=Distance.COSINE))
        client.upsert("users", points=[PointStruct(id=i, vector=row.tolist(), payload={"primary_field": str(i % 3)})
                                         for i, row in enumerate(x)])
        with patch("qdrant_client.QdrantClient", return_value=client):
            vectors, fields, ids, _, digest = b.fetch_qdrant("unused", "users")
        args = argparse.Namespace(seed=42, queries=5, top_k=3, repeats=1, clusters=3,
                                  collection="users", hnsw_ef=128)
        with threadpool_limits(limits=1):
            summary, _, _ = b.benchmark_search(vectors, fields, ids, args, client)
        remote = summary.loc[summary.method == "qdrant"].iloc[0]
        self.assertAlmostEqual(remote.recall_at_k, 1)
        self.assertEqual(len(digest), 64)
        client.close()

    def test_ties_self_exclusion_and_full_probe(self):
        x = normalize(np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32))
        truth = b.exact_search(x, 0, 1)
        self.assertEqual(truth.tolist(), [1])
        self.assertEqual(b.recall_with_ties(x, 0, np.array([2]), truth, 1), 1.0)
        result, count = b.cluster_search(x, 0, 3, np.array([[1, 0], [0, 1]]),
                                         [np.array([0, 1, 2]), np.array([3])], 2)
        self.assertEqual(result.tolist(), [1, 2, 3])
        self.assertEqual(count, 3)
        self.assertNotIn(0, result)

    def test_pruning_can_miss_true_neighbors(self):
        x = normalize(np.array([[1, 0], [.99, .01], [.8, .2], [0, 1]], dtype=np.float32))
        # The closest real neighbor is deliberately in a different candidate cluster.
        inverted = [np.array([0, 2]), np.array([1, 3])]
        centers = np.array([[1., 0], [0, 1]])
        truth = b.exact_search(x, 0, 1)
        pruned, _ = b.cluster_search(x, 0, 1, centers, inverted, 1)
        all_clusters, _ = b.cluster_search(x, 0, 1, centers, inverted, 2)
        self.assertNotEqual(pruned.tolist(), truth.tolist())
        self.assertEqual(all_clusters.tolist(), truth.tolist())
        self.assertEqual(b.recall_with_ties(x, 0, pruned, truth, 1), 0)

    def test_original_space_is_shared_by_all_cluster_evaluations(self):
        x = normalize(sparse.csr_matrix(np.random.default_rng(7).uniform(size=(40, 12))))
        args = argparse.Namespace(seed=42, eval_samples=25, cluster_repeats=1, svd_dim=4, clusters=3)
        seen = []

        def evaluate(features, labels, metric):
            seen.append(features.copy())
            self.assertEqual(metric, "cosine")
            return .1

        with threadpool_limits(limits=1), patch.object(b, "silhouette_score", side_effect=evaluate):
            summary, detail = b.benchmark_clusters(x, x, None, args, "videos")
        self.assertEqual(len(seen), 3)
        for matrix in seen[1:]:
            np.testing.assert_array_equal(matrix.toarray(), seen[0].toarray())
        self.assertTrue((detail.total_s >= detail.fit_s).all())
        self.assertEqual(set(summary.dimensions), {4, 12})

    def test_lfs_pointer_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.csv"
            path.write_text("version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(ValueError, "git lfs pull"):
                b.check_file(path)

    def test_cli_outputs_real_metrics_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            users = root / "users.csv"
            rows = [{"user_id": i, "primary_field": f"field{i % 3}",
                     "interest_fields": json.dumps([f"field{i % 3}"]),
                     "interest_tags": json.dumps([f"tag{i % 9}", f"token{i}"])} for i in range(45)]
            pd.DataFrame(rows).to_csv(users, index=False)
            events = root / "events.parquet"
            pd.DataFrame([{"video_id": str(v), "user_id": u, "event_type": "watch"}
                          for v in range(50) for u in range(15) if (v + u) % 3 != 0]).to_parquet(events)
            out = root / "results"
            process = subprocess.run([sys.executable, str(Path(b.__file__)), "--users", str(users),
                                      "--events", str(events), "--out", str(out), "--clusters", "3",
                                      "--queries", "5", "--repeats", "2", "--cluster-repeats", "1",
                                      "--eval-samples", "25", "--svd-dim", "4", "--video-limit", "40"],
                                     capture_output=True, text=True, timeout=90)
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            summary = pd.read_csv(out / "search_summary.csv")
            full = summary.loc[summary.method == "cluster_probe_3"].iloc[0]
            self.assertAlmostEqual(full.recall_at_k, 1)
            self.assertTrue((summary.mean_ms > 0).all())
            self.assertTrue((summary.mean_returned <= 10).all())
            metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["videos"]["n"], 40)
            self.assertEqual(len(metadata["inputs"]), 2)
            self.assertTrue((out / "report.md").exists())
            detail = pd.read_csv(out / "search_queries.csv")
            self.assertEqual(len(detail), len(summary) * 5 * 2)


if __name__ == "__main__":
    unittest.main()

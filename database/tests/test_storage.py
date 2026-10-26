import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from database.import_data import DATASETS, rows_from_csv, source_path, transform_row


class ImportValidationTests(unittest.TestCase):
    def event(self, **changes):
        row = {"round": "1", "user_id": "42", "video_id": "7477203555078786350", "event_type": "watch"}
        return {**row, **changes}

    def test_video_id_keeps_full_precision(self):
        values = transform_row(DATASETS["events"], self.event())
        self.assertEqual(values, (1, 42, "7477203555078786350", "watch"))

    def test_invalid_events_rejected(self):
        for changes in ({"event_type": "delete"}, {"round": "0"}, {"user_id": "-1"},
                        {"video_id": ""}, {"user_id": ""}, {"event_type": None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                transform_row(DATASETS["events"], self.event(**changes))

    def test_prediction_values_and_id(self):
        vid, encoded = transform_row(DATASETS["predictions"], {
            "video_id": "7477203555078786350", "round_1": "2", "pred_total_play": "5.25"})
        self.assertEqual(vid, "7477203555078786350")
        self.assertEqual(json.loads(encoded), {"round_1": 2, "pred_total_play": 5.25})

    def test_prediction_nonfinite_rejected(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                transform_row(DATASETS["predictions"], {"video_id": "1", "pred_total_play": value})

    def test_user_interest_must_be_string_array(self):
        user = {"user_id": "42", "primary_field": "Music", "gamma": "0.5", "interest_fields": '["Music"]',
                "interest_tags": '["song"]', "total_watched": "20", "total_liked": "3"}
        self.assertEqual(json.loads(transform_row(DATASETS["users"], user)[3]), ["Music"])
        for bad in ('{"song": 1}', '[12]', 'broken'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                transform_row(DATASETS["users"], {**user, "interest_tags": bad})

    def test_lfs_pointer_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.csv"
            path.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "git lfs pull"):
                list(rows_from_csv(path, DATASETS["events"]))

    def test_missing_columns_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.csv"
            path.write_text("user_id,video_id\n42,123\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "缺少列"):
                list(rows_from_csv(path, DATASETS["events"]))

    def test_bad_row_includes_location(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.csv"
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=DATASETS["events"].columns)
                writer.writeheader()
                writer.writerow(self.event(event_type="invalid"))
            with self.assertRaisesRegex(ValueError, "第 2 行"):
                list(rows_from_csv(path, DATASETS["events"]))

    def test_real_csv_takes_priority_over_parquet(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / DATASETS["events"].path
            path.parent.mkdir(parents=True)
            path.write_text("round,user_id,video_id,event_type\n", encoding="utf-8")
            path.with_suffix(".parquet").write_bytes(b"PAR1")
            self.assertEqual(source_path(root, DATASETS["events"]), path)

    def test_real_parquet_replaces_lfs_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / DATASETS["events"].path
            path.parent.mkdir(parents=True)
            path.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
            path.with_suffix(".parquet").write_bytes(b"PAR1")
            self.assertEqual(source_path(root, DATASETS["events"]), path.with_suffix(".parquet"))


class RepositoryTests(unittest.TestCase):
    def test_user_query_is_filtered_and_parameterized(self):
        from database.repository import user_watch_events
        with patch("database.repository.query_frame") as query:
            user_watch_events(42)
        sql, params = query.call_args.args
        self.assertIn("WHERE history.user_id=%s", sql)
        self.assertIn("user_event_history", sql)
        self.assertIn("unnest(history.watch_video_keys, history.watch_rounds)", sql)
        self.assertEqual(params, (42,))
        self.assertTrue(query.call_args.kwargs["require_events"])

    def test_video_aggregation_is_server_side_and_parameterized(self):
        import pandas as pd
        from database.repository import video_watch_counts
        hostile_id = "1'; DROP TABLE users; --"
        with patch("database.repository.query_frame", return_value=pd.DataFrame({"round": [1, 2], "count": [4, 7]})) as query:
            result = video_watch_counts(hostile_id)
        sql, params = query.call_args.args
        self.assertNotIn(hostile_id, sql)
        self.assertEqual(params, (hostile_id,))
        self.assertIn("video_round_stats", sql)
        self.assertNotIn("watch_events", sql)
        self.assertTrue(query.call_args.kwargs["require_events"])
        self.assertEqual(result.to_dict(), {1: 4, 2: 7})

    def test_empty_video_history_returns_empty_series(self):
        import pandas as pd
        from database.repository import video_watch_counts
        with patch("database.repository.query_frame", return_value=pd.DataFrame(columns=["round", "count"])):
            self.assertTrue(video_watch_counts("missing").empty)


class ConfigurationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("psycopg_pool"), "需要连接池依赖")
    def test_read_pool_is_reused_and_checks_connection_health(self):
        from database.config import read_pool
        read_pool.cache_clear()
        try:
            with patch("database.config.database_url", return_value="postgresql://placeholder"), \
                 patch("psycopg_pool.ConnectionPool") as factory, patch("database.config.atexit.register"):
                self.assertIs(read_pool(), read_pool())
                factory.assert_called_once()
                self.assertEqual(factory.call_args.kwargs["check"], factory.check_connection)
        finally:
            read_pool.cache_clear()

    def test_default_is_database(self):
        from database.config import backend
        with patch.dict("os.environ", {}, clear=True), patch("database.config.load_environment"):
            self.assertEqual(backend(), "supabase")

    def test_csv_requires_explicit_configuration(self):
        from database.config import backend
        with patch.dict("os.environ", {"DATA_BACKEND": "csv"}), patch("database.config.load_environment"):
            self.assertEqual(backend(), "csv")

    def test_invalid_backend_rejected(self):
        from database.config import backend
        with patch.dict("os.environ", {"DATA_BACKEND": "typo"}), patch("database.config.load_environment"):
            with self.assertRaises(ValueError):
                backend()


if __name__ == "__main__":
    unittest.main()

import sqlite3
import unittest
from collections import Counter
from contextlib import closing

from database.capacity import CapacityError, ensure_capacity
from database.compact_events import allocate_video_keys, history_rows, prepare_events, stats_rows, verify_totals


class CompactEventTests(unittest.TestCase):
    def setUp(self):
        self.keys = {"7477203555078786350": 1, "7056981264364080389": 2}
        self.rows = [
            (1, 42, "7477203555078786350", "watch"),
            (1, 42, "7477203555078786350", "watch"),  # exact duplicate
            (1, 42, "7477203555078786350", "like"),
            (2, 42, "7477203555078786350", "watch"),  # repeat on another day
            (2, 99, "7477203555078786350", "watch"),  # different user
            (3, 99, "7056981264364080389", "like"),   # like without a watch remains a like
        ]

    def test_history_reconstructs_every_unique_event(self):
        with closing(sqlite3.connect(":memory:")) as local:
            report = prepare_events(local, iter(self.rows), self.keys, {42, 99})
            inverse = {key: vid for vid, key in self.keys.items()}
            restored = set()
            for uid, watch_keys, watch_rounds, like_keys, like_rounds in history_rows(local):
                restored.update((rnd, uid, inverse[key], "watch") for key, rnd in zip(watch_keys, watch_rounds))
                restored.update((rnd, uid, inverse[key], "like") for key, rnd in zip(like_keys, like_rounds))
            self.assertEqual(restored, set(self.rows))
            self.assertEqual(report["source_rows"], 6)
            self.assertEqual(report["unique_events"], 5)
            self.assertEqual((report["watch_count"], report["like_count"]), (3, 2))

    def test_video_statistics_equal_original_group_counts(self):
        with closing(sqlite3.connect(":memory:")) as local:
            report = prepare_events(local, iter(self.rows), self.keys, {42, 99})
            expected = Counter((self.keys[vid], rnd, kind) for rnd, uid, vid, kind in set(self.rows))
            restored = {}
            for key, rounds, watches, likes in stats_rows(local):
                for rnd, watch, like in zip(rounds, watches, likes):
                    if watch:
                        restored[key, rnd, "watch"] = watch
                    if like:
                        restored[key, rnd, "like"] = like
            self.assertEqual(restored, expected)
            verify_totals(local, report)

    def test_empty_event_snapshot_is_valid(self):
        with closing(sqlite3.connect(":memory:")) as local:
            report = prepare_events(local, iter([]), {}, set())
            self.assertEqual(report["unique_events"], 0)
            self.assertEqual(list(history_rows(local)), [])
            self.assertEqual(list(stats_rows(local)), [])
            verify_totals(local, report)

    def test_missing_user_or_video_rejected_before_upload(self):
        for keys, users in ((self.keys, {99}), ({}, {42, 99})):
            with self.subTest(keys=keys, users=users), closing(sqlite3.connect(":memory:")) as local:
                with self.assertRaisesRegex(ValueError, "不存在"):
                    prepare_events(local, iter(self.rows), keys, users)

    def test_invalid_round_and_kind_rejected(self):
        for row in ((0, 42, "7477203555078786350", "watch"),
                    (2**31, 42, "7477203555078786350", "watch"),
                    (1, 42, "7477203555078786350", "bad")):
            with self.subTest(row=row), closing(sqlite3.connect(":memory:")) as local:
                with self.assertRaises(ValueError):
                    prepare_events(local, iter([row]), self.keys, {42})

    def test_video_codes_stay_stable_when_catalog_grows(self):
        self.assertEqual(allocate_video_keys(["new", *self.keys], self.keys), {**self.keys, "new": 3})


class CapacityTests(unittest.TestCase):
    def test_budget_guard_counts_projected_growth(self):
        from unittest.mock import MagicMock
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = (400_000_000,)
        self.assertEqual(ensure_capacity(conn, 450_000_000, 40_000_000), 400_000_000)
        with self.assertRaises(CapacityError):
            ensure_capacity(conn, 450_000_000, 60_000_000)


if __name__ == "__main__":
    unittest.main()

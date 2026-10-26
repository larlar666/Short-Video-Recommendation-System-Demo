"""Lossless event packing with disk-backed deduplication before any event upload."""
from itertools import groupby, islice


def allocate_video_keys(video_ids, existing):
    mapping = dict(existing)
    next_key = max(mapping.values(), default=0) + 1
    for video_id in sorted(set(video_ids) - mapping.keys()):
        if next_key > 2_147_483_647:
            raise ValueError("视频字典超出 integer 范围")
        mapping[video_id] = next_key
        next_key += 1
    return mapping


def prepare_events(local, rows, video_keys, user_ids, progress=None):
    local.execute("PRAGMA temp_store=FILE")
    local.execute("PRAGMA cache_size=-32768")
    local.execute("""CREATE TABLE events (
        user_id INTEGER NOT NULL, round INTEGER NOT NULL, video_key INTEGER NOT NULL,
        kind INTEGER NOT NULL, PRIMARY KEY (user_id, round, video_key, kind)
    ) WITHOUT ROWID""")
    count = 0
    iterator = iter(rows)
    while batch := list(islice(iterator, 10000)):
        packed = []
        for rnd, uid, vid, event_type in batch:
            if uid not in user_ids:
                raise ValueError(f"事件引用不存在的用户：{uid}")
            if vid not in video_keys:
                raise ValueError(f"事件引用不存在的视频：{vid}")
            if not 0 < rnd <= 2_147_483_647 or event_type not in {"watch", "like"}:
                raise ValueError("事件轮次或类型无效")
            packed.append((uid, rnd, video_keys[vid], int(event_type == "like")))
        local.executemany("INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?)", packed)
        count += len(batch)
        if progress and count % 500000 == 0:
            progress(count)
    local.execute("CREATE INDEX events_video_round ON events(video_key, round, kind)")
    local.commit()
    kinds = dict(local.execute("SELECT kind, count(*) FROM events GROUP BY kind"))
    watches, likes = kinds.get(0, 0), kinds.get(1, 0)
    users = local.execute("SELECT count(DISTINCT user_id) FROM events").fetchone()[0]
    videos = local.execute("SELECT count(DISTINCT video_key) FROM events").fetchone()[0]
    pairs = local.execute("SELECT count(*) FROM (SELECT 1 FROM events GROUP BY video_key, round)").fetchone()[0]
    return {"source_rows": count, "unique_events": watches + likes, "watch_count": watches,
            "like_count": likes, "users": users, "videos": videos, "round_pairs": pairs,
            "array_bytes": (watches + likes) * 8 + pairs * 12}


def history_rows(local):
    rows = local.execute("SELECT user_id, round, video_key, kind FROM events ORDER BY user_id, round, video_key, kind")
    for uid, events in groupby(rows, key=lambda row: row[0]):
        watch_keys, watch_rounds, like_keys, like_rounds = [], [], [], []
        for _, rnd, video_key, kind in events:
            keys, rounds = (like_keys, like_rounds) if kind else (watch_keys, watch_rounds)
            keys.append(video_key)
            rounds.append(rnd)
        yield (uid, watch_keys, watch_rounds, like_keys, like_rounds)


def stats_rows(local):
    rows = local.execute("""SELECT video_key, round, sum(kind=0), sum(kind=1)
        FROM events GROUP BY video_key, round ORDER BY video_key, round""")
    for key, stats in groupby(rows, key=lambda row: row[0]):
        rounds, watches, likes = [], [], []
        for _, rnd, watch_count, like_count in stats:
            rounds.append(rnd)
            watches.append(watch_count)
            likes.append(like_count)
        yield (key, rounds, watches, likes)


def verify_totals(local, report):
    history_watch = history_like = 0
    for _, keys, _, likes, _ in history_rows(local):
        history_watch += len(keys)
        history_like += len(likes)
    stat_watch = stat_like = 0
    for _, _, watches, likes in stats_rows(local):
        stat_watch += sum(watches)
        stat_like += sum(likes)
    if (history_watch, history_like) != (stat_watch, stat_like) or (
        history_watch, history_like
    ) != (report["watch_count"], report["like_count"]):
        raise ValueError("压缩历史与按轮统计总数不一致，停止导入")


def import_compact_events(conn, path, batch_size, budget_bytes):
    import sqlite3
    import tempfile
    from contextlib import closing
    from pathlib import Path
    from database.capacity import ensure_capacity
    from database.import_data import DATASETS, checksum, rows_from_file, write_rows

    digest = checksum(path)
    with conn.transaction():
        if not conn.execute("SELECT to_regclass('recommender.event_snapshot')").fetchone()[0]:
            raise ValueError("请先运行 python -m database.migrate，应用 002 紧凑行为表迁移")
        snapshot = conn.execute("SELECT sha256 FROM recommender.event_snapshot WHERE singleton").fetchone()
        if snapshot and snapshot[0] == digest:
            print("events: 紧凑行为快照已导入，跳过")
            return
        video_ids = {row[0] for row in conn.execute("SELECT id FROM recommender.videos")}
        users = {row[0] for row in conn.execute("SELECT user_id FROM recommender.users")}
        existing = dict(conn.execute("SELECT video_id, video_key FROM recommender.event_video_keys"))
    video_keys = allocate_video_keys(video_ids, existing)
    print("events: 在本机校验、去重并整理历史与统计，暂不上传原始日志", flush=True)
    with tempfile.TemporaryDirectory(prefix="video-events-") as folder:
        with closing(sqlite3.connect(Path(folder) / "events.sqlite")) as local:
            report = prepare_events(local, rows_from_file(path, DATASETS["events"]), video_keys, users,
                                    lambda count: print(f"events: 本地处理 {count:,} 行", flush=True))
            verify_totals(local, report)
            if checksum(path) != digest:
                raise ValueError("源文件在整理期间发生变化，请停止其他数据生成任务后重试")
            print(f"events: {report['unique_events']:,} 个唯一事件 → {report['users']:,} 条用户历史 + "
                  f"{report['videos']:,} 条视频统计；数组原始数据约 {report['array_bytes']/1_000_000:.1f} MB", flush=True)
            with conn.transaction():
                conn.execute("SELECT pg_advisory_xact_lock(74201902)")
                latest = dict(conn.execute("SELECT video_id, video_key FROM recommender.event_video_keys"))
                if latest != existing:
                    raise ValueError("其他任务已修改视频字典，请重试本次导入")
                # Includes space for dictionary, row overhead, indexes, plus a generous array reserve.
                growth = report["array_bytes"] * 2 + len(video_keys) * 256 + report["users"] * 512 + report["videos"] * 256
                ensure_capacity(conn, budget_bytes, growth)
                new_keys = ((key, vid) for vid, key in video_keys.items() if vid not in existing)
                write_rows(conn, "event_video_keys", ("video_key", "video_id"), ("video_key",),
                           new_keys, batch_size, budget_bytes)
                conn.execute("TRUNCATE recommender.user_event_history, recommender.video_round_stats")
                write_rows(conn, "user_event_history", ("user_id", "watch_video_keys", "watch_rounds",
                           "like_video_keys", "like_rounds"), ("user_id",), history_rows(local), batch_size, budget_bytes)
                write_rows(conn, "video_round_stats", ("video_key", "rounds", "watch_counts", "like_counts"),
                           ("video_key",), stats_rows(local), batch_size, budget_bytes)
                conn.execute("""INSERT INTO recommender.event_snapshot
                    (singleton, sha256, source_rows, unique_events, watch_count, like_count)
                    VALUES (true, %s, %s, %s, %s, %s)
                    ON CONFLICT (singleton) DO UPDATE SET sha256=EXCLUDED.sha256,
                    source_rows=EXCLUDED.source_rows, unique_events=EXCLUDED.unique_events,
                    watch_count=EXCLUDED.watch_count, like_count=EXCLUDED.like_count, imported_at=now()""",
                    (digest, report["source_rows"], report["unique_events"], report["watch_count"], report["like_count"]))
                conn.execute("""INSERT INTO recommender.import_runs(dataset, sha256, row_count)
                    VALUES ('events_compact_v1', %s, %s) ON CONFLICT (dataset, sha256)
                    DO UPDATE SET row_count=EXCLUDED.row_count, imported_at=now()""", (digest, report["source_rows"]))
                ensure_capacity(conn, budget_bytes)
            print("events: 紧凑历史与统计已一起提交，原始 watch_events 表保持为空", flush=True)


def main():
    """Offline full-data verification; never reads .env or connects to Supabase."""
    import argparse
    import json
    import sqlite3
    import tempfile
    from contextlib import closing
    from pathlib import Path
    from database.config import ROOT
    from database.import_data import DATASETS, rows_from_csv, rows_from_file, source_path

    parser = argparse.ArgumentParser(description="本地验证紧凑历史与视频统计，不连接云端")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    args = parser.parse_args()
    videos = {row[0] for row in rows_from_csv(source_path(args.data_root, DATASETS["videos"]), DATASETS["videos"])}
    users = {row[0] for row in rows_from_csv(source_path(args.data_root, DATASETS["users"]), DATASETS["users"])}
    with tempfile.TemporaryDirectory(prefix="video-events-check-") as folder:
        with closing(sqlite3.connect(Path(folder) / "events.sqlite")) as local:
            report = prepare_events(local, rows_from_file(source_path(args.data_root, DATASETS["events"]),
                                    DATASETS["events"]), allocate_video_keys(videos, {}), users,
                                    lambda count: print(f"本地校验 {count:,} 行", flush=True))
            verify_totals(local, report)
            report["temporary_sqlite_bytes"] = (Path(folder) / "events.sqlite").stat().st_size
            print(json.dumps(report, ensure_ascii=False, indent=2))
            print("用户历史与视频统计的观看、点赞总数一致；此命令未连接数据库。")


if __name__ == "__main__":
    main()

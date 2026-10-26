"""Stream CSV artifacts into PostgreSQL with COPY, validation and idempotent upserts.

python -m database.import_data --validate-only
python -m database.import_data --datasets videos users events
"""
import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from database.config import ROOT, connect
from database.capacity import ensure_capacity, estimate_batch_bytes


@dataclass(frozen=True)
class Dataset:
    path: str
    table: str
    columns: tuple
    keys: tuple


DATASETS = {
    "videos": Dataset("F1/data/tiktok_meta_100k.csv", "videos", (
        "id", "desc", "challenges", "duration", "digg_count", "comment_count", "play_count",
        "share_count", "collect_count", "music_id", "user_id", "poi_name", "city", "poi_category",
        "preference_tags", "fields"), ("id",)),
    "users": Dataset("F2/data/users_sim.csv", "users", (
        "user_id", "primary_field", "gamma", "interest_fields", "interest_tags", "total_watched",
        "total_liked"), ("user_id",)),
    "popularity": Dataset("F1/data/video_popularity_init.csv", "video_popularity", (
        "id", "H_behavior", "Q_valid", "Q", "R", "H0", "H0_norm"), ("id",)),
    "events": Dataset("F2/data/watch_events.csv", "watch_events", (
        "round", "user_id", "video_id", "event_type"), ("user_id", "round", "video_id", "event_type")),
    "predictions": Dataset("F5/data/video_series_pred.csv", "video_predictions", (
        "video_id", "metrics"), ("video_id",)),
    "video_clusters": Dataset("F6/data/video_clusters.csv", "video_clusters", (
        "video_id", "cluster_id"), ("video_id",)),
    "user_clusters": Dataset("F7/data/user_clusters.csv", "user_clusters", (
        "user_id", "cluster_id", "primary_field"), ("user_id",)),
}
LIST_COLUMNS = {"preference_tags", "fields", "interest_fields", "interest_tags"}
INT_COLUMNS = {"digg_count", "comment_count", "play_count", "share_count", "collect_count",
               "total_watched", "total_liked", "round", "cluster_id"}
FLOAT_COLUMNS = {"duration", "gamma", "H_behavior", "Q", "R", "H0", "H0_norm"}


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("数值必须有限")
    return number


def transform_row(dataset, row):
    if dataset.table == "video_predictions":
        metrics = {key: finite_float(value) for key, value in row.items()
                   if key != "video_id" and value not in (None, "")}
        if not row.get("video_id"):
            raise ValueError("缺少 video_id")
        return (row["video_id"], json.dumps(metrics, allow_nan=False))
    values = []
    for column in dataset.columns:
        value = row[column]
        if value is None:
            raise ValueError(f"{column} 缺少值，请检查 CSV 列数")
        if column in LIST_COLUMNS:
            parsed = json.loads(value or "[]")
            if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
                raise ValueError(f"{column} 必须为字符串数组")
            value = json.dumps(parsed, ensure_ascii=False)
        elif column == "challenges":
            try:
                parsed = json.loads(value or "[]")
            except json.JSONDecodeError:
                parsed = value
            value = json.dumps(parsed, ensure_ascii=False)
        elif column in INT_COLUMNS or (column == "user_id" and dataset.table != "videos"):
            # Never route 64-bit IDs through a float: TikTok IDs exceed 2**53.
            if column == "user_id" and not value:
                raise ValueError("user_id 不可为空")
            value = int(value or "0")
            if value < 0 or (column == "round" and value == 0):
                raise ValueError(f"{column} 超出允许范围")
        elif column in FLOAT_COLUMNS:
            value = None if column == "Q" and not value else finite_float(value or 0)
            if column in {"duration", "gamma"} and value < 0:
                raise ValueError(f"{column} 不可为负")
            if column == "H0_norm" and not 0 <= value <= 1:
                raise ValueError("H0_norm 必须在 0 到 1 之间")
        elif column == "Q_valid":
            if value.lower() not in {"true", "false", "1", "0"}:
                raise ValueError("Q_valid 必须为布尔值")
            value = value.lower() in {"true", "1"}
        elif column in {"id", "video_id", "primary_field"} and not value:
            raise ValueError(f"{column} 不可为空")
        elif column == "event_type" and value not in {"watch", "like"}:
            raise ValueError("event_type 必须为 watch 或 like")
        values.append(value)
    return tuple(values)


def rows_from_csv(path, dataset):
    with path.open(encoding="utf-8-sig", newline="") as source:
        if source.readline().startswith("version https://git-lfs.github.com/spec/"):
            raise ValueError(f"{path.name} 是 Git LFS 指针，请先执行 git lfs pull")
        source.seek(0)
        reader = csv.DictReader(source)
        required = set(dataset.columns) if dataset.table != "video_predictions" else {"video_id", "pred_total_play"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} 缺少列：{sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            try:
                yield transform_row(dataset, row)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path.name} 第 {line} 行格式错误：{exc}") from exc


def rows_from_file(path, dataset):
    if path.suffix != ".parquet":
        yield from rows_from_csv(path, dataset)
        return
    import pyarrow.parquet as pq
    source = pq.ParquetFile(path)
    missing = set(dataset.columns) - set(source.schema.names)
    if missing:
        raise ValueError(f"{path.name} 缺少列：{sorted(missing)}")
    line = 0
    for batch in source.iter_batches(batch_size=10000, columns=list(dataset.columns)):
        for row in batch.to_pylist():
            line += 1
            try:
                yield transform_row(dataset, {key: "" if value is None else str(value)
                                               for key, value in row.items()})
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path.name} 第 {line} 条格式错误：{exc}") from exc


def source_path(root, dataset):
    path = root / dataset.path
    if dataset.table == "watch_events":
        parquet = path.with_suffix(".parquet")
        csv_is_pointer = False
        if path.exists():
            with path.open("rb") as source:
                csv_is_pointer = source.read(80).startswith(b"version https://git-lfs.github.com/spec/")
        if (not path.exists() or csv_is_pointer) and parquet.exists():
            with parquet.open("rb") as source:
                if source.read(4) == b"PAR1":
                    return parquet
    return path


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_rows(conn, table_name, column_names, key_names, rows, batch_size, budget_bytes):
    from psycopg import sql
    table = sql.Identifier("recommender", table_name)
    columns = sql.SQL(", ").join(map(sql.Identifier, column_names))
    keys = sql.SQL(", ").join(map(sql.Identifier, key_names))
    updates = [column for column in column_names if column not in key_names]
    conflict = (sql.SQL("DO UPDATE SET ") + sql.SQL(", ").join(
        sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c)) for c in updates
    )) if updates else sql.SQL("DO NOTHING")
    conn.execute(sql.SQL("CREATE TEMP TABLE import_stage (LIKE {} INCLUDING DEFAULTS) ON COMMIT DROP").format(table))
    count = 0
    iterator = iter(rows)
    while batch := list(islice(iterator, batch_size)):
        key_indexes = [column_names.index(key) for key in key_names]
        unique_rows = {tuple(row[index] for index in key_indexes): row for row in batch}
        ensure_capacity(conn, budget_bytes, estimate_batch_bytes(unique_rows.values()))
        with conn.cursor() as cur:
            with cur.copy(sql.SQL("COPY import_stage ({}) FROM STDIN").format(columns)) as copy:
                for row in unique_rows.values():
                    copy.write_row(row)
        conn.execute(sql.SQL("INSERT INTO {} ({}) SELECT {} FROM import_stage WHERE true "
                             "ON CONFLICT ({}) {}")
                     .format(table, columns, columns, keys, conflict))
        conn.execute("TRUNCATE import_stage")
        ensure_capacity(conn, budget_bytes)
        count += len(batch)
        print(f"{table_name}: 已写入 {count:,} 条紧凑记录" if table_name in {
            "user_event_history", "video_round_stats"
        } else f"{table_name}: 已处理 {count:,} 行", flush=True)
    conn.execute("DROP TABLE import_stage")
    return count


def import_dataset(conn, name, path, batch_size=2000, budget_bytes=450_000_000):
    if name == "events":
        from database.compact_events import import_compact_events
        return import_compact_events(conn, path, batch_size, budget_bytes)
    dataset = DATASETS[name]
    digest = checksum(path)
    # One transaction per dataset: failed tables roll back completely, completed tables remain.
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(74201902)")
        if conn.execute("SELECT 1 FROM recommender.import_runs WHERE dataset=%s AND sha256=%s",
                        (name, digest)).fetchone():
            print(f"{name}: 内容已导入，跳过")
            return
        count = write_rows(conn, dataset.table, dataset.columns, dataset.keys,
                           rows_from_file(path, dataset), batch_size, budget_bytes)
        conn.execute("INSERT INTO recommender.import_runs(dataset, sha256, row_count) VALUES (%s,%s,%s)",
                     (name, digest, count))
    print(f"{name}: 提交成功")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--max-db-mb", type=int, default=450,
                        help="数据库软上限（十进制 MB），默认 450，为免费额度预留空间")
    parser.add_argument("--validate-only", action="store_true", help="只校验所有行，不连接数据库")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size 必须大于 0")
    if args.max_db_mb < 1:
        parser.error("max-db-mb 必须大于 0")
    # Apply dependency order even when user lists datasets in another order.
    selected = [(name, source_path(args.data_root, spec)) for name, spec in DATASETS.items() if name in args.datasets]
    for name, path in selected:
        if not path.is_file():
            raise FileNotFoundError(f"缺少 {path}；请恢复数据或通过 --datasets 选择已有的数据")
        next(iter(rows_from_file(path, DATASETS[name])), None)
    if args.validate_only:
        for name, path in selected:
            count = sum(1 for _ in rows_from_file(path, DATASETS[name]))
            print(f"{name}: 校验通过 {count:,} 行")
        return
    with connect() as conn:
        # Autocommit ensures import_dataset owns each explicit transaction.
        conn.autocommit = True
        for name, path in selected:
            import_dataset(conn, name, path, args.batch_size, args.max_db_mb * 1_000_000)
        size = conn.execute("SELECT pg_database_size(current_database())").fetchone()[0]
        print(f"导入完成：数据库当前占用 {size / 1_000_000:.1f} MB；请同时查看 Dashboard 磁盘用量。")


if __name__ == "__main__":
    main()

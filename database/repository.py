"""Queries used by Streamlit. Event filters and aggregation run in PostgreSQL."""
import pandas as pd

from database.config import read_connection

TABLES = {
    "videos": "videos", "users": "users", "series": "video_predictions",
    "video_clusters": "video_clusters", "user_clusters": "user_clusters",
}


def query_frame(statement, params=(), *, require_events=False):
    with read_connection() as conn:
        if require_events and not conn.execute(
            "SELECT 1 FROM recommender.event_snapshot WHERE singleton"
        ).fetchone():
            raise ValueError("尚未导入行为历史，请运行 python -m database.import_data --datasets events")
        with conn.cursor() as cur:
            cur.execute(statement, params)
            columns = [col.name for col in cur.description]
            return pd.DataFrame.from_records(cur.fetchall(), columns=columns)


def load_catalog():
    from psycopg import sql
    result = {}
    with read_connection() as conn:
        with conn.cursor() as cur:
            for key, table in TABLES.items():
                cur.execute(sql.SQL("SELECT * FROM recommender.{}").format(sql.Identifier(table)))
                rows = cur.fetchall()
                if not rows:
                    continue
                columns = [col.name for col in cur.description]
                frame = pd.DataFrame.from_records(rows, columns=columns)
                if key == "series":
                    frame = pd.DataFrame([
                        {**row["metrics"], "video_id": row["video_id"]}
                        for row in frame.to_dict("records")
                    ])
                result[key] = frame
    return result


def user_watch_events(user_id):
    return query_frame("""SELECT item.round, history.user_id, keys.video_id, 'watch' AS event_type
        FROM recommender.user_event_history AS history
        CROSS JOIN LATERAL unnest(history.watch_video_keys, history.watch_rounds) AS item(video_key, round)
        JOIN recommender.event_video_keys AS keys ON keys.video_key=item.video_key
        WHERE history.user_id=%s ORDER BY item.round, keys.video_id""", (int(user_id),), require_events=True)


def video_watch_counts(video_id):
    frame = query_frame("""SELECT item.round, item.count
        FROM recommender.event_video_keys AS keys
        JOIN recommender.video_round_stats AS stats ON stats.video_key=keys.video_key
        CROSS JOIN LATERAL unnest(stats.rounds, stats.watch_counts) AS item(round, count)
        WHERE keys.video_id=%s AND item.count>0 ORDER BY item.round""", (str(video_id),), require_events=True)
    return frame.set_index("round")["count"]

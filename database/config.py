import os
import atexit
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_environment():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)


def backend():
    load_environment()
    value = os.getenv("DATA_BACKEND", "supabase").strip().lower()
    if value not in {"supabase", "csv"}:
        raise ValueError("DATA_BACKEND 必须为 supabase 或 csv")
    return value


def database_url():
    load_environment()
    url = os.getenv("DATABASE_URL", "").strip()
    if not url or "YOUR_PASSWORD" in url or "PROJECT_REF" in url:
        raise ValueError("请在项目 .env 中设置 DATABASE_URL，参照 .env.example")
    return url


def connect():
    import psycopg
    url = database_url()
    return psycopg.connect(url, connect_timeout=10, application_name="short-video-recommender",
                           prepare_threshold=None)


@lru_cache(maxsize=1)
def read_pool():
    """Reuse server connections across Streamlit reruns; scripts keep direct connections."""
    from psycopg_pool import ConnectionPool
    pool = ConnectionPool(
        conninfo=database_url(), name="recommender-reads",
        min_size=0, max_size=3, timeout=20, max_idle=300,
        check=ConnectionPool.check_connection,
        kwargs={"connect_timeout": 10, "application_name": "short-video-recommender",
                "prepare_threshold": None},
    )
    atexit.register(pool.close)
    return pool


def read_connection():
    return read_pool().connection()

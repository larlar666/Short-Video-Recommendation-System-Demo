"""Run with python -m database.migrate; each SQL migration is atomic."""
import hashlib
from pathlib import Path

from database.config import connect


def migrate(conn):
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(74201901)")
        conn.execute("CREATE SCHEMA IF NOT EXISTS recommender")
        conn.execute("REVOKE ALL ON SCHEMA recommender FROM PUBLIC")
        conn.execute("""CREATE TABLE IF NOT EXISTS recommender.schema_migrations (
            version text PRIMARY KEY, sha256 text NOT NULL,
            applied_at timestamptz NOT NULL DEFAULT now())""")
        for path in sorted((Path(__file__).parent / "migrations").glob("*.sql")):
            source = path.read_text(encoding="utf-8")
            checksum = hashlib.sha256(source.encode()).hexdigest()
            old = conn.execute("SELECT sha256 FROM recommender.schema_migrations WHERE version=%s",
                               (path.name,)).fetchone()
            if old:
                if old[0] != checksum:
                    raise ValueError(f"已应用的迁移被修改：{path.name}，请新增迁移文件")
                continue
            conn.execute(source, prepare=False)
            conn.execute("INSERT INTO recommender.schema_migrations(version, sha256) VALUES (%s,%s)",
                         (path.name, checksum))
            print(f"已应用 {path.name}")


if __name__ == "__main__":
    with connect() as connection:
        migrate(connection)

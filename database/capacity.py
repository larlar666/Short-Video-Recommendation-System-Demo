"""A conservative import guard, not a substitute for provider disk/WAL monitoring."""


class CapacityError(RuntimeError):
    pass


def ensure_capacity(conn, budget_bytes, growth_bytes=0):
    size = conn.execute("SELECT pg_database_size(current_database())").fetchone()[0]
    if size + growth_bytes > budget_bytes:
        raise CapacityError(
            f"容量保护停止写入：当前数据库 {size / 1_000_000:.1f} MB，"
            f"预计新增峰值 {growth_bytes / 1_000_000:.1f} MB，"
            f"导入上限 {budget_bytes / 1_000_000:.0f} MB。"
            "当前数据集会回滚；不要反复重试，请先检查云端用量。"
        )
    return size


def estimate_batch_bytes(rows):
    # Reserve space for heap, TOAST and indexes, without relying on compression.
    def value_bytes(value):
        if isinstance(value, (list, tuple)):
            return 32 + sum(value_bytes(item) for item in value)
        if isinstance(value, int):
            return 8
        if value is None:
            return 1
        return len(str(value).encode("utf-8")) + 8
    return sum(256 + 2 * sum(value_bytes(value) for value in row) for row in rows)

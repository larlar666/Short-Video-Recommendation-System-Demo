"""UI diagnostics that never echo a DSN or connection error text."""


def error_summary(error):
    kind = type(error).__name__
    sqlstate = getattr(error, "sqlstate", None)
    if kind in {"ConnectionTimeout", "PoolTimeout", "OperationalError", "InterfaceError"} and (
        sqlstate is None or str(sqlstate).startswith("08")
    ):
        return "数据库连接超时或中断。请保持可用网络后重试，无需重新导入数据。"
    if isinstance(error, ImportError):
        return "缺少数据库运行依赖，请执行 python -m pip install -r requirements-db.txt 后重启应用。"
    return "处理失败，请查看诊断信息；详细异常已记录在启动应用的终端。"


def error_code(error):
    text = f"异常类型：{type(error).__name__}"
    state = getattr(error, "sqlstate", None)
    if state:
        text += f"\nSQLSTATE：{state}"
    return text

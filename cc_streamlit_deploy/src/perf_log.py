"""
perf_log.py — 性能日志模块

记录每个主要函数耗时到 logs/performance.log
"""

import csv
import os
import time
from datetime import datetime
from pathlib import Path
from functools import wraps

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_PATH = LOGS_DIR / "performance.log"

_LOGGING_ENABLED = True


def init_log():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "function_name", "duration_seconds", "status", "note"])
    print(f"[perf_log] Log initialized: {LOG_PATH}")


def log_event(function_name: str, duration: float, status: str = "OK", note: str = ""):
    if not _LOGGING_ENABLED:
        return
    init_log()
    with open(LOG_PATH, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            datetime.now().isoformat(),
            function_name,
            f"{duration:.3f}",
            status,
            note[:200],
        ])


def print_timing(function_name: str, duration: float):
    """打印到终端便于实时观察性能。"""
    status = "OK" if duration < 1.0 else "SLOW" if duration < 5.0 else "CRITICAL"
    icon = "⚡" if status == "OK" else "🐢" if status == "SLOW" else "🔴"
    print(f"  {icon} {function_name}: {duration:.2f}s [{status}]")


def timed(func):
    """装饰器：自动记录函数耗时。"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        try:
            result = func(*args, **kwargs)
            dt = time.time() - t0
            log_event(func.__name__, dt, "OK")
            return result
        except Exception as e:
            dt = time.time() - t0
            log_event(func.__name__, dt, "ERROR", str(e))
            raise
    return wrapper


def set_logging(enabled: bool):
    global _LOGGING_ENABLED
    _LOGGING_ENABLED = enabled


def get_recent_logs(n: int = 20) -> list:
    """获取最近 n 条日志。"""
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[-n:]


def print_summary(n: int = 10):
    """打印最近 n 条性能摘要。"""
    logs = get_recent_logs(n)
    if not logs:
        print("[perf_log] No logs yet.")
        return
    print(f"\n{'='*60}")
    print(f"  Performance Log Summary (last {len(logs)} entries)")
    print(f"{'='*60}")
    print(f"{'Function':<35} {'Duration':<12} {'Status':<10}")
    print(f"{'-'*35} {'-'*12} {'-'*10}")
    for row in logs:
        dur = row.get("duration_seconds", "")
        status = row.get("status", "")
        func = row.get("function_name", "")
        print(f"  {func:<35} {dur:<12} {status:<10}")
    print(f"{'='*60}")

"""Persistent, resumable, idempotent literature task orchestration."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.oa_literature import top_up_open_access
from src.stage1_store import BASE_DIR


TASK_STATUSES = {
    "PENDING",
    "SEARCHING",
    "DOWNLOADING",
    "DEEP_READING",
    "INDEXING",
    "COMPLETED",
    "PARTIAL",
    "FAILED",
    "MANUAL_ACTION_REQUIRED",
}
TERMINAL_STATUSES = {
    "COMPLETED",
    "PARTIAL",
    "FAILED",
    "MANUAL_ACTION_REQUIRED",
}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def task_path(base_dir: Path = BASE_DIR) -> Path:
    return base_dir / "data" / "tasks" / "literature_tasks.jsonl"


def _read_tasks(base_dir: Path = BASE_DIR) -> List[Dict[str, Any]]:
    path = task_path(base_dir)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_tasks(rows: List[Dict[str, Any]], base_dir: Path = BASE_DIR) -> None:
    path = task_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


def stable_task_id(query: str) -> str:
    digest = hashlib.sha256(" ".join(query.lower().split()).encode()).hexdigest()
    return f"LIT_{digest[:20].upper()}"


def create_literature_task(
    query: str,
    *,
    batch_size: int = 3,
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    if not 3 <= int(batch_size) <= 5:
        raise ValueError("batch_size must be between 3 and 5")
    task_id = stable_task_id(query)
    rows = _read_tasks(base_dir)
    for row in rows:
        if row.get("task_id") == task_id:
            return row
    task = {
        "task_id": task_id,
        "query": query,
        "status": "PENDING",
        "created_at": _now(),
        "started_at": "",
        "completed_at": "",
        "candidate_count": 0,
        "downloaded_count": 0,
        "deep_read_count": 0,
        "indexed_count": 0,
        "error": "",
        "retry_count": 0,
        "batch_size": int(batch_size),
        "checkpoint": {
            "last_completed_phase": "",
            "downloaded_paper_ids": [],
            "topup_result": {},
        },
    }
    rows.append(task)
    _write_tasks(rows, base_dir)
    return task


def get_task(task_id: str, base_dir: Path = BASE_DIR) -> Optional[Dict[str, Any]]:
    return next(
        (row for row in _read_tasks(base_dir) if row.get("task_id") == task_id),
        None,
    )


def update_task(
    task_id: str,
    *,
    base_dir: Path = BASE_DIR,
    **updates: Any,
) -> Dict[str, Any]:
    rows = _read_tasks(base_dir)
    for index, row in enumerate(rows):
        if row.get("task_id") != task_id:
            continue
        status = updates.get("status", row.get("status"))
        if status not in TASK_STATUSES:
            raise ValueError(f"Unsupported task status: {status}")
        changed = {**row, **updates}
        rows[index] = changed
        _write_tasks(rows, base_dir)
        return changed
    raise KeyError(task_id)


def retry_with_exponential_backoff(
    operation: Callable[[], Dict[str, Any]],
    *,
    max_retries: int = 3,
    base_delay_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Dict[str, Any], int]:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return operation(), attempt
        except (requests_retryable_errors()) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            sleep(base_delay_seconds * (2**attempt))
    assert last_error is not None
    raise last_error


def requests_retryable_errors() -> tuple[type[BaseException], ...]:
    import requests

    return (
        requests.Timeout,
        requests.ConnectionError,
        requests.HTTPError,
    )


def process_literature_task(
    task_id: str,
    *,
    base_dir: Path = BASE_DIR,
    topup: Callable[..., Dict[str, Any]] = top_up_open_access,
    sleep: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    task = get_task(task_id, base_dir)
    if task is None:
        raise KeyError(task_id)
    if task.get("status") == "COMPLETED":
        return task
    checkpoint = dict(task.get("checkpoint") or {})
    if checkpoint.get("last_completed_phase") == "INDEXING":
        return update_task(
            task_id,
            base_dir=base_dir,
            status="COMPLETED",
            completed_at=task.get("completed_at") or _now(),
        )
    if not task.get("started_at"):
        task = update_task(
            task_id,
            base_dir=base_dir,
            status="SEARCHING",
            started_at=_now(),
            error="",
        )
    else:
        task = update_task(
            task_id, base_dir=base_dir, status="SEARCHING", error=""
        )

    try:
        result, retries = retry_with_exponential_backoff(
            lambda: topup(
                task["query"],
                max_downloads=int(task.get("batch_size") or 3),
                base_dir=base_dir,
            ),
            sleep=sleep,
        )
        task = update_task(
            task_id,
            base_dir=base_dir,
            status="DOWNLOADING",
            retry_count=int(task.get("retry_count") or 0) + retries,
            candidate_count=int(result.get("candidate_count") or 0),
            checkpoint={
                **checkpoint,
                "last_completed_phase": "SEARCHING",
                "topup_result": result,
            },
        )
        downloads = result.get("downloaded") or []
        downloaded_count = sum(
            row.get("status") in {"DEEP_READ_COMPLETE", "DEEP_READ_PARTIAL"}
            for row in downloads
        )
        deep_read_count = sum(
            row.get("status") == "DEEP_READ_COMPLETE" for row in downloads
        )
        task = update_task(
            task_id,
            base_dir=base_dir,
            status="DEEP_READING",
            downloaded_count=downloaded_count,
            checkpoint={
                **task["checkpoint"],
                "last_completed_phase": "DOWNLOADING",
            },
        )
        new_paper_ids = list(result.get("new_paper_ids") or [])
        task = update_task(
            task_id,
            base_dir=base_dir,
            status="INDEXING",
            deep_read_count=deep_read_count,
            checkpoint={
                **task["checkpoint"],
                "last_completed_phase": "DEEP_READING",
                "downloaded_paper_ids": new_paper_ids,
            },
        )
        manual = any(
            row.get("manual_download_required") for row in downloads
        )
        final_status = (
            "COMPLETED"
            if result.get("status") in {"COMPLETED", "NOT_REQUIRED"}
            else "MANUAL_ACTION_REQUIRED"
            if manual and not new_paper_ids
            else "PARTIAL"
        )
        return update_task(
            task_id,
            base_dir=base_dir,
            status=final_status,
            completed_at=_now(),
            indexed_count=len(new_paper_ids),
            error="; ".join(result.get("errors") or []),
            checkpoint={
                **task["checkpoint"],
                "last_completed_phase": "INDEXING",
            },
        )
    except Exception as exc:
        return update_task(
            task_id,
            base_dir=base_dir,
            status="FAILED",
            completed_at=_now(),
            error=str(exc),
            retry_count=int(task.get("retry_count") or 0) + 1,
            checkpoint={
                **checkpoint,
                "last_completed_phase": checkpoint.get(
                    "last_completed_phase", ""
                ),
            },
        )


def run_pending_tasks(
    *,
    base_dir: Path = BASE_DIR,
    limit: int = 1,
) -> List[Dict[str, Any]]:
    pending = [
        row
        for row in _read_tasks(base_dir)
        if row.get("status")
        in {
            "PENDING",
            "SEARCHING",
            "DOWNLOADING",
            "DEEP_READING",
            "INDEXING",
            "FAILED",
        }
    ]
    return [
        process_literature_task(row["task_id"], base_dir=base_dir)
        for row in pending[: max(0, limit)]
    ]

"""Persistent, resumable, idempotent literature task orchestration."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.oa_literature import top_up_open_access
from src.stage1_store import BASE_DIR


TASK_STATUSES = {
    "PENDING",
    "RUNNING",
    "SEARCHING",
    "DOWNLOADING",
    "DEEP_READING",
    "REINDEXING",
    "NOT_REQUIRED",
    # Legacy values remain readable so interrupted Stage-3 tasks can resume.
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
    "NOT_REQUIRED",
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


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "").strip().lower())


def stable_task_id(query: str, task_type: str = "automatic_topup") -> str:
    payload = f"{task_type}|{normalize_query(query)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"LIT_{digest[:20].upper()}"


def create_literature_task(
    query: str,
    *,
    task_type: str = "automatic_topup",
    source: str = "stage3",
    max_papers: Optional[int] = None,
    manual_override: bool = False,
    evidence_status_before: str = "",
    batch_size: Optional[int] = None,
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    if task_type not in {"automatic_topup", "manual_topup"}:
        raise ValueError("task_type must be automatic_topup or manual_topup")
    if task_type == "manual_topup":
        manual_override = True
    if max_papers is None:
        max_papers = int(batch_size) if batch_size is not None else 1
    max_papers = int(max_papers)
    if max_papers < 1 or max_papers > 5:
        raise ValueError("max_papers must be between 1 and 5")
    if batch_size is not None and not 1 <= int(batch_size) <= 5:
        raise ValueError("batch_size must be between 1 and 5")
    normalized = normalize_query(query)
    if not normalized:
        raise ValueError("query must not be empty")
    task_id = stable_task_id(query, task_type)
    rows = _read_tasks(base_dir)
    for row in rows:
        if row.get("task_id") == task_id:
            return row
    task = {
        "task_id": task_id,
        "task_type": task_type,
        "query": query,
        "normalized_query": normalized,
        "source": source,
        "status": "PENDING",
        "created_at": _now(),
        "started_at": "",
        "completed_at": "",
        "candidate_count": 0,
        "oa_candidate_count": 0,
        "downloaded_count": 0,
        "duplicate_rejected_count": 0,
        "paywall_rejected_count": 0,
        "deep_read_count": 0,
        "indexed_count": 0,
        "max_papers": max_papers,
        "manual_override": bool(manual_override),
        "evidence_status_before": evidence_status_before,
        "attempt_count": 0,
        "last_error": "",
        "error": "",
        "retry_count": 0,
        # Kept for compatibility with existing Stage-3 callers.
        "batch_size": max_papers,
        "checkpoint": {
            "last_completed_phase": "",
            "downloaded_paper_ids": [],
            "result_summary": {},
        },
    }
    rows.append(task)
    _write_tasks(rows, base_dir)
    return task


def list_literature_tasks(
    base_dir: Path = BASE_DIR,
    *,
    newest_first: bool = True,
) -> List[Dict[str, Any]]:
    rows = _read_tasks(base_dir)
    return list(reversed(rows)) if newest_first else rows


def compact_task_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep task JSON small: no full PDF text, retrieval rows, or deep-read payloads."""
    downloads = list(result.get("downloaded") or [])
    return {
        "status": result.get("status") or "",
        "message": result.get("message") or "",
        "candidate_count": int(result.get("candidate_count") or 0),
        "oa_candidate_count": int(result.get("oa_candidate_count") or 0),
        "downloaded_count": sum(
            row.get("status") in {"DEEP_READ_COMPLETE", "DEEP_READ_PARTIAL"}
            for row in downloads
        ),
        "duplicate_rejected_count": int(
            result.get("duplicate_rejected_count") or 0
        ),
        "paywall_rejected_count": int(result.get("paywall_rejected_count") or 0),
        "new_paper_ids": list(result.get("new_paper_ids") or []),
        "source_results": list(result.get("source_results") or []),
        "errors": [str(value)[:500] for value in result.get("errors") or []],
        "cache_invalidated": bool(result.get("cache_invalidated")),
    }


def create_topup_task_from_evidence(
    query: str,
    evidence: Dict[str, Any],
    *,
    manual: bool = False,
    source: str = "streamlit_ui",
    base_dir: Path = BASE_DIR,
) -> Optional[Dict[str, Any]]:
    status = str(
        evidence.get("evidence_status")
        or (evidence.get("evidence_sufficiency") or {}).get("status")
        or ""
    )
    if not manual and status not in {"INSUFFICIENT", "PARTIALLY_SUFFICIENT"}:
        return None
    return create_literature_task(
        query,
        task_type="manual_topup" if manual else "automatic_topup",
        source=source,
        max_papers=1,
        manual_override=manual,
        evidence_status_before=status,
        base_dir=base_dir,
    )


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
    if task.get("status") in {"COMPLETED", "NOT_REQUIRED"}:
        return task
    checkpoint = dict(task.get("checkpoint") or {})
    if checkpoint.get("last_completed_phase") in {"INDEXING", "REINDEXING"}:
        return update_task(
            task_id,
            base_dir=base_dir,
            status="COMPLETED",
            completed_at=task.get("completed_at") or _now(),
        )
    task = update_task(
        task_id,
        base_dir=base_dir,
        status="RUNNING",
        started_at=task.get("started_at") or _now(),
        attempt_count=int(task.get("attempt_count") or 0) + 1,
        last_error="",
        error="",
        checkpoint={
            **checkpoint,
            "current_phase": "RUNNING",
        },
    )

    def progress(phase: str, payload: Optional[Dict[str, Any]] = None) -> None:
        nonlocal task
        phase = str(phase or "").upper()
        if phase == "INDEXING":
            phase = "REINDEXING"
        if phase not in {
            "SEARCHING",
            "DOWNLOADING",
            "DEEP_READING",
            "REINDEXING",
        }:
            return
        payload = dict(payload or {})
        downloaded_ids = list(
            payload.get("downloaded_paper_ids")
            or (task.get("checkpoint") or {}).get("downloaded_paper_ids")
            or []
        )
        task = update_task(
            task_id,
            base_dir=base_dir,
            status=phase,
            candidate_count=int(
                payload.get("candidate_count", task.get("candidate_count") or 0)
            ),
            oa_candidate_count=int(
                payload.get(
                    "oa_candidate_count", task.get("oa_candidate_count") or 0
                )
            ),
            downloaded_count=int(
                payload.get(
                    "downloaded_count", task.get("downloaded_count") or 0
                )
            ),
            deep_read_count=int(
                payload.get(
                    "deep_read_count", task.get("deep_read_count") or 0
                )
            ),
            checkpoint={
                **(task.get("checkpoint") or {}),
                "current_phase": phase,
                "last_completed_phase": str(
                    payload.get("last_completed_phase")
                    or (task.get("checkpoint") or {}).get(
                        "last_completed_phase", ""
                    )
                ),
                "downloaded_paper_ids": downloaded_ids,
            },
        )

    try:
        parameters = inspect.signature(topup).parameters
        supports_kwargs = any(
            value.kind == inspect.Parameter.VAR_KEYWORD
            for value in parameters.values()
        )
        topup_kwargs: Dict[str, Any] = {
            "max_downloads": int(task.get("max_papers") or task.get("batch_size") or 1),
            "base_dir": base_dir,
            "manual_override": bool(task.get("manual_override")),
            "evidence_status_before": task.get("evidence_status_before") or "",
            "progress_callback": progress,
        }
        if not supports_kwargs:
            topup_kwargs = {
                key: value
                for key, value in topup_kwargs.items()
                if key in parameters
            }
        progress("SEARCHING")
        result, retries = retry_with_exponential_backoff(
            lambda: topup(task["query"], **topup_kwargs),
            sleep=sleep,
        )
        summary = compact_task_result(result)
        task = update_task(
            task_id,
            base_dir=base_dir,
            status=task.get("status") or "RUNNING",
            retry_count=int(task.get("retry_count") or 0) + retries,
            candidate_count=summary["candidate_count"],
            oa_candidate_count=summary["oa_candidate_count"],
            duplicate_rejected_count=summary["duplicate_rejected_count"],
            paywall_rejected_count=summary["paywall_rejected_count"],
            checkpoint={
                **(task.get("checkpoint") or {}),
                "result_summary": summary,
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
        new_paper_ids = list(result.get("new_paper_ids") or [])
        result_status = str(result.get("status") or "PARTIAL")
        if result_status == "NOT_REQUIRED":
            final_status = "NOT_REQUIRED"
        elif result_status == "COMPLETED":
            final_status = "COMPLETED"
        elif result_status == "FAILED":
            final_status = "FAILED"
        else:
            final_status = "PARTIAL"
        result_errors = [str(value) for value in result.get("errors") or []]
        last_error = "; ".join(result_errors)
        return update_task(
            task_id,
            base_dir=base_dir,
            status=final_status,
            completed_at=_now(),
            downloaded_count=downloaded_count,
            deep_read_count=deep_read_count,
            indexed_count=len(new_paper_ids),
            last_error=last_error,
            error=last_error,
            checkpoint={
                **(task.get("checkpoint") or {}),
                "current_phase": final_status,
                "last_completed_phase": (
                    "REINDEXING" if new_paper_ids else "SEARCHING"
                ),
                "downloaded_paper_ids": new_paper_ids,
                "result_summary": summary,
            },
        )
    except Exception as exc:
        return update_task(
            task_id,
            base_dir=base_dir,
            status="FAILED",
            completed_at=_now(),
            last_error=str(exc),
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
            "RUNNING",
            "SEARCHING",
            "DOWNLOADING",
            "DEEP_READING",
            "REINDEXING",
            "INDEXING",
            "FAILED",
        }
    ]
    return [
        process_literature_task(row["task_id"], base_dir=base_dir)
        for row in pending[: max(0, limit)]
    ]

"""Persistent single-worker PDF inbox watcher."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import fitz

from src.api_keys import get_deepseek_settings
from src.auto_oa_pipeline import evaluate_auto_rag_gate, index_auto_validated_paper
from src.deep_read_pipeline import deep_read_pdf
from src.full_library_deep_read import _postprocess_artifacts
from src.metadata_gate import run_metadata_gate
from src.stage1_store import (
    load_paper_manifest,
    register_pdf_path,
    update_paper_library_status,
)


STAGES = (
    "DISCOVERED", "VALIDATING", "DEDUPLICATING", "EXTRACTING_METADATA",
    "DEEP_READING", "EXTRACTING_EVIDENCE", "EXTRACTING_CONDITIONS",
    "EXTRACTING_FORMULAS", "QUALITY_GATE", "BUILDING_RAG",
    "FORMAL_INDEXED", "REJECTED_AND_DELETED",
)
TERMINAL = {"FORMAL_INDEXED", "REJECTED_AND_DELETED"}
TEMPORARY_SUFFIXES = (".crdownload", ".part", ".tmp", ".download")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def watch_paths(base_dir: Path) -> dict[str, Path]:
    system = base_dir / "data" / "system"
    return {
        "inbox": base_dir / "paper" / "pdfs",
        "queue": system / "pdf_watch_queue.json",
        "status": system / "pdf_watch_status.json",
        "control": system / "pdf_watch_control.json",
    }


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_queue(base_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "pdf-watch-1.0",
        "inbox": str(watch_paths(base_dir)["inbox"].resolve()),
        "tasks": [],
        "observations": {},
        "updated_at": _now(),
    }


def load_queue(base_dir: Path) -> dict[str, Any]:
    return _read_json(watch_paths(base_dir)["queue"], _default_queue(base_dir))


def _save_queue(base_dir: Path, queue: dict[str, Any]) -> None:
    queue["updated_at"] = _now()
    _atomic_json(watch_paths(base_dir)["queue"], queue)


def set_control(action: str, *, base_dir: Path) -> dict[str, Any]:
    normalized = action.upper()
    if normalized not in {"RUN", "PAUSE", "STOP"}:
        raise ValueError("watcher control must be RUN, PAUSE or STOP")
    payload = {"action": normalized, "updated_at": _now()}
    _atomic_json(watch_paths(base_dir)["control"], payload)
    return payload


def _control(base_dir: Path) -> str:
    return str(
        _read_json(watch_paths(base_dir)["control"], {"action": "RUN"}).get(
            "action", "RUN"
        )
    ).upper()


def _file_can_be_opened(path: Path) -> bool:
    try:
        with path.open("rb+"):
            return True
    except OSError:
        return False


def discover_stable_files(
    *,
    base_dir: Path,
    stable_checks: int = 3,
    debounce_seconds: float = 1.0,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Record one observation and enqueue only files stable across three checks."""
    paths = watch_paths(base_dir)
    paths["inbox"].mkdir(parents=True, exist_ok=True)
    queue = load_queue(base_dir)
    observations = queue.setdefault("observations", {})
    tasks = queue.setdefault("tasks", [])
    active_paths = {
        str(task.get("path") or "").casefold()
        for task in tasks
        if task.get("stage") not in TERMINAL
    }
    terminal_fingerprints = {
        (str(task.get("path") or "").casefold(), str(task.get("sha256") or ""))
        for task in tasks
        if task.get("stage") in TERMINAL
    }
    known_canonical = {
        str(Path(str(row.get("canonical_pdf_path") or "")).resolve()).casefold():
        str(row.get("file_hash_sha256") or "")
        for row in load_paper_manifest(base_dir)
        if row.get("canonical_pdf_path") and row.get("file_hash_sha256")
    }
    timestamp = float(now_epoch if now_epoch is not None else time.time())
    enqueued = 0
    for path in sorted(paths["inbox"].iterdir()):
        if not path.is_file() or path.suffix.casefold() != ".pdf":
            continue
        if path.name.casefold().endswith(TEMPORARY_SUFFIXES):
            continue
        resolved = str(path.resolve())
        stat = path.stat()
        fingerprint = f"{stat.st_size}:{stat.st_mtime_ns}"
        previous = dict(observations.get(resolved) or {})
        unchanged = int(previous.get("unchanged_checks") or 0) + 1 if previous.get("fingerprint") == fingerprint else 1
        first_stable = (
            float(previous["first_stable_epoch"])
            if previous.get("fingerprint") == fingerprint
            and "first_stable_epoch" in previous
            else timestamp
        )
        observations[resolved] = {
            "fingerprint": fingerprint,
            "unchanged_checks": unchanged,
            "first_stable_epoch": first_stable,
            "last_seen_epoch": timestamp,
        }
        if unchanged < stable_checks or timestamp - first_stable < debounce_seconds:
            continue
        if resolved.casefold() in active_paths or not _file_can_be_opened(path):
            continue
        file_hash = _sha256(path)
        if known_canonical.get(resolved.casefold()) == file_hash:
            observations[resolved]["processed_sha256"] = file_hash
            continue
        if (resolved.casefold(), file_hash) in terminal_fingerprints:
            continue
        task_id = "PDFWATCH_" + hashlib.sha256(
            f"{resolved.casefold()}:{file_hash}".encode("utf-8")
        ).hexdigest()[:20].upper()
        if any(task.get("task_id") == task_id for task in tasks):
            continue
        tasks.append(
            {
                "task_id": task_id,
                "path": resolved,
                "filename": path.name,
                "sha256": file_hash,
                "stage": "DISCOVERED",
                "page_current": 0,
                "page_total": 0,
                "created_at": _now(),
                "updated_at": _now(),
                "last_error_cn": "",
                "deepseek_call_count": 0,
            }
        )
        active_paths.add(resolved.casefold())
        enqueued += 1
    _save_queue(base_dir, queue)
    return {"enqueued": enqueued, "task_count": len(tasks)}


def validate_pdf(path: Path) -> dict[str, Any]:
    result = {"passed": False, "reason_cn": "", "page_count": 0, "sha256": ""}
    try:
        if path.suffix.casefold() != ".pdf":
            result["reason_cn"] = "文件扩展名不是 PDF"
            return result
        content = path.read_bytes()
        if len(content) < 512:
            result["reason_cn"] = "文件过小或为空"
            return result
        if not content.startswith(b"%PDF"):
            result["reason_cn"] = "文件头不是有效 PDF"
            return result
        if content.lstrip().lower().startswith((b"<html", b"<!doctype html")):
            result["reason_cn"] = "文件是 HTML，不是真实 PDF"
            return result
        result["sha256"] = hashlib.sha256(content).hexdigest()
        with fitz.open(stream=content, filetype="pdf") as document:
            result["page_count"] = int(document.page_count)
            if document.page_count < 1:
                result["reason_cn"] = "PDF 没有有效页面"
                return result
            text = "\n".join(
                document[index].get_text("text")
                for index in range(min(document.page_count, 4))
            ).strip()
            if document.page_count == 1 and len(text) < 400:
                result["reason_cn"] = "PDF 只有封面、目录或内容不足"
                return result
            if len(text) < 80:
                result["reason_cn"] = "PDF 正文无法可靠读取"
                return result
        result["passed"] = True
        return result
    except (OSError, RuntimeError, ValueError) as exc:
        result["reason_cn"] = f"PDF 无法打开：{type(exc).__name__}"
        return result


def _update_task(base_dir: Path, task_id: str, **updates: Any) -> dict[str, Any]:
    queue = load_queue(base_dir)
    task = next(item for item in queue["tasks"] if item["task_id"] == task_id)
    task.update(updates)
    task["updated_at"] = _now()
    _save_queue(base_dir, queue)
    _write_status(base_dir)
    return task


def _recycle(path: Path, base_dir: Path) -> None:
    from scripts.finalize_formal_library import send_to_recycle_bin

    if path.exists():
        send_to_recycle_bin(path, base_dir=base_dir)


def _cleanup_rejected(base_dir: Path) -> None:
    from scripts.finalize_formal_library import apply_plan, build_plan

    apply_plan(base_dir, build_plan(base_dir))


def _record_success(base_dir: Path, paper_id: str) -> None:
    paths = watch_paths(base_dir)
    whitelist_path = base_dir / "data" / "system" / "formal_rag_whitelist.json"
    payload = _read_json(whitelist_path, {"paper_ids": []})
    payload["paper_ids"] = list(dict.fromkeys([*(payload.get("paper_ids") or []), paper_id]))
    payload["paper_count"] = len(payload["paper_ids"])
    payload["generated_at"] = _now()
    payload["schema_version"] = "formal-rag-whitelist-2.0"
    _atomic_json(whitelist_path, payload)
    dataset = {
        "version": hashlib.sha256(
            (base_dir / "data" / "rag" / "manifest.json").read_bytes()
        ).hexdigest(),
        "updated_at": _now(),
    }
    _atomic_json(paths["status"].parent / "dataset_version.json", dataset)


def process_task(
    task: dict[str, Any],
    *,
    base_dir: Path,
    processor: Callable[[Path, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Process one task.  A processor hook keeps end-to-end tests isolated."""
    task_id = str(task["task_id"])
    path = Path(str(task["path"]))
    _update_task(base_dir, task_id, stage="VALIDATING")
    validation = validate_pdf(path)
    if not validation["passed"]:
        _recycle(path, base_dir)
        return _update_task(
            base_dir, task_id, stage="REJECTED_AND_DELETED",
            last_error_cn=validation["reason_cn"], page_total=validation["page_count"],
        )
    _update_task(base_dir, task_id, stage="DEDUPLICATING", page_total=validation["page_count"])
    current = load_paper_manifest(base_dir)
    duplicate = next(
        (row for row in current if str(row.get("file_hash_sha256") or "") == validation["sha256"]),
        None,
    )
    if duplicate:
        canonical = Path(str(duplicate.get("canonical_pdf_path") or "")).resolve()
        if path.resolve() != canonical:
            _recycle(path, base_dir)
        return _update_task(
            base_dir, task_id, stage="REJECTED_AND_DELETED",
            last_error_cn="已识别为完全重复 PDF，未重复精读", duplicate_of=duplicate.get("paper_id"),
            deepseek_call_count=0,
        )
    if processor is not None:
        result = processor(path, task)
        if result.get("status") == "FORMAL_INDEXED":
            updates = dict(result)
            updates.pop("status", None)
            return _update_task(
                base_dir, task_id, stage="FORMAL_INDEXED", **updates
            )
        _recycle(path, base_dir)
        return _update_task(
            base_dir, task_id, stage="REJECTED_AND_DELETED",
            last_error_cn=str(result.get("reason_cn") or "质量门禁未通过"),
        )

    _update_task(base_dir, task_id, stage="EXTRACTING_METADATA")
    registration = register_pdf_path(path, source_type="PDF_WATCHER", base_dir=base_dir)
    paper_id = str(registration.get("paper_id") or "")
    if not paper_id:
        _recycle(path, base_dir)
        return _update_task(
            base_dir, task_id, stage="REJECTED_AND_DELETED",
            last_error_cn="无法建立可靠 canonical 文献身份",
        )
    metadata = run_metadata_gate(base_dir, [paper_id], workers=1)
    if not metadata.get("passed_count"):
        _cleanup_rejected(base_dir)
        return _update_task(
            base_dir, task_id, stage="REJECTED_AND_DELETED", paper_id=paper_id,
            last_error_cn="题名、作者、年份、DOI 或文献类型未通过元数据门禁",
        )

    _update_task(base_dir, task_id, stage="DEEP_READING", paper_id=paper_id)
    settings = get_deepseek_settings()
    deep = deep_read_pdf(
        path, paper_id=paper_id,
        title=str(registration.get("title") or ""), base_dir=base_dir,
        use_deepseek=settings.configured,
    )
    _update_task(
        base_dir, task_id, page_current=int(deep.get("processed_page_count") or 0),
        page_total=int(deep.get("real_page_count") or validation["page_count"]),
        deepseek_call_count=int((deep.get("deepseek_usage") or {}).get("api_call_count") or 0),
    )
    if not deep.get("deep_read_complete"):
        _cleanup_rejected(base_dir)
        return _update_task(
            base_dir, task_id, stage="REJECTED_AND_DELETED",
            last_error_cn="逐页精读未完整通过",
        )
    for stage in ("EXTRACTING_EVIDENCE", "EXTRACTING_CONDITIONS", "EXTRACTING_FORMULAS"):
        _update_task(base_dir, task_id, stage=stage)
    _postprocess_artifacts(base_dir, paper_id, str(registration.get("title") or ""))
    _update_task(base_dir, task_id, stage="QUALITY_GATE")
    gate = evaluate_auto_rag_gate(paper_id, base_dir=base_dir)
    if not gate["passed"]:
        _cleanup_rejected(base_dir)
        return _update_task(
            base_dir, task_id, stage="REJECTED_AND_DELETED",
            last_error_cn="证据质量门禁未通过",
        )
    _update_task(base_dir, task_id, stage="BUILDING_RAG")
    indexed = index_auto_validated_paper(paper_id, base_dir=base_dir)
    if indexed.get("status") != "INDEXED_STAGE3_UNIFIED":
        _cleanup_rejected(base_dir)
        return _update_task(
            base_dir, task_id, stage="REJECTED_AND_DELETED",
            last_error_cn="统一 RAG 增量更新失败",
        )
    update_paper_library_status(paper_id, "FORMAL", base_dir=base_dir)
    _record_success(base_dir, paper_id)
    return _update_task(
        base_dir, task_id, stage="FORMAL_INDEXED", paper_id=paper_id,
        title=str(registration.get("title") or ""), last_error_cn="",
    )


def _write_status(base_dir: Path, *, running: bool | None = None) -> dict[str, Any]:
    queue = load_queue(base_dir)
    tasks = queue.get("tasks") or []
    active = next((task for task in tasks if task.get("stage") not in TERMINAL), {})
    successes = [task for task in tasks if task.get("stage") == "FORMAL_INDEXED"]
    rejected = [task for task in tasks if task.get("stage") == "REJECTED_AND_DELETED"]
    payload = {
        "watching": bool(running) if running is not None else _control(base_dir) == "RUN",
        "paused": _control(base_dir) == "PAUSE",
        "pid": os.getpid() if running else int(_read_json(watch_paths(base_dir)["status"], {}).get("pid") or 0),
        "current_file": str(active.get("filename") or ""),
        "current_stage": str(active.get("stage") or ""),
        "page_current": int(active.get("page_current") or 0),
        "page_total": int(active.get("page_total") or 0),
        "waiting_count": sum(task.get("stage") not in TERMINAL for task in tasks),
        "last_success_title": str(successes[-1].get("title") or "") if successes else "",
        "rejected_deleted_count": len(rejected),
        "updated_at": _now(),
    }
    _atomic_json(watch_paths(base_dir)["status"], payload)
    return payload


def public_status(base_dir: Path) -> dict[str, Any]:
    status = _read_json(watch_paths(base_dir)["status"], {})
    return {
        "自动监听": "已暂停" if status.get("paused") else "运行中" if status.get("watching") else "已停止",
        "当前文件": status.get("current_file") or "无",
        "当前阶段": status.get("current_stage") or "空闲",
        "页面进度": f"{int(status.get('page_current') or 0)}/{int(status.get('page_total') or 0)}",
        "等待队列": int(status.get("waiting_count") or 0),
        "最近成功": status.get("last_success_title") or "无",
        "最近删除": int(status.get("rejected_deleted_count") or 0),
    }


def run_watcher(
    *, base_dir: Path, once: bool = False, poll_interval: float = 1.0,
    stable_checks: int = 3, processor: Callable[[Path, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    set_control("RUN", base_dir=base_dir)
    _write_status(base_dir, running=True)
    idle_stable_rounds = 0
    processed = 0
    try:
        while _control(base_dir) != "STOP":
            if _control(base_dir) == "PAUSE":
                _write_status(base_dir, running=True)
                if once:
                    break
                time.sleep(poll_interval)
                continue
            discovered = discover_stable_files(
                base_dir=base_dir, stable_checks=stable_checks,
                debounce_seconds=max(0.0, poll_interval * (stable_checks - 1)),
            )
            queue = load_queue(base_dir)
            pending = next(
                (task for task in queue["tasks"] if task.get("stage") not in TERMINAL), None
            )
            if pending:
                process_task(pending, base_dir=base_dir, processor=processor)
                processed += 1
                idle_stable_rounds = 0
            else:
                idle_stable_rounds += 1
            _write_status(base_dir, running=True)
            if once and idle_stable_rounds >= stable_checks and not discovered["enqueued"]:
                break
            time.sleep(poll_interval)
    finally:
        _write_status(base_dir, running=False)
    return {"processed": processed, "status": public_status(base_dir)}


def ensure_background_watcher(base_dir: Path) -> bool:
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("TITANIUM_WATCHER_CHILD"):
        return False
    paths = watch_paths(base_dir)
    if not paths["inbox"].exists():
        return False
    prior = _read_json(paths["status"], {})
    pid = int(prior.get("pid") or 0)
    if pid:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            pass
    env = dict(os.environ)
    env["TITANIUM_WATCHER_CHILD"] = "1"
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    subprocess.Popen(
        [sys.executable, str(base_dir / "app.py"), "watch-pdfs"],
        cwd=str(base_dir), env=env, creationflags=flags,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return True

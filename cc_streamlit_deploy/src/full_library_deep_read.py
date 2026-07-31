"""Persistent, resumable full-library PDF deep-reading orchestration.

Scientific extraction remains in :mod:`src.deep_read_pipeline`; this module
only inventories physical PDFs, deduplicates logical works, persists task/page
checkpoints, applies the existing Stage-2/Stage-3 gates, and writes audit reports.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import fitz

from src.auto_oa_pipeline import evaluate_auto_rag_gate, index_auto_validated_paper
from src.deep_read_pipeline import deep_read_pdf, deep_read_paths
from src.domain_scope import OUT_OF_SCOPE, classify_literature_scope
from src.metadata_service import fetch_metadata, is_valid_title
from src.stage1_store import (
    BASE_DIR,
    clean_pdf_title,
    extract_basic_pdf_metadata,
    load_paper_manifest,
    load_trusted_evidence_rows,
    normalize_doi,
    normalize_title,
    register_pdf_path,
    register_metadata_record,
    sha256_file,
    update_paper_extraction_status,
    update_paper_library_status,
    update_paper_rag_status,
    update_paper_verified_metadata,
    validate_pdf_path,
)
from src.unified_rag import rag_paths


QUEUE_VERSION = "full-library-deep-read-1.0"
TERMINAL_STATES = {
    "COMPLETED", "NEEDS_HUMAN_REVIEW", "SKIPPED_DUPLICATE", "FAILED_RETRYABLE"
}
RUNNING_STATES = {
    "PDF_VALIDATED", "METADATA_VALIDATED", "PAGE_EXTRACTION_RUNNING",
    "PAGE_EXTRACTION_COMPLETE", "DEEP_READING", "EVIDENCE_AUDITING",
    "QUALITY_GATING", "INDEXING",
}
TASK_ROOT_RELATIVE = Path("data") / "tasks" / "full_library_deep_read"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def queue_paths(base_dir: Path = BASE_DIR) -> Dict[str, Path]:
    root = base_dir / TASK_ROOT_RELATIVE
    return {
        "root": root,
        "inventory": root / "inventory.json",
        "queue": root / "queue.json",
        "control": root / "control.json",
        "events": root / "events.jsonl",
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(6):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _event(base_dir: Path, task_id: str, status: str, detail: str = "") -> None:
    path = queue_paths(base_dir)["events"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "timestamp": _now(), "task_id": task_id, "status": status,
            "detail": detail,
        }, ensure_ascii=False) + "\n")


def _pdf_text_fingerprint(path: Path) -> tuple[str, int, bool]:
    digest = hashlib.sha256()
    characters = 0
    extractable = False
    with fitz.open(path) as document:
        for page in document:
            text = " ".join(page.get_text("text").split()).lower()
            characters += len(text)
            if len(text) >= 25:
                extractable = True
            digest.update(text.encode("utf-8", errors="ignore"))
    return digest.hexdigest(), characters, extractable


def _year(value: str) -> str:
    match = re.search(r"\b(?:19|20)\d{2}\b", str(value or ""))
    return match.group(0) if match else ""


def _first_author(value: str) -> str:
    first = re.split(r"[;,]", str(value or ""))[0]
    return normalize_title(first)


def inventory_pdfs(pdf_dir: Path | str, *, base_dir: Path = BASE_DIR) -> Dict[str, Any]:
    """Build a real-page inventory and hierarchical logical-document dedup map."""
    root = Path(pdf_dir)
    if not root.is_absolute():
        root = (base_dir / root).resolve()
    paths = sorted(
        (path for path in root.rglob("*.pdf") if path.is_file()),
        key=lambda value: str(value).lower(),
    ) if root.exists() else []
    manifests = load_paper_manifest(base_dir)
    by_hash = {str(row.get("file_hash_sha256") or ""): row for row in manifests}
    rows: List[Dict[str, Any]] = []
    for path in paths:
        validation = validate_pdf_path(path)
        file_hash = sha256_file(path)
        metadata: Dict[str, Any] = {}
        fingerprint, chars, extractable = "", 0, False
        if validation.get("pdf_valid"):
            metadata = extract_basic_pdf_metadata(path.read_bytes(), path.name)
            try:
                fingerprint, chars, extractable = _pdf_text_fingerprint(path)
            except Exception:
                extractable = False
        current = by_hash.get(file_hash) or {}
        title = str(current.get("title") or metadata.get("title") or "").strip()
        authors = str(current.get("authors") or metadata.get("authors") or "").strip()
        publication = str(
            current.get("publication_date") or metadata.get("publication_date") or ""
        )
        doi = normalize_doi(current.get("doi") or metadata.get("doi") or "")
        status_path = deep_read_paths(base_dir, str(current.get("paper_id") or ""))["status"]
        deep_status = _read_json(status_path, {}) if current else {}
        rows.append({
            "original_filename": path.name,
            "local_path": str(path.resolve()),
            "file_size": path.stat().st_size,
            "sha256": file_hash,
            "valid_pdf": bool(validation.get("pdf_valid")),
            "validation_error": str(validation.get("error") or ""),
            "real_page_count": int(validation.get("real_page_count") or 0),
            "text_extractable": extractable,
            "extracted_character_count": chars,
            "text_fingerprint": fingerprint,
            "title": title,
            "authors": authors,
            "year": _year(publication),
            "doi": doi,
            "canonical_paper_id": str(current.get("paper_id") or ""),
            "current_deep_read_status": str(deep_status.get("status") or current.get("extraction_status") or "NOT_STARTED"),
            "current_evidence_count": int(deep_status.get("evidence_count") or 0),
            "current_rag_status": str(current.get("rag_status") or "NOT_INDEXED"),
        })

    # Union physical files by exact hash, DOI, title/year/author, then full-text fingerprint.
    parent = list(range(len(rows)))
    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left
    keys: Dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        title_key = normalize_title(row["title"]) if is_valid_title(row["title"]) else ""
        logical_keys = [("sha256", row["sha256"])]
        if row["doi"]:
            logical_keys.append(("doi", row["doi"]))
        if title_key and row["year"] and _first_author(row["authors"]):
            logical_keys.append(("title_year_author", f"{title_key}|{row['year']}|{_first_author(row['authors'])}"))
        if row["text_fingerprint"]:
            logical_keys.append(("text_fingerprint", row["text_fingerprint"]))
        for key in logical_keys:
            if key in keys:
                union(index, keys[key])
            else:
                keys[key] = index
    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row)
    logical: List[Dict[str, Any]] = []
    for members in groups.values():
        best = max(members, key=lambda row: (
            row["valid_pdf"], row["text_extractable"], row["extracted_character_count"],
            row["real_page_count"], row["file_size"],
        ))
        hashes = sorted({row["sha256"] for row in members})
        # The selected primary file hash remains stable when metadata is repaired.
        # DOI/title/fingerprint decide grouping, but never task checkpoint identity.
        logical_id = str(best["sha256"])[:20].upper()
        logical.append({
            "logical_document_id": f"LOGICAL_{logical_id}",
            "primary_pdf": best,
            "versions": members,
            "physical_file_count": len(members),
            "unique_hash_count": len(hashes),
            "exact_duplicate_count": len(members) - len(hashes),
            "different_version_count": max(0, len(hashes) - 1),
        })
    result = {
        "schema_version": QUEUE_VERSION,
        "pdf_dir": str(root),
        "scanned_at": _now(),
        "pdf_file_count": len(rows),
        "unique_hash_count": len({row["sha256"] for row in rows}),
        "logical_document_count": len(logical),
        "exact_duplicate_count": len(rows) - len({row["sha256"] for row in rows}),
        "different_version_count": sum(row["different_version_count"] for row in logical),
        "valid_pdf_count": sum(row["valid_pdf"] for row in rows),
        "invalid_pdf_count": sum(not row["valid_pdf"] for row in rows),
        "total_pages": sum(group["primary_pdf"]["real_page_count"] for group in logical if group["primary_pdf"]["valid_pdf"]),
        "files": rows,
        "logical_documents": logical,
    }
    return result


def build_full_library_queue(
    pdf_dir: Path | str,
    *,
    base_dir: Path = BASE_DIR,
    resume: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    inventory = inventory_pdfs(pdf_dir, base_dir=base_dir)
    if dry_run:
        return {"inventory": inventory, "tasks": []}
    paths = queue_paths(base_dir)
    old = _read_json(paths["queue"], {}) if resume else {}
    old_by_logical = {
        str(row.get("logical_document_id")): row for row in old.get("tasks") or []
    }
    tasks = []
    for group in inventory["logical_documents"]:
        primary = group["primary_pdf"]
        existing = old_by_logical.get(group["logical_document_id"], {})
        status = str(existing.get("status") or "PENDING")
        if status in RUNNING_STATES:
            status = "FAILED_RETRYABLE"
            existing["last_error"] = "PROCESS_TERMINATED_DURING_ACTIVE_STAGE"
        task = {
            **existing,
            "task_id": str(existing.get("task_id") or f"FULLREAD_{group['logical_document_id']}"),
            "logical_document_id": group["logical_document_id"],
            "primary_pdf_path": primary["local_path"],
            "original_filename": primary["original_filename"],
            "sha256": primary["sha256"],
            "real_page_count": primary["real_page_count"],
            "valid_pdf": primary["valid_pdf"],
            "text_extractable": primary["text_extractable"],
            "versions": [row["local_path"] for row in group["versions"]],
            "status": status,
            "retry_count": int(existing.get("retry_count") or 0),
            "max_retries": 3,
            "processed_pages": int(existing.get("processed_pages") or 0),
            "updated_at": _now(),
        }
        tasks.append(task)
    queue = {
        "schema_version": QUEUE_VERSION, "pdf_dir": inventory["pdf_dir"],
        "created_at": old.get("created_at") or _now(), "updated_at": _now(),
        "tasks": tasks,
    }
    _atomic_json(paths["inventory"], inventory)
    _atomic_json(paths["queue"], queue)
    if not paths["control"].exists():
        _atomic_json(paths["control"], {"action": "RUN", "updated_at": _now()})
    return {"inventory": inventory, "tasks": tasks}


def load_full_library_queue(base_dir: Path = BASE_DIR) -> Dict[str, Any]:
    return _read_json(queue_paths(base_dir)["queue"], {"tasks": []})


def set_queue_control(action: str, *, base_dir: Path = BASE_DIR) -> Dict[str, Any]:
    action = action.upper()
    if action not in {"RUN", "PAUSE", "STOP_AFTER_CURRENT", "STOP_NOW"}:
        raise ValueError("Unsupported queue control action")
    value = {"action": action, "updated_at": _now()}
    _atomic_json(queue_paths(base_dir)["control"], value)
    return value


def _save_task(base_dir: Path, task_id: str, **updates: Any) -> Dict[str, Any]:
    path = queue_paths(base_dir)["queue"]
    queue = _read_json(path, {"tasks": []})
    target: Dict[str, Any] = {}
    for task in queue.get("tasks") or []:
        if task.get("task_id") == task_id:
            task.update(updates)
            task["updated_at"] = _now()
            target = task
            break
    queue["updated_at"] = _now()
    _atomic_json(path, queue)
    if "status" in updates:
        _event(base_dir, task_id, str(updates["status"]), str(updates.get("last_error") or ""))
    return target


RESEARCH_FIELDS = {
    "research_object": ("material_and_powder",), "titanium_grade": ("material_and_powder",),
    "manufacturing_process": ("lpbf_process",), "additive_method": ("lpbf_process",),
    "process_parameters": ("lpbf_process",), "build_orientation": ("build_orientation",),
    "heat_treatment": ("heat_treatment_and_hip",), "hip_state": ("heat_treatment_and_hip",),
    "surface_state": ("surface_state_and_roughness",), "surface_roughness": ("surface_state_and_roughness",),
    "microstructure": ("microstructure",), "alpha_beta_phase": ("microstructure",),
    "residual_stress": ("residual_stress",), "defect_type": ("pores_and_defects",),
    "pores_lof_inclusions": ("pores_and_defects",), "defect_size": ("pores_and_defects",),
    "defect_location": ("pores_and_defects",), "crack_initiation_site": ("crack_initiation_site",),
    "crack_growth_behavior": ("crack_growth_and_paris",), "fatigue_regime": ("fatigue_test_conditions",),
    "loading_mode": ("fatigue_test_conditions",), "stress_ratio_R": ("fatigue_test_conditions",),
    "stress_or_strain_amplitude": ("fatigue_test_conditions",), "frequency": ("fatigue_test_conditions",),
    "temperature": ("fatigue_test_conditions",), "environment": ("fatigue_test_conditions",),
    "specimen_geometry": ("fatigue_test_conditions",), "sample_count": ("statistical_data",),
    "fatigue_life_Nf": ("fatigue_life_and_limit",), "fatigue_strength": ("fatigue_life_and_limit",),
    "sn_data": ("fatigue_life_and_limit", "equations_and_models"),
    "epsilon_n_data": ("fatigue_life_and_limit", "equations_and_models"),
    "dadn_delta_k_data": ("crack_growth_and_paris", "equations_and_models"),
    "delta_k_threshold": ("crack_growth_and_paris",), "fractography": ("crack_initiation_site",),
    "sem_ebsd_ct": ("microstructure", "pores_and_defects"),
    "main_conclusions": ("fatigue_life_and_limit", "crack_growth_and_paris"),
    "boundary_conditions": ("fatigue_test_conditions",),
    "limitations": ("limitations_and_future_work",), "future_work": ("limitations_and_future_work",),
}


def _postprocess_artifacts(base_dir: Path, paper_id: str, title: str) -> Dict[str, int]:
    paths = deep_read_paths(base_dir, paper_id)
    scans = _read_json(paths["scans"], {})
    status = _read_json(paths["status"], {})
    fields: Dict[str, Any] = {}
    audited = bool(status.get("variable_sweep_complete") and status.get("missing_audit_complete"))
    for field, categories in RESEARCH_FIELDS.items():
        evidence = []
        values = []
        for category in categories:
            scan = scans.get(category) or {}
            evidence.extend((scan.get("original_evidence") or [])[:8])
            values.extend((scan.get("structured_values") or [])[:20])
        fields[field] = {
            "status": "EXTRACTED" if evidence or values else ("NOT_REPORTED" if audited else "NOT_EXTRACTED"),
            "evidence": evidence[:12], "structured_values": values[:30],
        }
    _atomic_json(paths["root"] / "research_fields.json", {
        "paper_id": paper_id, "title": title, "fields": fields,
        "sequential_scan_complete": bool(status.get("sequential_scan_complete")),
        "variable_sweep_complete": bool(status.get("variable_sweep_complete")),
        "missing_audit_complete": bool(status.get("missing_audit_complete")),
    })
    equations = []
    if paths["equations"].exists():
        for line in paths["equations"].read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                original = str(row.get("original_text") or "")
                row.update({
                    "original_formula": original, "normalized_formula": " ".join(original.split()),
                    "variable_definitions": {}, "units": re.findall(r"\b(?:MPa|GPa|Hz|cycles?|mm|µm|μm|%)\b", original, re.I),
                    "parameter_values": re.findall(r"[-+]?\d+(?:\.\d+)?(?:\s*[eE][-+]?\d+)?", original),
                    "fitting_conditions": {}, "applicable_material": "", "applicable_load": "",
                    "section": "", "original_context": original,
                    "is_literature_formula": True,
                    "needs_visual_review": not bool(re.search(r"=", original)),
                })
                equations.append(row)
        with paths["equations"].open("w", encoding="utf-8") as handle:
            for row in equations:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    visual = []
    for artifact_name, id_key, caption_key in (
        ("figures", "figure_id", "caption"), ("tables", "table_id", "title")
    ):
        artifact_path = paths[artifact_name]
        if not artifact_path.exists():
            continue
        for line in artifact_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "REVIEW_REQUIRED" not in str(row.get("review_status") or ""):
                continue
            visual.append({
                "paper_id": paper_id, "paper_title": title,
                "page_number": row.get("page_number"), "item_id": row.get(id_key),
                "caption": row.get(caption_key), "review_status": "NEEDS_VISUAL_REVIEW",
                "review_request": "Confirm values/curves/microstructural features from the original visual.",
                "importance": "May contain quantitative fatigue evidence not reliable from text alone.",
            })
    visual_path = paths["root"] / "visual_review_items.jsonl"
    with visual_path.open("w", encoding="utf-8") as handle:
        for row in visual:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"formula_count": len(equations), "visual_review_count": len(visual)}


def _canonical(base_dir: Path, paper_id: str) -> Dict[str, Any]:
    return next((row for row in load_paper_manifest(base_dir) if row.get("paper_id") == paper_id), {})


def _control(base_dir: Path) -> str:
    return str(_read_json(queue_paths(base_dir)["control"], {"action": "RUN"}).get("action") or "RUN")


def run_full_library_queue(
    pdf_dir: Path | str,
    *,
    base_dir: Path = BASE_DIR,
    resume: bool = True,
    retry_failed: bool = False,
    only_unread: bool = False,
    limit: Optional[int] = None,
    concurrency: int = 1,
    stop_after_pages: Optional[int] = None,
) -> Dict[str, Any]:
    if concurrency not in {1, 2}:
        raise ValueError("concurrency must be 1 or 2")
    # Canonical manifest and unified RAG are atomically rewritten shared stores;
    # process tasks serially even when a future caller allows two extraction workers.
    built = build_full_library_queue(pdf_dir, base_dir=base_dir, resume=resume)
    set_queue_control("RUN", base_dir=base_dir)
    tasks = built["tasks"]
    if retry_failed:
        for task in tasks:
            if task["status"] in {"FAILED_RETRYABLE", "PAUSED"}:
                _save_task(base_dir, task["task_id"], status="PENDING", last_error="")
    processed_tasks = 0
    pages_this_run = 0
    for snapshot in list(load_full_library_queue(base_dir).get("tasks") or []):
        task_id = snapshot["task_id"]
        task = next(row for row in load_full_library_queue(base_dir)["tasks"] if row["task_id"] == task_id)
        if task["status"] == "COMPLETED" and (resume or only_unread):
            continue
        if task["status"] == "SKIPPED_DUPLICATE":
            continue
        if task["status"] == "NEEDS_HUMAN_REVIEW":
            continue
        if task["status"] == "FAILED_RETRYABLE" and not retry_failed:
            continue
        if limit is not None and processed_tasks >= limit:
            break
        if _control(base_dir) in {"PAUSE", "STOP_NOW"}:
            break
        started = time.monotonic()
        if not task["valid_pdf"]:
            _save_task(base_dir, task_id, status="NEEDS_HUMAN_REVIEW", last_error="INVALID_PDF")
            continue
        _save_task(base_dir, task_id, status="PDF_VALIDATED", started_at=_now(), last_error="")
        try:
            registration = register_pdf_path(
                Path(task["primary_pdf_path"]), source_type="FULL_LIBRARY_DEEP_READ", base_dir=base_dir
            )
            paper_id = str(registration.get("paper_id") or "")
            canonical_row = _canonical(base_dir, paper_id)
            title = clean_pdf_title(canonical_row.get("title") or registration.get("title") or "")
            local_metadata = extract_basic_pdf_metadata(
                Path(task["primary_pdf_path"]).read_bytes(),
                Path(task["primary_pdf_path"]).name,
            )
            local_title = clean_pdf_title(local_metadata.get("title") or "")
            if (
                is_valid_title(local_title)
                and (
                    not is_valid_title(title)
                    or "文章编号" in str(canonical_row.get("title") or "")
                    or "Article ID" in str(canonical_row.get("title") or "")
                )
            ):
                    local_metadata["title"] = local_title
                    local_metadata["metadata_source"] = "PDF_FIRST_PAGE_TYPOGRAPHY"
                    update_paper_verified_metadata(
                        paper_id, local_metadata, base_dir=base_dir
                    )
                    canonical_row = _canonical(base_dir, paper_id)
                    title = str(canonical_row.get("title") or "").strip()
            if registration.get("doi") and (
                not is_valid_title(title)
                or not str(registration.get("authors") or "").strip()
                or not _year(str(registration.get("publication_date") or ""))
            ):
                try:
                    verified = fetch_metadata(str(registration["doi"]), timeout=8)
                    if verified.get("status") == "VALID":
                        register_metadata_record(
                            verified,
                            source_record_id=f"FULLREAD_METADATA_{task['sha256'][:16]}",
                            source_type=str(verified.get("metadata_source") or "SCHOLARLY_API"),
                            base_dir=base_dir,
                        )
                        update_paper_verified_metadata(
                            paper_id, verified, base_dir=base_dir
                        )
                        title = str(_canonical(base_dir, paper_id).get("title") or verified.get("title") or title)
                except Exception:
                    pass
            _save_task(base_dir, task_id, canonical_paper_id=paper_id, canonical_title=title)
            metadata_review_required = not is_valid_title(title)
            _save_task(
                base_dir, task_id,
                status="PDF_VALIDATED" if metadata_review_required else "METADATA_VALIDATED",
                metadata_review_required=metadata_review_required,
            )

            def progress(page: int, total: int, _record: Any) -> None:
                nonlocal pages_this_run
                pages_this_run += 1
                _save_task(
                    base_dir, task_id, status="PAGE_EXTRACTION_RUNNING",
                    processed_pages=page, real_page_count=total,
                )

            def stop() -> bool:
                action = _control(base_dir)
                if action in {"PAUSE", "STOP_NOW"}:
                    return True
                return stop_after_pages is not None and pages_this_run >= stop_after_pages

            result = deep_read_pdf(
                Path(task["primary_pdf_path"]), paper_id=paper_id, title=title,
                base_dir=base_dir, progress_callback=progress, should_stop=stop,
            )
            if result.get("status") == "INTERRUPTED":
                _save_task(base_dir, task_id, status="PAUSED", processed_pages=result.get("processed_page_count", 0), last_error="CONTROLLED_INTERRUPTION")
                break
            if not result.get("deep_read_complete"):
                retry = int(task.get("retry_count") or 0) + 1
                state = "FAILED_RETRYABLE" if retry < 3 else "NEEDS_HUMAN_REVIEW"
                _save_task(base_dir, task_id, status=state, retry_count=retry, last_error=str(result.get("error") or "DEEP_READ_INCOMPLETE"))
                continue
            _save_task(base_dir, task_id, status="PAGE_EXTRACTION_COMPLETE", processed_pages=result.get("processed_page_count", 0))
            _save_task(base_dir, task_id, status="DEEP_READING")
            extras = _postprocess_artifacts(base_dir, paper_id, title)
            _save_task(base_dir, task_id, status="EVIDENCE_AUDITING", **extras)
            _save_task(base_dir, task_id, status="QUALITY_GATING")
            gate = evaluate_auto_rag_gate(paper_id, base_dir=base_dir)
            if metadata_review_required and "NEEDS_METADATA_REVIEW" not in gate["reasons"]:
                gate["reasons"].append("NEEDS_METADATA_REVIEW")
                gate["passed"] = False
            if not gate["passed"]:
                update_paper_library_status(paper_id, "CANDIDATE", base_dir=base_dir)
                update_paper_rag_status(paper_id, "NOT_INDEXED", base_dir)
                _save_task(base_dir, task_id, status="NEEDS_HUMAN_REVIEW", gate=gate, last_error=";".join(gate["reasons"]), elapsed_seconds=round(time.monotonic() - started, 3))
                processed_tasks += 1
                continue
            update_paper_library_status(paper_id, "FORMAL", base_dir=base_dir)
            _save_task(base_dir, task_id, status="INDEXING", gate=gate)
            rag_manifest = _read_json(rag_paths(base_dir)["manifest"], {})
            already_indexed = (
                str(_canonical(base_dir, paper_id).get("rag_status") or "")
                == "INDEXED_STAGE3_UNIFIED"
                and paper_id in set(rag_manifest.get("paper_ids") or [])
            )
            indexed = (
                {"status": "INDEXED_STAGE3_UNIFIED", "idempotent_reuse": True}
                if already_indexed
                else index_auto_validated_paper(paper_id, base_dir=base_dir)
            )
            if indexed.get("status") != "INDEXED_STAGE3_UNIFIED":
                update_paper_library_status(paper_id, "CANDIDATE", base_dir=base_dir)
                _save_task(base_dir, task_id, status="NEEDS_HUMAN_REVIEW", last_error="STAGE3_INDEX_FAILED", index_result=indexed)
            else:
                _save_task(base_dir, task_id, status="COMPLETED", index_result=indexed, elapsed_seconds=round(time.monotonic() - started, 3), completed_at=_now())
            processed_tasks += 1
        except Exception as exc:
            current = next((row for row in load_full_library_queue(base_dir)["tasks"] if row["task_id"] == task_id), task)
            retry = int(current.get("retry_count") or 0) + 1
            state = "FAILED_RETRYABLE" if retry < 3 else "NEEDS_HUMAN_REVIEW"
            _save_task(base_dir, task_id, status=state, retry_count=retry, last_error=f"{type(exc).__name__}: {exc}", elapsed_seconds=round(time.monotonic() - started, 3))
        if _control(base_dir) == "STOP_AFTER_CURRENT":
            break
    report = generate_full_library_report(base_dir=base_dir)
    return {"processed_tasks": processed_tasks, "concurrency": concurrency, "report": report}


def queue_summary(*, base_dir: Path = BASE_DIR) -> Dict[str, Any]:
    queue = load_full_library_queue(base_dir)
    inventory = _read_json(queue_paths(base_dir)["inventory"], {})
    tasks = queue.get("tasks") or []
    counts = Counter(str(row.get("status") or "PENDING") for row in tasks)
    return {
        "pdf_file_count": int(inventory.get("pdf_file_count") or 0),
        "logical_document_count": int(inventory.get("logical_document_count") or 0),
        "exact_duplicate_count": int(inventory.get("exact_duplicate_count") or 0),
        "total_pages": int(inventory.get("total_pages") or 0),
        "completed": counts["COMPLETED"],
        "running": sum(counts[state] for state in RUNNING_STATES),
        "pending": counts["PENDING"],
        "failed": counts["FAILED_RETRYABLE"],
        "needs_human_review": counts["NEEDS_HUMAN_REVIEW"],
        "completed_pages": sum(int(row.get("processed_pages") or 0) for row in tasks),
        "evidence_count": sum(int(row.get("gate", {}).get("evidence_count") or 0) for row in tasks),
        "indexed": sum(str(row.get("index_result", {}).get("status") or "") == "INDEXED_STAGE3_UNIFIED" for row in tasks),
        "control": _control(base_dir),
        "tasks": tasks,
    }


def audit_full_library_metadata(*, base_dir: Path = BASE_DIR) -> Dict[str, Any]:
    """Verify DOI metadata, repair PDF titles, and demote unconfirmed records."""
    queue = load_full_library_queue(base_dir)
    verified = repaired_from_pdf = demoted = failures = 0
    for task in queue.get("tasks") or []:
        paper_id = str(task.get("canonical_paper_id") or "")
        if not paper_id:
            continue
        paper = _canonical(base_dir, paper_id)
        path = Path(str(task.get("primary_pdf_path") or ""))
        local_metadata: Dict[str, Any] = {}
        if path.exists():
            try:
                local_metadata = extract_basic_pdf_metadata(path.read_bytes(), path.name)
            except OSError:
                local_metadata = {}
        doi = normalize_doi(paper.get("doi") or local_metadata.get("doi") or "")
        resolved: Dict[str, Any] = {}
        if doi:
            try:
                resolved = fetch_metadata(doi, timeout=8)
            except Exception:
                resolved = {}
        if resolved.get("status") == "VALID" and is_valid_title(resolved.get("title")):
            update_paper_verified_metadata(paper_id, resolved, base_dir=base_dir)
            verified += 1
        else:
            local_title = clean_pdf_title(local_metadata.get("title") or "")
            current_title = clean_pdf_title(paper.get("title") or "")
            if is_valid_title(local_title) and (
                not is_valid_title(current_title)
                or len(local_title) < len(current_title) * 0.8
                or "文章编号" in str(paper.get("title") or "")
            ):
                local_metadata["title"] = local_title
                local_metadata["metadata_source"] = "PDF_FIRST_PAGE_TYPOGRAPHY"
                update_paper_verified_metadata(paper_id, local_metadata, base_dir=base_dir)
                repaired_from_pdf += 1
        paper = _canonical(base_dir, paper_id)
        title_ok = is_valid_title(paper.get("title"))
        authors_ok = bool(str(paper.get("authors") or "").strip())
        if not title_ok or not authors_ok:
            reasons = []
            if not title_ok:
                reasons.append("NEEDS_METADATA_REVIEW:TITLE")
            if not authors_ok:
                reasons.append("NEEDS_METADATA_REVIEW:AUTHORS")
            update_paper_library_status(paper_id, "CANDIDATE", base_dir=base_dir)
            update_paper_rag_status(paper_id, "NOT_INDEXED", base_dir)
            _save_task(
                base_dir, task["task_id"], status="NEEDS_HUMAN_REVIEW",
                canonical_title=str(paper.get("title") or ""),
                last_error=";".join(reasons),
            )
            demoted += 1
        else:
            _save_task(
                base_dir, task["task_id"],
                canonical_title=str(paper.get("title") or ""),
                canonical_authors=str(paper.get("authors") or ""),
                canonical_year=str(paper.get("publication_date") or "UNKNOWN"),
                metadata_verified=bool(resolved.get("status") == "VALID"),
            )
        if doi and resolved.get("status") != "VALID":
            failures += 1
    return {
        "verified_by_api": verified,
        "repaired_from_pdf": repaired_from_pdf,
        "demoted_for_metadata": demoted,
        "doi_verification_failures": failures,
    }


def generate_full_library_report(*, base_dir: Path = BASE_DIR) -> Dict[str, Any]:
    inventory = _read_json(queue_paths(base_dir)["inventory"], {})
    queue = load_full_library_queue(base_dir)
    manifests = {str(row.get("paper_id") or ""): row for row in load_paper_manifest(base_dir)}
    evidence = load_trusted_evidence_rows(base_dir)
    evidence_by_paper: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in evidence:
        evidence_by_paper[str(row.get("paper_id") or "")].append(row)
    details = []
    formula_count = visual_count = 0
    for task in queue.get("tasks") or []:
        paper_id = str(task.get("canonical_paper_id") or "")
        paper = manifests.get(paper_id) or {}
        status = _read_json(deep_read_paths(base_dir, paper_id)["status"], {}) if paper_id else {}
        extras = _postprocess_artifacts(base_dir, paper_id, str(paper.get("title") or task.get("canonical_title") or "")) if paper_id and status.get("deep_read_complete") else {"formula_count": 0, "visual_review_count": 0}
        formula_count += extras["formula_count"]
        visual_count += extras["visual_review_count"]
        paper_evidence = evidence_by_paper.get(paper_id, [])
        directness = Counter(str(row.get("directness") or "") for row in paper_evidence)
        details.append({
            "canonical_paper_id": paper_id,
            "title": str(paper.get("title") or task.get("canonical_title") or ""),
            "filename": task.get("original_filename"),
            "pages": int(status.get("real_page_count") or task.get("real_page_count") or 0),
            "processed_pages": int(status.get("processed_page_count") or task.get("processed_pages") or 0),
            "page_coverage_ratio": float(status.get("page_coverage_ratio") or 0.0),
            "evidence_count": len(paper_evidence),
            "direct_count": directness["DIRECT"], "indirect_count": directness["INDIRECT"],
            "inferred_count": directness["INFERRED"], "mention_only_count": directness["MENTION_ONLY"],
            "quality_gate": (task.get("gate") or {}).get("passed", False),
            "library_status": paper.get("library_status") or "",
            "rag_status": paper.get("rag_status") or "NOT_INDEXED",
            "domain_scope": paper.get("domain_scope") or "",
            "task_status": task.get("status"), "failure_reason": task.get("last_error") or "",
            "elapsed_seconds": task.get("elapsed_seconds") or 0,
            "formula_count": extras["formula_count"], "visual_review_count": extras["visual_review_count"],
        })
    directness_total = Counter()
    for row in details:
        directness_total["DIRECT"] += row["direct_count"]
        directness_total["INDIRECT"] += row["indirect_count"]
        directness_total["INFERRED"] += row["inferred_count"]
        directness_total["MENTION_ONLY"] += row["mention_only_count"]
    total_pages = sum(row["pages"] for row in details)
    completed_pages = sum(row["processed_pages"] if row["page_coverage_ratio"] == 1.0 else 0 for row in details)
    report = {
        "generated_at": _now(), "schema_version": QUEUE_VERSION,
        "scanned_pdf_count": int(inventory.get("pdf_file_count") or 0),
        "unique_logical_document_count": int(inventory.get("logical_document_count") or 0),
        "exact_duplicate_count": int(inventory.get("exact_duplicate_count") or 0),
        "different_version_count": int(inventory.get("different_version_count") or 0),
        "valid_pdf_count": int(inventory.get("valid_pdf_count") or 0),
        "damaged_pdf_count": int(inventory.get("invalid_pdf_count") or 0),
        "total_pages": total_pages, "fully_read_pages": completed_pages,
        "page_coverage_ratio": round(completed_pages / total_pages, 6) if total_pages else 0.0,
        "completed_paper_count": sum(row["page_coverage_ratio"] == 1.0 for row in details),
        "failed_paper_count": sum(row["task_status"] == "FAILED_RETRYABLE" for row in details),
        "needs_human_review_count": sum(row["task_status"] == "NEEDS_HUMAN_REVIEW" for row in details),
        "evidence_record_count": sum(row["evidence_count"] for row in details),
        "direct_count": directness_total["DIRECT"], "indirect_count": directness_total["INDIRECT"],
        "inferred_count": directness_total["INFERRED"], "mention_only_count": directness_total["MENTION_ONLY"],
        "formula_record_count": formula_count, "needs_visual_review_count": visual_count,
        "formal_paper_count": sum(row["library_status"] == "FORMAL" for row in details),
        "indexed_paper_count": sum(row["rag_status"] == "INDEXED_STAGE3_UNIFIED" for row in details),
        "out_of_scope_count": sum(row["domain_scope"] == OUT_OF_SCOPE for row in details),
        "papers": details,
    }
    outputs = base_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    _atomic_json(outputs / "full_library_deep_read_report.json", report)
    columns = list(details[0]) if details else ["canonical_paper_id", "title", "filename"]
    with (outputs / "full_library_deep_read_report.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(details)
    lines = [
        "# 本地全量PDF全文深度精读报告", "",
        f"- 扫描PDF总数：{report['scanned_pdf_count']}",
        f"- 唯一逻辑文献：{report['unique_logical_document_count']}",
        f"- 完全重复：{report['exact_duplicate_count']}",
        f"- 总页数：{report['total_pages']}",
        f"- 完整读取页数：{report['fully_read_pages']}",
        f"- 页面覆盖率：{report['page_coverage_ratio']:.2%}",
        f"- EvidenceRecord：{report['evidence_record_count']}", "", "## 逐篇结果", "",
    ]
    for row in details:
        lines.append(
            f"- {row['title'] or '[NEEDS_METADATA_REVIEW]'} | {row['filename']} | "
            f"{row['pages']}页 | 覆盖率 {row['page_coverage_ratio']:.2%} | "
            f"Evidence {row['evidence_count']} | Gate {row['quality_gate']} | "
            f"{row['library_status']} | {row['rag_status']} | {row['failure_reason']}"
        )
    (outputs / "full_library_deep_read_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report

"""Audited, non-destructive-by-default canonical library state changes."""

from __future__ import annotations

import json
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.metadata_gate import atomic_jsonl, read_jsonl


TARGET_AUDIT = "PAPER_1EF9D65943440F0C"


def reconcile_persistent_selection(
    persisted_ids: Sequence[str], page_ids: Sequence[str], page_selected_ids: Sequence[str],
) -> list[str]:
    """Replace the current page selection while retaining selections on other pages."""
    persisted = {str(value) for value in persisted_ids if value}
    persisted.difference_update(str(value) for value in page_ids if value)
    persisted.update(str(value) for value in page_selected_ids if value)
    return sorted(persisted)


def load_canonical_aliases(base_dir: Path) -> dict[str, str]:
    aliases: dict[str, str] = {}
    root = Path(base_dir) / "data" / "deep_read_aliases"
    if not root.exists():
        return aliases
    for path in root.glob("*/MIGRATION.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        alias = str(row.get("alias_paper_id") or "")
        canonical = str(row.get("canonical_paper_id") or "")
        if alias and canonical and alias != canonical:
            aliases[alias] = canonical
    return aliases


def migrate_persistent_selection(
    persisted_ids: Sequence[str],
    valid_ids: Sequence[str],
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve old IDs and remove selections that are no longer selectable."""
    valid = {str(value) for value in valid_ids if value}
    mapping = dict(aliases or {})
    migrated: dict[str, str] = {}
    removed: list[str] = []
    selected: set[str] = set()
    for raw in dict.fromkeys(str(value) for value in persisted_ids if value):
        resolved = raw
        visited: set[str] = set()
        while resolved in mapping and resolved not in visited:
            visited.add(resolved)
            resolved = mapping[resolved]
        if resolved in valid:
            selected.add(resolved)
            if resolved != raw:
                migrated[raw] = resolved
        else:
            removed.append(raw)
    return {
        "selected_ids": sorted(selected),
        "migrated": migrated,
        "removed": sorted(removed),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_log(base_dir: Path, events: list[dict[str, Any]]) -> None:
    path = base_dir / "data/operation_logs/library_operations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def update_records(
    paper_ids: Sequence[str], updates: dict[str, Any], *, reason: str,
    action: str, base_dir: Path,
) -> dict[str, Any]:
    requested = {str(value) for value in paper_ids if value}
    path = base_dir / "data/paper_manifest.jsonl"
    rows = read_jsonl(path); updated = []; events = []
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        if paper_id not in requested:
            continue
        old_state = {key: row.get(key) for key in updates}
        row.update(updates); row["updated_at"] = _now(); updated.append(paper_id)
        events.append({
            "timestamp": _now(), "paper_id": paper_id, "action": action,
            "old_state": old_state, "new_state": {key: row.get(key) for key in updates},
            "reason": reason,
        })
    if updated:
        atomic_jsonl(path, rows); _append_log(base_dir, events)
    return {"status": "COMPLETED", "updated": updated, "not_found": sorted(requested - set(updated)), "events": events}


def _eligible_for_formal(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if not row.get("pdf_valid"): reasons.append("PDF_NOT_VALID")
    if not row.get("deep_read_complete"): reasons.append("DEEP_READ_INCOMPLETE")
    if row.get("evidence_status") not in {"AUTO_VALIDATED", "HUMAN_CONFIRMED"}: reasons.append("EVIDENCE_NOT_VALIDATED")
    if row.get("metadata_gate_status") != "PASSED" and row.get("paper_id") != TARGET_AUDIT: reasons.append("METADATA_GATE_NOT_PASSED")
    if row.get("domain_scope") == "OUT_OF_SCOPE": reasons.append("OUT_OF_SCOPE")
    return not reasons, reasons


def add_to_formal(paper_ids: Sequence[str], *, reason: str, base_dir: Path) -> dict[str, Any]:
    rows = {row["paper_id"]: row for row in read_jsonl(base_dir / "data/paper_manifest.jsonl")}
    accepted, rejected = [], {}
    for paper_id in dict.fromkeys(str(value) for value in paper_ids if value):
        row = rows.get(paper_id)
        if not row: rejected[paper_id] = ["NOT_FOUND"]; continue
        passed, reasons = _eligible_for_formal(row)
        if passed: accepted.append(paper_id)
        else: rejected[paper_id] = reasons
    result = update_records(accepted, {"library_status": "FORMAL", "quarantine_reason": ""}, reason=reason, action="ADD_TO_FORMAL", base_dir=base_dir)
    result["rejected"] = rejected; return result


def remove_from_formal(paper_ids: Sequence[str], *, reason: str, base_dir: Path) -> dict[str, Any]:
    return update_records(paper_ids, {"library_status": "COMPLETE_NOT_INDEXED", "rag_status": "NOT_INDEXED_MANUAL"}, reason=reason, action="REMOVE_FROM_FORMAL", base_dir=base_dir)


def set_scope(paper_ids: Sequence[str], scope: str, *, reason: str, base_dir: Path) -> dict[str, Any]:
    updates = {"domain_scope": scope}
    if scope == "OUT_OF_SCOPE":
        updates.update({"library_status": "OUT_OF_SCOPE", "rag_status": "NOT_INDEXED_OUT_OF_SCOPE"})
    return update_records(paper_ids, updates, reason=reason, action=f"SET_SCOPE_{scope}", base_dir=base_dir)


def archive(paper_ids: Sequence[str], *, reason: str, base_dir: Path) -> dict[str, Any]:
    return update_records(paper_ids, {"library_status": "ARCHIVED", "rag_status": "NOT_INDEXED_ARCHIVED"}, reason=reason, action="ARCHIVE", base_dir=base_dir)


def rebuild_current_formal_rag(base_dir: Path) -> dict[str, Any]:
    from src.unified_rag import build_unified_rag
    rows = read_jsonl(base_dir / "data/paper_manifest.jsonl")
    by_id = {str(row.get("paper_id") or ""): row for row in rows}
    whitelist_path = base_dir / "data/system/formal_rag_whitelist.json"
    try:
        whitelist_payload = json.loads(whitelist_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        whitelist_payload = {}
    confirmed = {
        str(value) for value in whitelist_payload.get("paper_ids") or [] if value
    }
    newly_qualified = {
        str(row.get("paper_id") or "")
        for row in rows
        if row.get("library_status") == "FORMAL"
        and row.get("deep_read_complete")
        and row.get("evidence_status") in {"AUTO_VALIDATED", "HUMAN_CONFIRMED"}
    }
    requested = confirmed if confirmed else newly_qualified
    ids = [
        paper_id
        for paper_id in sorted(requested)
        if (row := by_id.get(paper_id))
        and row.get("library_status") == "FORMAL"
        and row.get("deep_read_complete")
        and row.get("domain_scope") != "OUT_OF_SCOPE"
    ]
    return build_unified_rag(ids, base_dir=base_dir)


def export_records(paper_ids: Sequence[str], *, base_dir: Path) -> bytes:
    requested = {str(value) for value in paper_ids if value}
    rows = [row for row in read_jsonl(base_dir / "data/paper_manifest.jsonl") if row.get("paper_id") in requested]
    return json.dumps({"exported_at": _now(), "records": rows}, ensure_ascii=False, indent=2).encode("utf-8")


def audit_selected_evidence(paper_ids: Sequence[str], *, base_dir: Path) -> dict[str, Any]:
    requested = {str(value) for value in paper_ids if value}
    path = base_dir / "data/evidence/trusted_evidence.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if str(row.get("canonical_paper_id") or row.get("paper_id") or "") in requested]
    normalized = Counter(" ".join(str(row.get("original_text") or "").casefold().split()) for row in rows)
    report = {
        "paper_ids": sorted(requested), "evidence_count": len(rows),
        "exact_duplicate_count": sum(value - 1 for value in normalized.values() if value > 1),
        "reference_or_toc_count": sum("reference" in str(row.get("section") or "").casefold() for row in rows),
        "directness_counts": dict(Counter(str(row.get("directness") or "") for row in rows)),
        "status": "AUDITED_NO_DESTRUCTIVE_CHANGE",
    }
    output = base_dir / "outputs/selected_evidence_reaudit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_log(base_dir, [{"timestamp": _now(), "paper_id": paper_id, "action": "REAUDIT_EVIDENCE", "old_state": {}, "new_state": {"audit_report": str(output)}, "reason": "USER_TRIGGERED_REAUDIT"} for paper_id in requested])
    return report


def deletion_impact(paper_ids: Sequence[str], *, base_dir: Path) -> dict[str, Any]:
    requested = {str(value) for value in paper_ids if value}
    manifest = [
        row for row in read_jsonl(base_dir / "data/paper_manifest.jsonl")
        if str(row.get("paper_id") or "") in requested
    ]
    evidence_path = base_dir / "data/evidence/trusted_evidence.csv"
    with evidence_path.open("r", encoding="utf-8-sig", newline="") as stream:
        evidence = [
            row for row in csv.DictReader(stream)
            if str(row.get("canonical_paper_id") or row.get("paper_id") or "")
            in requested
        ]
    formula_count = 0
    for paper_id in requested:
        for filename in ("equations.jsonl", "formula_records.jsonl"):
            path = base_dir / "data/deep_read" / paper_id / filename
            if path.exists():
                formula_count += len(read_jsonl(path))
                break
    return {
        "paper_count": len(manifest),
        "titles": [str(row.get("title") or "题名未报告") for row in manifest],
        "pdf_count": sum(
            bool(row.get("canonical_pdf_path"))
            and Path(str(row.get("canonical_pdf_path"))).is_file()
            for row in manifest
        ),
        "evidence_count": len(evidence),
        "formula_count": formula_count,
    }


def delete_formal_papers(
    paper_ids: Sequence[str], *, confirmation: str, base_dir: Path
) -> dict[str, Any]:
    """Remove selected formal papers and send managed assets to Recycle Bin."""
    if confirmation.strip() != "DELETE":
        raise ValueError("删除确认文本必须为 DELETE")
    requested = {str(value) for value in paper_ids if value}
    whitelist_path = base_dir / "data/system/formal_rag_whitelist.json"
    payload = json.loads(whitelist_path.read_text(encoding="utf-8"))
    current = [str(value) for value in payload.get("paper_ids") or []]
    unknown = requested - set(current)
    if not requested or unknown:
        raise ValueError("只能删除当前正式文献库中的有效选择")
    impact = deletion_impact(requested, base_dir=base_dir)
    retained = [paper_id for paper_id in current if paper_id not in requested]

    from src.unified_rag import build_unified_rag
    from scripts.finalize_formal_library import apply_plan, build_plan

    rag = build_unified_rag(retained, base_dir=base_dir)
    if int(rag.get("paper_count") or -1) != len(retained):
        raise RuntimeError("删除中止：统一 RAG 未能安全重建")
    payload["paper_ids"] = retained
    payload["paper_count"] = len(retained)
    payload["generated_at"] = _now()
    temporary = whitelist_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(whitelist_path)
    cleanup = apply_plan(base_dir, build_plan(base_dir))
    return {"status": "COMPLETED", "impact": impact, "rag": rag, "cleanup": cleanup}

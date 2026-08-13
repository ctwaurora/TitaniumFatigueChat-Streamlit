"""Fail-closed protection for verified Formal PDF assets.

The lock manifest is the authority for the reproducible dataset.  Historical
manifest records remain readable, but no cleanup, quarantine, deduplication or
index build may silently delete, move, replace or re-hash a locked PDF.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


LOCK_RELATIVE = Path("data/system/formal_pdf_lock_manifest.json")
LOCK_POINTER_RELATIVE = Path("data/system/active_formal_pdf_lock.json")
AUDIT_RELATIVE = Path("data/audit/formal_pdf_deletion_audit.jsonl")


class FormalProvenanceViolation(RuntimeError):
    """Raised before a write when a locked Formal PDF is absent or changed."""


class FormalDeleteDenied(PermissionError):
    """Raised when a destructive operation targets a Formal asset."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _manifest_rows(base_dir: Path) -> list[dict[str, Any]]:
    path = base_dir / "data" / "paper_manifest.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_formal_pdf_lock(base_dir: Path) -> dict[str, Any]:
    root = Path(base_dir).resolve()
    pointer = _load_json(root / LOCK_POINTER_RELATIVE, {})
    relative = str(pointer.get("manifest_path") or "").strip()
    if relative:
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise FormalProvenanceViolation("FORMAL_PROVENANCE_VIOLATION:LOCK_POINTER_ESCAPES_PROJECT") from exc
        if not candidate.is_file():
            raise FormalProvenanceViolation("FORMAL_PROVENANCE_VIOLATION:ACTIVE_LOCK_MISSING")
        return _load_json(candidate, {})
    return _load_json(root / LOCK_RELATIVE, {})


def validate_dataset_manifest_relation(base_dir: Path) -> dict[str, Any]:
    """Prove that Verified v1 is exactly Historical minus metadata-only rows."""
    base_dir = Path(base_dir).resolve()
    historical_path = base_dir / "data/system/historical_snapshot_manifest.json"
    verified_path = base_dir / "data/system/verified_dataset_v1_candidate_manifest.json"
    lock_path = base_dir / LOCK_RELATIVE
    if not (historical_path.is_file() and verified_path.is_file() and lock_path.is_file()):
        return {"status": "DATASET_MANIFESTS_NOT_CONFIGURED"}
    historical = _load_json(historical_path, {})
    verified = _load_json(verified_path, {})
    # The immutable v1 relation is always verified against the immutable v1
    # lock.  A v1.1 active lock is validated separately below.
    lock = _load_json(lock_path, {})
    historical_ids = {str(value) for value in historical.get("paper_ids") or [] if value}
    verified_ids = {
        str(row.get("document_id") or "") for row in verified.get("papers") or []
        if row.get("document_id")
    }
    metadata_ids = {
        str(value) for value in verified.get("metadata_only_historical_ids") or [] if value
    }
    lock_ids = {
        str(row.get("document_id") or "") for row in lock.get("documents") or []
        if row.get("formal") is True and row.get("document_id")
    }
    expected_counts = (
        int(historical.get("historical_rag_paper_count") or -1),
        int(verified.get("verified_paper_count") or -1),
        int(verified.get("metadata_only_historical_count") or -1),
    )
    violations = []
    if expected_counts != (len(historical_ids), len(verified_ids), len(metadata_ids)):
        violations.append("DECLARED_COUNTS_DO_NOT_MATCH_IDS")
    if verified_ids != lock_ids:
        violations.append("VERIFIED_IDS_DO_NOT_MATCH_FORMAL_LOCK")
    if verified_ids | metadata_ids != historical_ids or verified_ids & metadata_ids:
        violations.append("HISTORICAL_VERIFIED_METADATA_PARTITION_INVALID")
    if violations:
        raise FormalProvenanceViolation(
            "FORMAL_PROVENANCE_VIOLATION:DATASET_MANIFEST_RELATION:" + ",".join(violations)
        )
    result = {
        "status": "PASS",
        "historical_rag": len(historical_ids),
        "verified": len(verified_ids),
        "metadata_only_historical": len(metadata_ids),
    }
    from src.dataset_versioning import active_dataset_ids
    active_ids = active_dataset_ids(base_dir)
    active_lock = load_formal_pdf_lock(base_dir)
    active_lock_ids = {
        str(row.get("document_id") or "") for row in active_lock.get("documents") or []
        if row.get("formal") is True and row.get("document_id")
    }
    if active_ids != active_lock_ids:
        raise FormalProvenanceViolation("FORMAL_PROVENANCE_VIOLATION:ACTIVE_DATASET_LOCK_MISMATCH")
    return result


def _locked_documents(base_dir: Path) -> list[dict[str, Any]]:
    payload = load_formal_pdf_lock(base_dir)
    return [dict(row) for row in payload.get("documents") or [] if row.get("formal") is True]


def _fallback_formal_documents(base_dir: Path) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in _manifest_rows(base_dir):
        paper_id = str(row.get("paper_id") or "")
        if not paper_id:
            continue
        merged.setdefault(paper_id, {}).update(
            {key: value for key, value in row.items() if value not in (None, "", [], {})}
        )
    output = []
    for paper_id, row in merged.items():
        if row.get("library_status") != "FORMAL":
            continue
        output.append({
            "document_id": paper_id,
            "canonical_path": row.get("canonical_pdf_path") or row.get("source_path") or "",
            "canonical_sha256": row.get("file_hash_sha256") or "",
            "expected_filename": Path(str(row.get("canonical_pdf_path") or "")).name,
            "version_status": "HISTORICAL_FORMAL_MANIFEST",
            "formal": True,
        })
    return output


def formal_documents(base_dir: Path) -> list[dict[str, Any]]:
    locked = _locked_documents(base_dir)
    if locked:
        return locked
    return _fallback_formal_documents(base_dir)


def validate_formal_pdf_locks(
    base_dir: Path,
    *,
    requested_document_ids: Iterable[str] | None = None,
    require_lock: bool = False,
) -> dict[str, Any]:
    """Validate every lock before any formal mutation or index write."""
    base_dir = Path(base_dir).resolve()
    validate_dataset_manifest_relation(base_dir)
    lock = load_formal_pdf_lock(base_dir)
    if not lock:
        if require_lock:
            raise FormalProvenanceViolation("FORMAL_PROVENANCE_VIOLATION:LOCK_MANIFEST_MISSING")
        return {"status": "LOCK_NOT_CONFIGURED", "verified": 0, "violations": []}
    documents = _locked_documents(base_dir)
    violations: list[dict[str, str]] = []
    locked_ids = {str(row.get("document_id") or "") for row in documents}
    manifest_by_id = {
        str(row.get("paper_id") or ""): row for row in _manifest_rows(base_dir)
        if row.get("paper_id")
    }
    for row in documents:
        document_id = str(row.get("document_id") or "")
        path = Path(str(row.get("canonical_path") or ""))
        expected_hash = str(row.get("canonical_sha256") or "").lower()
        expected_name = str(row.get("expected_filename") or "")
        if not path.is_file():
            violations.append({"document_id": document_id, "reason": "PDF_MISSING", "path": str(path)})
            continue
        manifest_row = manifest_by_id.get(document_id)
        if manifest_row is None:
            violations.append({
                "document_id": document_id,
                "reason": "LOCK_ID_MISSING_FROM_PAPER_MANIFEST",
                "path": str(path),
            })
        else:
            manifest_path = Path(str(manifest_row.get("canonical_pdf_path") or ""))
            if (
                manifest_row.get("library_status") != "FORMAL"
                or not manifest_path
                or manifest_path.resolve() != path.resolve()
            ):
                violations.append({
                    "document_id": document_id,
                    "reason": "LOCK_MANIFEST_RELATION_INVALID",
                    "path": str(path),
                })
        if expected_name and path.name != expected_name:
            violations.append({"document_id": document_id, "reason": "FILENAME_CHANGED", "path": str(path)})
        actual = _sha256(path)
        if not expected_hash or actual != expected_hash:
            violations.append({
                "document_id": document_id,
                "reason": "SHA256_CHANGED",
                "path": str(path),
                "expected_sha256": expected_hash,
                "actual_sha256": actual,
            })
    requested = {str(value) for value in requested_document_ids or [] if value}
    unlocked_requested = sorted(requested - locked_ids)
    if requested and unlocked_requested:
        violations.extend(
            {"document_id": value, "reason": "REQUESTED_FORMAL_ID_NOT_LOCKED", "path": ""}
            for value in unlocked_requested
        )
    if violations:
        raise FormalProvenanceViolation(
            "FORMAL_PROVENANCE_VIOLATION:" + json.dumps(violations, ensure_ascii=False, sort_keys=True)
        )
    return {"status": "PASS", "verified": len(documents), "violations": []}


def _path_targets_document(path: Path, row: dict[str, Any]) -> bool:
    raw = str(row.get("canonical_path") or "")
    if not raw:
        return False
    canonical = Path(raw).resolve()
    resolved = path.resolve()
    return resolved == canonical or (resolved.is_dir() and canonical.is_relative_to(resolved))


def formal_document_for_path(path: Path, base_dir: Path) -> dict[str, Any] | None:
    base_dir = Path(base_dir).resolve()
    manifest_by_id = {
        str(row.get("paper_id") or ""): row for row in _manifest_rows(base_dir)
        if row.get("paper_id")
    }
    for row in formal_documents(base_dir):
        if _path_targets_document(Path(path), row):
            return row
        manifest = manifest_by_id.get(str(row.get("document_id") or ""), {})
        for related in manifest.get("linked_versions") or []:
            related_row = {**row, "canonical_path": related}
            if _path_targets_document(Path(path), related_row):
                return {**row, "matched_related_version": str(Path(related).resolve())}
    return None


def formal_document_for_id(document_id: str, base_dir: Path) -> dict[str, Any] | None:
    wanted = str(document_id or "")
    for row in formal_documents(Path(base_dir).resolve()):
        if str(row.get("document_id") or "") == wanted:
            return row
    return None


def _reference_counts(base_dir: Path, document_id: str) -> dict[str, int]:
    rag = 0
    rag_root = base_dir / "data" / "rag"
    if rag_root.exists():
        for path in rag_root.glob("*_documents.jsonl"):
            try:
                rag += path.read_text(encoding="utf-8", errors="ignore").count(document_id)
            except OSError:
                pass
    evidence = 0
    evidence_path = base_dir / "data" / "evidence" / "trusted_evidence.csv"
    if evidence_path.is_file():
        with evidence_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("canonical_paper_id") or row.get("paper_id") or "") == document_id:
                    evidence += 1
    condition = 0
    condition_path = base_dir / "data" / "evidence" / "condition_evidence_records.jsonl"
    if condition_path.is_file():
        for line in condition_path.read_text(encoding="utf-8").splitlines():
            if document_id in line:
                condition += 1
    return {"rag_references": rag, "evidence_references": evidence, "condition_references": condition}


def _append_audit(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def authorize_destructive_action(
    path: Path,
    *,
    base_dir: Path,
    action: str,
    allow_formal_delete: bool = False,
    reason: str = "",
    operator: str = "",
) -> dict[str, Any] | None:
    """Deny a Formal delete/move unless a fully audited override is explicit."""
    base_dir = Path(base_dir).resolve()
    target = Path(path).resolve()
    document = formal_document_for_path(target, base_dir)
    if document is None:
        return None
    document_id = str(document.get("document_id") or "")
    if not allow_formal_delete:
        raise FormalDeleteDenied(f"DENY_DELETE_FORMAL_DOCUMENT:{document_id}:{target}")
    if not str(reason).strip() or not str(operator).strip():
        raise FormalDeleteDenied("FORMAL_DELETE_OVERRIDE_REQUIRES_REASON_AND_OPERATOR")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = base_dir / "backups" / "formal_delete_override" / timestamp / document_id
    backup_root.mkdir(parents=True, exist_ok=False)
    canonical = Path(str(document.get("canonical_path") or target)).resolve()
    protected_source = target if target.is_file() else canonical
    backup_path = backup_root / protected_source.name
    shutil.copy2(protected_source, backup_path)
    actual_hash = _sha256(protected_source)
    record = {
        "event": "FORMAL_DELETE_OVERRIDE_AUTHORIZED",
        "action": str(action),
        "document_id": document_id,
        "canonical_path": str(canonical),
        "target_path": str(target),
        "sha256": actual_hash,
        "expected_sha256": str(document.get("canonical_sha256") or ""),
        "reason": str(reason).strip(),
        "operator": str(operator).strip() or os.environ.get("USERNAME", "UNKNOWN"),
        "timestamp": _now(),
        "backup_path": str(backup_path.resolve()),
        "references": _reference_counts(base_dir, document_id),
        "status": "AUTHORIZED_PENDING_ACTION",
    }
    _append_audit(base_dir / AUDIT_RELATIVE, record)
    return record


def deny_formal_state_transition(
    document_id: str,
    *,
    base_dir: Path,
    action: str,
    allow_formal_delete: bool = False,
    reason: str = "",
    operator: str = "",
) -> dict[str, Any] | None:
    document = formal_document_for_id(document_id, base_dir)
    if document is None:
        return None
    path = Path(str(document.get("canonical_path") or ""))
    return authorize_destructive_action(
        path,
        base_dir=base_dir,
        action=action,
        allow_formal_delete=allow_formal_delete,
        reason=reason,
        operator=operator,
    )

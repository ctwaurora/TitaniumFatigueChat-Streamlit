"""Single-source corpus statistics with explicit, non-overlapping definitions.

The UI, CLI audits and reports must not use one number for physical PDF files,
unique hashes, logical works, canonical metadata records and formal RAG papers.
This module reads the durable full-library inventory and writes one versioned
snapshot that is cheap to load during Streamlit reruns.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "corpus-statistics-2.1"
PDF_ROOTS = ("paper/pdfs", "papers", "early_papers", "followup_papers")
FINAL_STATES = (
    "FORMAL_INDEXED",
    "COMPLETE_NOT_INDEXED",
    "PENDING_PROCESSING",
    "PROCESSING_FAILED",
    "NEEDS_HUMAN_REVIEW",
    "PDF_NOT_ACQUIRED",
    "RELATED_VERSION",
    "OUT_OF_SCOPE",
    "ARCHIVED",
    "DELETED",
)

NON_ACTIVE_PRIMARY_STATES = {"RELATED_VERSION", "ARCHIVED", "DELETED"}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        return sum(bool(line.strip()) for line in stream)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_paths(base_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for relative in PDF_ROOTS:
        root = base_dir / relative
        if root.exists():
            paths.extend(path for path in root.rglob("*.pdf") if path.is_file())
    return sorted(paths, key=lambda value: str(value).casefold())


def corpus_statistics_signature(base_dir: Path) -> str:
    """Fast invalidation signature; no PDF content is read here."""
    paths: list[Path] = [
        base_dir / "data" / "paper_manifest.jsonl",
        base_dir / "data" / "pdf_files.jsonl",
        base_dir / "data" / "rag" / "manifest.json",
        base_dir / "data" / "evidence" / "trusted_evidence.csv",
        base_dir / "data" / "tasks" / "full_library_deep_read" / "inventory.json",
        base_dir / "data" / "tasks" / "full_library_deep_read" / "queue.json",
        *_pdf_paths(base_dir),
    ]
    deep_root = base_dir / "data" / "deep_read"
    if deep_root.exists():
        paths.extend(deep_root.glob("*/extraction_status.json"))
        paths.extend(deep_root.glob("*/equations.jsonl"))
        paths.extend(deep_root.glob("*/visual_review_items.jsonl"))
    payload: list[str] = []
    for path in paths:
        try:
            status = path.stat()
        except OSError:
            continue
        payload.append(f"{path.resolve()}|{status.st_size}|{status.st_mtime_ns}")
    return hashlib.sha256("\n".join(sorted(payload)).encode("utf-8")).hexdigest()


def _deep_statuses(base_dir: Path) -> list[dict[str, Any]]:
    root = base_dir / "data" / "deep_read"
    return [
        status
        for path in root.glob("*/extraction_status.json")
        if isinstance(status := _read_json(path, {}), dict)
    ] if root.exists() else []


def _trusted_evidence_count(base_dir: Path) -> int:
    path = base_dir / "data" / "evidence" / "trusted_evidence.csv"
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def _artifact_count(base_dir: Path, filename: str) -> int:
    root = base_dir / "data" / "deep_read"
    return sum(_line_count(path) for path in root.glob(f"*/{filename}")) if root.exists() else 0


def classify_paper_final_state(
    row: dict[str, Any],
    *,
    rag_ids: set[str],
    deep_ids: set[str],
    queue_terminal_states: dict[str, str] | None = None,
) -> str:
    """Assign exactly one active terminal state to a canonical record.

    The precedence protects the formal RAG first, then removes version/scope
    records from processing queues. A failed extraction without an acquired
    PDF is a PDF acquisition state, not a PDF processing failure.
    """
    paper_id = str(row.get("paper_id") or "")
    if str(row.get("duplicate_status") or "") == "RELATED_VERSION":
        return "RELATED_VERSION"
    if str(row.get("library_status") or "") == "ARCHIVED":
        return "ARCHIVED"
    if (
        paper_id in rag_ids
        and paper_id in deep_ids
        and str(row.get("library_status") or "") == "FORMAL"
    ):
        return "FORMAL_INDEXED"
    if str(row.get("domain_scope") or "") == "OUT_OF_SCOPE":
        return "OUT_OF_SCOPE"
    if not bool(row.get("pdf_valid")):
        return "PDF_NOT_ACQUIRED"
    queue_state = str((queue_terminal_states or {}).get(paper_id) or "")
    if queue_state == "MANUAL_VISUAL_REVIEW_REQUIRED":
        return "NEEDS_HUMAN_REVIEW"
    if queue_state == "COMPLETE_NOT_INDEXED":
        return "COMPLETE_NOT_INDEXED"
    if str(row.get("extraction_status") or "") == "FAILED":
        return "PROCESSING_FAILED"
    if str(row.get("evidence_status") or "") == "NEEDS_HUMAN_REVIEW":
        return "NEEDS_HUMAN_REVIEW"
    if paper_id in deep_ids or bool(row.get("deep_read_complete")):
        return "COMPLETE_NOT_INDEXED"
    return "PENDING_PROCESSING"


def paper_final_states(
    manifest: Iterable[dict[str, Any]],
    *,
    rag_ids: set[str],
    deep_ids: set[str],
    queue_terminal_states: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    return [
        {
            "paper_id": str(row.get("paper_id") or ""),
            "final_state": classify_paper_final_state(
                row,
                rag_ids=rag_ids,
                deep_ids=deep_ids,
                queue_terminal_states=queue_terminal_states,
            ),
        }
        for row in manifest
        if row.get("paper_id")
    ]


def _deleted_audit_count(base_dir: Path) -> int:
    payload = _read_json(
        base_dir / "outputs" / "deleted_failed_and_useless_papers.json", {}
    )
    rows = payload.get("deleted_records") or payload.get("records") or []
    return len(rows) if isinstance(rows, list) else 0


def _deleted_audit_rows(base_dir: Path) -> list[dict[str, Any]]:
    payload = _read_json(
        base_dir / "outputs" / "deleted_failed_and_useless_papers.json", {}
    )
    rows = payload.get("deleted_records") or payload.get("records") or []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _alias_audit_rows(base_dir: Path) -> list[dict[str, Any]]:
    root = base_dir / "data" / "deep_read_aliases"
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/MIGRATION.json")):
        payload = _read_json(path, {})
        if isinstance(payload, dict) and payload.get("alias_paper_id"):
            rows.append(payload)
    return rows


def compute_corpus_statistics(base_dir: Path) -> dict[str, Any]:
    """Compute the authoritative statistics snapshot.

    Logical-work grouping comes from the durable Stage-2 inventory. Physical
    file/SHA counts are refreshed from all configured PDF roots. This operation
    is explicit and is never run on every Streamlit rerun.
    """
    base_dir = Path(base_dir).resolve()
    inventory_path = (
        base_dir / "data" / "tasks" / "full_library_deep_read" / "inventory.json"
    )
    inventory = _read_json(inventory_path, {})
    pdf_paths = _pdf_paths(base_dir)
    hashes: dict[str, list[str]] = {}
    root_counts: Counter[str] = Counter()
    for path in pdf_paths:
        relative = path.relative_to(base_dir)
        root_name = relative.parts[0]
        if root_name == "paper" and len(relative.parts) > 1:
            root_name = "paper/pdfs"
        root_counts[root_name] += 1
        try:
            hashes.setdefault(_sha256(path), []).append(path.relative_to(base_dir).as_posix())
        except OSError:
            continue

    manifest = _read_jsonl(base_dir / "data" / "paper_manifest.jsonl")
    logical_documents = list(inventory.get("logical_documents") or [])
    inventory_hashes = {
        str(row.get("sha256") or "")
        for row in inventory.get("files") or []
        if row.get("sha256")
    }

    # Version relationships found by the later metadata gate are recorded as
    # canonical rows, not retroactively written into the frozen Stage-2
    # inventory. Count those explicit reconciliations exactly once.
    post_inventory_related = {
        str(row.get("paper_id") or "")
        for row in manifest
        if row.get("duplicate_status") == "RELATED_VERSION"
        and row.get("version_relation")
        and row.get("duplicate_of")
    }
    # Active logical counts follow the canonical manifest. The frozen Stage-2
    # inventory remains an audit source, but must not keep deleted records in
    # current UI/CLI statistics.
    valid_manifest_identities = {
        str(row.get("file_hash_sha256") or row.get("paper_id") or "")
        for row in manifest
        if row.get("pdf_valid")
        and row.get("duplicate_status") != "RELATED_VERSION"
        and row.get("library_status") != "ARCHIVED"
    }
    logical_count = len(valid_manifest_identities) if manifest else len(hashes)
    valid_logical_count = logical_count
    pdf_assets = _read_jsonl(base_dir / "data" / "pdf_files.jsonl")
    rag_manifest = _read_json(base_dir / "data" / "rag" / "manifest.json", {})
    rag_ids = {str(value) for value in rag_manifest.get("paper_ids") or [] if value}
    deep_statuses = _deep_statuses(base_dir)
    deep_ids = {
        str(row.get("paper_id") or "")
        for row in deep_statuses
        if row.get("deep_read_complete") and row.get("paper_id")
    }
    manifest_by_id = {str(row.get("paper_id") or ""): row for row in manifest}
    formal_indexed_ids = {
        paper_id
        for paper_id in rag_ids
        if str((manifest_by_id.get(paper_id) or {}).get("library_status") or "") == "FORMAL"
        and paper_id in deep_ids
    }
    queue = _read_json(
        base_dir / "data" / "tasks" / "full_library_deep_read" / "queue.json", {}
    )
    terminal_counts = Counter(
        str(row.get("terminal_state") or "") for row in queue.get("tasks") or []
    )
    queue_terminal_states = {
        str(row.get("canonical_paper_id") or ""): str(
            row.get("terminal_state") or ""
        )
        for row in queue.get("tasks") or []
        if row.get("canonical_paper_id")
    }
    final_state_rows = paper_final_states(
        manifest,
        rag_ids=rag_ids,
        deep_ids=deep_ids,
        queue_terminal_states=queue_terminal_states,
    )
    final_state_counts = Counter(row["final_state"] for row in final_state_rows)
    deleted_audit_rows = _deleted_audit_rows(base_dir)
    alias_audit_rows = _alias_audit_rows(base_dir)
    final_state_counts["DELETED"] = len(deleted_audit_rows)
    for state in FINAL_STATES:
        final_state_counts.setdefault(state, 0)

    active_final_state_rows = [
        row
        for row in final_state_rows
        if row["final_state"] not in NON_ACTIVE_PRIMARY_STATES
    ]
    archived_count = final_state_counts["ARCHIVED"]
    inventory_primary_ids = {
        str((row.get("primary_pdf") or {}).get("canonical_paper_id") or "")
        for row in logical_documents
        if (row.get("primary_pdf") or {}).get("canonical_paper_id")
    }
    deleted_ids = {
        str(row.get("paper_id") or row.get("canonical_id") or "")
        for row in deleted_audit_rows
        if row.get("paper_id") or row.get("canonical_id")
    }
    deleted_inventory_overlap = inventory_primary_ids & deleted_ids
    historical_inventory_primary_count = max(
        0, len(logical_documents) - len(post_inventory_related)
    )

    reconciliation_rows: list[dict[str, Any]] = []
    final_state_by_id = {
        row["paper_id"]: row["final_state"] for row in final_state_rows
    }
    for row in manifest:
        paper_id = str(row.get("paper_id") or "")
        state = final_state_by_id.get(paper_id, "")
        record_class = (
            "RELATED_VERSION"
            if state == "RELATED_VERSION"
            else "ARCHIVED"
            if state == "ARCHIVED"
            else "ACTIVE_CANONICAL_PRIMARY"
        )
        reconciliation_rows.append(
            {
                "record_id": paper_id,
                "record_class": record_class,
                "final_state": state,
                "library_status": str(row.get("library_status") or ""),
                "pdf_valid": bool(row.get("pdf_valid")),
                "canonical_of": str(row.get("duplicate_of") or ""),
                "source": "data/paper_manifest.jsonl",
            }
        )
    for row in deleted_audit_rows:
        reconciliation_rows.append(
            {
                "record_id": str(row.get("paper_id") or row.get("canonical_id") or ""),
                "record_class": "DELETED_AUDIT_ONLY",
                "final_state": "DELETED",
                "library_status": "DELETED",
                "pdf_valid": bool(row.get("pdf_valid")),
                "canonical_of": "",
                "source": "outputs/deleted_failed_and_useless_papers.json",
            }
        )
    for row in alias_audit_rows:
        reconciliation_rows.append(
            {
                "record_id": str(row.get("alias_paper_id") or ""),
                "record_class": "ALIAS_OLD_ID",
                "final_state": "ALIAS",
                "library_status": str(row.get("status") or "ARCHIVED_ALIAS"),
                "pdf_valid": False,
                "canonical_of": str(row.get("canonical_paper_id") or ""),
                "source": "data/deep_read_aliases/*/MIGRATION.json",
            }
        )

    counts = {
        "pdf_file_count": len(pdf_paths),
        "unique_pdf_sha256_count": len(hashes),
        "duplicate_pdf_file_count": sum(max(0, len(paths) - 1) for paths in hashes.values()),
        "pdf_asset_count": len({row.get("pdf_file_id") for row in pdf_assets if row.get("pdf_file_id")}),
        "acquired_logical_literature_count": logical_count,
        "current_logical_literature_count": len(active_final_state_rows) if manifest else logical_count,
        "active_canonical_primary_record_count": len(active_final_state_rows) if manifest else logical_count,
        "canonical_paper_record_count": len(manifest_by_id),
        "archived_count": archived_count,
        "alias_old_id_count": len(alias_audit_rows),
        "historical_pre_cleanup_acquired_primary_count": historical_inventory_primary_count,
        "manual_visual_review_required_count": terminal_counts[
            "MANUAL_VISUAL_REVIEW_REQUIRED"
        ],
        "pdf_valid_logical_count": valid_logical_count,
        "deep_read_complete_count": len(deep_ids),
        "rag_paper_count": len(rag_ids),
        "evidence_record_count": _trusted_evidence_count(base_dir),
        "formula_record_count": _artifact_count(base_dir, "equations.jsonl"),
        "visual_review_item_count": _artifact_count(base_dir, "visual_review_items.jsonl"),
        "formal_indexed_count": final_state_counts["FORMAL_INDEXED"],
        "complete_not_indexed_count": final_state_counts["COMPLETE_NOT_INDEXED"],
        "pending_processing_count": final_state_counts["PENDING_PROCESSING"],
        "processing_failed_count": final_state_counts["PROCESSING_FAILED"],
        "needs_human_review_count": final_state_counts["NEEDS_HUMAN_REVIEW"],
        "pdf_not_acquired_count": final_state_counts["PDF_NOT_ACQUIRED"],
        "related_version_count": final_state_counts["RELATED_VERSION"],
        "out_of_scope_count": final_state_counts["OUT_OF_SCOPE"],
        "deleted_count": final_state_counts["DELETED"],
    }
    definitions = {
        "pdf_file_count": "Physical *.pdf files across the four configured PDF roots.",
        "unique_pdf_sha256_count": "Unique physical PDF byte identities.",
        "pdf_asset_count": "Unique pdf_file_id rows in data/pdf_files.jsonl.",
        "acquired_logical_literature_count": "Logical works represented by acquired PDFs after DOI/metadata/fingerprint version grouping.",
        "canonical_paper_record_count": "All canonical manifest records, including no-PDF candidates and quarantined records.",
        "formal_indexed_count": "FORMAL canonical papers that are deep-read complete and present in the unified RAG manifest.",
        "current_logical_literature_count": "Active canonical primary records, including metadata-only PDF_NOT_ACQUIRED records; excludes RELATED_VERSION, ARCHIVED, DELETED and alias/audit-only identities.",
        "active_canonical_primary_record_count": "Same active-primary population as current_logical_literature_count.",
        "historical_pre_cleanup_acquired_primary_count": "Frozen inventory logical documents minus explicit related versions, before later useless-record cleanup.",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "signature": corpus_statistics_signature(base_dir),
        "counts": counts,
        "definitions": definitions,
        "sources": {
            "logical_inventory": "data/tasks/full_library_deep_read/inventory.json",
            "manifest": "data/paper_manifest.jsonl",
            "pdf_assets": "data/pdf_files.jsonl",
            "rag_manifest": "data/rag/manifest.json",
        },
        "validation": {
            "inventory_unique_hash_count": int(inventory.get("unique_hash_count") or 0),
            "filesystem_unique_hash_count": len(hashes),
            "inventory_covers_all_unique_hashes": inventory_hashes == set(hashes),
            "pdf_root_counts": dict(root_counts),
            "manifest_status_counts": dict(
                Counter(str(row.get("library_status") or "") for row in manifest)
            ),
        "final_state_counts": {
            state: int(final_state_counts[state]) for state in FINAL_STATES
        },
        "active_final_state_sum": sum(
            int(final_state_counts[state])
            for state in FINAL_STATES
            if state not in NON_ACTIVE_PRIMARY_STATES
        ),
        "active_final_states_explain_current_total": sum(
            int(final_state_counts[state])
            for state in FINAL_STATES
            if state not in NON_ACTIVE_PRIMARY_STATES
        ) == (len(active_final_state_rows) if manifest else logical_count),
        "paper_final_states": final_state_rows,
        "record_reconciliation": reconciliation_rows,
        "record_class_counts": dict(
            Counter(row["record_class"] for row in reconciliation_rows)
        ),
        "historical_count_reconciliation": {
            "frozen_inventory_logical_documents": len(logical_documents),
            "minus_related_versions": len(post_inventory_related),
            "historical_pre_cleanup_acquired_primary_count": historical_inventory_primary_count,
            "minus_later_deleted_inventory_records": len(deleted_inventory_overlap),
            "current_acquired_active_primary_count": valid_logical_count,
            "current_metadata_only_active_primary_count": sum(
                row["final_state"] == "PDF_NOT_ACQUIRED"
                for row in active_final_state_rows
            ),
            "current_active_canonical_primary_count": len(active_final_state_rows),
        },
        "terminal_state_counts": dict(terminal_counts),
        "post_inventory_related_paper_ids": sorted(post_inventory_related),
        "duplicate_pdf_relationships": [
            {"sha256": sha, "paths": paths}
            for sha, paths in sorted(hashes.items())
            if len(paths) > 1
        ],
    },
    }


def get_corpus_statistics(
    base_dir: Path,
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    base_dir = Path(base_dir).resolve()
    snapshot_path = base_dir / "data" / "system" / "corpus_statistics.json"
    signature = corpus_statistics_signature(base_dir)
    cached = _read_json(snapshot_path, {})
    if (
        not refresh
        and cached.get("schema_version") == SCHEMA_VERSION
        and cached.get("signature") == signature
    ):
        return cached
    result = compute_corpus_statistics(base_dir)
    _atomic_json(snapshot_path, result)
    return result


def statistics_counts(base_dir: Path, *, refresh: bool = False) -> dict[str, int]:
    return dict(get_corpus_statistics(base_dir, refresh=refresh).get("counts") or {})


def read_corpus_statistics_snapshot(base_dir: Path) -> dict[str, Any]:
    """Read the durable snapshot without scanning PDF/deep-read directories.

    Streamlit reruns use this function.  Mutating workflows explicitly refresh
    the snapshot after a successful commit.
    """
    path = Path(base_dir).resolve() / "data" / "system" / "corpus_statistics.json"
    cached = _read_json(path, {})
    if cached.get("schema_version") == SCHEMA_VERSION:
        return cached
    return get_corpus_statistics(Path(base_dir).resolve(), refresh=True)

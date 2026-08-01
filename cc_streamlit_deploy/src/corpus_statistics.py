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


SCHEMA_VERSION = "corpus-statistics-1.0"
PDF_ROOTS = ("paper/pdfs", "papers", "early_papers", "followup_papers")


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
    if inventory_path.exists() and inventory:
        inventory_logical_count = int(inventory.get("logical_document_count") or len(logical_documents))
        inventory_valid_logical_count = sum(
            bool((row.get("primary_pdf") or {}).get("valid_pdf")) for row in logical_documents
        )
        inventory_related_version_count = int(inventory.get("different_version_count") or 0)
        related_version_count = inventory_related_version_count + len(post_inventory_related)
        logical_count = max(0, inventory_logical_count - len(post_inventory_related))
        valid_logical_count = max(0, inventory_valid_logical_count - len(post_inventory_related))
    else:
        valid_manifest_identities = {
            str(row.get("file_hash_sha256") or row.get("paper_id") or "")
            for row in manifest
            if row.get("pdf_valid") and row.get("duplicate_status") != "RELATED_VERSION"
        }
        logical_count = len(valid_manifest_identities) or len(hashes)
        valid_logical_count = logical_count
        related_version_count = len(post_inventory_related)
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

    counts = {
        "pdf_file_count": len(pdf_paths),
        "unique_pdf_sha256_count": len(hashes),
        "duplicate_pdf_file_count": sum(max(0, len(paths) - 1) for paths in hashes.values()),
        "pdf_asset_count": len({row.get("pdf_file_id") for row in pdf_assets if row.get("pdf_file_id")}),
        "acquired_logical_literature_count": logical_count,
        "canonical_paper_record_count": len(manifest_by_id),
        "related_version_count": related_version_count,
        "formal_indexed_count": len(formal_indexed_ids),
        "complete_not_indexed_count": len(deep_ids - rag_ids),
        "manual_visual_review_required_count": terminal_counts[
            "MANUAL_VISUAL_REVIEW_REQUIRED"
        ],
        "pdf_not_acquired_count": sum(not bool(row.get("pdf_valid")) for row in manifest),
        "pdf_valid_logical_count": valid_logical_count,
        "deep_read_complete_count": len(deep_ids),
        "rag_paper_count": len(rag_ids),
        "evidence_record_count": _trusted_evidence_count(base_dir),
        "formula_record_count": _artifact_count(base_dir, "equations.jsonl"),
        "visual_review_item_count": _artifact_count(base_dir, "visual_review_items.jsonl"),
        "pending_or_failed_acquired_count": max(0, logical_count - len(rag_ids)),
    }
    definitions = {
        "pdf_file_count": "Physical *.pdf files across the four configured PDF roots.",
        "unique_pdf_sha256_count": "Unique physical PDF byte identities.",
        "pdf_asset_count": "Unique pdf_file_id rows in data/pdf_files.jsonl.",
        "acquired_logical_literature_count": "Logical works represented by acquired PDFs after DOI/metadata/fingerprint version grouping.",
        "canonical_paper_record_count": "All canonical manifest records, including no-PDF candidates and quarantined records.",
        "formal_indexed_count": "FORMAL canonical papers that are deep-read complete and present in the unified RAG manifest.",
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

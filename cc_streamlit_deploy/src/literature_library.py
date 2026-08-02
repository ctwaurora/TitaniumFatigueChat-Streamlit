"""Canonical literature-library read model and state transitions."""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.domain_scope import OUT_OF_SCOPE, classify_literature_scope
from src.metadata_service import fetch_metadata, is_valid_title
from src.stage1_store import (
    BASE_DIR,
    TRUSTED_EVIDENCE_PATH,
    extract_basic_pdf_metadata,
    load_paper_manifest,
    normalize_doi,
    normalize_title,
    register_metadata_record,
    update_paper_domain_scope,
    update_paper_library_status,
)


INVALID_METADATA = "INVALID_METADATA"
VALID_METADATA = "VALID"
QUARANTINED = "QUARANTINED"
HUMAN_CONFIRMED = "HUMAN_CONFIRMED"
HUMAN_REVISION_REQUIRED = "HUMAN_REVISION_REQUIRED"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_manifest(base_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path = base_dir / "data" / "paper_manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temp.replace(path)


def trusted_evidence_rows(base_dir: Path = BASE_DIR) -> List[Dict[str, str]]:
    path = base_dir / "data" / "evidence" / "trusted_evidence.csv"
    return _read_csv(path)


def _deep_read_status(base_dir: Path, paper_id: str) -> Dict[str, Any]:
    return _read_json(
        base_dir / "data" / "deep_read" / paper_id / "extraction_status.json",
        {},
    )


def _rag_paper_ids(base_dir: Path) -> set[str]:
    payload = _read_json(base_dir / "data" / "rag" / "manifest.json", {})
    return {str(value) for value in payload.get("paper_ids") or [] if value}


def _canonical_snapshot(base_dir: Path) -> Dict[str, Any]:
    payload = _read_json(
        base_dir / "data" / "canonical_literature_snapshot.json",
        {},
    )
    if payload.get("schema_version") != "canonical-library-snapshot-1.0":
        return {}
    return payload


def _candidate_source_url(row: Dict[str, Any]) -> str:
    return str(
        row.get("source_url")
        or row.get("url")
        or row.get("landing_page")
        or row.get("pdf_url")
        or ""
    ).strip()


def load_candidate_records(base_dir: Path = BASE_DIR) -> List[Dict[str, Any]]:
    """Normalize candidates and quarantine invalid/duplicate metadata in memory."""
    path = base_dir / "data" / "candidate_papers.csv"
    rows = _read_csv(path)
    manifest = load_paper_manifest(base_dir)
    formal_dois = {
        normalize_doi(row.get("doi") or "")
        for row in manifest
        if str(row.get("library_status") or "").upper() == "FORMAL"
        and normalize_doi(row.get("doi") or "")
    }
    # Legacy formal CSV remains metadata enrichment only, never a count source.
    formal_rows = _read_csv(base_dir / "data" / "literature_database.csv")
    formal_dois.update(
        normalize_doi(row.get("doi") or "")
        for row in formal_rows
        if normalize_doi(row.get("doi") or "")
    )
    formal_titles = {
        normalize_title(row.get("title") or "")
        for row in formal_rows
        if is_valid_title(row.get("title"))
    }

    first_candidate_by_doi: Dict[str, str] = {}
    normalized: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        candidate_id = str(
            row.get("candidate_id") or row.get("paper_id") or f"CAND_{index:04d}"
        ).strip()
        raw_title = str(row.get("title") or "").strip()
        title = " ".join(raw_title.split()) if is_valid_title(raw_title) else ""
        doi = normalize_doi(row.get("doi") or "")
        title_key = normalize_title(title)
        authors = " ".join(str(row.get("authors") or "").split()).strip()
        year = str(row.get("year") or "").strip()
        duplicate_of = ""
        duplicate_status = "UNIQUE"
        if doi and doi in formal_dois:
            duplicate_status = "DUPLICATE"
            duplicate_of = next(
                (
                    str(item.get("paper_id") or "")
                    for item in manifest
                    if normalize_doi(item.get("doi") or "") == doi
                ),
                "FORMAL_DOI",
            )
        elif title_key and title_key in formal_titles:
            duplicate_status = "DUPLICATE"
            duplicate_of = "FORMAL_TITLE"
        elif doi and doi in first_candidate_by_doi:
            duplicate_status = "DUPLICATE"
            duplicate_of = first_candidate_by_doi[doi]
        elif doi:
            first_candidate_by_doi[doi] = candidate_id

        source_url = _candidate_source_url(row)
        metadata_source = str(
            row.get("metadata_source") or row.get("source_database") or ""
        ).strip()
        raw_oa_status = str(row.get("oa_status") or "").strip()
        if not raw_oa_status and str(row.get("is_open_access") or "").strip():
            raw_oa_status = (
                "OPEN"
                if str(row.get("is_open_access") or "").lower() == "true"
                else "CLOSED"
            )
        validation_errors = []
        if not title:
            validation_errors.append("MISSING_OR_PLACEHOLDER_TITLE")
        if not authors or authors.lower() in {"nan", "none", "unknown"}:
            validation_errors.append("MISSING_AUTHORS")
        if not re.search(r"\b(19|20)\d{2}\b", year) and year.upper() != "UNKNOWN":
            validation_errors.append("MISSING_YEAR")
        if not doi and not source_url.lower().startswith("https://"):
            validation_errors.append("MISSING_DOI_OR_TRUSTED_SOURCE_URL")
        if not metadata_source:
            validation_errors.append("MISSING_METADATA_SOURCE")
        if not raw_oa_status:
            validation_errors.append("MISSING_OA_STATUS")
        validation = VALID_METADATA if not validation_errors else INVALID_METADATA
        quarantine_reason = ";".join(validation_errors)
        normalized.append(
            {
                **row,
                "paper_id": candidate_id,
                "candidate_id": candidate_id,
                "canonical_paper_id": duplicate_of if duplicate_status == "DUPLICATE" else "",
                "title": title,
                "authors": authors,
                "year": year,
                "doi": doi,
                "source_url": source_url,
                "metadata_source": metadata_source,
                "oa_status": raw_oa_status,
                "pdf_status": str(
                    row.get("pdf_status")
                    or row.get("full_text_status")
                    or "PDF_NOT_ACQUIRED"
                ),
                "duplicate_status": duplicate_status,
                "duplicate_of": duplicate_of,
                "validation_status": validation,
                "quarantine_reason": quarantine_reason,
                "selectable": validation == VALID_METADATA and duplicate_status == "UNIQUE",
                "source": "candidate",
                "type_code": str(row.get("paper_type_primary") or "candidate"),
            }
        )
    return normalized


def invalid_candidate_records(base_dir: Path = BASE_DIR) -> List[Dict[str, Any]]:
    return [
        row
        for row in load_candidate_records(base_dir)
        if row["validation_status"] == INVALID_METADATA
    ]


def valid_candidate_records(base_dir: Path = BASE_DIR) -> List[Dict[str, Any]]:
    return [
        row
        for row in load_candidate_records(base_dir)
        if row["validation_status"] == VALID_METADATA
        and row["duplicate_status"] == "UNIQUE"
    ]


def canonical_pdf_records(base_dir: Path = BASE_DIR) -> List[Dict[str, Any]]:
    """Return one UI record per DOI/title/SHA canonical PDF group."""
    # Imported lazily to keep this domain module usable without Streamlit.
    from src.data_cache import build_pdf_duplicate_inventory

    inventory = build_pdf_duplicate_inventory(base_dir)
    manifest = load_paper_manifest(base_dir)
    by_id = {str(row.get("paper_id") or ""): row for row in manifest}
    by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        file_hash = str(row.get("file_hash_sha256") or "")
        if file_hash:
            by_hash[file_hash].append(row)
    evidence_by_paper: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in trusted_evidence_rows(base_dir):
        evidence_by_paper[str(row.get("paper_id") or "")].append(row)
    rag_ids = _rag_paper_ids(base_dir)

    records: List[Dict[str, Any]] = []
    for group in inventory["groups"].values():
        group_ids = [str(value) for value in group.get("paper_ids") or [] if value]
        candidates = [by_id[value] for value in group_ids if value in by_id]
        for file_hash in group.get("hashes") or []:
            candidates.extend(by_hash.get(str(file_hash), []))
        candidates = list(
            {str(row.get("paper_id") or id(row)): row for row in candidates}.values()
        )
        candidates.sort(
            key=lambda row: (
                bool(row.get("deep_read_complete")),
                bool(row.get("pdf_valid")),
                bool(row.get("title")),
            ),
            reverse=True,
        )
        primary = dict(candidates[0]) if candidates else {}
        paper_id = str(
            primary.get("paper_id")
            or group.get("canonical_paper_id")
            or ""
        )
        paths = [Path(value) for value in group.get("paths") or []]
        title = str(primary.get("title") or "").strip()
        authors = str(primary.get("authors") or "").strip()
        doi = normalize_doi(primary.get("doi") or "")
        publication_date = str(primary.get("publication_date") or "")
        if (not is_valid_title(title) or not doi) and paths:
            try:
                metadata = extract_basic_pdf_metadata(paths[0].read_bytes(), paths[0].name)
            except OSError:
                metadata = {}
            if not is_valid_title(title) and is_valid_title(metadata.get("title")):
                title = str(metadata["title"]).strip()
            authors = authors or str(metadata.get("authors") or "")
            doi = doi or normalize_doi(metadata.get("doi") or "")
            publication_date = publication_date or str(
                metadata.get("publication_date") or ""
            )
        deep_status = _deep_read_status(base_dir, paper_id)
        deep_complete = bool(
            deep_status.get("deep_read_complete")
            or primary.get("deep_read_complete")
        )
        evidence = evidence_by_paper.get(paper_id, [])
        review_counts = Counter(
            str(row.get("review_status") or "") for row in evidence
        )
        rag_status = (
            "INDEXED_STAGE3_UNIFIED"
            if paper_id in rag_ids
            else "NOT_INDEXED"
        )
        evidence_status = str(primary.get("evidence_status") or "")
        if evidence_status not in {"AUTO_VALIDATED", HUMAN_CONFIRMED}:
            evidence_status = (
                HUMAN_CONFIRMED
                if review_counts[HUMAN_CONFIRMED] == len(evidence) and evidence
                else "AUTO_VALIDATED"
                if evidence
                and deep_complete
                and rag_status == "INDEXED_STAGE3_UNIFIED"
                else "PENDING_CONFIRMATION"
                if evidence
                else "NO_EVIDENCE"
            )
        scope = classify_literature_scope(
            {
                **primary,
                "title": title,
                "abstract": " ".join(
                    str(row.get("original_text") or row.get("claim") or "")
                    for row in evidence[:12]
                ),
            }
        )
        if str(primary.get("domain_scope") or "") in {
            "CORE",
            "CONTEXT",
            OUT_OF_SCOPE,
        }:
            scope["domain_scope"] = str(primary["domain_scope"])
            scope["scope_reason"] = str(
                primary.get("scope_reason") or scope["scope_reason"]
            )
        library_status = str(primary.get("library_status") or "CANDIDATE")
        if (
            library_status != QUARANTINED
            and bool(primary.get("pdf_valid"))
            and deep_complete
            and evidence
            and evidence_status in {"AUTO_VALIDATED", HUMAN_CONFIRMED}
            and rag_status == "INDEXED_STAGE3_UNIFIED"
            and scope["domain_scope"] != OUT_OF_SCOPE
        ):
            library_status = "FORMAL"
        records.append(
            {
                **primary,
                "paper_id": paper_id,
                "canonical_paper_id": paper_id,
                "title": title if is_valid_title(title) else "",
                "authors": authors,
                "year": (
                    publication_date[:4]
                    if publication_date[:4].isdigit()
                    else str(primary.get("year") or "")
                ),
                "doi": doi,
                "source": "formal",
                "source_url": str(primary.get("source_path") or ""),
                "type_code": str(primary.get("paper_type_primary") or "other"),
                "validation_status": (
                    VALID_METADATA if is_valid_title(title) else INVALID_METADATA
                ),
                "duplicate_status": "UNIQUE",
                "linked_versions": list(dict.fromkeys(group.get("paths") or [])),
                "pdf_status": "PDF_VALID",
                "pdf_valid": True,
                "real_page_count": int(
                    deep_status.get("real_page_count")
                    or primary.get("real_page_count")
                    or 0
                ),
                "deep_read_complete": deep_complete,
                "deep_read_status": (
                    "COMPLETED" if deep_complete else str(deep_status.get("status") or "PENDING")
                ),
                "evidence_count": len(evidence),
                "evidence_status": evidence_status,
                "rag_status": rag_status,
                "library_status": library_status,
                "domain_scope": scope["domain_scope"],
                "scope_reason": scope["scope_reason"],
                "selectable": bool(
                    is_valid_title(title)
                    and deep_complete
                    and evidence
                    and evidence_status in {"AUTO_VALIDATED", HUMAN_CONFIRMED}
                    and library_status == "FORMAL"
                    and rag_status == "INDEXED_STAGE3_UNIFIED"
                    and scope["domain_scope"] != OUT_OF_SCOPE
                ),
                "quarantine_reason": str(primary.get("quarantine_reason") or ""),
            }
        )
    snapshot_rows = [
        dict(row)
        for row in (_canonical_snapshot(base_dir).get("records") or [])
        if is_valid_title(row.get("title"))
    ]
    # Runtime records override a read-only deployment snapshot with the same
    # DOI or normalized title.
    merged: Dict[str, Dict[str, Any]] = {}
    for row in [*snapshot_rows, *records]:
        doi = normalize_doi(row.get("doi") or "")
        title = normalize_title(row.get("title") or "")
        key = (
            f"doi:{doi}"
            if doi
            else f"title:{title}"
            if title
            else f"id:{row.get('paper_id')}"
        )
        merged[key] = row
    return sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("title") or "").lower(),
            str(row.get("paper_id") or ""),
        ),
    )


def canonical_library_records(base_dir: Path = BASE_DIR) -> List[Dict[str, Any]]:
    """Build every library view from the single canonical paper manifest."""
    manifest = load_paper_manifest(base_dir)
    # The manifest already owns PDF/canonical identity.  Re-hashing every PDF
    # just to render the first page made a cold library view take several
    # seconds.  Only aggregate the trusted-evidence CSV here.
    evidence_counts = Counter(
        str(row.get("canonical_paper_id") or row.get("paper_id") or "")
        for row in trusted_evidence_rows(base_dir)
    )
    records: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for source in manifest:
        paper_id = str(source.get("paper_id") or "")
        row = dict(source)
        row["evidence_count"] = evidence_counts.get(paper_id, 0)
        title = str(row.get("title") or "").strip()
        authors = " ".join(str(row.get("authors") or "").split())
        publication_date = str(
            row.get("publication_date") or row.get("year") or ""
        )
        year_match = re.search(r"\b(19|20)\d{2}\b", publication_date)
        year = year_match.group(0) if year_match else ""
        doi = normalize_doi(row.get("doi") or "")
        source_url = str(
            row.get("source_url") or row.get("source_path") or ""
        ).strip()
        trusted_url = source_url.startswith("https://")
        metadata_source = str(
            row.get("metadata_source") or row.get("source_type") or ""
        ).strip()
        scope = classify_literature_scope(row)
        if str(row.get("domain_scope") or "") in {
            "CORE",
            "CONTEXT",
            OUT_OF_SCOPE,
        }:
            scope["domain_scope"] = str(row["domain_scope"])
            scope["scope_reason"] = str(
                row.get("scope_reason") or scope["scope_reason"]
            )
        metadata_valid = bool(
            is_valid_title(title)
            and authors
            and authors.lower() not in {"nan", "none", "unknown"}
            and year
            and (doi or trusted_url or row.get("doi_status") == "NOT_AVAILABLE_VERIFIED")
            and metadata_source
        )
        library_status = str(row.get("library_status") or "CANDIDATE")
        formal_ready = bool(
            is_valid_title(title)
            and row.get("pdf_valid")
            and row.get("deep_read_complete")
            and int(row.get("evidence_count") or 0) > 0
            and row.get("evidence_status")
            in {"AUTO_VALIDATED", HUMAN_CONFIRMED}
            and row.get("rag_status") == "INDEXED_STAGE3_UNIFIED"
            and scope["domain_scope"] != OUT_OF_SCOPE
        )
        if row.get("duplicate_status") == "RELATED_VERSION":
            library_status = "COMPLETE_NOT_INDEXED"
        elif (
            (not metadata_valid and not formal_ready)
            or row.get("duplicate_status") not in {
            "", "UNIQUE", None
            }
        ):
            library_status = QUARANTINED
        elif library_status == "ARCHIVED":
            pass
        elif formal_ready:
            library_status = "FORMAL"
        elif library_status == "FORMAL":
            library_status = "CANDIDATE"
        key = f"doi:{doi}" if doi else f"title:{normalize_title(title)}"
        if key in seen:
            library_status = QUARANTINED
            row["quarantine_reason"] = "UNMERGED_CANONICAL_DUPLICATE"
            key = f"id:{paper_id}"
        seen.add(key)
        records.append(
            {
                **row,
                "paper_id": paper_id,
                "title": title if is_valid_title(title) else "",
                "authors": authors,
                "year": year,
                "doi": doi,
                "source_url": source_url,
                "metadata_source": metadata_source,
                "domain_scope": scope["domain_scope"],
                "scope_reason": scope["scope_reason"],
                "library_status": library_status,
                "validation_status": (
                    VALID_METADATA
                    if metadata_valid or formal_ready
                    else INVALID_METADATA
                ),
                "source": (
                    "formal"
                    if library_status == "FORMAL"
                    else "candidate"
                    if library_status not in {QUARANTINED, "ARCHIVED"}
                    else "archived"
                    if library_status == "ARCHIVED"
                    else "quarantined"
                ),
                "pdf_status": (
                    "PDF_VALID" if row.get("pdf_valid") else "PDF_NOT_ACQUIRED"
                ),
                "deep_read_complete": bool(row.get("deep_read_complete")),
                "evidence_count": int(row.get("evidence_count") or 0),
                "rag_status": str(row.get("rag_status") or "NOT_INDEXED"),
                "selectable": bool(
                    library_status == "FORMAL" and formal_ready
                ),
                "type_code": str(row.get("paper_type_primary") or "other"),
            }
        )
    return sorted(
        records,
        key=lambda row: (
            str(row.get("library_status") or ""),
            str(row.get("title") or "").lower(),
            str(row.get("paper_id") or ""),
        ),
    )


def library_statistics(base_dir: Path = BASE_DIR) -> Dict[str, int]:
    from src.corpus_statistics import statistics_counts

    records = canonical_library_records(base_dir)
    corpus = statistics_counts(base_dir)
    result = {
        "unique_literature": corpus["current_logical_literature_count"],
        "acquired_logical_literature": corpus["acquired_logical_literature_count"],
        "canonical_paper_records": corpus["canonical_paper_record_count"],
        "pdf_files": corpus["pdf_file_count"],
        "unique_pdf_sha256": corpus["unique_pdf_sha256_count"],
        "pdf_assets": corpus["pdf_asset_count"],
        "related_versions": corpus["related_version_count"],
        "archived": corpus["archived_count"],
        "alias_old_ids": corpus["alias_old_id_count"],
        "historical_pre_cleanup_acquired_primary": corpus[
            "historical_pre_cleanup_acquired_primary_count"
        ],
        "candidate_metadata": sum(
            row.get("library_status") not in {"FORMAL", QUARANTINED}
            for row in records
        ),
        "pdf_acquired": corpus["pdf_valid_logical_count"],
        "pdf_not_acquired": corpus["pdf_not_acquired_count"],
        "deep_read_complete": corpus["deep_read_complete_count"],
        "evidence_pending": sum(
            bool(row.get("evidence_count"))
            and row.get("evidence_status")
            not in {"AUTO_VALIDATED", HUMAN_CONFIRMED}
            for row in records
        ),
        "evidence_confirmed": sum(
            row.get("evidence_status") in {"AUTO_VALIDATED", HUMAN_CONFIRMED}
            for row in records
        ),
        "rag_indexed": corpus["rag_paper_count"],
        "formal_indexed": corpus["formal_indexed_count"],
        "complete_not_indexed": corpus["complete_not_indexed_count"],
        "pending_processing": corpus["pending_processing_count"],
        "processing_failed": corpus["processing_failed_count"],
        "needs_human_review": corpus["needs_human_review_count"],
        "related_versions": corpus["related_version_count"],
        "out_of_scope": corpus["out_of_scope_count"],
        "deleted": corpus["deleted_count"],
        "invalid_metadata": sum(
            row.get("library_status") == QUARANTINED for row in records
        ),
    }
    snapshot = _canonical_snapshot(base_dir)
    if snapshot and result["unique_literature"] == 0:
        stored = snapshot.get("statistics") or {}
        for key in (
            "unique_literature",
            "pdf_acquired",
            "deep_read_complete",
            "evidence_pending",
            "evidence_confirmed",
            "rag_indexed",
            "pending_or_failed",
        ):
            result[key] = int(stored.get(key) or 0)
    return result


def eligible_paper_ids(
    paper_ids: Iterable[str],
    *,
    base_dir: Path = BASE_DIR,
    require_confirmation: bool = False,
) -> Dict[str, Any]:
    records = {row["paper_id"]: row for row in canonical_pdf_records(base_dir)}
    eligible: List[str] = []
    rejected: Dict[str, List[str]] = {}
    for paper_id in dict.fromkeys(str(value) for value in paper_ids if value):
        row = records.get(paper_id)
        reasons: List[str] = []
        if row is None:
            reasons.append("STALE_OR_DELETED_SELECTION_REMOVED")
        else:
            if row.get("snapshot_read_only"):
                reasons.append("RUNTIME_ARTIFACTS_UNAVAILABLE")
            if row.get("validation_status") != VALID_METADATA:
                reasons.append(INVALID_METADATA)
            if not row.get("pdf_valid"):
                reasons.append("VALID_PDF_REQUIRED")
            if not row.get("deep_read_complete"):
                reasons.append("DEEP_READ_REQUIRED")
            if int(row.get("evidence_count") or 0) < 1:
                reasons.append("EVIDENCE_REQUIRED")
            if (
                row.get("library_status") != "FORMAL"
                and not (
                    require_confirmation
                    and row.get("evidence_status")
                    in {"AUTO_VALIDATED", HUMAN_CONFIRMED}
                )
            ):
                reasons.append("FORMAL_LIBRARY_REQUIRED")
            if row.get("quarantine_reason"):
                reasons.append(QUARANTINED)
            if row.get("domain_scope") == OUT_OF_SCOPE:
                reasons.append("OUT_OF_TITANIUM_FATIGUE_SCOPE")
            if require_confirmation and row.get("evidence_status") not in {
                "AUTO_VALIDATED",
                HUMAN_CONFIRMED,
            }:
                reasons.append("EVIDENCE_CONFIRMATION_REQUIRED")
        if reasons:
            rejected[paper_id] = reasons
        else:
            eligible.append(paper_id)
    return {"eligible": eligible, "rejected": rejected}


def set_evidence_review_status(
    paper_id: str,
    status: str,
    *,
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    if status not in {HUMAN_CONFIRMED, HUMAN_REVISION_REQUIRED}:
        raise ValueError("Unsupported evidence review status")
    path = base_dir / "data" / "evidence" / "trusted_evidence.csv"
    rows = _read_csv(path)
    changed = 0
    now = _now()
    for row in rows:
        if str(row.get("paper_id") or "") != paper_id:
            continue
        row["review_status"] = status
        row["updated_at"] = now
        changed += 1
    if changed:
        _write_csv(path, rows)
    return {"paper_id": paper_id, "status": status, "updated_count": changed}


def quarantine_record(
    record_id: str,
    reason: str,
    *,
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    manifest = load_paper_manifest(base_dir)
    for row in manifest:
        if str(row.get("paper_id") or "") == record_id:
            row["library_status"] = QUARANTINED
            row["quarantine_reason"] = reason or "MANUAL_QUARANTINE"
            row["updated_at"] = _now()
            _write_manifest(base_dir, manifest)
            return {"status": QUARANTINED, "record_id": record_id}
    candidate_path = base_dir / "data" / "candidate_papers.csv"
    candidates = _read_csv(candidate_path)
    for row in candidates:
        if str(row.get("candidate_id") or row.get("paper_id") or "") == record_id:
            row["validation_status"] = INVALID_METADATA
            row["quarantine_reason"] = reason or "MANUAL_QUARANTINE"
            row["last_updated"] = _now()
            _write_csv(candidate_path, candidates)
            return {"status": QUARANTINED, "record_id": record_id}
    return {"status": "NOT_FOUND", "record_id": record_id}


def archive_canonical_records(
    paper_ids: Sequence[str], *, base_dir: Path = BASE_DIR
) -> Dict[str, Any]:
    requested = set(str(value) for value in paper_ids if value)
    rows = load_paper_manifest(base_dir)
    archived: List[str] = []
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        if paper_id not in requested:
            continue
        row["library_status"] = "ARCHIVED"
        row["updated_at"] = _now()
        archived.append(paper_id)
    if archived:
        _write_manifest(base_dir, rows)
    return {"status": "COMPLETED", "archived": archived}


def permanently_delete_canonical_records(
    paper_ids: Sequence[str], *, base_dir: Path = BASE_DIR
) -> Dict[str, Any]:
    """Delete only explicitly selected canonical records and rebuild Stage-3."""
    requested = set(str(value) for value in paper_ids if value)
    manifest = load_paper_manifest(base_dir)
    targets = [
        row for row in manifest if str(row.get("paper_id") or "") in requested
    ]
    kept = [
        row for row in manifest if str(row.get("paper_id") or "") not in requested
    ]
    if not targets:
        return {"status": "NOT_FOUND", "deleted": []}

    kept_paths = {
        str(Path(value).resolve())
        for row in kept
        for value in [
            row.get("canonical_pdf_path"),
            *(row.get("linked_versions") or []),
        ]
        if value
    }
    root = base_dir.resolve()
    deleted_files: List[str] = []
    for row in targets:
        for value in [
            row.get("canonical_pdf_path"),
            *(row.get("linked_versions") or []),
        ]:
            if not value:
                continue
            path = Path(value).resolve()
            if (
                path.is_file()
                and path.is_relative_to(root)
                and str(path) not in kept_paths
            ):
                path.unlink()
                deleted_files.append(path.name)
        deep_dir = (base_dir / "data" / "deep_read" / str(row["paper_id"])).resolve()
        if deep_dir.is_dir() and deep_dir.is_relative_to(root):
            shutil.rmtree(deep_dir)

    _write_manifest(base_dir, kept)
    # Retain physical-file rows through the canonical JSONL loader.
    from src.stage1_store import load_pdf_file_records

    pdf_rows = [
        row
        for row in load_pdf_file_records(base_dir)
        if str(row.get("paper_id") or "") not in requested
    ]
    _write_jsonl(base_dir / "data" / "pdf_files.jsonl", pdf_rows)
    evidence_path = base_dir / "data" / "evidence" / "trusted_evidence.csv"
    evidence = [
        row
        for row in _read_csv(evidence_path)
        if str(row.get("paper_id") or "") not in requested
    ]
    _write_csv(evidence_path, evidence)

    from src.unified_rag import build_unified_rag

    remaining_formal = [
        row["paper_id"]
        for row in canonical_library_records(base_dir)
        if row.get("library_status") == "FORMAL"
    ]
    rag = build_unified_rag(remaining_formal, base_dir=base_dir)
    return {
        "status": "COMPLETED",
        "deleted": sorted(requested),
        "deleted_pdf_files": deleted_files,
        "rag": rag,
    }


def delete_invalid_candidate(record_id: str, *, base_dir: Path = BASE_DIR) -> bool:
    path = base_dir / "data" / "candidate_papers.csv"
    rows = _read_csv(path)
    kept = [
        row
        for row in rows
        if str(row.get("candidate_id") or row.get("paper_id") or "") != record_id
    ]
    if len(kept) == len(rows):
        return False
    _write_csv(path, kept)
    return True


def add_doi_or_url(
    value: str,
    *,
    base_dir: Path = BASE_DIR,
    acquire_oa_pdf: bool = True,
    session: Any = None,
) -> Dict[str, Any]:
    metadata = fetch_metadata(value, session=session)
    if metadata.get("validation_status") != VALID_METADATA:
        return metadata
    scope = classify_literature_scope(metadata)
    metadata.update(scope)
    registration = register_metadata_record(
        metadata,
        source_record_id=metadata["doi"],
        source_type=f"{metadata['metadata_source'].upper()}_METADATA",
        library_status=(
            QUARANTINED
            if scope["domain_scope"] == OUT_OF_SCOPE
            else "CANDIDATE"
        ),
        base_dir=base_dir,
    )
    result = {**metadata, **registration}
    if (
        scope["domain_scope"] != OUT_OF_SCOPE
        and acquire_oa_pdf
        and metadata.get("is_oa") is True
        and metadata.get("pdf_url")
    ):
        from src.oa_literature import download_and_deep_read

        oa_candidate = {
            "title": metadata["title"],
            "authors": metadata.get("authors", ""),
            "date": metadata.get("year", ""),
            "doi": metadata["doi"],
            "landing_page": metadata.get("source_url", ""),
            "pdf_url": metadata["pdf_url"],
            "license": metadata.get("license", ""),
            "is_oa": True,
            "oa_status": metadata.get("oa_status", ""),
        }
        result["oa_ingest"] = download_and_deep_read(
            oa_candidate,
            base_dir=base_dir,
            session=session,
        )
        paper_id = str(result["oa_ingest"].get("paper_id") or "")
        if paper_id:
            from src.auto_oa_pipeline import index_auto_validated_paper

            indexed = index_auto_validated_paper(paper_id, base_dir=base_dir)
            result["oa_index"] = indexed
            if indexed.get("status") == "INDEXED_STAGE3_UNIFIED":
                update_paper_library_status(
                    paper_id, "FORMAL", base_dir=base_dir
                )
            else:
                reasons = (indexed.get("gate") or {}).get("reasons") or []
                update_paper_library_status(
                    paper_id,
                    QUARANTINED,
                    quarantine_reason=";".join(reasons)
                    or "OA_FULLTEXT_PIPELINE_FAILED",
                    base_dir=base_dir,
                )
    return result


def repair_candidate_metadata(
    record_id: str,
    value: str,
    *,
    base_dir: Path = BASE_DIR,
    session: Any = None,
) -> Dict[str, Any]:
    """Backfill one quarantined candidate from OpenAlex/Crossref in place."""
    metadata = fetch_metadata(value, session=session)
    if metadata.get("validation_status") != VALID_METADATA:
        return metadata
    path = base_dir / "data" / "candidate_papers.csv"
    rows = _read_csv(path)
    for row in rows:
        current_id = str(row.get("candidate_id") or row.get("paper_id") or "")
        if current_id != record_id:
            continue
        row.update(
            {
                "title": metadata["title"],
                "authors": metadata.get("authors", ""),
                "year": metadata.get("year", ""),
                "doi": metadata.get("doi", ""),
                "url": metadata.get("source_url", ""),
                "source_url": metadata.get("source_url", ""),
                "source_database": metadata.get("metadata_source", ""),
                "metadata_source": metadata.get("metadata_source", ""),
                "oa_status": metadata.get("oa_status", "UNKNOWN"),
                "pdf_status": metadata.get("pdf_status", "PDF_NOT_ACQUIRED"),
                "validation_status": VALID_METADATA,
                "quarantine_reason": "",
                "last_updated": _now(),
            }
        )
        _write_csv(path, rows)
        return {
            **metadata,
            "status": "UPDATED",
            "candidate_id": record_id,
        }
    return {
        "status": "NOT_FOUND",
        "validation_status": INVALID_METADATA,
        "candidate_id": record_id,
    }


def backfill_invalid_candidates(
    *,
    base_dir: Path = BASE_DIR,
    max_workers: int = 6,
) -> Dict[str, Any]:
    """Attempt real-source backfill once per unique DOI/URL, then write once."""
    invalid = invalid_candidate_records(base_dir)
    lookup_by_id = {
        str(row["candidate_id"]): str(
            row.get("doi") or row.get("source_url") or ""
        ).strip()
        for row in invalid
    }
    unique_inputs = sorted({value for value in lookup_by_id.values() if value})
    fetched: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), 10))) as pool:
        pending = {pool.submit(fetch_metadata, value): value for value in unique_inputs}
        for future in as_completed(pending):
            value = pending[future]
            try:
                fetched[value] = future.result()
            except Exception as exc:  # Network failure remains auditable, never guessed.
                fetched[value] = {
                    "status": "LOOKUP_FAILED",
                    "validation_status": INVALID_METADATA,
                    "error": str(exc),
                }

    path = base_dir / "data" / "candidate_papers.csv"
    rows = _read_csv(path)
    updated: List[str] = []
    failed: Dict[str, str] = {}
    skipped: List[str] = []
    for row in rows:
        record_id = str(row.get("candidate_id") or row.get("paper_id") or "")
        if record_id not in lookup_by_id:
            continue
        lookup = lookup_by_id[record_id]
        if not lookup:
            skipped.append(record_id)
            continue
        metadata = fetched.get(lookup) or {}
        if metadata.get("validation_status") != VALID_METADATA:
            failed[record_id] = str(
                metadata.get("error") or metadata.get("status") or "NOT_FOUND"
            )
            continue
        row.update(
            {
                "title": metadata["title"],
                "authors": metadata.get("authors", ""),
                "year": metadata.get("year", ""),
                "doi": metadata.get("doi", ""),
                "url": metadata.get("source_url", ""),
                "source_url": metadata.get("source_url", ""),
                "source_database": metadata.get("metadata_source", ""),
                "metadata_source": metadata.get("metadata_source", ""),
                "oa_status": metadata.get("oa_status", "UNKNOWN"),
                "pdf_status": metadata.get("pdf_status", "PDF_NOT_ACQUIRED"),
                "validation_status": VALID_METADATA,
                "quarantine_reason": "",
                "last_updated": _now(),
            }
        )
        updated.append(record_id)
    if updated:
        _write_csv(path, rows)
    return {
        "status": "COMPLETED",
        "attempted_unique_inputs": len(unique_inputs),
        "updated": updated,
        "failed": failed,
        "skipped_without_identifier": skipped,
        "sources": sorted(
            {
                str(value.get("metadata_source") or "")
                for value in fetched.values()
                if value.get("metadata_source")
            }
        ),
    }


def ingest_uploaded_pdf(
    content: bytes,
    filename: str,
    *,
    base_dir: Path = BASE_DIR,
    metadata_override: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    from src.pdf_upload_handler import process_uploaded_pdf
    from src.auto_oa_pipeline import index_auto_validated_paper

    result = process_uploaded_pdf(
        content,
        filename,
        base_dir=base_dir,
        metadata_override=metadata_override,
    )
    result["pipeline_executed"] = bool(
        result.get("pdf_valid")
        and result.get("paper_id")
        and result.get("real_page_count")
        and result.get("page_record_path")
    )
    scope = classify_literature_scope(result)
    result.update(scope)
    if result.get("paper_id"):
        update_paper_domain_scope(
            str(result["paper_id"]),
            scope["domain_scope"],
            scope_reason=scope["scope_reason"],
            base_dir=base_dir,
        )
    authors = " ".join(str(result.get("authors") or "").split())
    year = str(result.get("year") or "")
    doi = normalize_doi(result.get("doi") or "")
    metadata_errors = []
    if scope["domain_scope"] == OUT_OF_SCOPE:
        metadata_errors.append("OUT_OF_TITANIUM_FATIGUE_SCOPE")
    if not is_valid_title(result.get("title")):
        metadata_errors.append("TITLE_REQUIRED")
    if not authors or authors.lower() in {"nan", "none", "unknown"}:
        metadata_errors.append("AUTHORS_REQUIRED")
    if not re.search(r"\b(19|20)\d{2}\b", year):
        metadata_errors.append("YEAR_REQUIRED")
    if not doi:
        metadata_errors.append("DOI_REQUIRED_FOR_AUTOMATIC_INDEXING")
    result["metadata_validation_status"] = (
        VALID_METADATA if not metadata_errors else INVALID_METADATA
    )
    result["metadata_validation_errors"] = metadata_errors
    result["evidence_status"] = "NEEDS_HUMAN_REVIEW"
    result["rag_status"] = "NOT_INDEXED"
    if result["pipeline_executed"] and not metadata_errors:
        indexed = index_auto_validated_paper(
            str(result["paper_id"]),
            base_dir=base_dir,
        )
        gate = indexed.get("gate") or {}
        result["quality_gate"] = "PASSED" if gate.get("passed") else "FAILED"
        result["quality_gate_reasons"] = gate.get("reasons") or []
        result["evidence_status"] = gate.get(
            "evidence_status", "NEEDS_HUMAN_REVIEW"
        )
        result["rag_status"] = indexed.get("status", "NOT_INDEXED")
        if result["rag_status"] == "INDEXED_STAGE3_UNIFIED":
            result["library_status"] = "FORMAL"
            update_paper_library_status(
                str(result["paper_id"]), "FORMAL", base_dir=base_dir
            )
        else:
            result["library_status"] = QUARANTINED
            update_paper_library_status(
                str(result["paper_id"]),
                QUARANTINED,
                quarantine_reason=";".join(
                    str(value) for value in result["quality_gate_reasons"]
                ),
                base_dir=base_dir,
            )
    else:
        result["quality_gate"] = "FAILED"
        result["quality_gate_reasons"] = metadata_errors or [
            "UPLOAD_PIPELINE_INCOMPLETE"
        ]
        result["library_status"] = QUARANTINED
        if result.get("paper_id"):
            update_paper_library_status(
                str(result["paper_id"]),
                QUARANTINED,
                quarantine_reason=";".join(
                    str(value) for value in result["quality_gate_reasons"]
                ),
                base_dir=base_dir,
            )
    return result


def rebuild_unified_rag(
    paper_ids: Sequence[str],
    *,
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    gate = eligible_paper_ids(
        paper_ids,
        base_dir=base_dir,
        require_confirmation=True,
    )
    if not gate["eligible"]:
        return {
            "status": "BLOCKED",
            "indexed": [],
            "rejected": gate["rejected"],
        }
    from src.unified_rag import build_unified_rag

    result = build_unified_rag(gate["eligible"], base_dir=base_dir)
    return {
        "status": "COMPLETED",
        "indexed": gate["eligible"],
        "rejected": gate["rejected"],
        "result": result,
    }

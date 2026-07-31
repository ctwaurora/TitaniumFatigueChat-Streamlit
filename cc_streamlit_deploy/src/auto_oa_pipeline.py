"""Synchronous cloud-safe OA discovery, deep reading, gating, and RAG indexing."""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

import requests

from src.metadata_service import is_valid_title
from src.oa_literature import (
    download_and_deep_read,
    oa_download_allowed,
    search_crossref_oa,
    search_openalex_oa,
    search_semantic_scholar_oa,
    search_unpaywall,
)
from src.stage1_store import (
    BASE_DIR,
    is_title_derived_evidence,
    load_paper_manifest,
    load_trusted_evidence_rows,
    normalize_doi,
    normalize_title,
    register_metadata_record,
    update_paper_extraction_status,
    update_paper_library_status,
)
from src.unified_rag import build_unified_rag, rag_paths


ProgressCallback = Callable[[str, Dict[str, Any]], None]
PHASES = (
    "SEARCHING",
    "METADATA_VALIDATED",
    "DOWNLOADING",
    "PDF_VALIDATED",
    "DEDUPLICATING",
    "DEEP_READING",
    "QUALITY_GATING",
    "REINDEXING",
    "COMPLETED",
)
VALID_SOURCES = {"OpenAlex", "Crossref", "Unpaywall", "Semantic Scholar"}
INVALID_TEXT = {"", "nan", "none", "null", "n/a", "na", "unknown", "untitled"}

TOPIC_TERMS = {
    "疲劳与裂纹萌生": ("fatigue", "crack initiation", "fracture origin"),
    "孔隙与缺陷": ("pore", "porosity", "defect", "lack of fusion", "keyhole"),
    "表面状态": ("surface", "roughness", "as-built", "machined", "polished"),
    "热处理与HIP": ("heat treatment", "hip", "hot isostatic", "anneal"),
    "裂纹扩展": ("crack growth", "fcgr", "paris", "da/dn"),
}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _emit(
    callback: Optional[ProgressCallback],
    phase: str,
    **payload: Any,
) -> None:
    if callback:
        callback(phase, {"phase": phase, "recorded_at": _now(), **payload})


def _year(value: Any) -> str:
    match = re.search(r"\b(19|20)\d{2}\b", str(value or ""))
    return match.group(0) if match else ""


def _valid_authors(value: Any) -> bool:
    text = " ".join(str(value or "").split()).strip()
    return len(text) >= 2 and text.lower() not in INVALID_TEXT


def _trusted_url(value: Any) -> bool:
    parsed = urlparse(str(value or "").strip())
    return parsed.scheme == "https" and bool(parsed.netloc)


def validate_oa_metadata(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Validate real source metadata without synthesizing any identity field."""
    normalized = dict(candidate)
    normalized["metadata_source"] = str(
        candidate.get("metadata_source") or candidate.get("source") or ""
    ).strip()
    normalized["source_url"] = str(
        candidate.get("source_url") or candidate.get("landing_page") or ""
    ).strip()
    normalized["doi"] = normalize_doi(candidate.get("doi") or "")
    recognized_year = _year(candidate.get("year") or candidate.get("date"))
    normalized["year"] = recognized_year or "UNKNOWN"
    normalized["oa_status"] = str(candidate.get("oa_status") or "").strip()
    reasons: List[str] = []
    if not is_valid_title(candidate.get("title")):
        reasons.append("TITLE_REQUIRED")
    if not _valid_authors(candidate.get("authors")):
        reasons.append("AUTHORS_REQUIRED")
    if not recognized_year and normalized["year"] != "UNKNOWN":
        reasons.append("YEAR_REQUIRED")
    if not normalized["doi"] and not _trusted_url(normalized["source_url"]):
        reasons.append("DOI_OR_TRUSTED_SOURCE_URL_REQUIRED")
    if normalized["metadata_source"] not in VALID_SOURCES:
        reasons.append("METADATA_SOURCE_REQUIRED")
    if candidate.get("is_oa") is not True or not normalized["oa_status"]:
        reasons.append("OA_STATUS_REQUIRED")
    allowed, oa_reason = oa_download_allowed(candidate)
    if not allowed:
        reasons.append(oa_reason)
    normalized["validation_status"] = (
        "VALID_METADATA" if not reasons else "INVALID_METADATA"
    )
    normalized["validation_errors"] = list(dict.fromkeys(reasons))
    return normalized


def _matches_topic(candidate: Dict[str, Any], topic_filter: str) -> bool:
    terms = TOPIC_TERMS.get(topic_filter)
    if not terms:
        return True
    haystack = normalize_title(
        f"{candidate.get('title', '')} {candidate.get('abstract', '')}"
    )
    return any(normalize_title(term) in haystack for term in terms)


def _is_core_paper(candidate: Dict[str, Any]) -> bool:
    text = normalize_title(
        f"{candidate.get('title', '')} {candidate.get('abstract', '')}"
    )
    material = any(term in text for term in ("ti 6al 4v", "ti6al4v", "titanium alloy"))
    process = any(
        term in text
        for term in (
            "l pbf",
            "lpbf",
            "slm",
            "laser powder bed fusion",
            "selective laser melting",
            "additive manufactur",
        )
    )
    fatigue = any(
        term in text
        for term in ("fatigue", "crack initiation", "crack growth", "fracture origin")
    )
    return material and process and fatigue


def _candidate_rank(candidate: Dict[str, Any], query: str) -> tuple[int, int, int, str]:
    title = normalize_title(candidate.get("title") or "")
    query_terms = set(normalize_title(query).split())
    score = sum(term in title for term in query_terms)
    review_penalty = int(any(term in title for term in ("review", "overview")))
    return (
        review_penalty,
        -score,
        -int(bool(normalize_doi(candidate.get("doi") or ""))),
        title,
    )


def _existing_rag_ids(base_dir: Path) -> List[str]:
    manifest = _read_json(rag_paths(base_dir)["manifest"])
    return [str(value) for value in manifest.get("paper_ids") or [] if value]


def evaluate_auto_rag_gate(
    paper_id: str,
    *,
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    """Require complete Stage-2 provenance before automatic Stage-3 indexing."""
    manifests = {
        str(row.get("paper_id") or ""): row for row in load_paper_manifest(base_dir)
    }
    paper = manifests.get(paper_id) or {}
    status_path = base_dir / "data" / "deep_read" / paper_id / "extraction_status.json"
    pages_path = base_dir / "data" / "deep_read" / paper_id / "page_records.jsonl"
    status = _read_json(status_path)
    pages = _read_jsonl(pages_path)
    evidence = [
        row
        for row in load_trusted_evidence_rows(base_dir)
        if str(row.get("paper_id") or "") == paper_id
    ]
    real_pages = int(status.get("real_page_count") or paper.get("real_page_count") or 0)
    page_by_number = {
        int(row.get("page_number") or 0): str(row.get("cleaned_text") or "")
        for row in pages
    }
    reasons: List[str] = []
    if paper.get("pdf_valid") is not True:
        reasons.append("VALID_PDF_REQUIRED")
    if status.get("sequential_scan_complete") is not True:
        reasons.append("SEQUENTIAL_SCAN_INCOMPLETE")
    if float(status.get("page_coverage_ratio") or 0.0) != 1.0:
        reasons.append("PAGE_COVERAGE_NOT_COMPLETE")
    if not status.get("deep_read_complete"):
        reasons.append("DEEP_READ_INCOMPLETE")
    if real_pages < 1 or len(pages) != real_pages:
        reasons.append("REAL_PAGE_COUNT_MISMATCH")
    if not evidence:
        reasons.append("EVIDENCE_REQUIRED")
    for row in evidence:
        directness = str(row.get("directness") or "").upper()
        try:
            page_number = int(float(row.get("page_number") or 0))
        except (TypeError, ValueError):
            page_number = 0
        original = " ".join(str(row.get("original_text") or "").split())
        section = str(row.get("section") or "").strip()
        page_text = " ".join(page_by_number.get(page_number, "").split())
        if directness == "INVALID":
            reasons.append("INVALID_EVIDENCE")
        if directness == "DIRECT" and is_title_derived_evidence(row):
            reasons.append("TITLE_OR_FILENAME_DERIVED_DIRECT_EVIDENCE")
        if not 1 <= page_number <= real_pages:
            reasons.append("ILLEGAL_EVIDENCE_PAGE")
        if not original or not section:
            reasons.append("INCOMPLETE_EVIDENCE_PROVENANCE")
        if original and page_text and normalize_title(original) not in normalize_title(page_text):
            reasons.append("EVIDENCE_TEXT_NOT_FOUND_ON_PAGE")
    passed = not reasons
    evidence_status = "AUTO_VALIDATED" if passed else "NEEDS_HUMAN_REVIEW"
    update_paper_extraction_status(
        paper_id,
        extraction_status=str(status.get("status") or "FAILED"),
        deep_read_complete=bool(status.get("deep_read_complete")),
        page_record_path=str(pages_path.resolve()) if pages_path.exists() else "",
        page_coverage_ratio=float(status.get("page_coverage_ratio") or 0.0),
        evidence_status=evidence_status,
        base_dir=base_dir,
    )
    return {
        "passed": passed,
        "evidence_status": evidence_status,
        "index_status": "INDEXED_STAGE3_UNIFIED" if passed else "NOT_INDEXED",
        "reasons": list(dict.fromkeys(reasons)),
        "real_page_count": real_pages,
        "page_record_count": len(pages),
        "evidence_count": len(evidence),
        "sequential_scan_complete": bool(status.get("sequential_scan_complete")),
        "page_coverage_ratio": float(status.get("page_coverage_ratio") or 0.0),
    }


def index_auto_validated_paper(
    paper_id: str,
    *,
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    gate = evaluate_auto_rag_gate(paper_id, base_dir=base_dir)
    if not gate["passed"]:
        return {"status": "NOT_INDEXED", "gate": gate}
    paper_ids = list(dict.fromkeys([*_existing_rag_ids(base_dir), paper_id]))
    result = build_unified_rag(paper_ids, base_dir=base_dir)
    actually_indexed = paper_id in set(_existing_rag_ids(base_dir))
    gate["index_status"] = (
        "INDEXED_STAGE3_UNIFIED" if actually_indexed else "NOT_INDEXED"
    )
    return {
        "status": gate["index_status"],
        "gate": gate,
        "rag": result,
    }


def _invalid_log(base_dir: Path, candidate: Dict[str, Any]) -> None:
    _append_jsonl(
        base_dir / "data" / "oa" / "invalid_metadata.jsonl",
        {
            **candidate,
            "validation_status": "INVALID_METADATA",
            "recorded_at": _now(),
        },
    )


def discover_oa_candidates(
    query: str,
    *,
    max_results_per_source: int = 8,
    session: Optional[requests.Session] = None,
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    """Search independent legal metadata sources; one failure never stops others."""
    rows: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    errors: List[str] = []
    operations = (
        ("OpenAlex", search_openalex_oa),
        ("Crossref", search_crossref_oa),
        ("Semantic Scholar", search_semantic_scholar_oa),
    )
    for source, operation in operations:
        try:
            found = operation(
                query,
                max_results=max_results_per_source,
                session=session,
            )
            rows.extend(found)
            sources.append({"source": source, "status": "OK", "count": len(found)})
        except Exception as exc:
            error = f"{source}:{type(exc).__name__}:{exc}"
            errors.append(error)
            sources.append({"source": source, "status": "ERROR", "count": 0, "error": str(exc)})

    email = os.environ.get("UNPAYWALL_EMAIL", "")
    if email and "@" in email:
        doi_values = list(
            dict.fromkeys(
                normalize_doi(row.get("doi") or "")
                for row in rows
                if normalize_doi(row.get("doi") or "")
            )
        )[:max_results_per_source]
        count = 0
        for doi in doi_values:
            try:
                resolved = search_unpaywall(doi, email=email, session=session)
                if resolved:
                    rows.append(resolved)
                    count += 1
            except Exception as exc:
                errors.append(f"Unpaywall:{doi}:{type(exc).__name__}:{exc}")
        sources.append({"source": "Unpaywall", "status": "OK", "count": count})
    else:
        sources.append(
            {
                "source": "Unpaywall",
                "status": "SKIPPED",
                "count": 0,
                "message": "UNPAYWALL_EMAIL_NOT_CONFIGURED",
            }
        )

    unique: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        checked = validate_oa_metadata(row)
        if checked["validation_status"] != "VALID_METADATA":
            _invalid_log(base_dir, checked)
            continue
        key = checked["doi"] or normalize_title(checked.get("title") or "")
        unique.setdefault(key, checked)
    return {
        "candidates": list(unique.values()),
        "source_results": sources,
        "errors": errors,
    }


def run_auto_oa_discovery(
    query: str,
    *,
    max_new: int = 1,
    topic_filter: str = "疲劳与裂纹萌生",
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    core_only: bool = True,
    base_dir: Path = BASE_DIR,
    session: Optional[requests.Session] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Run the complete bounded workflow in the active web request."""
    query = " ".join(str(query or "").split())
    if not query:
        return {"status": "INVALID_INPUT", "results": [], "error": "SEARCH_TOPIC_REQUIRED"}
    max_new = max(1, min(int(max_new), 3))
    _emit(progress_callback, "SEARCHING", query=query)
    discovery = discover_oa_candidates(
        query,
        max_results_per_source=max(6, max_new * 4),
        session=session,
        base_dir=base_dir,
    )
    filtered: List[Dict[str, Any]] = []
    for candidate in discovery["candidates"]:
        year = _year(candidate.get("year"))
        if year_from and year and int(year) < int(year_from):
            continue
        if year_to and year and int(year) > int(year_to):
            continue
        if not _matches_topic(candidate, topic_filter):
            continue
        if core_only and not _is_core_paper(candidate):
            continue
        filtered.append(candidate)
    filtered.sort(key=lambda row: _candidate_rank(row, query))
    _emit(
        progress_callback,
        "METADATA_VALIDATED",
        valid_candidates=len(filtered),
        discovered_candidates=len(discovery["candidates"]),
    )

    results: List[Dict[str, Any]] = []
    completed = 0
    download_attempt_count = 0
    download_success_count = 0
    for candidate in filtered[:max_new]:
        registration = register_metadata_record(
            {
                **candidate,
                "source_path": (
                    candidate.get("source_url")
                    or candidate.get("landing_page")
                    or ""
                ),
            },
            source_record_id=(
                normalize_doi(candidate.get("doi") or "")
                or normalize_title(candidate.get("title") or "")
            ),
            source_type=f"{str(candidate.get('metadata_source') or candidate.get('source') or 'OA').upper()}_METADATA",
            library_status="CANDIDATE",
            base_dir=base_dir,
        )
        detail: Dict[str, Any] = {
            "paper_id": str(registration.get("paper_id") or ""),
            "title": candidate.get("title") or "",
            "authors": candidate.get("authors") or "",
            "year": candidate.get("year") or "UNKNOWN",
            "doi": candidate.get("doi") or "",
            "metadata_source": candidate.get("metadata_source") or candidate.get("source") or "",
            "oa_source": candidate.get("oa_source") or candidate.get("metadata_source") or candidate.get("source") or "",
            "oa_status": candidate.get("oa_status") or "",
            "source_url": candidate.get("source_url") or candidate.get("landing_page") or "",
            "pdf_url": candidate.get("pdf_url") or "",
            "download_status": "NOT_STARTED",
            "downloaded_pdf": False,
            "file_size": 0,
            "file_hash_sha256": "",
            "saved_location": "",
            "real_page_count": 0,
            "processed_page_count": 0,
            "deep_read_status": "NOT_STARTED",
            "evidence_count": 0,
            "direct_evidence_count": 0,
            "indirect_evidence_count": 0,
            "mention_only_count": 0,
            "quality_gate": "NOT_RUN",
            "evidence_status": "NOT_EXTRACTED",
            "library_status": "CANDIDATE",
            "rag_status": "NOT_INDEXED",
            "failure_reason": "",
            "phases": ["SEARCHING", "METADATA_VALIDATED"],
        }
        _emit(progress_callback, "DOWNLOADING", title=detail["title"])
        detail["phases"].append("DOWNLOADING")
        download_attempt_count += 1
        ingested = download_and_deep_read(
            candidate,
            base_dir=base_dir,
            session=session,
        )
        detail["download_status"] = str(ingested.get("status") or "FAILED")
        detail["downloaded_pdf"] = bool(
            ingested.get("downloaded_pdf") and ingested.get("pdf_valid")
        )
        if detail["downloaded_pdf"]:
            download_success_count += 1
        detail["file_size"] = int(ingested.get("file_size") or 0)
        detail["file_hash_sha256"] = str(
            ingested.get("file_hash_sha256") or ""
        )
        detail["saved_location"] = (
            "CANONICAL_PDF_STORAGE" if detail["downloaded_pdf"] else ""
        )
        detail["failure_reason"] = str(
            ingested.get("failure_reason") or ingested.get("reason") or ""
        )
        if not ingested.get("paper_id"):
            if (
                ingested.get("status") == "FAILED"
                and detail.get("paper_id")
            ):
                detail["library_status"] = "QUARANTINED"
                update_paper_library_status(
                    str(detail["paper_id"]),
                    "QUARANTINED",
                    quarantine_reason=detail["failure_reason"]
                    or "INVALID_DOWNLOADED_PDF",
                    base_dir=base_dir,
                )
            results.append(detail)
            continue
        paper_id = str(ingested["paper_id"])
        detail["paper_id"] = paper_id
        detail["real_page_count"] = int(ingested.get("real_page_count") or 0)
        detail["processed_page_count"] = int(
            ingested.get("processed_page_count")
            or ingested.get("page_record_count")
            or 0
        )
        detail["deep_read_status"] = str(ingested.get("status") or "FAILED")
        detail["evidence_count"] = int(ingested.get("evidence_count") or 0)
        detail["direct_evidence_count"] = int(
            ingested.get("direct_evidence_count") or 0
        )
        detail["indirect_evidence_count"] = int(
            ingested.get("indirect_evidence_count") or 0
        )
        detail["mention_only_count"] = int(
            ingested.get("mention_only_count") or 0
        )
        detail["http_status"] = int((ingested.get("http") or {}).get("status_code") or 0)
        detail["content_type"] = str((ingested.get("http") or {}).get("content_type") or "")
        for phase in ("PDF_VALIDATED", "DEDUPLICATING", "DEEP_READING"):
            detail["phases"].append(phase)
            _emit(progress_callback, phase, paper_id=paper_id, title=detail["title"])
        if ingested.get("status") == "DEEP_READ_COMPLETE":
            update_paper_library_status(
                paper_id, "DEEP_READ_COMPLETE", base_dir=base_dir
            )
        detail["phases"].append("QUALITY_GATING")
        _emit(progress_callback, "QUALITY_GATING", paper_id=paper_id)
        gate = evaluate_auto_rag_gate(paper_id, base_dir=base_dir)
        detail["quality_gate"] = "PASSED" if gate["passed"] else "FAILED"
        detail["evidence_status"] = gate["evidence_status"]
        detail["evidence_count"] = gate["evidence_count"]
        detail["gate_reasons"] = gate["reasons"]
        if gate["passed"]:
            update_paper_library_status(
                paper_id, "AUTO_VALIDATED", base_dir=base_dir
            )
            detail["phases"].append("REINDEXING")
            _emit(progress_callback, "REINDEXING", paper_id=paper_id)
            indexed = index_auto_validated_paper(paper_id, base_dir=base_dir)
            detail["rag_status"] = indexed["status"]
            if indexed["status"] == "INDEXED_STAGE3_UNIFIED":
                completed += 1
                detail["library_status"] = "FORMAL"
                update_paper_library_status(
                    paper_id, "FORMAL", base_dir=base_dir
                )
                detail["phases"].append("COMPLETED")
                _emit(progress_callback, "COMPLETED", paper_id=paper_id)
            else:
                detail["failure_reason"] = "STAGE3_INDEX_BUILD_REJECTED"
        else:
            detail["failure_reason"] = ";".join(gate["reasons"])
            detail["library_status"] = "QUARANTINED"
            update_paper_library_status(
                paper_id,
                "QUARANTINED",
                quarantine_reason=detail["failure_reason"],
                base_dir=base_dir,
            )
        results.append(detail)

    status = "COMPLETED" if completed else "NO_PAPER_INDEXED"
    return {
        "status": status,
        "query": query,
        "max_new": max_new,
        "completed_count": completed,
        "search_candidate_count": len(discovery["candidates"]),
        "oa_candidate_count": len(filtered),
        "download_attempt_count": download_attempt_count,
        "download_success_count": download_success_count,
        "candidate_count": len(filtered),
        "results": results,
        "source_results": discovery["source_results"],
        "errors": discovery["errors"],
        "synchronous": True,
        "background_continues_after_page_close": False,
    }

"""Reproducible metadata quality gate for the formal literature corpus.

The gate deliberately separates locally extracted hints from publisher metadata.
It never treats a PDF creation timestamp, operating-system user name, or filename as
scholarly metadata.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote

import fitz
import requests

from src.metadata_service import USER_AGENT, is_valid_title
from src.stage1_store import normalize_doi, normalize_title


BAD_AUTHOR_VALUES = {
    "admin", "administrator", "author", "default", "hp", "lenovo", "mac",
    "microsoft office user", "user", "windows user", "yj", "unknown",
}
CROSSREF_TYPES = {
    "journal-article": "JOURNAL_ARTICLE",
    "proceedings-article": "CONFERENCE_PAPER",
    "book-chapter": "BOOK_CHAPTER",
    "book": "BOOK",
    "edited-book": "EDITED_BOOK",
    "dissertation": "THESIS_OR_DISSERTATION",
    "report": "REPORT",
    "posted-content": "PREPRINT",
    "dataset": "DATASET",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def authors_valid(value: Any) -> bool:
    authors = " ".join(str(value or "").split()).strip()
    if not authors or authors.casefold() in BAD_AUTHOR_VALUES:
        return False
    if any(token in authors.casefold() for token in ("windows user", "unknown author")):
        return False
    names = [part.strip() for part in re.split(r"\s*;\s*|\s+and\s+", authors) if part.strip()]
    if not names:
        return False
    for name in names:
        letters = re.findall(r"[^\W\d_]", name, flags=re.UNICODE)
        if len(letters) < 3:
            return False
        # A lone pair of initials is not a reliably identified author.
        if re.fullmatch(r"(?:[A-Za-z]\.?\s*){1,3}", name):
            return False
    return True


def valid_year(value: Any) -> str:
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", str(value or ""))
    if not match:
        return ""
    year = int(match.group(1))
    return str(year) if 1900 <= year <= datetime.now().year + 1 else ""


def doi_syntax_valid(value: Any) -> bool:
    doi = normalize_doi(str(value or ""))
    match = re.fullmatch(r"10\.\d{4,9}/(\S{3,})", doi, re.I)
    return bool(match and re.search(r"[A-Za-z0-9]{2}", match.group(1)))


def _title_score(expected: str, observed: str) -> float:
    left = normalize_title(expected)
    right = normalize_title(observed)
    if not left or not right:
        return 0.0
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    recall = len(left_tokens & right_tokens) / max(1, len(left_tokens))
    sequence = SequenceMatcher(None, left, right).ratio()
    return round(max(sequence, recall * 0.98), 4)


def pdf_front_matter(path: Path, pages: int = 2) -> dict[str, Any]:
    result = {"open_valid": False, "page_count": 0, "text": "", "dois": [], "error": ""}
    try:
        if path.read_bytes()[:5] != b"%PDF-":
            result["error"] = "INVALID_PDF_HEADER"
            return result
        with fitz.open(path) as document:
            result["open_valid"] = len(document) > 0
            result["page_count"] = len(document)
            result["text"] = "\n".join(
                document[index].get_text("text") for index in range(min(pages, len(document)))
            )
    except Exception as exc:  # malformed assets are held, never silently passed
        result["error"] = f"PDF_OPEN_FAILED:{type(exc).__name__}:{exc}"
        return result
    found = re.findall(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", result["text"], re.I)
    result["dois"] = list(dict.fromkeys(filter(None, (normalize_doi(value) for value in found))))
    return result


def _crossref_message_to_metadata(message: dict[str, Any]) -> dict[str, Any]:
    titles = message.get("title") or []
    authors_list = []
    for row in message.get("author") or []:
        name = " ".join(str(row.get(key) or "").strip() for key in ("given", "family")).strip()
        if name:
            authors_list.append(name)
    date_parts = (
        (message.get("published-print") or {}).get("date-parts")
        or (message.get("published-online") or {}).get("date-parts")
        or (message.get("issued") or {}).get("date-parts")
        or []
    )
    year = str(date_parts[0][0]) if date_parts and date_parts[0] else ""
    work_type = str(message.get("type") or "")
    title = " ".join(str(titles[0] if titles else "").split())
    if re.search(r"\b(review|systematic review|meta-analysis)\b", title, re.I):
        document_type = "REVIEW_ARTICLE"
    else:
        document_type = CROSSREF_TYPES.get(work_type, work_type.upper().replace("-", "_"))
    containers = message.get("container-title") or []
    return {
        "title": title,
        "authors": "; ".join(authors_list),
        "authors_list": authors_list,
        "year": valid_year(year),
        "doi": normalize_doi(message.get("DOI") or ""),
        "document_type": document_type,
        "publication_title": " ".join(str(containers[0] if containers else "").split()),
        "publisher": str(message.get("publisher") or "").strip(),
        "source_url": str(message.get("URL") or "").strip(),
        "crossref_type": work_type,
    }


def fetch_crossref(*, doi: str = "", title: str = "", timeout: int = 20) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    # The desktop runtime may expose a stale proxy endpoint.  Crossref is a
    # public metadata API, so use a dedicated direct session for deterministic
    # lookup rather than inheriting an unusable process proxy.
    session = requests.Session()
    session.trust_env = False
    try:
        if doi:
            url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
            response = session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            message = (response.json() or {}).get("message") or {}
            return {"status": "FOUND", "metadata": _crossref_message_to_metadata(message)}
        response = session.get(
            "https://api.crossref.org/works",
            params={"query.title": title, "rows": 3, "select": "DOI,title,author,published-print,published-online,issued,type,container-title,publisher,URL"},
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        candidates = []
        for message in ((response.json() or {}).get("message") or {}).get("items") or []:
            metadata = _crossref_message_to_metadata(message)
            metadata["query_title_score"] = _title_score(title, metadata.get("title") or "")
            candidates.append(metadata)
        candidates.sort(key=lambda row: float(row.get("query_title_score") or 0), reverse=True)
        if candidates and candidates[0]["query_title_score"] >= 0.90:
            return {"status": "FOUND", "metadata": candidates[0]}
        return {"status": "NOT_FOUND", "metadata": {}, "candidates": candidates}
    except (requests.RequestException, ValueError, KeyError) as exc:
        return {"status": "ERROR", "metadata": {}, "error": f"{type(exc).__name__}:{exc}"}


def _infer_local_document_type(title: str, text: str) -> str:
    sample = f"{title}\n{text[:12000]}".casefold()
    if re.search(r"\b(phd|doctoral|master'?s?)\s+(thesis|dissertation)\b", sample):
        return "THESIS_OR_DISSERTATION"
    if re.search(r"\b(nasa|technical|contractor)\s+report\b", sample):
        return "REPORT"
    if re.search(r"\b(review|systematic review|meta-analysis)\b", title, re.I):
        return "REVIEW_ARTICLE"
    if re.search(r"\b(proceedings|conference)\b", sample[:3000]):
        return "CONFERENCE_PAPER"
    if re.search(r"\b(arxiv|preprint)\b", sample[:3000]):
        return "PREPRINT"
    return ""


@dataclass
class GateResult:
    paper_id: str
    passed: bool
    updated_record: dict[str, Any]
    audit: dict[str, Any]


def audit_record(record: dict[str, Any], crossref_result: dict[str, Any]) -> GateResult:
    paper_id = str(record.get("paper_id") or "")
    path = Path(str(record.get("canonical_pdf_path") or record.get("source_path") or ""))
    front = pdf_front_matter(path) if path.is_file() else {"open_valid": False, "page_count": 0, "text": "", "dois": [], "error": "PDF_MISSING"}
    supplied_doi = normalize_doi(record.get("doi") or "")
    source = dict(crossref_result.get("metadata") or {})
    source_found = crossref_result.get("status") == "FOUND"
    source_title = str(source.get("title") or "")
    title_score = _title_score(source_title or str(record.get("title") or ""), str(front.get("text") or "")[:12000])
    supplied_doi_on_pdf = bool(supplied_doi and supplied_doi in set(front.get("dois") or []))
    source_doi = normalize_doi(source.get("doi") or "")
    doi_conflict = bool(supplied_doi and source_doi and supplied_doi != source_doi)
    title_confirmed = title_score >= 0.72
    source_matches_pdf = bool(source_found and not doi_conflict and (title_confirmed or supplied_doi_on_pdf))

    updated = dict(record)
    repairs: list[str] = []
    reasons: list[str] = []
    if source_matches_pdf:
        for key in ("title", "authors", "authors_list", "year", "document_type", "publication_title", "publisher", "source_url"):
            value = source.get(key)
            if value and updated.get(key) != value:
                updated[key] = value
                repairs.append(key)
        updated["publication_date"] = source.get("year") or updated.get("publication_date")
        updated["doi"] = source_doi
        updated["doi_status"] = "VERIFIED_PRESENT"
        updated["metadata_source"] = "CROSSREF_AND_PDF_FRONT_MATTER"
        updated["metadata_sources"] = ["Crossref", "PDF_FRONT_MATTER"]
        updated["metadata_confidence"] = "VERIFIED_API_PDF_MATCH"
        updated["metadata_verified"] = True
    else:
        local_type = str(updated.get("document_type") or "") or _infer_local_document_type(str(updated.get("title") or ""), str(front.get("text") or ""))
        updated["document_type"] = local_type
        if not supplied_doi and crossref_result.get("status") == "NOT_FOUND" and title_confirmed:
            updated["doi"] = ""
            updated["doi_status"] = "NOT_AVAILABLE_VERIFIED"
        if source_found and not source_matches_pdf:
            reasons.append("REGISTERED_METADATA_DOES_NOT_MATCH_PDF_FRONT_MATTER")
        elif crossref_result.get("status") == "ERROR":
            reasons.append("METADATA_SOURCE_UNAVAILABLE")
        else:
            reasons.append("NO_RELIABLE_REGISTERED_METADATA_MATCH")

    title_ok = is_valid_title(updated.get("title")) and (source_matches_pdf or title_confirmed)
    author_ok = authors_valid(updated.get("authors"))
    year_ok = bool(valid_year(updated.get("year") or updated.get("publication_date")))
    doi_ok = updated.get("doi_status") in {"VERIFIED_PRESENT", "NOT_AVAILABLE_VERIFIED"}
    type_ok = bool(updated.get("document_type"))
    pdf_ok = bool(front.get("open_valid")) and int(front.get("page_count") or 0) == int(updated.get("real_page_count") or 0)
    checks = {"pdf_valid": pdf_ok, "title_valid_and_confirmed": title_ok, "authors_valid": author_ok, "year_valid": year_ok, "doi_status_valid": doi_ok, "document_type_valid": type_ok}
    for name, passed in checks.items():
        if not passed:
            reasons.append(f"FAILED_{name.upper()}")
    passed = all(checks.values())
    if passed:
        updated["library_status"] = "FORMAL"
        updated["evidence_status"] = updated.get("evidence_status") or "AUTO_VALIDATED"
        updated["quarantine_reason"] = ""
    else:
        updated["library_status"] = "QUARANTINED"
        updated["evidence_status"] = "NEEDS_HUMAN_REVIEW"
        updated["rag_status"] = "NOT_INDEXED_METADATA_HOLD"
        updated["metadata_verified"] = False
        updated["quarantine_reason"] = ";".join(dict.fromkeys(reasons))
    updated["real_page_count"] = int(front.get("page_count") or updated.get("real_page_count") or 0)
    updated["metadata_gate_status"] = "PASSED" if passed else "FAILED"
    updated["metadata_gate_checked_at"] = utc_now()
    updated["updated_at"] = utc_now()
    audit = {
        "canonical_paper_id": paper_id,
        "before": {key: record.get(key) for key in ("title", "authors", "publication_date", "doi", "document_type", "metadata_source", "metadata_confidence", "library_status", "rag_status", "evidence_status")},
        "after": {key: updated.get(key) for key in ("title", "authors", "publication_date", "doi", "doi_status", "document_type", "publication_title", "metadata_source", "metadata_confidence", "library_status", "rag_status", "evidence_status")},
        "checks": checks,
        "passed": passed,
        "repairs": repairs,
        "reasons": list(dict.fromkeys(reasons)),
        "pdf_path": str(path),
        "pdf_page_count": int(front.get("page_count") or 0),
        "pdf_dois": front.get("dois") or [],
        "title_pdf_score": title_score,
        "metadata_lookup_status": crossref_result.get("status"),
    }
    return GateResult(paper_id, passed, updated, audit)


def run_metadata_gate(base_dir: Path, paper_ids: list[str], workers: int = 6) -> dict[str, Any]:
    manifest_path = base_dir / "data" / "paper_manifest.jsonl"
    rows = read_jsonl(manifest_path)
    by_id = {str(row.get("paper_id") or ""): row for row in rows}
    targets = [by_id[paper_id] for paper_id in paper_ids if paper_id in by_id]
    lookups: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {}
        for row in targets:
            doi = normalize_doi(row.get("doi") or "")
            key = str(row.get("paper_id") or "")
            futures[pool.submit(fetch_crossref, doi=doi if doi_syntax_valid(doi) else "", title="" if doi_syntax_valid(doi) else str(row.get("title") or ""))] = key
        for future in as_completed(futures):
            key = futures[future]
            try:
                lookups[key] = future.result()
            except Exception as exc:
                lookups[key] = {"status": "ERROR", "metadata": {}, "error": f"{type(exc).__name__}:{exc}"}
    results = [audit_record(row, lookups.get(str(row.get("paper_id") or ""), {"status": "ERROR", "metadata": {}})) for row in targets]
    updates = {result.paper_id: result.updated_record for result in results}
    atomic_jsonl(manifest_path, [updates.get(str(row.get("paper_id") or ""), row) for row in rows])
    audits = [result.audit for result in results]
    return {
        "generated_at": utc_now(),
        "target_count": len(targets),
        "passed_count": sum(result.passed for result in results),
        "held_count": sum(not result.passed for result in results),
        "metadata_repaired_count": sum(bool(result.audit["repairs"]) for result in results),
        "records": audits,
    }

"""Legal open-access literature discovery and Stage-2/Stage-3 integration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

import requests

from src.deep_read_pipeline import deep_read_pdf
from src.stage1_store import (
    BASE_DIR,
    load_paper_manifest,
    normalize_doi,
    normalize_title,
    register_pdf_bytes,
    semantic_duplicate_candidates,
    sha256_bytes,
    title_similarity,
    validate_pdf_bytes,
)
from src.unified_rag import (
    answer_research_question,
    build_unified_rag,
    identify_variables,
    rag_paths,
)


OPENALEX_URL = "https://api.openalex.org/works"
USER_AGENT = "TitaniumFatigueChat/3.0 (legal-open-access-research-client)"
OA_SCHEMA_VERSION = "stage3.0"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def oa_paths(base_dir: Path = BASE_DIR) -> Dict[str, Path]:
    root = base_dir / "data" / "oa"
    return {
        "root": root,
        "search_log": root / "search_results.jsonl",
        "download_log": root / "downloads.jsonl",
        "manual": root / "manual_download_required.jsonl",
        "temp": base_dir / "tmp" / "pdfs" / "oa",
        "pdf_dir": base_dir / "paper" / "pdfs",
    }


def generate_search_queries(question: str) -> List[str]:
    variables = identify_variables(question)
    base = "Ti-6Al-4V additive manufacturing fatigue"
    queries = [f"{base} {question}"]
    mappings = {
        "pore_size": "pore defect size fatigue life",
        "surface_distance": "defect distance surface crack initiation",
        "surface_state": "as-built machined surface fatigue mechanism",
        "crack_initiation": "crack initiation pore surface internal defect",
        "crack_growth": "fatigue crack growth Delta K da/dN",
        "paris_c": "Paris law C coefficient comparison",
        "paris_m": "Paris law m exponent comparison",
        "stress_ratio_R": "stress ratio R fatigue crack growth",
    }
    for variable in variables:
        if variable in mappings:
            queries.append(f"{base} {mappings[variable]}")
    return list(dict.fromkeys(queries))[:4]


def _openalex_authorships(work: Dict[str, Any]) -> str:
    names = []
    for item in work.get("authorships") or []:
        name = ((item.get("author") or {}).get("display_name") or "").strip()
        if name:
            names.append(name)
    return "; ".join(names)


def _location_candidate(work: Dict[str, Any], location: Dict[str, Any]) -> Dict[str, Any]:
    oa = work.get("open_access") or {}
    source = location.get("source") or {}
    return {
        "source": "OpenAlex",
        "openalex_id": work.get("id") or "",
        "title": work.get("display_name") or work.get("title") or "",
        "authors": _openalex_authorships(work),
        "date": work.get("publication_date") or "",
        "doi": normalize_doi(work.get("doi") or ""),
        "landing_page": location.get("landing_page_url") or work.get("doi") or "",
        "pdf_url": location.get("pdf_url") or "",
        "license": location.get("license") or "",
        "version": location.get("version") or "",
        "host_type": source.get("host_organization_name") or source.get("type") or "",
        "is_oa": bool(oa.get("is_oa")),
        "oa_status": oa.get("oa_status") or "",
        "retrieved_at": _now(),
        "data_version": OA_SCHEMA_VERSION,
    }


def search_openalex_oa(
    query: str,
    *,
    max_results: int = 10,
    timeout: int = 30,
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    client = session or requests.Session()
    response = client.get(
        OPENALEX_URL,
        params={
            "search": query,
            "filter": "is_oa:true,has_fulltext:true",
            "per-page": min(max(1, max_results), 50),
        },
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    candidates = []
    for work in response.json().get("results") or []:
        locations = [
            work.get("best_oa_location") or {},
            work.get("primary_location") or {},
            *(work.get("locations") or []),
        ]
        seen_urls = set()
        for location in locations:
            candidate = _location_candidate(work, location)
            url = candidate["pdf_url"]
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            candidates.append(candidate)
            break
    return candidates[:max_results]


def search_unpaywall(
    doi: str,
    *,
    email: str = "",
    timeout: int = 30,
    session: Optional[requests.Session] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve a DOI through Unpaywall when a real contact email is configured."""
    email = email or os.environ.get("UNPAYWALL_EMAIL", "")
    if not email or "@" not in email:
        return None
    client = session or requests.Session()
    response = client.get(
        f"https://api.unpaywall.org/v2/{normalize_doi(doi)}",
        params={"email": email},
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    work = response.json()
    best = work.get("best_oa_location") or {}
    return {
        "source": "Unpaywall",
        "title": work.get("title") or "",
        "authors": "; ".join(
            author.get("family", "") for author in work.get("z_authors") or []
            if author.get("family")
        ),
        "date": str(work.get("year") or ""),
        "doi": normalize_doi(work.get("doi") or doi),
        "landing_page": best.get("url") or "",
        "pdf_url": best.get("url_for_pdf") or "",
        "license": best.get("license") or "",
        "version": best.get("version") or "",
        "host_type": best.get("host_type") or "",
        "is_oa": bool(work.get("is_oa")),
        "oa_status": work.get("oa_status") or "",
        "retrieved_at": _now(),
        "data_version": OA_SCHEMA_VERSION,
    }


def oa_download_allowed(candidate: Dict[str, Any]) -> tuple[bool, str]:
    if candidate.get("is_oa") is not True:
        return False, "NOT_OPEN_ACCESS"
    pdf_url = str(candidate.get("pdf_url") or "").strip()
    if not pdf_url:
        return False, "NO_PUBLIC_PDF_URL"
    parsed = urlparse(pdf_url)
    if parsed.scheme != "https" or not parsed.netloc:
        return False, "UNSAFE_OR_NON_HTTPS_PDF_URL"
    license_value = str(candidate.get("license") or "").lower()
    oa_status = str(candidate.get("oa_status") or "").lower()
    # OpenAlex can label repository manuscripts "green" while the repository
    # omits a machine-readable license.  A public repository PDF URL plus the
    # OpenAlex OA flag is still a legal public source; provenance is retained.
    if not license_value and oa_status not in {"green", "gold", "diamond", "hybrid", "bronze"}:
        return False, "OA_RIGHTS_NOT_ESTABLISHED"
    return True, ""


def record_manual_download_required(
    candidate: Dict[str, Any],
    failure_reason: str,
    *,
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    row = {
        "title": candidate.get("title") or "",
        "authors": candidate.get("authors") or "",
        "date": candidate.get("date") or "",
        "doi": normalize_doi(candidate.get("doi") or ""),
        "landing_page": candidate.get("landing_page") or "",
        "failure_reason": failure_reason,
        "manual_download_required": True,
        "recorded_at": _now(),
    }
    _append_jsonl(oa_paths(base_dir)["manual"], row)
    return row


def candidate_duplicate_reason(
    candidate: Dict[str, Any],
    *,
    content_hash: str = "",
    base_dir: Path = BASE_DIR,
) -> str:
    doi = normalize_doi(candidate.get("doi") or "")
    title = str(candidate.get("title") or "")
    for paper in load_paper_manifest(base_dir):
        if doi and normalize_doi(paper.get("doi") or "") == doi:
            return f"DOI_DUPLICATE:{paper.get('paper_id')}"
        if content_hash and paper.get("file_hash_sha256") == content_hash:
            return f"HASH_DUPLICATE:{paper.get('paper_id')}"
        similarity = title_similarity(title, paper.get("title") or "")
        if title and similarity >= 0.94:
            return f"TITLE_NEAR_DUPLICATE:{paper.get('paper_id')}:{similarity:.3f}"
    return ""


def _download_bytes(
    url: str,
    *,
    timeout: int = 60,
    max_bytes: int = 100 * 1024 * 1024,
    session: Optional[requests.Session] = None,
) -> bytes:
    client = session or requests.Session()
    response = client.get(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/pdf,application/octet-stream;q=0.9",
        },
        timeout=timeout,
        stream=True,
        allow_redirects=True,
    )
    response.raise_for_status()
    if urlparse(response.url).scheme != "https":
        raise ValueError("DOWNLOAD_REDIRECTED_TO_NON_HTTPS")
    content = bytearray()
    for block in response.iter_content(chunk_size=1024 * 128):
        if not block:
            continue
        content.extend(block)
        if len(content) > max_bytes:
            raise ValueError("PDF_EXCEEDS_SIZE_LIMIT")
    return bytes(content)


def download_and_deep_read(
    candidate: Dict[str, Any],
    *,
    base_dir: Path = BASE_DIR,
    force_deep_read: bool = False,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    allowed, reason = oa_download_allowed(candidate)
    if not allowed:
        record_manual_download_required(candidate, reason, base_dir=base_dir)
        return {
            "status": "MANUAL_ACTION_REQUIRED",
            "manual_download_required": True,
            "failure_reason": reason,
            "candidate": candidate,
        }
    pre_duplicate = candidate_duplicate_reason(candidate, base_dir=base_dir)
    if pre_duplicate:
        return {"status": "DUPLICATE_SKIPPED", "reason": pre_duplicate, "candidate": candidate}
    try:
        content = _download_bytes(candidate["pdf_url"], session=session)
    except Exception as exc:
        record_manual_download_required(
            candidate, f"PUBLIC_PDF_DOWNLOAD_FAILED:{exc}", base_dir=base_dir
        )
        return {
            "status": "PARTIAL",
            "manual_download_required": True,
            "failure_reason": str(exc),
            "candidate": candidate,
        }
    return ingest_verified_oa_bytes(
        candidate,
        content,
        base_dir=base_dir,
        force_deep_read=force_deep_read,
    )


def ingest_verified_oa_file(
    candidate: Dict[str, Any],
    file_path: Path | str,
    *,
    base_dir: Path = BASE_DIR,
    force_deep_read: bool = False,
) -> Dict[str, Any]:
    """Ingest a browser/BITS-downloaded public file through the same OA gate."""
    allowed, reason = oa_download_allowed(candidate)
    if not allowed:
        record_manual_download_required(candidate, reason, base_dir=base_dir)
        return {
            "status": "MANUAL_ACTION_REQUIRED",
            "manual_download_required": True,
            "failure_reason": reason,
            "candidate": candidate,
        }
    path = Path(file_path).resolve()
    if not path.exists():
        return {"status": "FAILED", "failure_reason": "DOWNLOADED_FILE_NOT_FOUND"}
    return ingest_verified_oa_bytes(
        candidate,
        path.read_bytes(),
        base_dir=base_dir,
        force_deep_read=force_deep_read,
    )


def ingest_verified_oa_bytes(
    candidate: Dict[str, Any],
    content: bytes,
    *,
    base_dir: Path = BASE_DIR,
    force_deep_read: bool = False,
) -> Dict[str, Any]:
    """Validate, deduplicate, register and deep-read already downloaded bytes."""
    content_hash = sha256_bytes(content)
    duplicate = candidate_duplicate_reason(
        candidate, content_hash=content_hash, base_dir=base_dir
    )
    if duplicate:
        return {"status": "DUPLICATE_SKIPPED", "reason": duplicate, "candidate": candidate}
    validation = validate_pdf_bytes(content)
    if not validation.get("pdf_valid"):
        return {
            "status": "FAILED",
            "failure_reason": validation.get("error") or "INVALID_PDF",
            "candidate": candidate,
        }
    registered = register_pdf_bytes(
        content,
        original_filename=Path(urlparse(candidate["pdf_url"]).path).name or "oa_paper.pdf",
        source_path=candidate["pdf_url"],
        source_type="OA_AUTOMATIC_DOWNLOAD",
        metadata_override={
            "title": str(candidate.get("title") or ""),
            "authors": str(candidate.get("authors") or ""),
            "publication_date": str(candidate.get("date") or ""),
            "doi": normalize_doi(candidate.get("doi") or ""),
        },
        base_dir=base_dir,
    )
    if not registered.get("pdf_valid"):
        return {"status": "FAILED", "failure_reason": registered.get("error"), "candidate": candidate}
    canonical_path = Path(registered["canonical_pdf_path"])
    deep_read = deep_read_pdf(
        canonical_path,
        paper_id=registered["paper_id"],
        title=registered.get("title") or candidate.get("title") or "",
        base_dir=base_dir,
        force=force_deep_read,
    )
    log = {
        "status": (
            "DEEP_READ_COMPLETE" if deep_read.get("deep_read_complete") else "DEEP_READ_PARTIAL"
        ),
        "paper_id": registered["paper_id"],
        "title": registered.get("title") or candidate.get("title") or "",
        "doi": normalize_doi(candidate.get("doi") or ""),
        "pdf_url": candidate.get("pdf_url") or "",
        "license": candidate.get("license") or "",
        "canonical_pdf_path": str(canonical_path.resolve()),
        "file_hash_sha256": content_hash,
        "real_page_count": validation["real_page_count"],
        "deep_read": deep_read,
        "completed_at": _now(),
    }
    _append_jsonl(oa_paths(base_dir)["download_log"], log)
    return log


def top_up_open_access(
    question: str,
    *,
    max_downloads: int = 1,
    max_candidates_per_query: int = 5,
    base_dir: Path = BASE_DIR,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    before = answer_research_question(question, base_dir=base_dir)
    if before["evidence_sufficiency"]["status"] != "INSUFFICIENT":
        return {
            "status": "NOT_REQUIRED",
            "message": "Local evidence is not insufficient; OA top-up was not triggered.",
            "before": before,
            "downloaded": [],
            "after": before,
        }
    message = (
        "\u5f53\u524d\u672c\u5730\u8bc1\u636e\u4e0d\u8db3\uff0c"
        "\u6b63\u5728\u68c0\u7d22\u5f00\u653e\u83b7\u53d6\u6587\u732e\u3002"
    )
    candidates: List[Dict[str, Any]] = []
    errors = []
    for query in generate_search_queries(question):
        try:
            found = search_openalex_oa(
                query,
                max_results=max_candidates_per_query,
                session=session,
            )
            candidates.extend(found)
            _append_jsonl(
                oa_paths(base_dir)["search_log"],
                {
                    "query": query,
                    "candidate_count": len(found),
                    "searched_at": _now(),
                    "source": "OpenAlex",
                },
            )
        except Exception as exc:
            errors.append(f"{query}: {exc}")
    unique: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        key = normalize_doi(candidate.get("doi") or "") or normalize_title(
            candidate.get("title") or ""
        )
        if key:
            unique.setdefault(key, candidate)
    downloads = []
    for candidate in unique.values():
        if len([row for row in downloads if row.get("status") == "DEEP_READ_COMPLETE"]) >= max_downloads:
            break
        result = download_and_deep_read(
            candidate, base_dir=base_dir, session=session
        )
        downloads.append(result)
    new_papers = [
        row["paper_id"]
        for row in downloads
        if row.get("status") == "DEEP_READ_COMPLETE"
    ]
    current_manifest = json.loads(
        rag_paths(base_dir)["manifest"].read_text(encoding="utf-8")
    )
    if new_papers:
        build_unified_rag(
            [*(current_manifest.get("paper_ids") or []), *new_papers],
            base_dir=base_dir,
        )
    after = answer_research_question(question, base_dir=base_dir)
    after["whether_oa_topup_triggered"] = True
    return {
        "status": "COMPLETED" if new_papers else "PARTIAL",
        "message": message,
        "before": before,
        "queries": generate_search_queries(question),
        "candidate_count": len(unique),
        "downloaded": downloads,
        "new_paper_ids": new_papers,
        "errors": errors,
        "after": after,
    }

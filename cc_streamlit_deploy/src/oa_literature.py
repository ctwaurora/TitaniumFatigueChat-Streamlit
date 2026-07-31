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
CROSSREF_URL = "https://api.crossref.org/works"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
USER_AGENT = "TitaniumFatigueChat/3.0 (legal-open-access-research-client)"
OA_SCHEMA_VERSION = "stage3.0"
DEFAULT_TIMEOUT = (10, 30)


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
        "network_log": root / "network_requests.jsonl",
        "manual": root / "manual_download_required.jsonl",
        "temp": base_dir / "tmp" / "pdfs" / "oa",
        "pdf_dir": base_dir / "paper" / "pdfs",
    }


def _response_metadata(response: Any, source: str, route: str) -> Dict[str, Any]:
    headers = getattr(response, "headers", {}) or {}
    return {
        "source": source,
        "route": route,
        "status_code": int(getattr(response, "status_code", 200) or 0),
        "final_url": str(getattr(response, "url", "") or ""),
        "content_type": str(headers.get("Content-Type") or "").split(";")[0].strip().lower(),
        "content_length": str(headers.get("Content-Length") or ""),
        "recorded_at": _now(),
    }


def _request_with_resilience(
    method: str,
    url: str,
    *,
    source: str,
    session: Optional[requests.Session] = None,
    timeout: tuple[int, int] | int = DEFAULT_TIMEOUT,
    max_attempts: int = 3,
    acceptable_statuses: Sequence[int] = (),
    sleep: Any = time.sleep,
    network_events: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> tuple[Any, Dict[str, Any]]:
    """Request through configured networking, then direct-connect after ProxyError.

    Direct fallback is local to this request and never changes system proxy
    variables or the caller's session.
    """
    events = network_events if network_events is not None else []
    primary = session or requests.Session()
    routes: List[tuple[str, Any]] = [("configured", primary)]
    last_error: Optional[BaseException] = None
    route_index = 0
    while route_index < len(routes):
        route, client = routes[route_index]
        route_index += 1
        for attempt in range(max(1, int(max_attempts))):
            try:
                request_method = getattr(client, method.lower())
                response = request_method(url, timeout=timeout, **kwargs)
                metadata = _response_metadata(response, source, route)
                metadata["attempt"] = attempt + 1
                events.append(metadata)
                status = metadata["status_code"]
                if status in set(acceptable_statuses):
                    return response, metadata
                if status == 429:
                    retry_after = str(
                        (getattr(response, "headers", {}) or {}).get(
                            "Retry-After", ""
                        )
                    ).strip()
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = float(2**attempt)
                    if attempt + 1 < max_attempts:
                        sleep(max(0.0, min(delay, 30.0)))
                        continue
                if status >= 500 and attempt + 1 < max_attempts:
                    sleep(float(2**attempt))
                    continue
                response.raise_for_status()
                return response, metadata
            except requests.exceptions.ProxyError as exc:
                last_error = exc
                events.append(
                    {
                        "source": source,
                        "route": route,
                        "error_type": "ProxyError",
                        "error": str(exc)[:500],
                        "attempt": attempt + 1,
                        "recorded_at": _now(),
                    }
                )
                if route == "configured":
                    direct = requests.Session()
                    direct.trust_env = False
                    routes.append(("direct_without_environment_proxy", direct))
                break
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
            ) as exc:
                last_error = exc
                events.append(
                    {
                        "source": source,
                        "route": route,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                        "attempt": attempt + 1,
                        "recorded_at": _now(),
                    }
                )
                if attempt + 1 < max_attempts:
                    sleep(float(2**attempt))
                    continue
                break
    if last_error is not None:
        raise last_error
    raise requests.ConnectionError(f"{source} request failed without a response")


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
    timeout: tuple[int, int] | int = DEFAULT_TIMEOUT,
    session: Optional[requests.Session] = None,
    network_events: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    response, request_meta = _request_with_resilience(
        "get",
        OPENALEX_URL,
        source="OpenAlex",
        session=session,
        params={
            "search": query,
            "filter": "is_oa:true,has_fulltext:true",
            "per-page": min(max(1, max_results), 50),
        },
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
        network_events=network_events,
    )
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
            candidate["network_request"] = request_meta
            candidates.append(candidate)
            break
    return candidates[:max_results]


def search_unpaywall(
    doi: str,
    *,
    email: str = "",
    timeout: tuple[int, int] | int = DEFAULT_TIMEOUT,
    session: Optional[requests.Session] = None,
    network_events: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve a DOI through Unpaywall when a real contact email is configured."""
    email = email or os.environ.get("UNPAYWALL_EMAIL", "")
    if not email or "@" not in email:
        return None
    response, request_meta = _request_with_resilience(
        "get",
        f"https://api.unpaywall.org/v2/{normalize_doi(doi)}",
        source="Unpaywall",
        session=session,
        params={"email": email},
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
        network_events=network_events,
        acceptable_statuses=(404,),
    )
    if response.status_code == 404:
        return None
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
        "network_request": request_meta,
    }


def search_crossref_oa(
    query: str,
    *,
    max_results: int = 10,
    timeout: tuple[int, int] | int = DEFAULT_TIMEOUT,
    session: Optional[requests.Session] = None,
    network_events: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    response, request_meta = _request_with_resilience(
        "get",
        CROSSREF_URL,
        source="Crossref",
        session=session,
        params={
            "query.bibliographic": query,
            "rows": min(max(1, max_results), 50),
            "select": "DOI,title,author,published,link,license,URL",
        },
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
        network_events=network_events,
    )
    results: List[Dict[str, Any]] = []
    for work in ((response.json().get("message") or {}).get("items") or []):
        links = list(work.get("link") or [])
        pdf_link = next(
            (
                link.get("URL") or ""
                for link in links
                if "pdf" in str(link.get("content-type") or "").lower()
                and str(link.get("URL") or "").startswith("https://")
            ),
            "",
        )
        licenses = list(work.get("license") or [])
        license_url = str((licenses[0] if licenses else {}).get("URL") or "")
        title_rows = work.get("title") or []
        title = str(title_rows[0] if title_rows else "")
        authors = "; ".join(
            " ".join(
                part
                for part in (
                    str(author.get("given") or "").strip(),
                    str(author.get("family") or "").strip(),
                )
                if part
            )
            for author in work.get("author") or []
        )
        date_parts = (
            ((work.get("published") or {}).get("date-parts") or [[]])[0]
        )
        results.append(
            {
                "source": "Crossref",
                "title": title,
                "authors": authors,
                "date": "-".join(str(value) for value in date_parts),
                "doi": normalize_doi(work.get("DOI") or ""),
                "landing_page": work.get("URL") or "",
                "pdf_url": pdf_link,
                # Crossref links alone do not prove OA rights.
                "license": license_url,
                "version": "",
                "host_type": "publisher",
                "is_oa": bool(pdf_link and license_url),
                "oa_status": "gold" if pdf_link and license_url else "",
                "retrieved_at": _now(),
                "data_version": OA_SCHEMA_VERSION,
                "network_request": request_meta,
            }
        )
    return results[:max_results]


def search_semantic_scholar_oa(
    query: str,
    *,
    max_results: int = 10,
    timeout: tuple[int, int] | int = DEFAULT_TIMEOUT,
    session: Optional[requests.Session] = None,
    network_events: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    response, request_meta = _request_with_resilience(
        "get",
        SEMANTIC_SCHOLAR_URL,
        source="Semantic Scholar",
        session=session,
        params={
            "query": query,
            "limit": min(max(1, max_results), 50),
            "fields": "title,authors,year,externalIds,url,openAccessPdf",
        },
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
        network_events=network_events,
    )
    rows: List[Dict[str, Any]] = []
    for work in response.json().get("data") or []:
        open_pdf = work.get("openAccessPdf") or {}
        pdf_url = str(open_pdf.get("url") or "")
        rows.append(
            {
                "source": "Semantic Scholar",
                "title": work.get("title") or "",
                "authors": "; ".join(
                    str(author.get("name") or "")
                    for author in work.get("authors") or []
                    if author.get("name")
                ),
                "date": str(work.get("year") or ""),
                "doi": normalize_doi(
                    (work.get("externalIds") or {}).get("DOI") or ""
                ),
                "landing_page": work.get("url") or "",
                "pdf_url": pdf_url,
                "license": str(open_pdf.get("license") or ""),
                "version": "",
                "host_type": "repository",
                "is_oa": bool(pdf_url),
                "oa_status": "green" if pdf_url else "",
                "retrieved_at": _now(),
                "data_version": OA_SCHEMA_VERSION,
                "network_request": request_meta,
            }
        )
    return rows[:max_results]


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
            if not paper.get("pdf_valid") or not paper.get("canonical_pdf_path"):
                return ""
            return f"DOI_DUPLICATE:{paper.get('paper_id')}"
        if content_hash and paper.get("file_hash_sha256") == content_hash:
            return f"HASH_DUPLICATE:{paper.get('paper_id')}"
        similarity = title_similarity(title, paper.get("title") or "")
        if title and similarity >= 0.94:
            if not paper.get("pdf_valid") or not paper.get("canonical_pdf_path"):
                return ""
            return f"TITLE_NEAR_DUPLICATE:{paper.get('paper_id')}:{similarity:.3f}"
    return ""


def _download_bytes(
    url: str,
    *,
    timeout: tuple[int, int] | int = (10, 60),
    max_bytes: int = 100 * 1024 * 1024,
    min_bytes: int = 512,
    session: Optional[requests.Session] = None,
    network_events: Optional[List[Dict[str, Any]]] = None,
    return_metadata: bool = False,
) -> bytes | tuple[bytes, Dict[str, Any]]:
    response, request_meta = _request_with_resilience(
        "get",
        url,
        source="OA PDF",
        session=session,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/pdf,application/octet-stream;q=0.9",
        },
        timeout=timeout,
        stream=True,
        allow_redirects=True,
        network_events=network_events,
    )
    if urlparse(response.url).scheme != "https":
        raise ValueError("DOWNLOAD_REDIRECTED_TO_NON_HTTPS")
    content_type = request_meta.get("content_type") or ""
    if (
        content_type.startswith("text/")
        or content_type in {"application/json", "application/xml", "text/html"}
    ):
        raise ValueError(f"NON_PDF_CONTENT_TYPE:{content_type or 'missing'}")
    declared_length = request_meta.get("content_length") or ""
    if str(declared_length).isdigit() and int(declared_length) > max_bytes:
        raise ValueError("PDF_EXCEEDS_SIZE_LIMIT")
    content = bytearray()
    for block in response.iter_content(chunk_size=1024 * 128):
        if not block:
            continue
        content.extend(block)
        if len(content) > max_bytes:
            raise ValueError("PDF_EXCEEDS_SIZE_LIMIT")
    payload = bytes(content)
    if len(payload) < int(min_bytes):
        raise ValueError("PDF_BELOW_MINIMUM_SIZE")
    if not payload.startswith(b"%PDF"):
        raise ValueError("INVALID_PDF_HEADER")
    request_meta["downloaded_bytes"] = len(payload)
    if return_metadata:
        return payload, request_meta
    return payload


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
        network_events: List[Dict[str, Any]] = []
        content, download_metadata = _download_bytes(
            candidate["pdf_url"],
            session=session,
            network_events=network_events,
            return_metadata=True,
        )
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
        download_metadata=download_metadata,
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
    download_metadata: Optional[Dict[str, Any]] = None,
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
        "authors": str(candidate.get("authors") or ""),
        "year": str(candidate.get("year") or candidate.get("date") or ""),
        "metadata_source": str(
            candidate.get("metadata_source") or candidate.get("source") or ""
        ),
        "oa_source": str(
            candidate.get("oa_source")
            or candidate.get("metadata_source")
            or candidate.get("source")
            or ""
        ),
        "pdf_url": candidate.get("pdf_url") or "",
        "license": candidate.get("license") or "",
        "canonical_pdf_path": str(canonical_path.resolve()),
        "file_hash_sha256": content_hash,
        "file_size": len(content),
        "downloaded_pdf": True,
        "real_page_count": validation["real_page_count"],
        "page_record_count": int(deep_read.get("page_record_count") or 0),
        "processed_page_count": int(
            deep_read.get("processed_page_count")
            or deep_read.get("page_record_count")
            or 0
        ),
        "evidence_count": int(deep_read.get("evidence_count") or 0),
        "direct_evidence_count": int(
            deep_read.get("direct_evidence_count") or 0
        ),
        "indirect_evidence_count": int(
            deep_read.get("indirect_evidence_count") or 0
        ),
        "mention_only_count": int(deep_read.get("mention_only_count") or 0),
        "pdf_valid": True,
        "http": dict(download_metadata or {}),
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
    manual_override: bool = False,
    evidence_status_before: str = "",
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    max_downloads = max(1, min(int(max_downloads), 3))
    try:
        before = answer_research_question(question, base_dir=base_dir)
    except (FileNotFoundError, RuntimeError, ValueError):
        before = {
            "evidence_sufficiency": {"status": "INSUFFICIENT"},
            "results": [],
        }
    canonical_status = str(
        before.get("evidence_sufficiency", {}).get("status")
        or evidence_status_before
        or "INSUFFICIENT"
    )
    if not manual_override and canonical_status not in {
        "INSUFFICIENT",
        "PARTIALLY_SUFFICIENT",
    }:
        return {
            "status": "NOT_REQUIRED",
            "message": "Local evidence is sufficient; automatic OA top-up was not triggered.",
            "before": before,
            "downloaded": [],
            "after": before,
            "candidate_count": 0,
            "oa_candidate_count": 0,
            "duplicate_rejected_count": 0,
            "paywall_rejected_count": 0,
            "source_results": [],
            "manual_override": False,
            "errors": [],
        }
    message = (
        "本次补充由用户主动触发，并非系统判定当前证据必然不足。"
        if manual_override
        else "当前本地证据不足，正在检索开放获取文献。"
    )
    if progress_callback:
        progress_callback("SEARCHING", {"candidate_count": 0})
    candidates: List[Dict[str, Any]] = []
    errors: List[str] = []
    source_results: List[Dict[str, Any]] = []
    network_events: List[Dict[str, Any]] = []
    queries = generate_search_queries(question)

    def run_source(source: str, operation: Any, query: str) -> None:
        try:
            found = list(operation())
            candidates.extend(found)
            source_results.append(
                {
                    "source": source,
                    "query": query,
                    "status": "OK",
                    "candidate_count": len(found),
                }
            )
            _append_jsonl(
                oa_paths(base_dir)["search_log"],
                {
                    "query": query,
                    "candidate_count": len(found),
                    "searched_at": _now(),
                    "source": source,
                    "status": "OK",
                },
            )
        except Exception as exc:
            error = f"{source}:{query}:{type(exc).__name__}:{exc}"
            errors.append(error)
            source_results.append(
                {
                    "source": source,
                    "query": query,
                    "status": "ERROR",
                    "candidate_count": 0,
                    "error": str(exc)[:500],
                }
            )
            _append_jsonl(
                oa_paths(base_dir)["search_log"],
                {
                    "query": query,
                    "candidate_count": 0,
                    "searched_at": _now(),
                    "source": source,
                    "status": "ERROR",
                    "error": str(exc)[:500],
                },
            )

    for query in queries:
        run_source(
            "OpenAlex",
            lambda query=query: search_openalex_oa(
                query,
                max_results=max_candidates_per_query,
                session=session,
                network_events=network_events,
            ),
            query,
        )
    run_source(
        "Crossref",
        lambda: search_crossref_oa(
            question,
            max_results=max_candidates_per_query,
            session=session,
            network_events=network_events,
        ),
        question,
    )
    run_source(
        "Semantic Scholar",
        lambda: search_semantic_scholar_oa(
            question,
            max_results=max_candidates_per_query,
            session=session,
            network_events=network_events,
        ),
        question,
    )

    unpaywall_email = os.environ.get("UNPAYWALL_EMAIL", "")
    if not unpaywall_email or "@" not in unpaywall_email:
        warning = "未配置UNPAYWALL_EMAIL，本次跳过Unpaywall。"
        source_results.append(
            {
                "source": "Unpaywall",
                "query": "",
                "status": "SKIPPED",
                "candidate_count": 0,
                "message": warning,
            }
        )
    else:
        doi_values = list(
            dict.fromkeys(
                normalize_doi(candidate.get("doi") or "")
                for candidate in candidates
                if normalize_doi(candidate.get("doi") or "")
            )
        )[:max_candidates_per_query]
        for doi in doi_values:
            try:
                resolved = search_unpaywall(
                    doi,
                    email=unpaywall_email,
                    session=session,
                    network_events=network_events,
                )
                if resolved:
                    candidates.append(resolved)
                source_results.append(
                    {
                        "source": "Unpaywall",
                        "query": doi,
                        "status": "OK",
                        "candidate_count": int(bool(resolved)),
                    }
                )
            except Exception as exc:
                errors.append(f"Unpaywall:{doi}:{type(exc).__name__}:{exc}")
                source_results.append(
                    {
                        "source": "Unpaywall",
                        "query": doi,
                        "status": "ERROR",
                        "candidate_count": 0,
                        "error": str(exc)[:500],
                    }
                )

    for event in network_events:
        _append_jsonl(oa_paths(base_dir)["network_log"], event)
    unique: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        key = normalize_doi(candidate.get("doi") or "") or normalize_title(
            candidate.get("title") or ""
        )
        if key:
            unique.setdefault(key, candidate)
    allowed_candidates = [
        candidate
        for candidate in unique.values()
        if oa_download_allowed(candidate)[0]
    ]
    if progress_callback:
        progress_callback(
            "DOWNLOADING",
            {
                "candidate_count": len(unique),
                "oa_candidate_count": len(allowed_candidates),
                "last_completed_phase": "SEARCHING",
            },
        )
    downloads: List[Dict[str, Any]] = []
    duplicate_rejected_count = 0
    paywall_rejected_count = 0
    for candidate in unique.values():
        if len([row for row in downloads if row.get("status") == "DEEP_READ_COMPLETE"]) >= max_downloads:
            break
        allowed, denial_reason = oa_download_allowed(candidate)
        if not allowed:
            paywall_rejected_count += 1
            record_manual_download_required(
                candidate, denial_reason, base_dir=base_dir
            )
            continue
        result = download_and_deep_read(
            candidate, base_dir=base_dir, session=session
        )
        downloads.append(result)
        if result.get("status") == "DUPLICATE_SKIPPED":
            duplicate_rejected_count += 1
        if result.get("manual_download_required"):
            paywall_rejected_count += 1
        if result.get("status") == "DEEP_READ_COMPLETE" and progress_callback:
            progress_callback(
                "DEEP_READING",
                {
                    "candidate_count": len(unique),
                    "oa_candidate_count": len(allowed_candidates),
                    "downloaded_count": 1,
                    "deep_read_count": 1,
                    "downloaded_paper_ids": [result.get("paper_id")],
                    "last_completed_phase": "DEEP_READING",
                },
            )
    deep_read_papers = [
        row["paper_id"]
        for row in downloads
        if row.get("status") == "DEEP_READ_COMPLETE"
    ]
    from src.auto_oa_pipeline import evaluate_auto_rag_gate

    gates = {
        paper_id: evaluate_auto_rag_gate(paper_id, base_dir=base_dir)
        for paper_id in deep_read_papers
    }
    new_papers = [
        paper_id for paper_id, gate in gates.items() if gate.get("passed")
    ]
    current_manifest = json.loads(
        rag_paths(base_dir)["manifest"].read_text(encoding="utf-8")
    ) if rag_paths(base_dir)["manifest"].exists() else {"paper_ids": []}
    cache_invalidated = False
    if new_papers:
        if progress_callback:
            progress_callback(
                "REINDEXING",
                {
                    "downloaded_count": len(new_papers),
                    "deep_read_count": len(new_papers),
                    "downloaded_paper_ids": new_papers,
                    "last_completed_phase": "DEEP_READING",
                },
            )
        build_unified_rag(
            list(
                dict.fromkeys(
                    [*(current_manifest.get("paper_ids") or []), *new_papers]
                )
            ),
            base_dir=base_dir,
        )
        try:
            from src.data_cache import mark_data_changed

            mark_data_changed()
            cache_invalidated = True
        except Exception as exc:
            errors.append(f"cache_invalidation:{type(exc).__name__}:{exc}")
    try:
        after = answer_research_question(question, base_dir=base_dir)
    except (FileNotFoundError, RuntimeError, ValueError):
        after = {
            "evidence_sufficiency": {"status": "INSUFFICIENT"},
            "results": [],
        }
    after["whether_oa_topup_triggered"] = True
    if progress_callback and new_papers:
        progress_callback(
            "REINDEXING",
            {
                "downloaded_count": len(new_papers),
                "deep_read_count": len(new_papers),
                "downloaded_paper_ids": new_papers,
                "last_completed_phase": "REINDEXING",
            },
        )
    return {
        "status": "COMPLETED" if new_papers else "PARTIAL",
        "message": message,
        "manual_override": bool(manual_override),
        "evidence_status_before": canonical_status,
        "before": before,
        "queries": queries,
        "candidate_count": len(unique),
        "oa_candidate_count": len(allowed_candidates),
        "downloaded": downloads,
        "new_paper_ids": new_papers,
        "automatic_quality_gates": gates,
        "duplicate_rejected_count": duplicate_rejected_count,
        "paywall_rejected_count": paywall_rejected_count,
        "source_results": source_results,
        "network_events": network_events,
        "cache_invalidated": cache_invalidated,
        "errors": errors,
        "after": after,
    }

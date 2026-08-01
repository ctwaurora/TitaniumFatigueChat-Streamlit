"""Real scholarly-metadata lookup used by the literature library."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

import requests

from src.stage1_store import normalize_doi


USER_AGENT = "TitaniumFatigueChat/3.1 (scholarly metadata verification)"
PLACEHOLDER_TITLES = {"", "nan", "none", "null", "n/a", "na", "unknown", "untitled"}


def is_valid_title(value: object) -> bool:
    title = " ".join(str(value or "").split()).strip()
    lowered = title.lower()
    if lowered in PLACEHOLDER_TITLES or len(title) < 4:
        return False
    if re.match(r"^(?:https?://)?(?:dx\.)?doi\.org/10\.", lowered):
        return False
    if re.fullmatch(r"(?:doi\s*:\s*)?10\.\d{4,9}/\S+", lowered):
        return False
    if lowered.startswith(("http://", "https://", "www.")):
        return False
    if re.fullmatch(r"[a-z]{0,5}\d{5,}[a-z0-9._-]*", lowered):
        return False
    if lowered.startswith((
        "university of ", "accepted manuscript", "author's personal copy",
        "elsevier editorial system", "researchgate", "materials research, vol.",
        "a sheffield hallam university thesis",
        "microsoft word - ", "type of the paper ",
    )):
        return False
    if lowered.endswith((".doc", ".docx", ".pdf")):
        return False
    if lowered.startswith(("arxiv:", "proceedings of ")):
        return False
    if re.match(r"^(?:metals|materials|fatigue|journal)\s+20\d{2}\s*,\s*\d+", lowered):
        return False
    if len(re.findall(r"[^\W\d_]", title, flags=re.UNICODE)) < 4:
        return False
    # A Latin-script scholarly title is not a single identifier/header token.
    if re.search(r"[a-z]", lowered) and not re.search(r"[\u3400-\u9fff]", title):
        if len(re.findall(r"[a-z0-9]+", lowered)) < 4:
            return False
    return True


def doi_from_input(value: str) -> str:
    raw = unquote(str(value or "").strip())
    doi = normalize_doi(raw)
    if doi:
        return doi
    parsed = urlparse(raw)
    if parsed.netloc.lower() in {"doi.org", "dx.doi.org"}:
        return normalize_doi(parsed.path.lstrip("/"))
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", raw, re.I)
    return normalize_doi(match.group(0)) if match else ""


def _crossref_metadata(
    doi: str,
    *,
    session: Optional[requests.Session] = None,
    timeout: int = 15,
) -> Optional[Dict[str, Any]]:
    client = session or requests.Session()
    response = client.get(
        f"https://api.crossref.org/works/{doi}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    message = (response.json() or {}).get("message") or {}
    titles = message.get("title") or []
    title = titles[0] if titles else ""
    authors = "; ".join(
        " ".join(
            part
            for part in (str(row.get("given") or ""), str(row.get("family") or ""))
            if part
        )
        for row in message.get("author") or []
    )
    date_parts = (
        (message.get("published-print") or {}).get("date-parts")
        or (message.get("published-online") or {}).get("date-parts")
        or (message.get("issued") or {}).get("date-parts")
        or []
    )
    year = str(date_parts[0][0]) if date_parts and date_parts[0] else ""
    if not is_valid_title(title):
        return None
    return {
        "title": " ".join(str(title).split()),
        "authors": authors,
        "year": year,
        "doi": normalize_doi(message.get("DOI") or doi),
        "source_url": str(message.get("URL") or f"https://doi.org/{doi}"),
        "metadata_source": "Crossref",
        "oa_status": "UNKNOWN",
        "pdf_url": "",
    }


def _openalex_metadata(
    doi: str,
    *,
    session: Optional[requests.Session] = None,
    timeout: int = 15,
) -> Optional[Dict[str, Any]]:
    client = session or requests.Session()
    response = client.get(
        "https://api.openalex.org/works",
        params={"filter": f"doi:{doi}", "per-page": 1},
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    results = (response.json() or {}).get("results") or []
    if not results:
        return None
    work = results[0]
    title = work.get("display_name") or work.get("title") or ""
    if not is_valid_title(title):
        return None
    authors = "; ".join(
        str((row.get("author") or {}).get("display_name") or "").strip()
        for row in work.get("authorships") or []
        if str((row.get("author") or {}).get("display_name") or "").strip()
    )
    oa = work.get("open_access") or {}
    location = work.get("best_oa_location") or {}
    return {
        "title": " ".join(str(title).split()),
        "authors": authors,
        "year": str(work.get("publication_year") or ""),
        "doi": normalize_doi(work.get("doi") or doi),
        "source_url": str(location.get("landing_page_url") or work.get("id") or ""),
        "metadata_source": "OpenAlex",
        "oa_status": str(oa.get("oa_status") or ("OA" if oa.get("is_oa") else "CLOSED")),
        "pdf_url": str(location.get("pdf_url") or ""),
        "license": str(location.get("license") or ""),
        "is_oa": bool(oa.get("is_oa")),
    }


def fetch_metadata(
    value: str,
    *,
    session: Optional[requests.Session] = None,
    timeout: int = 15,
) -> Dict[str, Any]:
    """Resolve a DOI/DOI URL against real APIs; never invent missing fields."""
    doi = doi_from_input(value)
    if not doi:
        return {
            "status": "INVALID_INPUT",
            "validation_status": "INVALID_METADATA",
            "error": "未识别到合法 DOI；普通网页 URL 需要包含可解析 DOI。",
        }

    errors = []
    for source in (_openalex_metadata, _crossref_metadata):
        try:
            metadata = source(doi, session=session, timeout=timeout)
        except (requests.RequestException, ValueError, KeyError) as exc:
            errors.append(f"{source.__name__}:{exc}")
            continue
        if metadata and is_valid_title(metadata.get("title")):
            metadata.update(
                {
                    "status": "VALID",
                    "validation_status": "VALID",
                    "pdf_status": (
                        "OA_PDF_AVAILABLE"
                        if metadata.get("pdf_url") and metadata.get("is_oa")
                        else "PDF_NOT_ACQUIRED"
                    ),
                }
            )
            return metadata
    return {
        "status": "NOT_FOUND",
        "validation_status": "INVALID_METADATA",
        "doi": doi,
        "error": "; ".join(errors) or "真实元数据源未返回可用题名。",
    }

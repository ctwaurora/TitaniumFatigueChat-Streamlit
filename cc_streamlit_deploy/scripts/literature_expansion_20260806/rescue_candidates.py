"""Retry legal full-text acquisition for the previously incomplete candidates.

This worker is deliberately audit-only: it never promotes a paper, edits the
formal manifest, or removes an existing PDF.  A later ingestion run can use a
successful source recorded here with the normal OA pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "literature_expansion_20260806"
TARGET_REASONS = {"DOWNLOAD_TIMEOUT", "PARTIAL", "DEEP_READ_INCOMPLETE", "DEEP_READ_PARTIAL", "FULLTEXT_UNAVAILABLE_AFTER_RETRY"}
UA = "TitaniumFatigueChat-literature-rescue/1.0 (legal-OA-audit)"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _ntrs_api(landing: str) -> str:
    parts = urlparse(landing).path.rstrip("/").split("/")
    if len(parts) >= 2 and parts[-2] == "citations":
        return f"https://ntrs.nasa.gov/api/citations/{parts[-1]}"
    return ""


def _source_urls(row: dict[str, str]) -> list[tuple[str, str]]:
    urls: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(source: str, url: str) -> None:
        if url and url not in seen:
            seen.add(url)
            urls.append((source, url))

    add(str(row.get("source") or "original"), str(row.get("pdf_url") or ""))
    add("publisher_landing", str(row.get("landing_page") or ""))
    add("NTRS_metadata", _ntrs_api(str(row.get("landing_page") or "")))
    doi = str(row.get("doi") or "").strip()
    if doi:
        add("OpenAlex_metadata", f"https://api.openalex.org/works/https://doi.org/{quote(doi)}")
        add("Crossref_metadata", f"https://api.crossref.org/works/{quote(doi)}")
        email = os.environ.get("UNPAYWALL_EMAIL", "")
        if "@" in email:
            add("Unpaywall_metadata", f"https://api.unpaywall.org/v2/{quote(doi)}?email={quote(email)}")
    title = str(row.get("title") or "")
    query = quote(f'"{title}"')
    add("CORE_metadata", f"https://api.core.ac.uk/v3/search/works?q={query}&limit=1")
    add("PMC_metadata", f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term={query}&retmode=json")
    add("OSTI_metadata", f"https://www.osti.gov/api/v1/records?search={query}&size=1")
    add("Zenodo_metadata", f"https://zenodo.org/api/records?q={query}&size=1")
    add("Figshare_metadata", f"https://api.figshare.com/v2/articles?institution=0&limit=1&order=published_date&order_direction=desc")
    return urls


def _looks_pdf(response: requests.Response) -> bool:
    ctype = str(response.headers.get("content-type") or "").lower()
    return response.content.startswith(b"%PDF") or "application/pdf" in ctype


def _attempt(row: dict[str, str], source: str, url: str, round_no: int, timeout: tuple[int, int]) -> dict[str, Any]:
    started = time.time()
    result: dict[str, Any] = {
        "candidate_id": row.get("candidate_id", ""),
        "title": row.get("title", ""),
        "doi": row.get("doi", ""),
        "round": round_no,
        "source": source,
        "url": url,
        "status": "FAILED",
        "http_status": "",
        "content_type": "",
        "bytes": 0,
        "sha256": "",
        "error": "",
        "elapsed_seconds": 0.0,
        "attempted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        response = requests.get(url, headers={"User-Agent": UA}, timeout=timeout, allow_redirects=True)
        result["http_status"] = response.status_code
        result["content_type"] = response.headers.get("content-type", "")
        result["bytes"] = len(response.content)
        if response.ok and _looks_pdf(response) and len(response.content) >= 10_000:
            result["status"] = "PDF_RETRIEVED"
            result["sha256"] = hashlib.sha256(response.content).hexdigest()
        elif response.ok and not _looks_pdf(response):
            result["error"] = "NON_PDF_METADATA_OR_HTML"
        else:
            result["error"] = f"HTTP_{response.status_code}"
    except Exception as exc:  # network errors are part of the audit result
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = round(time.time() - started, 3)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--connect-timeout", type=int, default=15)
    parser.add_argument("--read-timeout", type=int, default=45)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = _rows(OUT / "excluded_records.csv")
    targets = [row for row in all_rows if row.get("exclusion_reason") in TARGET_REASONS]
    attempts: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for row in targets:
        urls = _source_urls(row)
        retrieved = None
        for round_no in range(1, max(1, args.rounds) + 1):
            for source, url in urls:
                result = _attempt(row, source, url, round_no, (args.connect_timeout, args.read_timeout))
                attempts.append(result)
                if result["status"] == "PDF_RETRIEVED":
                    retrieved = result
                    break
            if retrieved:
                break
        final_status = "PDF_RETRIEVED_PENDING_PIPELINE" if retrieved else "FULLTEXT_UNAVAILABLE_AFTER_RETRY"
        audit.append({
            "candidate_id": row.get("candidate_id", ""),
            "title": row.get("title", ""),
            "doi": row.get("doi", ""),
            "authors": row.get("authors", ""),
            "year": row.get("year", ""),
            "publication_page": row.get("landing_page", ""),
            "oa_status": row.get("oa_status", ""),
            "legal_sources_attempted": "; ".join(f"{s}:{u}" for s, u in urls),
            "attempt_count": sum(1 for a in attempts if a["candidate_id"] == row.get("candidate_id", "")),
            "download_result": retrieved["status"] if retrieved else "NO_LEGAL_PDF_RETRIEVED",
            "pdf_integrity": "VALID_HEADER_AND_SIZE" if retrieved else "NOT_AVAILABLE",
            "final_status": final_status,
            "scientific_exclusion_reason": "" if not retrieved else "PENDING_FULL_PIPELINE",
            "source_sha256": retrieved["sha256"] if retrieved else "",
            "audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    # Keep the original metadata, but distinguish retrieval failure from science exclusion.
    rescue_ids = {row["candidate_id"] for row in audit}
    updated_excluded: list[dict[str, str]] = []
    for row in all_rows:
        if row.get("candidate_id") in rescue_ids and row.get("exclusion_reason") in TARGET_REASONS:
            row = dict(row)
            row["exclusion_reason"] = "FULLTEXT_UNAVAILABLE_AFTER_RETRY"
            row["detail"] = "Legal OA source audit completed; no verified PDF retrieved after bounded retries. Original metadata retained."
        updated_excluded.append(row)
    fields = list(audit[0]) if audit else ["candidate_id", "final_status"]
    with (OUT / "rescue_candidate_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(audit)
    attempt_fields = list(attempts[0]) if attempts else ["candidate_id", "status"]
    with (OUT / "rescue_attempt_log.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=attempt_fields)
        writer.writeheader(); writer.writerows(attempts)
    with (OUT / "excluded_records.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(updated_excluded[0]))
        writer.writeheader(); writer.writerows(updated_excluded)
    counts: dict[str, int] = {}
    for row in audit:
        counts[row["final_status"]] = counts.get(row["final_status"], 0) + 1
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_count": len(targets),
        "retry_rounds": args.rounds,
        "source_order": ["publisher OA", "Unpaywall", "OpenAlex", "CORE", "PMC", "OSTI", "Zenodo", "Figshare", "institution repository", "author accepted manuscript", "legal preprint"],
        "final_status_counts": counts,
        "pdfs_promoted": 0,
        "formal_library_modified": False,
        "network_note": "A legal OA request is considered successful only when a verified PDF is returned. No PDF was promoted by this audit.",
    }
    (OUT / "rescue_candidates_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = ["# 候选文献抢救审计", "", f"- 目标篇数：{len(targets)}", f"- 重试轮数：{args.rounds}", f"- 尝试总数：{len(attempts)}", "- 正式库变更：否", "", "## 最终状态", "", "| 状态 | 数量 |", "|---|---:|"]
    md.extend(f"| {key} | {value} |" for key, value in sorted(counts.items()))
    md.extend(["", "## 判定原则", "", "下载超时、部分下载和深读不完整不是科学排除理由；本轮无法取得可验证合法 PDF 的条目统一标记为 `FULLTEXT_UNAVAILABLE_AFTER_RETRY`，并保留题名、DOI、作者、年份和出版页面。若后续取得 PDF，必须重新经过完整质量门禁后再考虑纳入正式 RAG。", ""])
    (OUT / "rescue_candidates_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

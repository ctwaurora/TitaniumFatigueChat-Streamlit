from __future__ import annotations

from typing import Any

from .base import LiteratureSource, SourceCandidate, normalize_doi


class OSTISource(LiteratureSource):
    name = "OSTI"
    retry_statuses = (500, 502, 503, 504)
    max_retries = 2
    backoff_base_seconds = 2.0
    circuit_failure_threshold = 2
    circuit_cooldown_seconds = 120.0

    @staticmethod
    def _candidate(work: dict[str, Any], *, method: str, identifier: str) -> SourceCandidate:
        landing = ""
        fulltext = str(work.get("download_url") or work.get("fulltext_url") or "")
        for link in work.get("links") or []:
            relation = str((link or {}).get("rel") or "")
            href = str((link or {}).get("href") or "")
            if relation == "fulltext" and href.startswith("https://"):
                fulltext = href
            elif relation == "citation" and href.startswith("https://"):
                landing = href
        location = {
            "pdf_url": fulltext,
            "landing_page_url": landing,
            "host_type": "government_repository",
            "version": "OFFICIAL_GOVERNMENT_COPY",
            "license": "PUBLIC_ACCESS_LOCATION",
            "source": "OSTI",
        }
        return SourceCandidate(
            title=str(work.get("title") or ""),
            DOI=normalize_doi(work.get("doi")),
            authors=[str(value) for value in work.get("authors") or []],
            year=str(work.get("publication_date") or "UNKNOWN")[:4],
            journal=str(work.get("journal_name") or work.get("product_type") or "UNKNOWN"),
            source_database=["OSTI"],
            OA_status="OFFICIAL_GOVERNMENT_FULLTEXT" if fulltext else "METADATA_ONLY",
            OA_locations=[location] if fulltext or landing else [],
            pdf_candidate_url=fulltext,
            source_record_ids=[str(work.get("osti_id") or "")],
            version_provenance=[
                {
                    "source": "OSTI",
                    "version": "OFFICIAL_GOVERNMENT_COPY",
                    "host_type": "government_repository",
                    "license": "PUBLIC_ACCESS_LOCATION",
                }
            ] if fulltext else [],
            retrieval_provenance=[{"source": "OSTI", "method": method, "identifier": identifier}],
        )

    def search(self, query: str, *, since: str = "", limit: int = 25) -> list[SourceCandidate]:
        doi = normalize_doi(query)
        params: dict[str, Any] = {"rows": min(limit, 100)}
        method = "DOI_SEARCH" if doi.startswith("10.") else "SEARCH"
        identifier = doi if doi.startswith("10.") else query
        params["doi" if doi.startswith("10.") else "q"] = identifier
        if since:
            params["publication_date_start"] = since
        payload = self._get_json_cached(
            "https://www.osti.gov/api/v1/records",
            cache_key=f"{method.casefold()}:{identifier.casefold()}:{since}:{min(limit, 100)}",
            params=params,
            headers={"Accept": "application/json"},
        )
        rows = payload if isinstance(payload, list) else payload.get("results") or payload.get("records") or []
        return [self._candidate(work, method=method, identifier=identifier) for work in rows]

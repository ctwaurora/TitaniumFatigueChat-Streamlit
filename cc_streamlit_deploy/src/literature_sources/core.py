from __future__ import annotations

from typing import Any

from .base import LiteratureSource, SourceCandidate, normalize_doi, secret_value


class CORESource(LiteratureSource):
    name = "CORE"
    env_key = "CORE_API_KEY"
    # CORE documents five single calls per ten seconds.  This intentionally
    # stays slightly below that public limit, with or without a key.
    min_request_interval_seconds = 2.1
    max_retries = 1
    backoff_base_seconds = 2.0
    circuit_failure_threshold = 2
    circuit_cooldown_seconds = 90.0
    request_timeout_seconds = 20.0

    @property
    def configured(self) -> bool:
        # The official public API supports limited unauthenticated access.
        return True

    def _headers(self) -> dict[str, str]:
        key = secret_value(self.env_key)
        return {"Authorization": f"Bearer {key}"} if key else {}

    @staticmethod
    def _links(work: dict[str, Any]) -> list[dict[str, str]]:
        values: list[dict[str, str]] = []
        download = work.get("downloadUrl")
        downloads = download if isinstance(download, list) else [download]
        for url in downloads:
            if str(url or "").startswith("https://"):
                values.append(
                    {
                        "pdf_url": str(url),
                        "landing_page_url": "",
                        "host_type": "repository",
                        "version": "REPOSITORY_COPY",
                        "license": "OA_DISCOVERED_BY_CORE",
                        "source": "CORE",
                    }
                )
        for url in work.get("sourceFulltextUrls") or []:
            text = str(url or "")
            if text.startswith("https://") and not any(row["pdf_url"] == text for row in values):
                values.append(
                    {
                        "pdf_url": text,
                        "landing_page_url": "",
                        "host_type": "repository",
                        "version": "REPOSITORY_COPY",
                        "license": "OA_DISCOVERED_BY_CORE",
                        "source": "CORE",
                    }
                )
        for link in work.get("links") or []:
            url = str((link or {}).get("url") or "")
            if url.startswith("https://"):
                values.append(
                    {
                        "pdf_url": "",
                        "landing_page_url": url,
                        "host_type": "aggregator",
                        "version": "UNKNOWN",
                        "license": "UNKNOWN",
                        "source": "CORE",
                    }
                )
        return values

    @classmethod
    def _candidate(cls, work: dict[str, Any], *, method: str, identifier: str) -> SourceCandidate:
        locations = cls._links(work)
        pdf = next((row["pdf_url"] for row in locations if row.get("pdf_url")), "")
        authors = []
        for value in work.get("authors") or []:
            name = str(value.get("name") or "") if isinstance(value, dict) else str(value or "")
            if name and name not in authors:
                authors.append(name)
        journals = work.get("journals") or []
        journal = "UNKNOWN"
        if journals:
            first = journals[0]
            journal = str(first.get("title") or first.get("name") or first) if isinstance(first, dict) else str(first)
        return SourceCandidate(
            title=str(work.get("title") or ""),
            DOI=normalize_doi(work.get("doi")),
            authors=authors,
            year=str(work.get("yearPublished") or "UNKNOWN"),
            journal=journal,
            source_database=["CORE"],
            citation_count=int(work.get("citationCount") or 0),
            OA_status="CORE_FULLTEXT" if pdf else "METADATA_ONLY",
            OA_locations=locations,
            pdf_candidate_url=pdf,
            source_record_ids=[str(work.get("id") or "")],
            version_provenance=[
                {
                    "source": "CORE",
                    "version": str(row.get("version") or "UNKNOWN"),
                    "host_type": str(row.get("host_type") or "UNKNOWN"),
                    "license": str(row.get("license") or "UNKNOWN"),
                }
                for row in locations
            ],
            retrieval_provenance=[{"source": "CORE", "method": method, "identifier": identifier}],
        )

    def _query(self, query: str, limit: int) -> list[SourceCandidate]:
        payload = self._get_json_cached(
            "https://api.core.ac.uk/v3/search/works/",
            cache_key=f"search:{query.casefold()}:{min(limit, 100)}",
            params={"q": query, "limit": min(limit, 100), "exclude": "fullText"},
            headers=self._headers(),
        )
        return [self._candidate(work, method="SEARCH", identifier=query) for work in payload.get("results") or []]

    def resolve_doi(self, doi: str) -> SourceCandidate | None:
        normalized = normalize_doi(doi)
        if not normalized.startswith("10."):
            return None
        rows = self._query(f"doi:{normalized}", 1)
        exact = next((row for row in rows if normalize_doi(row.DOI) == normalized), None)
        if exact:
            exact.retrieval_provenance = [{"source": self.name, "method": "DOI_SEARCH", "identifier": normalized}]
        return exact

    def search(self, query: str, *, since: str = "", limit: int = 25) -> list[SourceCandidate]:
        normalized = normalize_doi(query)
        if normalized.startswith("10."):
            candidate = self.resolve_doi(normalized)
            return [candidate] if candidate else []
        return self._query(query, limit)

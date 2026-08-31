from __future__ import annotations

from typing import Any, Iterable

from .base import LiteratureSource, SourceCandidate, normalize_doi, secret_value


class OpenAlexSource(LiteratureSource):
    name = "OPENALEX"
    max_retries = 4
    backoff_base_seconds = 1.0
    circuit_failure_threshold = 2
    circuit_cooldown_seconds = 120.0

    def _auth_params(self) -> dict[str, str]:
        params: dict[str, str] = {}
        api_key = secret_value("OPENALEX_API_KEY")
        email = secret_value("OPENALEX_EMAIL") or secret_value("UNPAYWALL_EMAIL")
        if api_key:
            params["api_key"] = api_key
        if email:
            params["mailto"] = email
        return params

    @staticmethod
    def _candidate(work: dict[str, Any], *, method: str, identifier: str) -> SourceCandidate:
        locations: list[dict[str, str]] = []
        for location in work.get("locations") or []:
            pdf = str(location.get("pdf_url") or "")
            landing = str(location.get("landing_page_url") or "")
            if not (pdf or landing):
                continue
            source = location.get("source") or {}
            locations.append(
                {
                    "pdf_url": pdf,
                    "landing_page_url": landing,
                    "host_type": str(source.get("host_organization_name") or source.get("type") or "UNKNOWN"),
                    "version": str(location.get("version") or "UNKNOWN"),
                    "license": str(location.get("license") or "UNKNOWN"),
                    "source": "OPENALEX",
                    "is_oa": str(bool(location.get("is_oa"))).lower(),
                }
            )
        authors = [
            str((row.get("author") or {}).get("display_name") or "")
            for row in work.get("authorships") or []
        ]
        primary_source = ((work.get("primary_location") or {}).get("source") or {})
        best = work.get("best_oa_location") or {}
        best_pdf = str(best.get("pdf_url") or "")
        if not best_pdf:
            best_pdf = next((row["pdf_url"] for row in locations if row.get("pdf_url") and row.get("is_oa") == "true"), "")
        return SourceCandidate(
            title=str(work.get("display_name") or ""),
            DOI=normalize_doi(work.get("doi")),
            authors=[value for value in authors if value],
            year=str(work.get("publication_year") or "UNKNOWN"),
            journal=str(primary_source.get("display_name") or "UNKNOWN"),
            source_database=["OPENALEX"],
            citation_count=int(work.get("cited_by_count") or 0),
            OA_status=str((work.get("open_access") or {}).get("oa_status") or "UNKNOWN"),
            OA_locations=locations,
            pdf_candidate_url=best_pdf,
            references=[str(value) for value in work.get("referenced_works") or []],
            source_record_ids=[str(work.get("id") or "")],
            version_provenance=[
                {
                    "source": "OPENALEX",
                    "version": str(row.get("version") or "UNKNOWN"),
                    "host_type": str(row.get("host_type") or "UNKNOWN"),
                    "license": str(row.get("license") or "UNKNOWN"),
                }
                for row in locations
            ],
            retrieval_provenance=[{"source": "OPENALEX", "method": method, "identifier": identifier}],
        )

    def _works(self, params: dict[str, Any], *, cache_key: str) -> list[SourceCandidate]:
        request_params = {**params, **self._auth_params()}
        payload = self._get_json_cached(
            "https://api.openalex.org/works",
            cache_key=cache_key,
            params=request_params,
        )
        method = "DOI_BATCH" if str(params.get("filter") or "").startswith("doi:") else "SEARCH"
        identifier = str(params.get("filter") or params.get("search") or "")
        return [self._candidate(work, method=method, identifier=identifier) for work in payload.get("results") or []]

    def resolve_dois(self, dois: Iterable[str]) -> dict[str, SourceCandidate]:
        unique = list(dict.fromkeys(normalize_doi(value) for value in dois if normalize_doi(value).startswith("10.")))
        found: dict[str, SourceCandidate] = {}
        for offset in range(0, len(unique), 100):
            batch = unique[offset:offset + 100]
            joined = "|".join(batch)
            rows = self._works(
                {"filter": f"doi:{joined}", "per_page": 100},
                cache_key=f"doi_batch:{joined}",
            )
            for row in rows:
                if row.DOI:
                    found[normalize_doi(row.DOI)] = row
        return found

    def resolve_doi(self, doi: str) -> SourceCandidate | None:
        return self.resolve_dois([doi]).get(normalize_doi(doi))

    def search(self, query: str, *, since: str = "", limit: int = 25) -> list[SourceCandidate]:
        normalized = normalize_doi(query)
        if normalized.startswith("10."):
            candidate = self.resolve_doi(normalized)
            return [candidate] if candidate else []
        params: dict[str, Any] = {"search": query, "per_page": min(100, limit)}
        if since:
            params["filter"] = f"from_publication_date:{since}"
        return self._works(params, cache_key=f"search:{query.casefold()}:{since}:{min(100, limit)}")

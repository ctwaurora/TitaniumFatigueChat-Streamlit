from __future__ import annotations

import os
from typing import Any

from .base import LiteratureSource, SourceCandidate, normalize_doi


class OpenAlexSource(LiteratureSource):
    name = "OPENALEX"

    def search(self, query: str, *, since: str = "", limit: int = 25) -> list[SourceCandidate]:
        params: dict[str, Any] = {"search": query, "per-page": min(100, limit)}
        filters = []
        if since:
            filters.append(f"from_publication_date:{since}")
        if filters:
            params["filter"] = ",".join(filters)
        if os.getenv("OPENALEX_API_KEY"):
            params["api_key"] = os.environ["OPENALEX_API_KEY"]
        if os.getenv("UNPAYWALL_EMAIL"):
            params["mailto"] = os.environ["UNPAYWALL_EMAIL"]
        data = self._get("https://api.openalex.org/works", params=params).json()
        output = []
        for work in data.get("results") or []:
            locations = []
            for loc in work.get("locations") or []:
                pdf = str(loc.get("pdf_url") or "")
                landing = str(loc.get("landing_page_url") or "")
                if pdf or landing:
                    locations.append({"pdf_url": pdf, "landing_page_url": landing})
            authors = [
                str((row.get("author") or {}).get("display_name") or "")
                for row in work.get("authorships") or []
            ]
            source = ((work.get("primary_location") or {}).get("source") or {})
            best = work.get("best_oa_location") or {}
            output.append(SourceCandidate(
                title=str(work.get("display_name") or ""), DOI=normalize_doi(work.get("doi")),
                authors=[x for x in authors if x], year=str(work.get("publication_year") or "UNKNOWN"),
                journal=str(source.get("display_name") or "UNKNOWN"), source_database=[self.name],
                citation_count=int(work.get("cited_by_count") or 0),
                OA_status=str((work.get("open_access") or {}).get("oa_status") or "UNKNOWN"),
                OA_locations=locations, pdf_candidate_url=str(best.get("pdf_url") or ""),
                references=[str(x) for x in work.get("referenced_works") or []],
                source_record_ids=[str(work.get("id") or "")],
            ))
        return output

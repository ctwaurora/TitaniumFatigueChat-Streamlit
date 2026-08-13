from __future__ import annotations

from .base import LiteratureSource, SourceCandidate, normalize_doi


class OSTISource(LiteratureSource):
    name = "OSTI"

    def search(self, query: str, *, since: str = "", limit: int = 25) -> list[SourceCandidate]:
        params = {"q": query, "rows": min(limit, 100)}
        if since:
            params["publication_date_start"] = since
        data = self._get("https://www.osti.gov/api/v1/records", params=params).json()
        rows = data if isinstance(data, list) else data.get("results") or data.get("records") or []
        output = []
        for work in rows:
            links = work.get("links") or []
            pdf = str(work.get("download_url") or work.get("fulltext_url") or "")
            output.append(SourceCandidate(
                title=str(work.get("title") or ""), DOI=normalize_doi(work.get("doi")),
                authors=[str(x) for x in work.get("authors") or []], year=str(work.get("publication_date") or "UNKNOWN")[:4],
                journal=str(work.get("journal_name") or work.get("product_type") or "UNKNOWN"), source_database=[self.name],
                OA_status="OFFICIAL_GOVERNMENT_FULLTEXT" if pdf else "METADATA_ONLY",
                OA_locations=[{"landing_page_url": str(work.get("osti_id") or ""), "links": str(links)}],
                pdf_candidate_url=pdf, source_record_ids=[str(work.get("osti_id") or "")],
            ))
        return output

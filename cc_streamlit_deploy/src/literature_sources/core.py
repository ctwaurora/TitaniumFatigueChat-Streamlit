from __future__ import annotations

import os

from .base import LiteratureSource, SourceCandidate, normalize_doi


class CORESource(LiteratureSource):
    name = "CORE"
    env_key = "CORE_API_KEY"

    def search(self, query: str, *, since: str = "", limit: int = 25) -> list[SourceCandidate]:
        if not self.configured:
            return []
        headers = {"Authorization": f"Bearer {os.environ[self.env_key]}"}
        data = self._get("https://api.core.ac.uk/v3/search/works", params={"q": query, "limit": min(limit, 100)}, headers=headers).json()
        output = []
        for work in data.get("results") or []:
            links = [str(x) for x in work.get("downloadUrl") or []] if isinstance(work.get("downloadUrl"), list) else [str(work.get("downloadUrl") or "")]
            pdf = next((x for x in links if x), "")
            output.append(SourceCandidate(
                title=str(work.get("title") or ""), DOI=normalize_doi(work.get("doi")),
                authors=[str(x) for x in work.get("authors") or []], year=str(work.get("yearPublished") or "UNKNOWN"),
                journal=str(work.get("journals") or "UNKNOWN"), source_database=[self.name],
                citation_count=int(work.get("citationCount") or 0), OA_status="CORE_FULLTEXT" if pdf else "METADATA_ONLY",
                pdf_candidate_url=pdf, source_record_ids=[str(work.get("id") or "")],
            ))
        return output

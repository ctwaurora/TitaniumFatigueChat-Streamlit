from __future__ import annotations

import os

from .base import LiteratureSource, SourceCandidate, normalize_doi


class WebOfScienceSource(LiteratureSource):
    name = "WEB_OF_SCIENCE"
    env_key = "WOS_API_KEY"

    def search(self, query: str, *, since: str = "", limit: int = 25) -> list[SourceCandidate]:
        if not self.configured:
            return []
        data = self._get("https://api.clarivate.com/apis/wos-starter/v1/documents", params={"q": f'TS=("{query}")', "limit": min(limit, 50)}, headers={"X-ApiKey": os.environ[self.env_key]}).json()
        output = []
        for work in data.get("hits") or []:
            output.append(SourceCandidate(
                title=str(work.get("title") or ""), DOI=normalize_doi((work.get("identifiers") or {}).get("doi")),
                authors=[str(x) for x in work.get("names", {}).get("authors") or []], year=str(work.get("source", {}).get("publishYear") or "UNKNOWN"),
                journal=str(work.get("source", {}).get("sourceTitle") or "UNKNOWN"), source_database=[self.name],
                citation_count=int(work.get("citations") or 0), OA_status="METADATA_ONLY", source_record_ids=[str(work.get("uid") or "")],
            ))
        return output

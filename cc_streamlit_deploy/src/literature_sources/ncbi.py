from __future__ import annotations

import os

from .base import LiteratureSource, SourceCandidate


class NCBISource(LiteratureSource):
    name = "NCBI"

    def search(self, query: str, *, since: str = "", limit: int = 25) -> list[SourceCandidate]:
        params = {"db": "pmc", "term": query, "retmode": "json", "retmax": min(limit, 100), "tool": "TitaniumFatigueChat"}
        if os.getenv("NCBI_API_KEY"):
            params["api_key"] = os.environ["NCBI_API_KEY"]
        ids = self._get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params).json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        summaries = self._get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", params={"db": "pmc", "id": ",".join(ids), "retmode": "json"}).json().get("result", {})
        output = []
        for uid in ids:
            work = summaries.get(uid) or {}
            output.append(SourceCandidate(
                title=str(work.get("title") or ""), authors=[str(x.get("name") or "") for x in work.get("authors") or []],
                year=str(work.get("pubdate") or "UNKNOWN")[:4], journal=str(work.get("fulljournalname") or "UNKNOWN"),
                source_database=[self.name], OA_status="PMC_OPEN_ACCESS",
                OA_locations=[{"landing_page_url": f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{uid}/"}],
                source_record_ids=[f"PMC{uid}"],
            ))
        return output

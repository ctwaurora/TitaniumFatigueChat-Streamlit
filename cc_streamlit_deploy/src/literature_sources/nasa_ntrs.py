from __future__ import annotations

from .base import LiteratureSource, SourceCandidate, normalize_doi


class NASANTRSSource(LiteratureSource):
    name = "NASA_NTRS"

    def search(self, query: str, *, since: str = "", limit: int = 25) -> list[SourceCandidate]:
        payload = {"terms": query, "page": {"size": min(limit, 100), "from": 0}}
        data = self.session.post("https://ntrs.nasa.gov/api/citations/search", json=payload, timeout=30)
        data.raise_for_status()
        rows = data.json().get("results") or []
        output = []
        for wrapper in rows:
            work = wrapper.get("citation") or wrapper
            downloads = work.get("downloads") or []
            pdf = next((str(x.get("links", {}).get("original") or x.get("url") or "") for x in downloads if str(x.get("mimetype") or x.get("type") or "").casefold() in {"application/pdf", "pdf"}), "")
            output.append(SourceCandidate(
                title=str(work.get("title") or ""), DOI=normalize_doi(work.get("doi")),
                authors=[str(x.get("name") if isinstance(x, dict) else x) for x in work.get("authorAffiliations") or work.get("authors") or []],
                year=str(work.get("publicationDate") or "UNKNOWN")[:4], journal=str(work.get("stiType") or "NASA Technical Report"),
                source_database=[self.name], OA_status="OFFICIAL_GOVERNMENT_FULLTEXT" if pdf else "METADATA_ONLY",
                pdf_candidate_url=pdf, source_record_ids=[str(work.get("id") or "")],
            ))
        return output

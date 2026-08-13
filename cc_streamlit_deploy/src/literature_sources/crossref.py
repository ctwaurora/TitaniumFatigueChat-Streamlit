from __future__ import annotations

from typing import Any

from .base import LiteratureSource, SourceCandidate, normalize_doi


class CrossrefSource(LiteratureSource):
    name = "CROSSREF"

    def search(self, query: str, *, since: str = "", limit: int = 25) -> list[SourceCandidate]:
        params: dict[str, Any] = {"query.bibliographic": query, "rows": min(100, limit), "select": "DOI,title,author,published,container-title,is-referenced-by-count,link,URL,reference"}
        if since:
            params["filter"] = f"from-pub-date:{since}"
        items = self._get("https://api.crossref.org/works", params=params).json().get("message", {}).get("items", [])
        output = []
        for work in items:
            date_parts = ((work.get("published") or {}).get("date-parts") or [["UNKNOWN"]])[0]
            authors = [" ".join(filter(None, (a.get("given"), a.get("family")))) for a in work.get("author") or []]
            links = [{"pdf_url": str(x.get("URL") or ""), "content_type": str(x.get("content-type") or "")} for x in work.get("link") or []]
            pdf = next((x["pdf_url"] for x in links if "pdf" in x["content_type"].casefold()), "")
            refs = [normalize_doi(x.get("DOI")) for x in work.get("reference") or [] if x.get("DOI")]
            output.append(SourceCandidate(
                title=str((work.get("title") or [""])[0]), DOI=normalize_doi(work.get("DOI")),
                authors=[x for x in authors if x], year=str(date_parts[0]),
                journal=str((work.get("container-title") or ["UNKNOWN"])[0]), source_database=[self.name],
                citation_count=int(work.get("is-referenced-by-count") or 0),
                OA_status="POSSIBLE_OA" if pdf else "UNKNOWN", OA_locations=links,
                pdf_candidate_url=pdf, references=refs, source_record_ids=[str(work.get("URL") or "")],
            ))
        return output

from __future__ import annotations

import os

from .base import LiteratureSource, SourceCandidate, normalize_doi


class UnpaywallSource(LiteratureSource):
    name = "UNPAYWALL"
    env_key = "UNPAYWALL_EMAIL"

    def search(self, query: str, *, since: str = "", limit: int = 25) -> list[SourceCandidate]:
        # Unpaywall is a DOI resolver, not a keyword search API.
        doi = normalize_doi(query)
        if not self.configured or not doi.startswith("10."):
            return []
        work = self._get(f"https://api.unpaywall.org/v2/{doi}", params={"email": os.environ[self.env_key]}).json()
        locations = [{"pdf_url": str(x.get("url_for_pdf") or ""), "landing_page_url": str(x.get("url_for_landing_page") or ""), "host_type": str(x.get("host_type") or "")} for x in work.get("oa_locations") or []]
        best = work.get("best_oa_location") or {}
        return [SourceCandidate(
            title=str(work.get("title") or ""), DOI=doi,
            authors=[str(x.get("family") or x.get("given") or "") for x in work.get("z_authors") or []],
            year=str(work.get("year") or "UNKNOWN"), journal=str(work.get("journal_name") or "UNKNOWN"),
            source_database=[self.name], OA_status=str(work.get("oa_status") or "UNKNOWN"),
            OA_locations=locations, pdf_candidate_url=str(best.get("url_for_pdf") or ""),
            source_record_ids=[str(work.get("doi_url") or "")],
        )]

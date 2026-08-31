from __future__ import annotations

from typing import Any

from .base import LiteratureSource, SourceCandidate, normalize_doi, secret_value


VERSION_PRIORITY = {
    "publishedversion": 0,
    "version_of_record": 0,
    "acceptedversion": 1,
    "accepted_manuscript": 1,
    "submittedversion": 2,
    "submitted_manuscript": 2,
}


def _normalized_version(value: Any) -> str:
    text = str(value or "UNKNOWN").strip()
    mapping = {
        "publishedVersion": "VERSION_OF_RECORD",
        "acceptedVersion": "ACCEPTED_MANUSCRIPT",
        "submittedVersion": "SUBMITTED_MANUSCRIPT",
    }
    return mapping.get(text, text.upper() if text else "UNKNOWN")


class UnpaywallSource(LiteratureSource):
    name = "UNPAYWALL"
    env_key = "UNPAYWALL_EMAIL"
    max_retries = 2
    backoff_base_seconds = 1.0

    def resolve_doi(self, doi: str) -> SourceCandidate | None:
        normalized = normalize_doi(doi)
        email = secret_value(self.env_key)
        if not email or "@" not in email or not normalized.startswith("10."):
            return None
        work = self._get_json_cached(
            f"https://api.unpaywall.org/v2/{normalized}",
            cache_key=f"doi:{normalized}",
            params={"email": email},
        )
        locations: list[dict[str, str]] = []
        for location in work.get("oa_locations") or []:
            pdf = str(location.get("url_for_pdf") or "")
            landing = str(location.get("url_for_landing_page") or "")
            if not (pdf or landing):
                continue
            locations.append(
                {
                    "pdf_url": pdf,
                    "landing_page_url": landing,
                    "host_type": str(location.get("host_type") or "UNKNOWN"),
                    "version": _normalized_version(location.get("version")),
                    "license": str(location.get("license") or "UNKNOWN"),
                    "source": self.name,
                    "endpoint_id": str(location.get("endpoint_id") or ""),
                }
            )
        locations.sort(
            key=lambda row: (
                VERSION_PRIORITY.get(str(row.get("version") or "").casefold(), 9),
                not bool(row.get("pdf_url")),
                str(row.get("host_type") or ""),
            )
        )
        best = next((row for row in locations if row.get("pdf_url")), {})
        return SourceCandidate(
            title=str(work.get("title") or ""),
            DOI=normalized,
            authors=[
                " ".join(
                    part for part in (str(row.get("given") or "").strip(), str(row.get("family") or "").strip())
                    if part
                )
                for row in work.get("z_authors") or []
            ],
            year=str(work.get("year") or "UNKNOWN"),
            journal=str(work.get("journal_name") or "UNKNOWN"),
            source_database=[self.name],
            OA_status=str(work.get("oa_status") or "UNKNOWN"),
            OA_locations=locations,
            pdf_candidate_url=str(best.get("pdf_url") or ""),
            source_record_ids=[str(work.get("doi_url") or "")],
            version_provenance=[
                {
                    "source": self.name,
                    "version": str(row.get("version") or "UNKNOWN"),
                    "host_type": str(row.get("host_type") or "UNKNOWN"),
                    "license": str(row.get("license") or "UNKNOWN"),
                }
                for row in locations
            ],
            retrieval_provenance=[{"source": self.name, "method": "DOI_RESOLUTION", "identifier": normalized}],
        )

    def search(self, query: str, *, since: str = "", limit: int = 25) -> list[SourceCandidate]:
        # The recovery workflow uses the official DOI endpoint; Unpaywall title
        # search is deliberately not used for identity matching.
        candidate = self.resolve_doi(query)
        return [candidate] if candidate else []

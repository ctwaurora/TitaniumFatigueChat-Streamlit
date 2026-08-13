"""Shared contracts for legal scholarly metadata connectors."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

import requests


USER_AGENT = "TitaniumFatigueChat/1.1 literature-maintenance (scholarly research)"


@dataclass
class SourceCandidate:
    candidate_id: str = ""
    title: str = ""
    DOI: str = ""
    authors: list[str] = field(default_factory=list)
    year: str = "UNKNOWN"
    journal: str = "UNKNOWN"
    source_database: list[str] = field(default_factory=list)
    citation_count: int = 0
    OA_status: str = "UNKNOWN"
    OA_locations: list[dict[str, str]] = field(default_factory=list)
    pdf_candidate_url: str = ""
    references: list[str] = field(default_factory=list)
    cited_by: list[str] = field(default_factory=list)
    topic: list[str] = field(default_factory=list)
    tier_candidate: str = "UNASSIGNED"
    retrieval_score: float = 0.0
    lifecycle_state: str = "DISCOVERED"
    source_record_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_doi(value: Any) -> str:
    text = str(value or "").strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.rstrip(" .")


def author_names(values: Any) -> list[str]:
    output: list[str] = []
    for value in values or []:
        if isinstance(value, str):
            name = value
        else:
            name = str(value.get("name") or value.get("display_name") or "")
        if name.strip() and name.strip() not in output:
            output.append(name.strip())
    return output


class LiteratureSource:
    name = "BASE"
    env_key = ""

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        # Scholarly APIs are public HTTPS endpoints.  A stale desktop proxy
        # must not silently turn discovery into an empty result set.
        self.session.trust_env = False
        self.session.headers.setdefault("User-Agent", USER_AGENT)

    @property
    def configured(self) -> bool:
        return not self.env_key or bool(os.getenv(self.env_key))

    def search(self, query: str, *, since: str = "", limit: int = 25) -> list[SourceCandidate]:
        raise NotImplementedError

    def _get(self, url: str, **kwargs: Any) -> requests.Response:
        response = self.session.get(url, timeout=30, **kwargs)
        response.raise_for_status()
        return response

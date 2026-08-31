"""Recover legal full-text locations for an existing v1.1 candidate registry."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

import requests

from src.literature_discovery import (
    LiteratureDiscoveryManager,
    assign_tier,
    legal_pdf_candidate,
    trusted_oa_pdf_location,
)
from src.literature_sources import CORESource, OpenAlexSource, OSTISource, SourceCandidate, UnpaywallSource
from src.literature_sources.base import normalize_doi


SOURCE_CANDIDATE_FIELDS = {field.name for field in fields(SourceCandidate)}


def candidate_from_dict(value: dict[str, Any]) -> SourceCandidate:
    selected = {key: value[key] for key in SOURCE_CANDIDATE_FIELDS if key in value}
    candidate = SourceCandidate(**selected)
    candidate.tier_candidate = candidate.tier_candidate or assign_tier(candidate.title)
    return candidate


def load_candidate_registry(path: Path) -> list[SourceCandidate]:
    rows: list[SourceCandidate] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(candidate_from_dict(value))
    return rows


def has_legal_fulltext(candidate: SourceCandidate) -> bool:
    return legal_pdf_candidate(candidate.pdf_candidate_url) or any(
        trusted_oa_pdf_location(location) for location in candidate.OA_locations
    )


class CandidateRecoveryManager:
    def __init__(
        self,
        *,
        cache_dir: Path,
        session: requests.Session | None = None,
        unpaywall: UnpaywallSource | None = None,
        core: CORESource | None = None,
        openalex: OpenAlexSource | None = None,
        osti: OSTISource | None = None,
    ) -> None:
        self.session = session or requests.Session()
        self.session.trust_env = False
        self.unpaywall = unpaywall or UnpaywallSource(self.session, cache_dir=cache_dir)
        self.core = core or CORESource(self.session, cache_dir=cache_dir)
        self.openalex = openalex or OpenAlexSource(self.session, cache_dir=cache_dir)
        self.osti = osti or OSTISource(self.session, cache_dir=cache_dir)
        self.merge = LiteratureDiscoveryManager._merge

    @staticmethod
    def _high_value(rows: Iterable[SourceCandidate]) -> list[SourceCandidate]:
        return [
            row for row in rows
            if row.tier_candidate != "OUT_OF_SCOPE" and normalize_doi(row.DOI).startswith("10.") and not has_legal_fulltext(row)
        ]

    def recover(self, rows: list[SourceCandidate]) -> tuple[list[SourceCandidate], list[dict[str, Any]]]:
        targets = self._high_value(rows)
        target_dois = [normalize_doi(row.DOI) for row in targets]
        openalex_results: dict[str, SourceCandidate] = {}
        openalex_error = ""
        try:
            openalex_results = self.openalex.resolve_dois(target_dois)
        except Exception as exc:
            openalex_error = f"{type(exc).__name__}:{str(exc)[:180]}"

        status_rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(targets, 1):
            doi = normalize_doi(candidate.DOI)
            status: dict[str, Any] = {
                "candidate_id": candidate.candidate_id,
                "DOI": doi,
                "previous_status": candidate.lifecycle_state,
                "previous_legal_fulltext": False,
                "Unpaywall_status": "NOT_CONFIGURED" if not self.unpaywall.configured else "NOT_FOUND",
                "CORE_status": "NOT_FOUND",
                "OpenAlex_status": "ERROR" if openalex_error else "NOT_FOUND",
                "OSTI_status": "NOT_APPLICABLE",
                "OA_locations": [],
                "download_status": "NOT_ATTEMPTED",
                "final_status": "NO_LEGAL_FULLTEXT_FOUND",
            }
            if self.unpaywall.configured:
                try:
                    value = self.unpaywall.resolve_doi(doi)
                    if value:
                        candidate = self.merge(candidate, value)
                        status["Unpaywall_status"] = "OA_LOCATION_FOUND" if has_legal_fulltext(value) else "METADATA_ONLY"
                except Exception as exc:
                    status["Unpaywall_status"] = f"ERROR:{type(exc).__name__}"

            value = openalex_results.get(doi)
            if value:
                candidate = self.merge(candidate, value)
                status["OpenAlex_status"] = "OA_LOCATION_FOUND" if has_legal_fulltext(value) else "METADATA_ONLY"

            try:
                value = self.core.resolve_doi(doi)
                if value:
                    candidate = self.merge(candidate, value)
                    status["CORE_status"] = "OA_LOCATION_FOUND" if has_legal_fulltext(value) else "METADATA_ONLY"
            except Exception as exc:
                status["CORE_status"] = f"ERROR:{type(exc).__name__}"

            # OSTI is an auxiliary source and is only queried when prior
            # metadata indicates a DOE/OSTI relationship.
            if "OSTI" in candidate.source_database or any(
                marker in candidate.title.casefold() for marker in ("national laboratory", "doe ", "department of energy")
            ):
                status["OSTI_status"] = "NOT_FOUND"
                try:
                    matches = self.osti.search(doi, limit=5)
                    value = next((row for row in matches if normalize_doi(row.DOI) == doi), None)
                    if value:
                        candidate = self.merge(candidate, value)
                        status["OSTI_status"] = "OA_LOCATION_FOUND" if has_legal_fulltext(value) else "METADATA_ONLY"
                except Exception as exc:
                    status["OSTI_status"] = f"ERROR:{type(exc).__name__}"

            status["OA_locations"] = [
                {
                    "source": str(location.get("source") or "UNKNOWN"),
                    "host_type": str(location.get("host_type") or "UNKNOWN"),
                    "version": str(location.get("version") or "UNKNOWN"),
                    "has_pdf": bool(location.get("pdf_url")),
                }
                for location in candidate.OA_locations
                if trusted_oa_pdf_location(location)
            ]
            status["final_status"] = "LEGAL_FULLTEXT_LOCATED" if has_legal_fulltext(candidate) else "NO_LEGAL_FULLTEXT_FOUND"
            status_rows.append(status)
            if index % 20 == 0:
                print(f"RECOVERY_PROGRESS {index}/{len(targets)}", flush=True)
        return rows, status_rows


def write_recovery_registry(path: Path, rows: Iterable[SourceCandidate]) -> None:
    path.write_text(
        "".join(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

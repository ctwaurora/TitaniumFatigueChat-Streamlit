"""Evidence-gap-driven discovery for the v1.1 candidate intake queue.

This module never promotes a work into the formal corpus.  It only discovers,
merges, validates public metadata, and optionally stages legal PDF candidates.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests

from src.literature_sources import (
    CORESource, CrossrefSource, NASANTRSSource, NCBISource, OpenAlexSource,
    OSTISource, SourceCandidate, UnpaywallSource, WebOfScienceSource,
)
from src.literature_sources.base import normalize_doi


LIFECYCLE = (
    "DISCOVERED", "METADATA_CROSSCHECKED", "DOI_VERIFIED", "OA_VERIFIED",
    "PDF_DOWNLOADED", "PDF_VALIDATED", "DEDUPLICATED", "FULLTEXT_SCREENED",
    "TIER_ASSIGNED", "EVIDENCE_EXTRACTED", "CONDITION_EXTRACTED", "FORMAL_ACCEPTED",
)

TOPIC_QUERIES: dict[str, list[str]] = {
    "defect": ["Ti-6Al-4V additive manufacturing fatigue defect size location", "L-PBF Ti64 pore lack of fusion fatigue XCT"],
    "surface": ["L-PBF Ti-6Al-4V surface roughness fatigue as-built machined", "additive Ti64 surface treatment fatigue initiation"],
    "hip": ["L-PBF Ti-6Al-4V hot isostatic pressing HIP fatigue", "Ti64 HIP surface condition fatigue interaction"],
    "heat_treatment": ["additive Ti-6Al-4V heat treatment microstructure fatigue", "L-PBF Ti64 alpha prime fatigue annealing"],
    "short_crack": ["Ti-6Al-4V short small crack fatigue growth closure", "additive Ti64 short crack long crack transition"],
    "long_crack": ["Ti-6Al-4V fatigue crack growth threshold Paris regime", "L-PBF Ti64 Delta K threshold crack closure"],
    "residual_stress": ["L-PBF Ti-6Al-4V residual stress fatigue relaxation", "additive Ti64 residual stress crack growth XRD"],
    "hcf_vhcf": ["L-PBF Ti-6Al-4V high cycle very high cycle fatigue", "Ti64 internal surface initiation VHCF"],
    "orientation_texture": ["L-PBF Ti-6Al-4V build orientation texture EBSD fatigue", "additive Ti64 crack path texture fatigue EBSD"],
    "stress_ratio": ["Ti-6Al-4V fatigue stress ratio R crack closure", "L-PBF Ti64 stress ratio fatigue crack growth"],
}

LEGAL_FULLTEXT_HOSTS = (
    "osti.gov", "nasa.gov", "ntrs.nasa.gov", "nist.gov", "ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov", "core.ac.uk", "springer.com", "link.springer.com",
    "sciencedirect.com", "wiley.com", "mdpi.com", "plos.org", "frontiersin.org",
    "zenodo.org", "figshare.com", "hal.science", ".edu", ".ac.uk", "repository",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def transition(current: str, target: str) -> str:
    if current not in LIFECYCLE or target not in LIFECYCLE:
        raise ValueError("UNKNOWN_LITERATURE_LIFECYCLE_STATE")
    if LIFECYCLE.index(target) != LIFECYCLE.index(current) + 1:
        raise ValueError(f"INVALID_LIFECYCLE_TRANSITION:{current}->{target}")
    return target


def assign_tier(title: str) -> str:
    text = str(title or "").casefold()
    ti64 = bool(re.search(r"ti[- ]?6al[- ]?4v|ti64|tc4", text))
    lpbf = bool(re.search(r"l[- ]?pbf|pbf[- ]?lb/?m|laser powder bed|selective laser melting|\bslm\b", text))
    near = bool(re.search(r"electron beam|\bebm\b|\bsebm\b|directed energy|\bded\b|waam|additive", text))
    mechanics = bool(re.search(r"short crack|small crack|crack closure|delta.?k|threshold|paris|kitagawa|el haddad|murakami|residual stress", text))
    if ti64 and lpbf:
        return "TIER1_CORE_DIRECT"
    if ti64 and near:
        return "TIER2_NEAR_DOMAIN"
    if ti64 or mechanics:
        return "TIER3_FOUNDATIONAL_MECHANICS"
    return "OUT_OF_SCOPE"


def legal_pdf_candidate(url: str) -> bool:
    parsed = urlparse(str(url or ""))
    host = parsed.netloc.casefold()
    return parsed.scheme == "https" and bool(host) and any(x in host for x in LEGAL_FULLTEXT_HOSTS)


class LiteratureDiscoveryManager:
    def __init__(self, *, sources: Iterable[Any] | None = None, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.trust_env = False
        self.sources = list(sources) if sources is not None else [
            OpenAlexSource(self.session), CrossrefSource(self.session), UnpaywallSource(self.session),
            CORESource(self.session), OSTISource(self.session), NASANTRSSource(self.session),
            NCBISource(self.session), WebOfScienceSource(self.session),
        ]
        self.source_status: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _key(row: SourceCandidate) -> str:
        return normalize_doi(row.DOI) or normalize_title(row.title)

    @staticmethod
    def _merge(left: SourceCandidate, right: SourceCandidate) -> SourceCandidate:
        for source in right.source_database:
            if source not in left.source_database:
                left.source_database.append(source)
        for field in ("authors", "OA_locations", "references", "cited_by", "source_record_ids", "topic"):
            current = getattr(left, field)
            for value in getattr(right, field):
                if value not in current:
                    current.append(value)
        for field in ("title", "DOI", "year", "journal", "pdf_candidate_url"):
            old, new = getattr(left, field), getattr(right, field)
            if old in ("", "UNKNOWN") and new not in ("", "UNKNOWN"):
                setattr(left, field, new)
        left.citation_count = max(left.citation_count, right.citation_count)
        if left.OA_status == "UNKNOWN" and right.OA_status != "UNKNOWN":
            left.OA_status = right.OA_status
        return left

    def discover(self, *, topic: str = "all", since: str = "", max_candidates: int = 300) -> list[SourceCandidate]:
        topics = TOPIC_QUERIES if topic == "all" else {topic: TOPIC_QUERIES[topic]}
        merged: dict[str, SourceCandidate] = {}
        query_count = sum(len(x) for x in topics.values())
        per_query = max(20, min(100, max_candidates // max(1, query_count) + 8))
        for source in self.sources:
            name = str(getattr(source, "name", type(source).__name__))
            if not getattr(source, "configured", True):
                self.source_status[name] = {"status": "SKIPPED_NOT_CONFIGURED", "results": 0}
                continue
            total, errors, consecutive_failures = 0, [], 0
            for topic_name, queries in topics.items():
                for query in queries:
                    if consecutive_failures >= 2:
                        break
                    try:
                        rows = source.search(query, since=since, limit=per_query)
                    except Exception as exc:
                        errors.append(f"{type(exc).__name__}:{str(exc)[:180]}")
                        consecutive_failures += 1
                        continue
                    consecutive_failures = 0
                    total += len(rows)
                    for row in rows:
                        row.topic = sorted(set(row.topic + [topic_name]))
                        row.tier_candidate = assign_tier(row.title)
                        row.retrieval_score = round(min(1.0, .25 + .10 * len(row.source_database) + .08 * len(row.topic) + .20 * (row.tier_candidate != "OUT_OF_SCOPE") + .12 * bool(row.DOI) + .12 * legal_pdf_candidate(row.pdf_candidate_url)), 4)
                        key = self._key(row)
                        if key:
                            merged[key] = self._merge(merged[key], row) if key in merged else row
                if consecutive_failures >= 2:
                    break
            self.source_status[name] = {"status": "OK" if not errors else "CIRCUIT_OPEN" if consecutive_failures >= 2 else "PARTIAL_ERROR", "results": total, "errors": sorted(set(errors))}
        ranked = sorted(merged.values(), key=lambda row: (-row.retrieval_score, -row.citation_count, normalize_title(row.title)))[:max_candidates]
        for index, row in enumerate(ranked, 1):
            row.candidate_id = f"V11-DISC-{index:04d}"
            if len(row.source_database) >= 2:
                row.lifecycle_state = "METADATA_CROSSCHECKED"
        return ranked

    def stage_legal_pdfs(self, rows: Iterable[SourceCandidate], incoming: Path, *, max_downloads: int = 25) -> list[dict[str, Any]]:
        incoming.mkdir(parents=True, exist_ok=True)
        attempts: list[dict[str, Any]] = []
        for row in rows:
            if len([x for x in attempts if x.get("status") == "PDF_VALIDATED"]) >= max_downloads:
                break
            url = row.pdf_candidate_url
            if row.tier_candidate == "OUT_OF_SCOPE" or not legal_pdf_candidate(url):
                continue
            item = {"candidate_id": row.candidate_id, "source_database": row.source_database, "status": "DOWNLOAD_FAILED", "reason": ""}
            try:
                response = self.session.get(url, timeout=45, allow_redirects=True)
                response.raise_for_status()
                payload = response.content
                if not payload.startswith(b"%PDF-") or len(payload) < 4096:
                    raise ValueError("NOT_A_VALID_PDF_PAYLOAD")
                digest = hashlib.sha256(payload).hexdigest()
                target = incoming / f"{row.candidate_id}_{digest[:12]}.pdf"
                target.write_bytes(payload)
                row.lifecycle_state = "PDF_VALIDATED"
                item.update({"status": "PDF_VALIDATED", "sha256": digest, "bytes": len(payload), "local_candidate_file": target.name})
            except Exception as exc:
                item["reason"] = f"{type(exc).__name__}:{str(exc)[:180]}"
            attempts.append(item)
        return attempts


def literature_expansion_suggestion(question: str, retrieval: dict[str, Any]) -> dict[str, Any] | None:
    sufficiency = str(retrieval.get("evidence_sufficiency") or retrieval.get("status") or "").upper()
    evidence_count = len(retrieval.get("results") or retrieval.get("evidence") or [])
    if evidence_count >= 3 and sufficiency not in {"INSUFFICIENT", "EVIDENCE_SPARSE"}:
        return None
    topics = [name for name, queries in TOPIC_QUERIES.items() if any(term in question.casefold() for query in queries for term in re.findall(r"[a-z]{4,}", query.casefold()))]
    return {
        "type": "LITERATURE_EXPANSION_SUGGESTION",
        "current_answer_uses_formal_rag_only": True,
        "background_download_started": False,
        "suggested_topics": sorted(set(topics)) or ["all"],
        "reason": "CURRENT_FORMAL_EVIDENCE_INSUFFICIENT",
    }


def write_discovery_audit(output: Path, rows: list[SourceCandidate], source_status: dict[str, Any], attempts: list[dict[str, Any]], *, dry_run: bool) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = output / "candidate_registry.jsonl"
    candidate_path.write_text("".join(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (output / "download_attempts.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in attempts), encoding="utf-8")
    summary = {
        "schema_version": "tfc-literature-expansion-v1.1",
        "generated_at": _utc_now(), "dry_run": dry_run,
        "formal_acceptance_performed": False,
        "candidate_count": len(rows),
        "tier_counts": dict(Counter(row.tier_candidate for row in rows)),
        "state_counts": dict(Counter(row.lifecycle_state for row in rows)),
        "legal_pdf_candidates": sum(legal_pdf_candidate(row.pdf_candidate_url) for row in rows),
        "pdf_validated": sum(row.get("status") == "PDF_VALIDATED" for row in attempts),
        "source_status": source_status,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary

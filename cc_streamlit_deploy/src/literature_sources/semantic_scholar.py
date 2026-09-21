"""Semantic Scholar Academic Graph connector for candidate discovery.

This module intentionally exposes discovery and citation-network operations
only.  It cannot promote papers into a formal registry or RAG index.
"""

from __future__ import annotations

import json
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .base import LiteratureSource, SourceCandidate, USER_AGENT, normalize_doi, secret_value


API_ROOT = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS = ",".join((
    "paperId", "corpusId", "title", "authors", "year", "venue",
    "publicationTypes", "citationCount", "referenceCount", "externalIds",
    "isOpenAccess", "openAccessPdf", "url",
))


class SemanticScholarConnector(LiteratureSource):
    name = "SEMANTIC_SCHOLAR"
    env_key = ""
    max_retries = 4
    backoff_base_seconds = 2.0
    min_request_interval_seconds = 1.0
    circuit_failure_threshold = 2
    circuit_cooldown_seconds = 300.0
    request_timeout_seconds = 45.0

    @property
    def api_key_configured(self) -> bool:
        return bool(secret_value("SEMANTIC_SCHOLAR_API_KEY"))

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        api_key = secret_value("SEMANTIC_SCHOLAR_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    def _json(self, path: str, *, params: dict[str, object] | None = None, cache_key: str) -> dict[str, Any]:
        cached = self._read_cache(cache_key)
        if isinstance(cached, dict):
            return cached
        if self._clock() < self._circuit_open_until:
            raise RuntimeError(f"{self.name}_CIRCUIT_COOLDOWN_ACTIVE")
        url = f"{API_ROOT}{path}"
        if params:
            url = f"{url}?{urlencode(params)}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_slot()
            try:
                with urlopen(Request(url, headers=self._headers()), timeout=self.request_timeout_seconds) as response:
                    self._last_request_at = self._clock()
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("SEMANTIC_SCHOLAR_RESPONSE_NOT_OBJECT")
                self._consecutive_failures = 0
                self._circuit_open_until = 0.0
                self._write_cache(cache_key, payload)
                return payload
            except (HTTPError, URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                retry_after = None
                if isinstance(exc, HTTPError):
                    raw = str(exc.headers.get("Retry-After") or "").strip()
                    try:
                        retry_after = max(0.0, float(raw)) if raw else None
                    except ValueError:
                        retry_after = None
                    if exc.code not in self.retry_statuses:
                        break
                if attempt < self.max_retries:
                    self._sleep(retry_after if retry_after is not None else self.backoff_base_seconds * (2 ** attempt) + self._random_value())
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.circuit_failure_threshold:
            self._circuit_open_until = self._clock() + self.circuit_cooldown_seconds
        if last_error:
            raise last_error
        raise RuntimeError(f"{self.name}_REQUEST_FAILED")

    @staticmethod
    def _candidate(work: dict[str, Any], *, method: str, identifier: str) -> SourceCandidate:
        external = work.get("externalIds") or {}
        pdf = work.get("openAccessPdf") or {}
        paper_id = str(work.get("paperId") or "")
        corpus_id = str(work.get("corpusId") or external.get("CorpusId") or "")
        arxiv_id = str(external.get("ArXiv") or "")
        doi = normalize_doi(external.get("DOI") or "")
        pdf_url = str(pdf.get("url") or "")
        location = {
            "pdf_url": pdf_url,
            "landing_page_url": str(work.get("url") or ""),
            "host_type": "semantic_scholar_resolved_oa",
            "version": "UNKNOWN",
            "license": str(pdf.get("license") or "UNKNOWN"),
            "source": "SEMANTIC_SCHOLAR",
            "is_oa": str(bool(work.get("isOpenAccess") or pdf_url)).lower(),
        }
        return SourceCandidate(
            candidate_id=f"S2-{paper_id}" if paper_id else "",
            title=str(work.get("title") or "").strip(),
            DOI=doi,
            authors=[str(author.get("name") or "").strip() for author in work.get("authors") or [] if str(author.get("name") or "").strip()],
            year=str(work.get("year") or "UNKNOWN"),
            journal=str(work.get("venue") or "UNKNOWN"),
            source_database=["SEMANTIC_SCHOLAR"],
            citation_count=int(work.get("citationCount") or 0),
            reference_count=int(work.get("referenceCount") or 0),
            OA_status="OPEN_ACCESS" if work.get("isOpenAccess") or pdf_url else "UNKNOWN",
            OA_locations=[location] if pdf_url else [],
            pdf_candidate_url=pdf_url,
            lifecycle_state="DISCOVERED",
            source_record_ids=[paper_id] if paper_id else [],
            retrieval_provenance=[{"source": "SEMANTIC_SCHOLAR", "method": method, "identifier": identifier}],
            arxiv_id=arxiv_id,
            publication_stage="PREPRINT" if arxiv_id and not doi else "UNKNOWN",
            semantic_scholar_id=paper_id,
            corpus_id=corpus_id,
            publication_types=[str(value) for value in work.get("publicationTypes") or []],
        )

    def search(self, query: str, *, since: str = "", limit: int = 25) -> list[SourceCandidate]:
        params: dict[str, object] = {"query": query, "limit": max(1, min(100, limit)), "fields": PAPER_FIELDS}
        if since[:4].isdigit():
            params["year"] = f"{since[:4]}-"
        payload = self._json("/paper/search", params=params, cache_key=f"search:{query.casefold()}:{since}:{params['limit']}")
        return [self._candidate(row, method="SEARCH", identifier=query) for row in payload.get("data") or []]

    def search_bulk(self, query: str, *, token: str = "", limit: int = 1000) -> tuple[list[SourceCandidate], str]:
        params: dict[str, object] = {"query": query, "fields": PAPER_FIELDS}
        if token:
            params["token"] = token
        payload = self._json("/paper/search/bulk", params=params, cache_key=f"bulk:{query.casefold()}:{token}")
        rows = [self._candidate(row, method="BULK_SEARCH", identifier=query) for row in (payload.get("data") or [])[:max(1, limit)]]
        return rows, str(payload.get("token") or "")

    def paper_detail(self, paper_id: str) -> SourceCandidate | None:
        payload = self._json(f"/paper/{quote(paper_id, safe=':')}", params={"fields": PAPER_FIELDS}, cache_key=f"detail:{paper_id}")
        return self._candidate(payload, method="DETAIL", identifier=paper_id) if payload.get("paperId") else None

    def _links(self, paper_id: str, relation: str, *, limit: int = 100) -> list[SourceCandidate]:
        payload = self._json(
            f"/paper/{quote(paper_id, safe=':')}/{relation}",
            params={"limit": max(1, min(1000, limit)), "fields": PAPER_FIELDS},
            cache_key=f"{relation}:{paper_id}:{limit}",
        )
        key = "citedPaper" if relation == "references" else "citingPaper"
        return [self._candidate(row.get(key) or {}, method=relation.upper(), identifier=paper_id) for row in payload.get("data") or [] if row.get(key)]

    def references(self, paper_id: str, *, limit: int = 100) -> list[SourceCandidate]:
        return self._links(paper_id, "references", limit=limit)

    def citations(self, paper_id: str, *, limit: int = 100) -> list[SourceCandidate]:
        return self._links(paper_id, "citations", limit=limit)

    def resolve_identifiers(self, identifiers: Iterable[str]) -> list[SourceCandidate]:
        output: list[SourceCandidate] = []
        for identifier in identifiers:
            candidate = self.paper_detail(identifier)
            if candidate:
                output.append(candidate)
        return output


SemanticScholarSource = SemanticScholarConnector

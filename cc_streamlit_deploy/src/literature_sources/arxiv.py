"""Official arXiv API connector for candidate discovery only.

The connector deliberately emits PREPRINT candidates.  It has no promotion or
RAG-ingestion capability; downloaded PDFs must still pass the existing formal
literature gates before a later v1.2 workflow may accept them.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .base import LiteratureSource, SourceCandidate, USER_AGENT, normalize_doi


ARXIV_API_URL = "https://export.arxiv.org/api/query"

ARXIV_TOPIC_QUERIES: dict[str, str] = {
    "Ti-6Al-4V fatigue": 'all:"Ti-6Al-4V" AND all:fatigue',
    "additive manufacturing titanium fatigue": 'all:"additive manufacturing" AND all:titanium AND all:fatigue',
    "LPBF titanium": '(all:LPBF OR all:"laser powder bed fusion") AND all:titanium',
    "fatigue crack growth titanium": 'all:"fatigue crack growth" AND all:titanium',
    "residual stress fatigue": 'all:"residual stress" AND all:fatigue',
    "machine learning fatigue prediction": 'all:"machine learning" AND all:"fatigue prediction"',
}

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _clean_text(value: str | None) -> str:
    return " ".join(str(value or "").split())


def _arxiv_id(entry_id: str) -> str:
    value = str(entry_id or "").rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", value, flags=re.I)


class ArxivConnector(LiteratureSource):
    """Discover arXiv preprints without granting formal-corpus status."""

    name = "ARXIV"
    max_retries = 3
    backoff_base_seconds = 2.0
    min_request_interval_seconds = 3.0
    circuit_failure_threshold = 2
    circuit_cooldown_seconds = 180.0
    request_timeout_seconds = 45.0

    def _get_atom_cached(self, *, cache_key: str, params: dict[str, object]) -> str:
        """Fetch Atom XML with caching and backoff.

        urllib is intentional here: the Windows desktop runtime exposes its
        managed HTTPS route through the OS networking layer, while requests'
        proxy discovery can resolve that route as a nonexistent file socket.
        """
        cached = self._read_cache(cache_key)
        if isinstance(cached, str):
            return cached
        if self._clock() < self._circuit_open_until:
            raise RuntimeError(f"{self.name}_CIRCUIT_COOLDOWN_ACTIVE")

        url = f"{ARXIV_API_URL}?{urlencode(params)}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_slot()
            try:
                request = Request(
                    url,
                    headers={"User-Agent": USER_AGENT, "Accept": "application/atom+xml"},
                )
                with urlopen(request, timeout=self.request_timeout_seconds) as response:
                    self._last_request_at = self._clock()
                    payload = response.read().decode("utf-8")
                self._consecutive_failures = 0
                self._circuit_open_until = 0.0
                self._write_cache(cache_key, payload)
                return payload
            except (HTTPError, URLError, TimeoutError, OSError, UnicodeError) as exc:
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
                    delay = retry_after if retry_after is not None else (
                        self.backoff_base_seconds * (2 ** attempt) + self._random_value()
                    )
                    self._sleep(delay)

        self._consecutive_failures += 1
        if self._consecutive_failures >= self.circuit_failure_threshold:
            self._circuit_open_until = self._clock() + self.circuit_cooldown_seconds
        if last_error:
            raise last_error
        raise RuntimeError(f"{self.name}_REQUEST_FAILED")

    @staticmethod
    def _candidate(entry: ET.Element, *, topic: str, query: str) -> SourceCandidate:
        entry_url = _clean_text(entry.findtext("atom:id", default="", namespaces=NS))
        arxiv_id = _arxiv_id(entry_url)
        published = _clean_text(entry.findtext("atom:published", default="", namespaces=NS))
        updated = _clean_text(entry.findtext("atom:updated", default="", namespaces=NS))
        authors = [
            _clean_text(author.findtext("atom:name", default="", namespaces=NS))
            for author in entry.findall("atom:author", NS)
        ]
        categories = [
            str(category.attrib.get("term") or "").strip()
            for category in entry.findall("atom:category", NS)
            if str(category.attrib.get("term") or "").strip()
        ]
        pdf_url = ""
        for link in entry.findall("atom:link", NS):
            href = str(link.attrib.get("href") or "").strip()
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = href
                break
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        doi = normalize_doi(entry.findtext("arxiv:doi", default="", namespaces=NS))
        journal_reference = _clean_text(entry.findtext("arxiv:journal_ref", default="", namespaces=NS))
        return SourceCandidate(
            candidate_id=f"ARXIV-{arxiv_id.replace('/', '-')}",
            title=_clean_text(entry.findtext("atom:title", default="", namespaces=NS)),
            DOI=doi,
            authors=[author for author in authors if author],
            year=published[:4] if re.fullmatch(r"\d{4}", published[:4]) else "UNKNOWN",
            journal=journal_reference or "arXiv",
            source_database=["ARXIV"],
            OA_status="ARXIV_PREPRINT_OA",
            OA_locations=[{
                "pdf_url": pdf_url,
                "landing_page_url": entry_url,
                "host_type": "preprint_repository",
                "version": "PREPRINT",
                "license": "ARXIV_DISTRIBUTION_LICENSE_OR_AUTHOR_SELECTED_LICENSE",
                "source": "ARXIV",
                "is_oa": "true",
            }],
            pdf_candidate_url=pdf_url,
            topic=[topic] if topic else [],
            lifecycle_state="DISCOVERED",
            source_record_ids=[f"arXiv:{arxiv_id}"],
            version_provenance=[{
                "source": "ARXIV",
                "version": "PREPRINT",
                "publication_stage": "PREPRINT",
                "journal_reference": journal_reference,
                "doi": doi,
            }],
            retrieval_provenance=[{"source": "ARXIV", "method": "API_SEARCH", "identifier": query}],
            abstract=_clean_text(entry.findtext("atom:summary", default="", namespaces=NS)),
            arxiv_id=arxiv_id,
            submitted_date=published,
            updated_date=updated,
            categories=categories,
            journal_reference=journal_reference,
            publication_stage="PREPRINT",
        )

    @classmethod
    def parse_feed(cls, xml_text: str, *, topic: str = "", query: str = "") -> list[SourceCandidate]:
        root = ET.fromstring(xml_text)
        return [cls._candidate(entry, topic=topic, query=query) for entry in root.findall("atom:entry", NS)]

    def search(self, query: str, *, since: str = "", limit: int = 25) -> list[SourceCandidate]:
        del since  # arXiv's Atom API has no stable submitted-date range filter.
        topic = query if query in ARXIV_TOPIC_QUERIES else ""
        api_query = ARXIV_TOPIC_QUERIES.get(query, query)
        capped_limit = max(1, min(100, int(limit)))
        xml_text = self._get_atom_cached(
            cache_key=f"search:{api_query.casefold()}:{capped_limit}",
            params={
                "search_query": api_query,
                "start": 0,
                "max_results": capped_limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )
        return self.parse_feed(xml_text, topic=topic, query=api_query)


# Backward-friendly source naming alongside the other connectors.
ArxivSource = ArxivConnector

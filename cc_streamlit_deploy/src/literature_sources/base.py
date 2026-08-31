"""Shared contracts for legal scholarly metadata connectors."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
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
    version_provenance: list[dict[str, str]] = field(default_factory=list)
    retrieval_provenance: list[dict[str, str]] = field(default_factory=list)

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


def secret_value(name: str) -> str:
    """Read a connector credential from the process or Streamlit Secrets."""
    value = str(os.getenv(name) or "").strip()
    if value:
        return value
    try:
        import streamlit as st

        return str(st.secrets.get(name, "") or "").strip()
    except Exception:
        return ""


class LiteratureSource:
    name = "BASE"
    env_key = ""

    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)
    max_retries = 2
    backoff_base_seconds = 1.0
    min_request_interval_seconds = 0.0
    circuit_failure_threshold = 3
    circuit_cooldown_seconds = 60.0
    request_timeout_seconds = 30.0

    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        cache_dir: Path | None = None,
        sleep: Any = time.sleep,
        clock: Any = time.monotonic,
        random_value: Any = random.random,
    ) -> None:
        self.session = session or requests.Session()
        # Scholarly APIs are public HTTPS endpoints.  A stale desktop proxy
        # must not silently turn discovery into an empty result set.
        self.session.trust_env = False
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._sleep = sleep
        self._clock = clock
        self._random_value = random_value
        self._last_request_at = 0.0
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    @property
    def configured(self) -> bool:
        return not self.env_key or bool(secret_value(self.env_key))

    def search(self, query: str, *, since: str = "", limit: int = 25) -> list[SourceCandidate]:
        raise NotImplementedError

    def _get(self, url: str, **kwargs: Any) -> requests.Response:
        response = self.session.get(url, timeout=30, **kwargs)
        response.raise_for_status()
        return response

    @property
    def circuit_state(self) -> str:
        return "OPEN" if self._clock() < self._circuit_open_until else "CLOSED"

    def _cache_path(self, key: str) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / self.name.casefold() / f"{digest}.json"

    def _read_cache(self, key: str) -> Any | None:
        path = self._cache_path(key)
        if not path or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload.get("response") if isinstance(payload, dict) else None

    def _write_cache(self, key: str, response: Any) -> None:
        path = self._cache_path(key)
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"source": self.name, "cache_key": key, "response": response}, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _retry_after_seconds(response: requests.Response) -> float | None:
        raw = str(response.headers.get("Retry-After") or "").strip()
        try:
            return max(0.0, float(raw)) if raw else None
        except ValueError:
            return None

    def _wait_for_rate_slot(self) -> None:
        remaining = self.min_request_interval_seconds - (self._clock() - self._last_request_at)
        if remaining > 0:
            self._sleep(remaining)

    def _get_json_cached(
        self,
        url: str,
        *,
        cache_key: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """GET JSON with persistent caching and a cooldown-based circuit breaker."""
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached
        now = self._clock()
        if now < self._circuit_open_until:
            raise RuntimeError(f"{self.name}_CIRCUIT_COOLDOWN_ACTIVE")

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_slot()
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.request_timeout_seconds,
                )
                self._last_request_at = self._clock()
                if response.status_code in self.retry_statuses:
                    retry_after = self._retry_after_seconds(response)
                    raise _RetryableHTTPError(response.status_code, retry_after)
                response.raise_for_status()
                payload = response.json()
                self._consecutive_failures = 0
                self._circuit_open_until = 0.0
                self._write_cache(cache_key, payload)
                return payload
            except (_RetryableHTTPError, requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    retry_after = exc.retry_after if isinstance(exc, _RetryableHTTPError) else None
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


class _RetryableHTTPError(requests.HTTPError):
    def __init__(self, status_code: int, retry_after: float | None = None) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.retry_after = retry_after

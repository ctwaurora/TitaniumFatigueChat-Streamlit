"""Secret-safe DeepSeek client with bounded, classified network handling."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Mapping, Optional
from urllib.parse import urlsplit

import requests

from src.api_keys import DeepSeekSettings, get_deepseek_settings


class DeepSeekConfigurationError(RuntimeError):
    pass


class DeepSeekRequestError(RuntimeError):
    """A safe, structured model-service failure.

    ``category`` is intentionally separate from the evidence quality gate so
    callers cannot mistake a transport/API failure for scientific uncertainty.
    """

    def __init__(
        self,
        message: str,
        *,
        category: str = "MODEL_SERVICE_ERROR",
        status_code: Optional[int] = None,
        retry_count: int = 0,
        elapsed_seconds: Optional[float] = None,
        proxy_used: Optional[bool] = None,
        base_url_masked: str = "",
        model: str = "",
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.retry_count = retry_count
        self.elapsed_seconds = elapsed_seconds
        self.proxy_used = proxy_used
        self.base_url_masked = base_url_masked
        self.model = model


def _mask_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}/..."
    except Exception:
        pass
    return "<configured>" if value else "<empty>"


def _safe_error_detail(response: requests.Response) -> str:
    """Extract a short provider error label without echoing request data."""
    try:
        body = response.json()
        error = body.get("error") if isinstance(body, Mapping) else None
        if isinstance(error, Mapping):
            for key in ("code", "type", "message"):
                value = str(error.get(key) or "").strip()
                if value:
                    return value[:160]
    except (ValueError, TypeError):
        pass
    return ""


class DeepSeekClient:
    def __init__(self, settings: Optional[DeepSeekSettings] = None):
        self.settings = settings or get_deepseek_settings()
        if not self.settings.api_key:
            raise DeepSeekConfigurationError("DEEPSEEK_API_KEY is not configured.")
        self._usage: Dict[str, int] = {
            "api_call_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "retry_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._direct_session = requests.Session()
        self._direct_session.trust_env = False
        self._bypass_broken_proxy = False

    @property
    def endpoint(self) -> str:
        return f"{self.settings.base_url.rstrip('/')}/chat/completions"

    def usage_snapshot(self) -> Dict[str, int]:
        return dict(self._usage)

    def _record_usage(self, payload: Any) -> None:
        usage = payload if isinstance(payload, Mapping) else {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            try:
                self._usage[key] += int(usage.get(key) or 0)
            except (TypeError, ValueError):
                continue

    @staticmethod
    def _category_for_status(status: int, detail: str = "") -> str:
        lowered = detail.casefold()
        if status in (401, 403):
            return "AUTHENTICATION_ERROR"
        if status == 429:
            return "RATE_LIMITED"
        if status in (502, 503, 504):
            return "UPSTREAM_SERVICE_ERROR"
        if status == 400 and any(token in lowered for token in ("balance", "quota", "credit", "insufficient")):
            return "QUOTA_OR_BALANCE_ERROR"
        if status == 404:
            return "ENDPOINT_OR_MODEL_ERROR"
        return "MODEL_SERVICE_ERROR"

    def _raise(
        self,
        message: str,
        *,
        category: str,
        status_code: Optional[int],
        retry_count: int,
        started: float,
        proxy_used: Optional[bool],
    ) -> None:
        self._usage["failure_count"] += 1
        raise DeepSeekRequestError(
            message,
            category=category,
            status_code=status_code,
            retry_count=retry_count,
            elapsed_seconds=round(time.perf_counter() - started, 4),
            proxy_used=proxy_used,
            base_url_masked=_mask_url(self.settings.base_url),
            model=self.settings.model,
        )

    def chat(
        self,
        messages: Iterable[Mapping[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 800,
        timeout: int = 60,
        connect_timeout: int = 10,
        max_retries: int = 3,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        started = time.perf_counter()
        last_error: Optional[BaseException] = None
        response: Optional[requests.Response] = None
        proxy_used: Optional[bool] = not self._bypass_broken_proxy
        # max_retries means retries after the initial request, capped at three.
        retry_limit = min(max(0, int(max_retries)), 3)
        attempts = retry_limit + 1
        request_kwargs = {
            "headers": {
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            "json": payload,
            "timeout": (connect_timeout, timeout),
        }
        for attempt in range(attempts):
            if attempt:
                self._usage["retry_count"] += 1
            self._usage["api_call_count"] += 1
            sender = self._direct_session.post if self._bypass_broken_proxy else requests.post
            proxy_used = not self._bypass_broken_proxy
            try:
                response = sender(self.endpoint, **request_kwargs)
            except requests.exceptions.ProxyError as exc:
                # A broken environment proxy is bypassed immediately and is
                # never allowed to contaminate the scientific quality status.
                last_error = exc
                self._bypass_broken_proxy = True
                self._usage["retry_count"] += 1
                try:
                    self._usage["api_call_count"] += 1
                    response = self._direct_session.post(self.endpoint, **request_kwargs)
                    proxy_used = False
                except requests.RequestException as direct_exc:
                    last_error = direct_exc
                    response = None
                    if attempt >= retry_limit:
                        self._raise("DeepSeek direct connection failed.", category="PROXY_CONNECTION_ERROR", status_code=None, retry_count=self._usage["retry_count"], started=started, proxy_used=False)
                    time.sleep(min(2 ** attempt, 30.0))
                    continue
            except requests.exceptions.ConnectTimeout as exc:
                last_error = exc
                response = None
                if attempt < retry_limit:
                    time.sleep(min(2 ** attempt, 30.0))
                    continue
                self._raise("DeepSeek connection timed out.", category="CONNECT_TIMEOUT", status_code=None, retry_count=self._usage["retry_count"], started=started, proxy_used=proxy_used)
            except requests.exceptions.ReadTimeout as exc:
                last_error = exc
                response = None
                if attempt < retry_limit:
                    time.sleep(min(2 ** attempt, 30.0))
                    continue
                self._raise("DeepSeek response timed out.", category="READ_TIMEOUT", status_code=None, retry_count=self._usage["retry_count"], started=started, proxy_used=proxy_used)
            except requests.RequestException as exc:
                last_error = exc
                response = None
                if attempt < retry_limit:
                    time.sleep(min(2 ** attempt, 30.0))
                    continue
                self._raise("DeepSeek network request failed.", category="NETWORK_ERROR", status_code=None, retry_count=self._usage["retry_count"], started=started, proxy_used=proxy_used)

            if response is None:
                continue
            if response.status_code == 200:
                break
            detail = _safe_error_detail(response)
            category = self._category_for_status(response.status_code, detail)
            retryable = response.status_code in (429, 502, 503, 504)
            if not retryable or attempt >= retry_limit:
                label = f"DeepSeek API returned HTTP {response.status_code}."
                if category == "QUOTA_OR_BALANCE_ERROR":
                    label = "DeepSeek account quota or balance is insufficient."
                self._raise(label, category=category, status_code=response.status_code, retry_count=self._usage["retry_count"], started=started, proxy_used=proxy_used)
            retry_after = response.headers.get("Retry-After", "")
            try:
                delay = float(retry_after)
            except (TypeError, ValueError):
                delay = float(2 ** attempt)
            time.sleep(min(max(delay, 0.0), 30.0))

        if response is None:
            self._raise("DeepSeek request failed.", category="MODEL_SERVICE_ERROR", status_code=None, retry_count=self._usage["retry_count"], started=started, proxy_used=proxy_used)
        if response.status_code != 200:
            self._raise(f"DeepSeek API returned HTTP {response.status_code}.", category=self._category_for_status(response.status_code), status_code=response.status_code, retry_count=self._usage["retry_count"], started=started, proxy_used=proxy_used)
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            self._raise("DeepSeek API returned an invalid response.", category="INVALID_RESPONSE", status_code=200, retry_count=self._usage["retry_count"], started=started, proxy_used=proxy_used)
        if not str(content).strip():
            self._raise("DeepSeek API returned an empty response.", category="INVALID_RESPONSE", status_code=200, retry_count=self._usage["retry_count"], started=started, proxy_used=proxy_used)
        self._record_usage(body.get("usage"))
        self._usage["success_count"] += 1
        return str(content).strip()

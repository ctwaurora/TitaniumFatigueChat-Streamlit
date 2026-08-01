"""Minimal DeepSeek chat client with secret-safe usage accounting."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Mapping, Optional

import requests

from src.api_keys import DeepSeekSettings, get_deepseek_settings


class DeepSeekConfigurationError(RuntimeError):
    pass


class DeepSeekRequestError(RuntimeError):
    pass


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
        """Return counters only; prompts, responses, and credentials are excluded."""
        return dict(self._usage)

    def _record_usage(self, payload: Any) -> None:
        usage = payload if isinstance(payload, Mapping) else {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            try:
                self._usage[key] += int(usage.get(key) or 0)
            except (TypeError, ValueError):
                continue

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
        response: Optional[requests.Response] = None
        last_error: Optional[BaseException] = None
        bypass_broken_proxy = self._bypass_broken_proxy
        attempts = max(1, max_retries)
        for attempt in range(attempts):
            if attempt:
                self._usage["retry_count"] += 1
            self._usage["api_call_count"] += 1
            try:
                sender = (
                    self._direct_session.post if bypass_broken_proxy else requests.post
                )
                response = sender(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.settings.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=(connect_timeout, timeout),
                )
                if response.status_code != 429:
                    break
                retry_after = response.headers.get("Retry-After", "")
                delay = (
                    float(retry_after)
                    if retry_after.replace(".", "", 1).isdigit()
                    else 2 ** attempt
                )
                if attempt + 1 < attempts:
                    time.sleep(min(delay, 30.0))
            except requests.exceptions.ProxyError as exc:
                last_error = exc
                response = None
                bypass_broken_proxy = True
                self._bypass_broken_proxy = True
                if attempt + 1 < attempts:
                    time.sleep(min(2 ** attempt, 30.0))
            except requests.RequestException as exc:
                last_error = exc
                response = None
                if attempt + 1 < attempts:
                    time.sleep(min(2 ** attempt, 30.0))

        if response is None:
            self._usage["failure_count"] += 1
            raise DeepSeekRequestError(
                "DeepSeek request failed; check network and service configuration."
            ) from last_error
        if response.status_code != 200:
            self._usage["failure_count"] += 1
            raise DeepSeekRequestError(
                f"DeepSeek API returned HTTP {response.status_code}."
            )
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            self._usage["failure_count"] += 1
            raise DeepSeekRequestError("DeepSeek API returned an invalid response.") from exc
        self._record_usage(body.get("usage"))
        self._usage["success_count"] += 1
        return str(content).strip()

"""Minimal DeepSeek chat client with secret-safe errors."""

from __future__ import annotations

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
            raise DeepSeekConfigurationError(
                "未配置 DEEPSEEK_API_KEY，请在 Streamlit Secrets 或环境变量中设置。"
            )

    @property
    def endpoint(self) -> str:
        return f"{self.settings.base_url.rstrip('/')}/chat/completions"

    def chat(
        self,
        messages: Iterable[Mapping[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 800,
        timeout: int = 30,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            response = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise DeepSeekRequestError("DeepSeek API 请求失败，请检查网络和服务配置。") from exc
        if response.status_code != 200:
            raise DeepSeekRequestError(
                f"DeepSeek API 返回状态码 {response.status_code}。"
            )
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise DeepSeekRequestError("DeepSeek API 返回了无法解析的响应。") from exc
        return str(content).strip()

"""Secure application configuration for Streamlit and CLI runtimes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
_UNSET = object()


def _streamlit_secrets() -> Mapping[str, Any]:
    try:
        import streamlit as st

        return st.secrets
    except Exception:
        return {}


def safe_get_secret(
    key: str,
    default: str = "",
    *,
    secrets: Any = _UNSET,
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    """Read a value from Streamlit secrets, then the environment."""
    source = _streamlit_secrets() if secrets is _UNSET else secrets
    try:
        value = source.get(key) if source is not None else None
    except Exception:
        value = None
    if value is not None and str(value).strip():
        return str(value).strip()
    env = os.environ if environ is None else environ
    return str(env.get(key, default) or default).strip()


def get_deepseek_api_key(
    *, secrets: Any = _UNSET, environ: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    value = safe_get_secret(
        "DEEPSEEK_API_KEY", secrets=secrets, environ=environ
    )
    return value or None


def get_deepseek_base_url(
    *, secrets: Any = _UNSET, environ: Optional[Mapping[str, str]] = None
) -> str:
    return safe_get_secret(
        "DEEPSEEK_BASE_URL",
        DEFAULT_DEEPSEEK_BASE_URL,
        secrets=secrets,
        environ=environ,
    ).rstrip("/")


def get_deepseek_model(
    *, secrets: Any = _UNSET, environ: Optional[Mapping[str, str]] = None
) -> str:
    return safe_get_secret(
        "DEEPSEEK_MODEL",
        DEFAULT_DEEPSEEK_MODEL,
        secrets=secrets,
        environ=environ,
    )


def get_app_password(
    *, secrets: Any = _UNSET, environ: Optional[Mapping[str, str]] = None
) -> str:
    return safe_get_secret(
        "APP_PASSWORD", secrets=secrets, environ=environ
    )


@dataclass(frozen=True)
class DeepSeekSettings:
    api_key: Optional[str]
    base_url: str
    model: str

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


def get_deepseek_settings(
    *, secrets: Any = _UNSET, environ: Optional[Mapping[str, str]] = None
) -> DeepSeekSettings:
    return DeepSeekSettings(
        api_key=get_deepseek_api_key(secrets=secrets, environ=environ),
        base_url=get_deepseek_base_url(secrets=secrets, environ=environ),
        model=get_deepseek_model(secrets=secrets, environ=environ),
    )

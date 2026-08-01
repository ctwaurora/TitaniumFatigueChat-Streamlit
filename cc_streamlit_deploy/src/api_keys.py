"""Secret-safe configuration shared by Streamlit and CLI runtimes.

The CLI never imports Streamlit.  Values are resolved in this order:
environment, project ``.env``, local ``.streamlit/secrets.toml``, then an
already-running Streamlit process' ``st.secrets`` object.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from dotenv import dotenv_values


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_UNSET = object()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _mapping_value(source: Any, key: str) -> str:
    """Read a top-level or ``deepseek`` table value without exposing it."""
    if source is None:
        return ""
    try:
        value = source.get(key)
    except Exception:
        value = None
    if _clean(value):
        return _clean(value)
    try:
        nested = source.get("deepseek") or source.get("DeepSeek") or {}
        value = nested.get(key) or nested.get(key.removeprefix("DEEPSEEK_").lower())
    except Exception:
        value = None
    return _clean(value)


def _dotenv_mapping(project_root: Path) -> Mapping[str, Any]:
    path = project_root / ".env"
    if not path.is_file():
        return {}
    try:
        # dotenv_values parses the file without copying secrets into os.environ.
        return dotenv_values(path)
    except (OSError, UnicodeError, ValueError):
        return {}


def _local_streamlit_secrets(project_root: Path) -> Mapping[str, Any]:
    path = project_root / ".streamlit" / "secrets.toml"
    if not path.is_file():
        return {}
    try:
        import toml

        value = toml.load(path)
        return value if isinstance(value, Mapping) else {}
    except (ImportError, OSError, UnicodeError, ValueError):
        return {}


def _runtime_streamlit_secrets() -> Mapping[str, Any]:
    """Use st.secrets only when Streamlit is already imported by the caller."""
    streamlit = sys.modules.get("streamlit")
    if streamlit is None:
        return {}


def _windows_persistent_environment(key: str) -> str:
    """Read a newly-set Windows environment value before the parent restarts."""
    if os.name != "nt":
        return ""
    try:
        import winreg

        locations = (
            (winreg.HKEY_CURRENT_USER, r"Environment"),
            (
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            ),
        )
        for hive, path in locations:
            try:
                with winreg.OpenKey(hive, path) as handle:
                    value, _kind = winreg.QueryValueEx(handle, key)
                if _clean(value):
                    return _clean(value)
            except OSError:
                continue
    except (ImportError, OSError):
        return ""
    return ""
    try:
        return streamlit.secrets
    except Exception:
        return {}


def resolve_config_value(
    key: str,
    default: str = "",
    *,
    secrets: Any = _UNSET,
    environ: Optional[Mapping[str, str]] = None,
    project_root: Optional[Path] = None,
) -> Tuple[str, str]:
    """Return ``(value, source)`` using the documented safe priority."""
    root = Path(project_root or PROJECT_ROOT).resolve()
    env = os.environ if environ is None else environ
    value = _clean(env.get(key))
    if value:
        return value, "environment"
    if environ is None:
        value = _windows_persistent_environment(key)
        if value:
            return value, "environment"

    value = _mapping_value(_dotenv_mapping(root), key)
    if value:
        return value, "dotenv"

    value = _mapping_value(_local_streamlit_secrets(root), key)
    if value:
        return value, "streamlit_secrets"

    runtime_secrets = _runtime_streamlit_secrets() if secrets is _UNSET else secrets
    value = _mapping_value(runtime_secrets, key)
    if value:
        return value, "streamlit_secrets"
    return _clean(default), "default" if _clean(default) else "none"


def safe_get_secret(
    key: str,
    default: str = "",
    *,
    secrets: Any = _UNSET,
    environ: Optional[Mapping[str, str]] = None,
    project_root: Optional[Path] = None,
) -> str:
    value, _source = resolve_config_value(
        key,
        default,
        secrets=secrets,
        environ=environ,
        project_root=project_root,
    )
    return value


def get_deepseek_api_key(
    *,
    secrets: Any = _UNSET,
    environ: Optional[Mapping[str, str]] = None,
    project_root: Optional[Path] = None,
) -> Optional[str]:
    value, _source = resolve_config_value(
        "DEEPSEEK_API_KEY",
        secrets=secrets,
        environ=environ,
        project_root=project_root,
    )
    return value or None


def get_deepseek_base_url(
    *,
    secrets: Any = _UNSET,
    environ: Optional[Mapping[str, str]] = None,
    project_root: Optional[Path] = None,
) -> str:
    value, _source = resolve_config_value(
        "DEEPSEEK_BASE_URL",
        DEFAULT_DEEPSEEK_BASE_URL,
        secrets=secrets,
        environ=environ,
        project_root=project_root,
    )
    return value.rstrip("/")


def get_deepseek_model(
    *,
    secrets: Any = _UNSET,
    environ: Optional[Mapping[str, str]] = None,
    project_root: Optional[Path] = None,
) -> str:
    value, _source = resolve_config_value(
        "DEEPSEEK_MODEL",
        DEFAULT_DEEPSEEK_MODEL,
        secrets=secrets,
        environ=environ,
        project_root=project_root,
    )
    return value


def get_app_password(
    *, secrets: Any = _UNSET, environ: Optional[Mapping[str, str]] = None
) -> str:
    return safe_get_secret("APP_PASSWORD", secrets=secrets, environ=environ)


@dataclass(frozen=True)
class DeepSeekSettings:
    api_key: Optional[str]
    base_url: str
    model: str
    source: str = "none"

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


def get_deepseek_settings(
    *,
    secrets: Any = _UNSET,
    environ: Optional[Mapping[str, str]] = None,
    project_root: Optional[Path] = None,
) -> DeepSeekSettings:
    api_key, source = resolve_config_value(
        "DEEPSEEK_API_KEY",
        secrets=secrets,
        environ=environ,
        project_root=project_root,
    )
    return DeepSeekSettings(
        api_key=api_key or None,
        base_url=get_deepseek_base_url(
            secrets=secrets, environ=environ, project_root=project_root
        ),
        model=get_deepseek_model(
            secrets=secrets, environ=environ, project_root=project_root
        ),
        source=source,
    )

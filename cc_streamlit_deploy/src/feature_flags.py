"""Small environment-backed switches for independently reversible research features."""

from __future__ import annotations

import os


def feature_enabled(name: str, *, default: bool = True) -> bool:
    """Return a boolean feature switch without exposing environment values."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off", "disabled"}

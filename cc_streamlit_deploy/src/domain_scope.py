"""Titanium-alloy fatigue literature scope classification."""

from __future__ import annotations

from typing import Any, Dict

from src.stage1_store import normalize_title


CORE = "CORE"
CONTEXT = "CONTEXT"
OUT_OF_SCOPE = "OUT_OF_SCOPE"

TI64_TERMS = (
    "ti 6al 4v",
    "ti6al4v",
    "ti 6al4v",
    "ti64",
    "tc4",
    "ti al6 v4",
    "tial6v4",
    "ti 6ai 4v",
    "ti6ai4v",
    "钛合金tc4",
    "tc4钛合金",
)
TITANIUM_TERMS = (
    *TI64_TERMS,
    "titanium",
    "titanium alloy",
    "titanium alloys",
    "钛合金",
)
FATIGUE_TERMS = (
    "fatigue",
    "low cycle",
    "high cycle",
    "very high cycle",
    "lcf",
    "hcf",
    "vhcf",
    "cyclic loading",
    "cyclic deformation",
    "strain life",
    "stress life",
    "s n curve",
    "crack initiation",
    "short crack",
    "crack growth",
    "fatigue crack",
    "疲劳",
    "循环载荷",
    "裂纹起裂",
    "裂纹萌生",
    "短裂纹",
    "裂纹扩展",
)


def classify_literature_scope(record: Dict[str, Any]) -> Dict[str, str]:
    """Classify by material and fatigue relevance, never by pore relevance."""
    text = normalize_title(
        " ".join(
            str(record.get(key) or "")
            for key in (
                "title",
                "abstract",
                "keywords",
                "material",
                "research_topic",
            )
        )
    )
    has_fatigue = any(normalize_title(term) in text for term in FATIGUE_TERMS)
    has_titanium = any(
        normalize_title(term) in text for term in TITANIUM_TERMS
    )
    has_ti64 = any(normalize_title(term) in text for term in TI64_TERMS)
    if not has_fatigue or not has_titanium:
        return {
            "domain_scope": OUT_OF_SCOPE,
            "scope_reason": "TITANIUM_FATIGUE_RELEVANCE_REQUIRED",
        }
    if has_ti64:
        return {
            "domain_scope": CORE,
            "scope_reason": "DIRECT_TI64_TC4_FATIGUE",
        }
    return {
        "domain_scope": CONTEXT,
        "scope_reason": "TRANSFERABLE_TITANIUM_ALLOY_FATIGUE",
    }


def is_allowed_in_titanium_fatigue_rag(record: Dict[str, Any]) -> bool:
    return classify_literature_scope(record)["domain_scope"] != OUT_OF_SCOPE

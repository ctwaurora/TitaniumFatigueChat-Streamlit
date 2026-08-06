"""Strict validation and comparison helpers for extracted fatigue formulas."""

from __future__ import annotations

import re
from typing import Any, Iterable


CONFIRMED = "CONFIRMED"
REJECTED = "REJECTED"
IMAGE_REVIEW = "IMAGE_REVIEW"
PENDING_REVIEW = "PENDING_REVIEW"

_SOURCE_EQUATION = re.compile(r"(?:equation|eq\.?)[\s\u00a0]*\(?\d+[a-z]?\)?", re.I)
_EQUATION_NUMBER = re.compile(r"(?:equation|eq\.?)?\s*\((\d+[a-z]?)\)", re.I)
_UNIT = re.compile(r"\b(?:MPa|GPa|Pa|Hz|kHz|mm|um|cycles?|m/cycle)\b|MPa.?m", re.I)
_DEFINITION = re.compile(r"\bwhere\b|\bdenotes\b|\bdefined as\b|\bis the\b", re.I)
_CONDITION = re.compile(r"\bR\s*=|stress ratio|temperature|environment|frequency|loading|valid|applicable|range|condition", re.I)
_RELATION = re.compile(r"da\s*/?\s*dN|Delta\s*K|Keff|Nf|sigma|epsilon|sqrt.?area|HV|U\s*\(|a0|C\s*\(|\bK\b", re.I)


def _normalized_formula(text: Any) -> str:
    return (
        " ".join(str(text or "").casefold().replace("−", "-").replace("∆", "Δ").split())
        .replace(" ", "")
    )


def formula_family(expression: Any, context: Any = "") -> str:
    """Classify from the equation itself; prose names are only tie-breakers."""
    compact = _normalized_formula(expression)
    prose = str(context or "").casefold()
    has_growth_rate = bool(re.search(r"da/?dn=", compact))
    has_delta_k = "δk" in compact or "Δk" in compact or "deltak" in compact

    if has_growth_rate and has_delta_k and all(token in compact for token in ("kmax", "kthr")):
        return "NASGRO"
    if has_growth_rate and has_delta_k and "kc" in compact and "/" in compact:
        return "Forman"
    if has_growth_rate and has_delta_k and ("(1-r)" in compact or "walker" in prose):
        return "Walker"
    if has_growth_rate and ("Δkeff" in compact or "δkeff" in compact or "keff" in compact or "kopen" in compact):
        return "Crack closure / Delta-K-effective"
    if has_growth_rate and has_delta_k and re.search(r"=c[\[(]?(?:[Δδ]k|deltak)", compact):
        return "Paris"
    if ("√area" in compact or "sqrtarea" in compact) and "hv" in compact and "=" in compact:
        return "Murakami"
    if "a0" in compact and ("a+a0" in compact or "a0+a" in compact) and ("Δk" in compact or "δk" in compact or "Δσ" in compact):
        return "El Haddad / Kitagawa-Takahashi"
    if ("Δεp" in compact or "δεp" in compact or "plasticstrain" in compact) and "εf" in compact and "nf" in compact:
        return "Coffin-Manson"
    if "nf" in compact and re.search(r"(?:σa|sigma_?a|stress|s)=", compact) and re.search(r"\(?(?:2\*)?nf\)?", compact):
        return "Basquin"
    if ("σmax" in compact or "sigmamax" in compact) and ("Δε" in compact or "δε" in compact or "strain" in compact) and "nf" in compact:
        return "SWT"
    if "σa" in compact and "σm" in compact and re.search(r"=1(?:\D|$)", compact):
        return "Goodman"
    if has_growth_rate and has_delta_k and ("a+a0" in compact or "hartman" in prose or "schijve" in prose):
        return "Short-crack correction"
    return "其他"


def _equation_excerpt(text: str) -> str:
    compact = " ".join(str(text or "").replace("\x00", " ").split())
    if not compact:
        return ""
    candidates = re.split(r"(?<=[.;])\s+", compact)
    equation_parts = [part for part in candidates if "=" in part and _RELATION.search(part)]
    if not equation_parts and "=" in compact and _RELATION.search(compact):
        equation_parts = [compact]
    if not equation_parts:
        return ""
    excerpt = equation_parts[0]
    if len(excerpt) > 500:
        equal_at = excerpt.find("=")
        excerpt = excerpt[max(0, equal_at - 150):equal_at + 350]
    return excerpt.strip()


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def validate_formula_candidate(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    context = str(
        row.get("context_before_after")
        or row.get("original_text")
        or row.get("short_excerpt")
        or row.get("claim")
        or ""
    )
    raw_formula = str(
        row.get("original_formula")
        or row.get("equation")
        or row.get("normalized_formula")
        or ""
    )
    combined = f"{raw_formula} {context}"
    excerpt = _equation_excerpt(raw_formula) or _equation_excerpt(context)
    family = formula_family(excerpt or raw_formula, context)
    equation_number = str(row.get("equation_number") or "").strip()
    if equation_number in {"", "未确认", "None"}:
        match = _EQUATION_NUMBER.search(combined)
        equation_number = match.group(1) if match else ""
    definitions = [
        item for item in _as_text_list(row.get("symbol_definitions") or row.get("parameters"))
        if "未" not in item and "not reported" not in item.casefold()
    ]
    units = [
        item for item in _as_text_list(row.get("symbol_units") or row.get("units") or row.get("parameter_units"))
        if "未" not in item and "not reported" not in item.casefold()
    ]
    conditions = [
        item for item in _as_text_list(row.get("applicable_conditions"))
        if "未" not in item and "not reported" not in item.casefold()
    ]
    source_title = str(row.get("paper_title") or row.get("title") or "").strip()
    page_number = row.get("page_number")
    image_required = str(row.get("manual_review_status") or row.get("review_status") or "").upper() in {
        "FORMULA_IMAGE_REVIEW_REQUIRED", "IMAGE_REVIEW", "MANUAL_VISUAL_REVIEW_REQUIRED"
    }
    control_chars = any(ord(char) < 32 and char not in "\n\r\t" for char in raw_formula)
    plain_test_value = bool(re.fullmatch(r"\s*(?:R|Nf|K|max|f)\s*=\s*[-+]?\d+(?:\.\d+)?(?:\s*\w+)?\s*", excerpt, re.I))

    checks = {
        "source_title": bool(source_title),
        "source_page": str(page_number or "").isdigit() and int(page_number) > 0,
        "equation_number": bool(equation_number) and bool(_SOURCE_EQUATION.search(combined) or _EQUATION_NUMBER.search(combined)),
        "equation_body": bool(excerpt) and not plain_test_value and 8 <= len(excerpt) <= 500,
        "recognized_family": family != "其他",
        "variable_definition": bool(definitions) and (bool(_DEFINITION.search(context)) or len(definitions) >= 2),
        "unit_context": bool(units) and (bool(_UNIT.search(context)) or bool(_UNIT.search(" ".join(units)))),
        "applicability_context": bool(conditions) or bool(_CONDITION.search(context)),
        "clean_text": not control_chars,
    }
    failed = [name for name, passed in checks.items() if not passed]
    source_confirmed = str(row.get("raw_review_status") or row.get("review_status") or "").upper() in {
        "CONFIRMED", "HUMAN_CONFIRMED", "VERIFIED"
    }
    if image_required:
        status = IMAGE_REVIEW
    elif source_confirmed and all(
        checks[key]
        for key in ("source_title", "source_page", "equation_body", "recognized_family", "clean_text")
    ):
        status = CONFIRMED
    elif not failed:
        status = CONFIRMED
    elif not checks["equation_body"] or control_chars:
        status = REJECTED
    else:
        status = PENDING_REVIEW

    result.update({
        "formula_family": family,
        "formula_expression": excerpt,
        "equation_number": equation_number or "未确认",
        "validation_status": status,
        "evidence_status": "已确认" if status == CONFIRMED else "待人工复核",
        "validation_checks": checks,
        "validation_failures": failed,
        "validation_basis": "来源页、公式编号、正文、变量、单位和适用范围的确定性联合门禁",
        "confirmed_variables": definitions,
        "confirmed_units": units,
        "confirmed_assumptions": [
            sentence.strip()
            for sentence in re.split(r"(?<=[.;])\s+", " ".join(context.split()))
            if re.search(r"assum|valid|applicable|condition|range", sentence, re.I)
        ][:4],
        "material_scope": row.get("material_scope") or source_title,
        "fatigue_stage": row.get("fatigue_stage") or ("裂纹扩展" if family in {"Paris", "Walker", "NASGRO", "Forman", "Crack closure / Delta-K-effective", "Short-crack correction"} else "疲劳寿命/极限"),
        "stress_ratio_scope": row.get("stress_ratio_scope") or next((item for item in conditions if re.search(r"\bR\s*=|stress ratio", item, re.I)), "原文片段未完整报告"),
        "direct_comparison_allowed": False,
    })
    return result


def validate_formula_collection(
    formulas: Iterable[dict[str, Any]],
    *,
    high_value_limit: int | None = None,
) -> list[dict[str, Any]]:
    validated = [validate_formula_candidate(dict(row)) for row in formulas]
    seen: set[tuple[str, str, str]] = set()
    for row in validated:
        signature = (
            str(row.get("paper_id") or ""),
            str(row.get("page_number") or ""),
            _normalized_formula(row.get("formula_expression")),
        )
        if signature in seen and row.get("validation_status") == CONFIRMED:
            row["validation_status"] = PENDING_REVIEW
            row["evidence_status"] = "待人工复核"
            row["validation_failures"] = [
                *(row.get("validation_failures") or []),
                "duplicate_formula",
            ]
        seen.add(signature)
    if high_value_limit is None:
        return validated
    priority = [row for row in validated if row["formula_family"] != "其他"]
    priority.sort(key=lambda row: (row["validation_status"] != CONFIRMED, str(row.get("paper_title") or row.get("title") or ""), str(row.get("formula_id") or "")))
    return priority[:high_value_limit]


def compare_confirmed_formulas(question: str, formulas: Iterable[dict[str, Any]]) -> str:
    confirmed = [
        row
        for row in validate_formula_collection(formulas)
        if row["validation_status"] == CONFIRMED
    ]
    query = str(question or "").casefold()
    selected = [
        row for row in confirmed
        if row["formula_family"].casefold() in query
        or any(term in query for term in str(row.get("formula_family") or "").casefold().split())
        or any(term in (str(row.get("formula_expression")) + str(row.get("context_before_after"))).casefold() for term in ("da/dn", "keff", "short crack", "long crack"))
    ][:8]
    if not selected:
        return "当前已确认公式库不能支持该项比较。"
    lines = ["### 已确认公式比较", ""]
    for row in selected:
        lines.extend([
            f"#### {row['formula_family']}",
            f"公式：{row.get('formula_expression') or row.get('original_formula')}",
            f"来源：{row.get('paper_title') or row.get('title')}；页码：{row.get('page_number')}；公式编号：{row.get('equation_number')}；Formula ID：{row.get('formula_id')}",
            f"变量与单位：{'；'.join(row.get('confirmed_variables') or [])}；{'；'.join(row.get('confirmed_units') or [])}",
            f"适用边界：{'；'.join(_as_text_list(row.get('applicable_conditions'))) or '仅限来源文献已报告条件'}",
            "是否可直接数值比较：否。必须先统一预测对象、单位、裂纹阶段、应力比和参数标定数据。",
            "",
        ])
    return "\n".join(lines).strip()

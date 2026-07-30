"""Compatibility layer for structured candidate hypothesis generation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.hypothesis_candidate import (
    build_candidate_hypothesis,
    build_candidate_hypothesis_markdown,
    format_candidate_hypothesis,
)


def detect_variable_complexity(
    question: str, ind_var: Optional[str], dep_var: Optional[str]
) -> bool:
    text = f"{question or ''} {ind_var or ''} {dep_var or ''}".lower()
    variable_terms = (
        "pore",
        "defect",
        "distance",
        "surface",
        "stress",
        "孔隙",
        "缺陷",
        "距表面",
        "表面",
        "应力",
    )
    return sum(term in text for term in variable_terms) >= 2


def match_hypothesis_templates(
    question: str, ind_var: Optional[str], dep_var: Optional[str]
) -> List[Dict[str, Any]]:
    return [build_candidate_hypothesis(question, ind_var, dep_var)]


def format_split_hypotheses(templates: List[Dict[str, Any]]) -> str:
    if not templates:
        return ""
    return "## 候选科学假设\n\n" + "\n\n---\n\n".join(
        format_candidate_hypothesis(template) for template in templates
    )


def generate_split_hypotheses(
    question: str, ind_var: Optional[str], dep_var: Optional[str]
) -> Optional[str]:
    if not question and not ind_var and not dep_var:
        return None
    return build_candidate_hypothesis_markdown(question, ind_var, dep_var)


def replace_old_hypothesis(
    question: str, ind_var: str, dep_var: str
) -> Tuple[bool, Optional[str]]:
    if not question and not ind_var and not dep_var:
        return False, None
    return True, build_candidate_hypothesis_markdown(question, ind_var, dep_var)

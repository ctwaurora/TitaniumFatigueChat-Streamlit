"""Build compact, traceable per-paper and cross-paper evidence bundles."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from src.research_topics import document_topics, identify_topics


EMPTY_VALUES = (None, "", "NOT_REPORTED", [], {})
COMPARISON_FIELDS = (
    "alloy_grade", "material", "manufacturing_process", "process", "heat_treatment",
    "hip", "surface_treatment", "surface_state", "stress_ratio_R", "fatigue_regime",
    "temperature", "environment", "build_orientation", "loading_mode",
)


def _clean_conditions(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if item not in EMPTY_VALUES}


def _text(row: dict[str, Any]) -> str:
    return str(row.get("claim") or row.get("original_text") or row.get("text") or "").strip()


def _normal_key(value: str) -> str:
    return re.sub(r"\W+", " ", value.casefold()).strip()


def is_plausible_formula(row: dict[str, Any]) -> bool:
    equation = str(row.get("equation") or row.get("formula") or "").strip()
    if not equation or len(equation) > 280 or not row.get("paper_id"):
        return False
    if not row.get("page_number") or not (row.get("doc_id") or row.get("formula_id")):
        return False
    relation = bool(re.search(r"(?:=|≈|≃|∝|≤|≥|<|>)", equation))
    operators = bool(re.search(r"[+\-*/^√∑Δσɛε()\[\]]", equation))
    symbol = bool(re.search(r"\b(?:da/dN|Nf|Delta\s*K|Paris|C|m|R|K|max|min)\b|[Δσɛεαβγ]", equation, re.I))
    words = re.findall(r"[A-Za-z]{3,}", equation)
    sentence_like = len(words) > 28 and equation.count(" ") > 35
    return relation and operators and symbol and not sentence_like


def _result_direction(text: str) -> str:
    lower = text.casefold()
    if any(term in lower for term in ("no significant", "not significant", "independent of", "无显著", "没有显著")):
        return "NO_SIGNIFICANT_EFFECT"
    if any(term in lower for term in ("decrease", "reduced", "lower", "下降", "降低", "减小")):
        return "DECREASE"
    if any(term in lower for term in ("increase", "improved", "higher", "上升", "提高", "增加")):
        return "INCREASE"
    return "CONDITION_DEPENDENT_OR_UNRESOLVED"


@dataclass
class PaperEvidenceSummary:
    paper_id: str
    title: str
    authors: str
    year: str
    roles: list[str]
    topics: list[str]
    study_object: str
    conditions: dict[str, Any]
    principal_claims: list[dict[str, Any]]
    mechanisms: list[str]
    formulas: list[dict[str, Any]]
    result_directions: list[str]
    applicability: list[str]
    limitations: list[str]


@dataclass
class CrossPaperSynthesis:
    consensus: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    condition_matches: list[str] = field(default_factory=list)
    condition_mismatches: list[str] = field(default_factory=list)
    covered_conditions: dict[str, list[str]] = field(default_factory=dict)
    missing_conditions: list[str] = field(default_factory=list)
    supported_conclusions: list[str] = field(default_factory=list)
    unsupported_conclusions: list[str] = field(default_factory=list)
    formula_comparability: list[str] = field(default_factory=list)
    mechanism_map: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceBundle:
    question: str
    topics: list[str]
    papers: list[PaperEvidenceSummary]
    formulas: list[dict[str, Any]]
    synthesis: CrossPaperSynthesis
    citation_index: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_json(self, max_chars: int = 18000) -> str:
        payload = json.dumps(self.as_dict(), ensure_ascii=False, separators=(",", ":"))
        return payload if len(payload) <= max_chars else payload[:max_chars] + "\n[EvidenceBundle truncated]"


def _merge_conditions(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        for key, item in _clean_conditions(row.get("experimental_conditions")).items():
            items = item if isinstance(item, (list, tuple, set)) else [item]
            for entry in items:
                text = str(entry).strip()
                if text and text not in values[key]:
                    values[key].append(text)
    return {key: items[0] if len(items) == 1 else items for key, items in values.items()}


def _condition_comparison(papers: list[PaperEvidenceSummary]) -> tuple[list[str], list[str], dict[str, list[str]]]:
    covered: dict[str, list[str]] = defaultdict(list)
    for paper in papers:
        for key in COMPARISON_FIELDS:
            item = paper.conditions.get(key)
            if item in EMPTY_VALUES:
                continue
            for value in item if isinstance(item, list) else [item]:
                if str(value) not in covered[key]:
                    covered[key].append(str(value))
    matches = [f"{key}={values[0]}" for key, values in covered.items() if len(values) == 1]
    mismatches = [f"{key}存在不可直接合并的条件：{' / '.join(values)}" for key, values in covered.items() if len(values) > 1]
    return matches, mismatches, dict(covered)


def build_evidence_bundle(
    question: str,
    supporting: list[dict[str, Any]],
    counter: list[dict[str, Any]],
    conditional: list[dict[str, Any]],
    retrieved_pool: list[dict[str, Any]] | None = None,
) -> EvidenceBundle:
    role_rows = (("SUPPORT", supporting), ("COUNTER", counter), ("CONDITION_DEPENDENT", conditional))
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for role, rows in role_rows:
        for row in rows:
            paper_id = str(row.get("paper_id") or "")
            claim = _text(row)
            key = (paper_id, _normal_key(claim))
            if not paper_id or not claim or key in seen:
                continue
            seen.add(key)
            grouped[paper_id].append((role, row))

    formula_source = list(retrieved_pool or []) + supporting + counter + conditional
    formula_rows: list[dict[str, Any]] = []
    formula_seen: set[str] = set()
    for row in formula_source:
        formula_id = str(row.get("doc_id") or row.get("formula_id") or "")
        if formula_id in formula_seen or not is_plausible_formula(row):
            continue
        formula_seen.add(formula_id)
        formula_rows.append({
            "formula_id": formula_id,
            "equation": str(row.get("equation") or row.get("formula")),
            "paper_id": str(row.get("paper_id")),
            "title": str(row.get("title") or ""),
            "page_number": row.get("page_number"),
            "section": str(row.get("section") or ""),
            "parameters": row.get("parameters") or [],
            "units": row.get("units") or [],
            "applicable_conditions": _clean_conditions(row.get("applicable_conditions") or row.get("experimental_conditions")),
        })

    papers: list[PaperEvidenceSummary] = []
    citation_index: dict[str, dict[str, Any]] = {}
    for paper_id, entries in grouped.items():
        rows = [row for _, row in entries]
        claims = []
        for role, row in entries[:6]:
            evidence_id = str(row.get("doc_id") or row.get("evidence_id") or "")
            claim = {
                "role": role,
                "claim": _text(row)[:650],
                "evidence_id": evidence_id,
                "page_number": row.get("page_number"),
                "section": str(row.get("section") or ""),
                "directness": str(row.get("directness") or ""),
                "conditions": _clean_conditions(row.get("experimental_conditions")),
            }
            claims.append(claim)
            if evidence_id:
                citation_index[evidence_id] = {
                    "paper_id": paper_id,
                    "title": str(row.get("title") or ""),
                    "page_number": row.get("page_number"),
                    "section": str(row.get("section") or ""),
                }
        conditions = _merge_conditions(rows)
        all_text = " ".join(_text(row) for row in rows)
        paper_formulas = [formula for formula in formula_rows if formula["paper_id"] == paper_id]
        mechanisms = [
            _text(row)[:350] for row in rows
            if re.search(r"mechanis|because|due to|attribut|机制|由于|归因", _text(row), re.I)
        ][:3]
        papers.append(PaperEvidenceSummary(
            paper_id=paper_id,
            title=str(rows[0].get("title") or "题名未报告"),
            authors=str(rows[0].get("authors") or "未报告"),
            year=str(rows[0].get("year") or "未报告"),
            roles=list(dict.fromkeys(role for role, _ in entries)),
            topics=document_topics(all_text, conditions),
            study_object=str(conditions.get("alloy_grade") or conditions.get("material") or "钛合金疲劳（具体牌号以证据条件为准）"),
            conditions=conditions,
            principal_claims=claims,
            mechanisms=mechanisms,
            formulas=paper_formulas,
            result_directions=list(dict.fromkeys(_result_direction(_text(row)) for row in rows)),
            applicability=[f"{key}={value}" for key, value in conditions.items() if key in COMPARISON_FIELDS],
            limitations=["未报告完整关键实验条件"] if len(conditions) < 3 else [],
        ))

    papers.sort(key=lambda paper: ("SUPPORT" not in paper.roles, paper.title.casefold()))
    matches, mismatches, covered = _condition_comparison(papers)
    directions = {direction for paper in papers for direction in paper.result_directions}
    consensus = [
        paper.principal_claims[0]["claim"]
        for paper in papers if "SUPPORT" in paper.roles and paper.principal_claims
    ][:4]
    conflicts = [
        paper.principal_claims[0]["claim"]
        for paper in papers if "COUNTER" in paper.roles and paper.principal_claims
    ][:4]
    if len(directions - {"CONDITION_DEPENDENT_OR_UNRESOLVED"}) > 1:
        conflicts.insert(0, "不同文献报告的结果方向不一致，必须按材料、处理、表面、载荷和疲劳阶段分层解释。")
    missing = [key for key in COMPARISON_FIELDS if key not in covered]
    formula_comparability = []
    if len(formula_rows) > 1:
        condition_sets = [formula["applicable_conditions"] for formula in formula_rows]
        formula_comparability.append(
            "公式条件一致后才可数值比较。" if all(item == condition_sets[0] for item in condition_sets[1:])
            else "这些公式的适用条件不同，不能直接进行数值比较。"
        )
    topics = identify_topics(question)
    synthesis = CrossPaperSynthesis(
        consensus=consensus,
        conflicts=conflicts,
        condition_matches=matches,
        condition_mismatches=mismatches,
        covered_conditions=covered,
        missing_conditions=missing,
        supported_conclusions=consensus,
        unsupported_conclusions=[f"缺少{key}条件，不能外推到该维度。" for key in missing[:5]],
        formula_comparability=formula_comparability,
        mechanism_map={
            "type": "QUALITATIVE_MECHANISM_MAP",
            "message": "当前证据只能形成定性机制主导区，尚不足以确定精确转换边界。",
            "candidate_axes": topics[:2],
        } if len(topics) >= 2 else {},
    )
    return EvidenceBundle(question, topics, papers, formula_rows, synthesis, citation_index)

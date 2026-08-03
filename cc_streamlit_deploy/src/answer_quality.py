"""Lightweight, evidence-aware quality gate for web research answers."""

from __future__ import annotations

import re
from typing import Any

from src.research_topics import query_mentions_pores


COUNT_TALK = re.compile(r"(?:提供|检索到|召回|使用|共有|共检索)\s*\d+\s*(?:条|篇).{0,10}(?:证据|文献)")
PLACEHOLDERS = ("目标因素", "问题中的目标因素", "某个因素", "疲劳响应发生变化")
PORE_PATTERN = re.compile(r"孔隙|孔洞|气孔|√area|\bpore\b|porosity", re.I)


SKILL_REQUIREMENTS = {
    "scientific_analysis_skill": ("结论", "机制", "条件", "反向", "不能确定"),
    "research_gap_skill": ("研究空白", "已有", "缺少", "反证", "最低成本"),
    "hypothesis_generation_skill": ("候选假设", "自变量", "因变量", "控制变量", "推翻", "验证"),
    "experiment_design_skill": ("研究对象", "核心假设", "自变量", "因变量", "控制变量", "分组", "加载条件", "数据分析", "推翻"),
}


def validate_answer_quality(
    answer: str,
    *,
    question: str,
    skill_name: str,
    evidence_bundle: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    if not str(answer).strip():
        failures.append("empty_answer")
    if COUNT_TALK.search(answer):
        failures.append("visible_evidence_count_talk")
    if any(term in answer for term in PLACEHOLDERS):
        failures.append("vague_placeholder")
    for term in SKILL_REQUIREMENTS.get(skill_name, ()):
        if term not in answer:
            failures.append(f"missing_section_or_concept:{term}")

    citations = evidence_bundle.get("citation_index") or {}
    known_ids = set(citations)
    cited_ids = set(re.findall(r"\b(?:EV|COND_EV|FORMULA)_[A-Z0-9_]+\b", answer))
    unknown_ids = sorted(cited_ids - known_ids - {
        str(item.get("formula_id")) for item in evidence_bundle.get("formulas") or []
    })
    if unknown_ids:
        failures.append("unknown_evidence_id:" + ",".join(unknown_ids[:3]))
    if known_ids and not (cited_ids & known_ids):
        failures.append("no_traceable_evidence_id")

    valid_pages = {
        str(item.get("page_number"))
        for item in [*citations.values(), *(evidence_bundle.get("formulas") or [])]
        if item.get("page_number")
    }
    cited_pages = set(re.findall(r"(?:p\.?|页码[：:]?)\s*(\d+)", answer, re.I))
    if cited_pages and not cited_pages.issubset(valid_pages):
        failures.append("invalid_page_reference")

    formulas = evidence_bundle.get("formulas") or []
    formula_topic = "FORMULA_MODEL" in (evidence_bundle.get("topics") or [])
    if formulas and (formula_topic or "公式" in question) and not any(
        str(item.get("formula_id")) in answer and str(item.get("equation")) in answer
        for item in formulas
    ):
        failures.append("missing_traceable_formula")
    if (
        not formulas
        and re.search(r"Formula ID\s*[：:]|(?<!并非)文献原公式\s*[：:]", answer)
        and "未检索到" not in answer
    ):
        failures.append("unverified_formula_claim")

    if not query_mentions_pores(question):
        pore_hits = len(PORE_PATTERN.findall(answer))
        headings = re.findall(r"(?m)^#{1,4}\s+(.+)$", answer)
        if pore_hits > 2 or any(PORE_PATTERN.search(heading) for heading in headings):
            failures.append("unsolicited_pore_centrality")

    if skill_name in {"hypothesis_generation_skill", "experiment_design_skill"}:
        falsifiers = len(re.findall(r"(?:推翻|证伪|置信区间|无法复现|相反)", answer))
        if falsifiers < 2:
            failures.append("insufficient_falsification_criteria")
    if "条件" not in answer and "适用范围" not in answer:
        failures.append("missing_condition_boundary")
    if not any(term in answer for term in ("反向", "反证", "冲突", "相反")):
        failures.append("counter_evidence_not_addressed")

    return {
        "passed": not failures,
        "failures": failures,
        "skill_name": skill_name,
        "citation_ids_checked": sorted(cited_ids),
        "pore_bias_checked": True,
    }


def precise_refusal(skill_name: str, failures: list[str], evidence_bundle: dict[str, Any]) -> str:
    missing = evidence_bundle.get("synthesis", {}).get("missing_conditions") or []
    details = "、".join(missing[:5]) or "可核验的页码、实验条件或反向证据"
    return (
        "## 直接结论\n\n"
        "本次生成未通过科研真实性门禁，因此不显示未经充分约束的结论。\n\n"
        "## 当前不能确定的内容\n\n"
        f"当前正式证据缺少或无法一致核对：{details}。需要补足这些证据后才能由 {skill_name} 继续生成。\n\n"
        "## 门禁原因\n\n" + "；".join(failures)
    )

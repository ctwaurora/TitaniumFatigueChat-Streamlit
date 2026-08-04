"""Lightweight, evidence-aware quality gate for web research answers."""

from __future__ import annotations

import re
from typing import Any

from src.research_topics import query_mentions_pores
from src.research_skills.common import is_usable_formula_record


COUNT_TALK = re.compile(r"(?:提供|检索到|召回|使用|共有|共检索)\s*\d+\s*(?:条|篇).{0,10}(?:证据|文献)")
EVIDENCE_COUNT_PHRASES = (
    "本地正式库提供", "条支持证据", "条反向证据", "条条件依赖证据",
    "Evidence数量", "支持证据数量", "反向证据数量", "条件证据数量",
    "当前使用多少篇文献", "本次调用了多少条证据", "检索池", "候选池", "reranker数量",
)
PLACEHOLDERS = ("目标因素", "问题中的目标因素", "某个因素", "疲劳响应发生变化")
PORE_PATTERN = re.compile(r"孔隙|孔洞|气孔|√area|\bpore\b|porosity", re.I)


SKILL_REQUIREMENTS = {
    "scientific_analysis_skill": ("结论", "机制", "条件", "反向", "不能确定"),
    "research_gap_skill": ("研究空白", "已有", "缺少", "反证", "最低成本"),
    "hypothesis_generation_skill": ("候选假设", "自变量", "因变量", "控制变量"),
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
    frame = evidence_bundle.get("query_frame") or {}
    if not str(answer).strip():
        failures.append("empty_answer")
    if COUNT_TALK.search(answer):
        failures.append("visible_evidence_count_talk")
    if any(term.casefold() in answer.casefold() for term in EVIDENCE_COUNT_PHRASES):
        failures.append("visible_evidence_count_talk")
    if any(term in answer for term in PLACEHOLDERS):
        failures.append("vague_placeholder")
    for term in SKILL_REQUIREMENTS.get(skill_name, ()):
        if term not in answer:
            failures.append(f"missing_section_or_concept:{term}")
    if skill_name == "hypothesis_generation_skill":
        if not any(term in answer for term in ("推翻", "证伪")):
            failures.append("missing_section_or_concept:falsification")
        if not any(term in answer for term in ("验证", "检验", "复现")):
            failures.append("missing_section_or_concept:validation")
    if skill_name == "scientific_analysis_skill":
        direct_block = answer.split("###", 1)[0]
        if not any(term in direct_block for term in ("提高", "降低", "增大", "减小", "相关", "影响", "作用")):
            failures.append("direct_answer_missing_scientific_conclusion")
        if not any(term in answer for term in ("机制", "驱动力", "应力集中", "裂纹", "组织")):
            failures.append("direct_answer_missing_mechanism")

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

    alloy = str(frame.get("alloy_grade") or "")
    if alloy and alloy.casefold() not in answer.casefold():
        failures.append("query_entity_missing:alloy_grade")
    entity_aliases = {
        "residual_stress": ("残余应力", "residual stress"),
        "alpha_lath_width": ("α片层", "片层宽度", "alpha lath"),
        "crack_growth_rate_da_dn": ("da/dN", "裂纹扩展速率"),
        "short_crack_growth_rate": ("短裂纹", "short crack"),
        "fatigue_life_Nf": ("疲劳寿命", "Nf"),
        "fatigue_limit": ("疲劳极限", "fatigue limit"),
    }
    # QueryFrame and the legacy variable mapper use different canonical names.
    # Normalize both to visible scientific terminology without weakening the
    # requirement that an answer explicitly addresses the queried entities.
    entity_aliases.update({
        "residual_stress": ("\u6b8b\u4f59\u5e94\u529b", "residual stress"),
        "alpha_lath_width": ("\u03b1\u7247\u5c42", "\u7247\u5c42\u5bbd\u5ea6", "alpha lath"),
        "microstructure": ("\u5fae\u89c2\u7ec4\u7ec7", "\u663e\u5fae\u7ec4\u7ec7", "microstructure"),
        "crack_growth_rate_da_dn": ("da/dn", "\u88c2\u7eb9\u6269\u5c55\u901f\u7387", "crack growth rate"),
        "da_dn": ("da/dn", "\u88c2\u7eb9\u6269\u5c55\u901f\u7387", "crack growth rate"),
        "short_crack_growth_rate": ("\u77ed\u88c2\u7eb9", "\u5c0f\u88c2\u7eb9", "short crack"),
        "short_crack": ("\u77ed\u88c2\u7eb9", "\u5c0f\u88c2\u7eb9", "short crack"),
        "long_crack": ("\u957f\u88c2\u7eb9", "long crack"),
        "fatigue_life_Nf": ("\u75b2\u52b3\u5bff\u547d", "nf", "fatigue life"),
        "fatigue_life": ("\u75b2\u52b3\u5bff\u547d", "nf", "fatigue life"),
        "fatigue_limit": ("\u75b2\u52b3\u6781\u9650", "\u75b2\u52b3\u5f3a\u5ea6", "fatigue limit"),
        "delta_k": ("\u0394k", "delta k", "\u5e94\u529b\u5f3a\u5ea6\u56e0\u5b50\u8303\u56f4"),
        "effective_delta_k": ("\u0394keff", "delta keff", "\u6709\u6548\u5e94\u529b\u5f3a\u5ea6\u56e0\u5b50"),
        "surface_roughness": ("\u8868\u9762\u7c97\u7cd9\u5ea6", "ra", "rz", "surface roughness"),
        "build_orientation": ("\u5efa\u9020\u65b9\u5411", "\u6210\u5f62\u65b9\u5411", "build orientation", "build direction"),
        "heat_treatment": ("\u70ed\u5904\u7406", "hip", "\u70ed\u7b49\u9759\u538b", "heat treatment"),
        "hip": ("hip", "\u70ed\u7b49\u9759\u538b", "hot isostatic"),
        "stress_amplitude": ("\u5e94\u529b\u5e45", "\u03c3a", "stress amplitude"),
        "stress_ratio": ("\u5e94\u529b\u6bd4", "r=", "stress ratio"),
        "pore_size": ("\u5b54\u9699\u5c3a\u5bf8", "\u7f3a\u9677\u5c3a\u5bf8", "pore size", "sqrt area"),
        "porosity": ("\u5b54\u9699\u7387", "porosity"),
        "pore_location": ("\u5b54\u9699\u4f4d\u7f6e", "\u8ddd\u8868\u9762\u8ddd\u79bb", "pore location"),
        "paris_c_m": ("paris", "c\u548cm", "c/m"),
        "crack_closure": ("\u88c2\u7eb9\u95ed\u5408", "crack closure"),
        "crack_initiation_life": ("\u8d77\u88c2\u5bff\u547d", "crack initiation life"),
    })
    for entity in [*(frame.get("independent_variables") or []), *(frame.get("dependent_variables") or [])]:
        aliases = entity_aliases.get(str(entity), (str(entity),))
        if not any(alias.casefold() in answer.casefold() for alias in aliases):
            failures.append(f"query_entity_missing:{entity}")

    valid_pages = {
        str(item.get("page_number"))
        for item in [*citations.values(), *(evidence_bundle.get("formulas") or [])]
        if item.get("page_number")
    }
    # A page citation must use an explicit ``p.`` or page label. Treating any
    # letter p followed by digits as a page misclassifies formula parameters.
    cited_pages = set(re.findall(r"(?:p\.|页码[：:]?)\s*(\d+)", answer, re.I))
    if cited_pages and not cited_pages.issubset(valid_pages):
        failures.append("invalid_page_reference")

    formulas = [
        item for item in evidence_bundle.get("formulas") or []
        if is_usable_formula_record(item)
    ]
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
    if frame.get("crack_stage") == "SHORT_CRACK" and formulas:
        long_crack_formula = any(
            str(item.get("crack_stage") or "").upper() in {"LONG_CRACK", "PARIS"}
            or "paris" in str(item.get("equation") or "").casefold()
            for item in formulas
        )
        if long_crack_formula and not any(term in answer for term in ("长裂纹基线", "不能直接", "不适用于短裂纹", "不可直接")):
            failures.append("long_crack_formula_applied_to_short_crack")

    if not query_mentions_pores(question):
        pore_hits = len(PORE_PATTERN.findall(answer))
        headings = re.findall(r"(?m)^#{1,4}\s+(.+)$", answer)
        if pore_hits > 2 or any(PORE_PATTERN.search(heading) for heading in headings):
            failures.append("unsolicited_pore_centrality")

    if skill_name in {"hypothesis_generation_skill", "experiment_design_skill"}:
        falsifiers = len(re.findall(r"(?:推翻|证伪|置信区间|无法复现|相反)", answer))
        if falsifiers < 2:
            failures.append("insufficient_falsification_criteria")
    if skill_name == "hypothesis_generation_skill":
        if not any(term in answer for term in ("文献原公式", "待拟合候选模型", "当前没有可核验公式")):
            failures.append("formula_provenance_not_distinguished")
        if "待拟合候选模型" in answer and re.search(r"β\d+\s*=\s*-?\d", answer):
            failures.append("fabricated_candidate_parameter")
        for term in ("数据要求", "拟合", "替代解释"):
            if term not in answer:
                failures.append(f"hypothesis_missing:{term}")
        has_model = bool(re.search(r"(?:β\d|log10?|log\(|Nf|da/dN).{0,120}=|=.{0,120}(?:β\d|log10?|Nf|da/dN)", answer, re.I | re.S))
        data_limited = "数据不足以确定函数形式" in answer or "不足以确定具体函数形式" in answer
        if not has_model and not data_limited:
            failures.append("hypothesis_missing_fittable_formula")
        if not all(term in answer for term in ("单位", "预测方向")):
            failures.append("hypothesis_missing_variable_units_or_direction")
    if skill_name == "research_gap_skill":
        if not any(term in answer for term in ("候选证据缺口", "真实研究空白", "本地文献库不足", "已有研究已经")):
            failures.append("gap_status_not_classified")
        if "缺失条件" not in answer and "缺口矩阵" not in answer:
            failures.append("gap_condition_matrix_missing")
        for term in ("已经解决的问题", "尚未解决的具体问题", "反向证据检索结果", "逐层缩小"):
            if term not in answer:
                failures.append(f"gap_missing:{term}")
        if not any(term in answer for term in (
            "A. 得到支持的具体研究空白",
            "B. 候选证据缺口，尚需外部文献验证",
            "C. 候选空白被反向证据推翻",
        )):
            failures.append("gap_status_not_one_of_ABC")
        vague_only = len(answer) < 500 and any(term in answer for term in ("关系尚不明确", "需要进一步研究", "文献不足"))
        if vague_only:
            failures.append("gap_overly_generic")
    if skill_name == "experiment_design_skill":
        for term in ("水平", "测量", "协变量", "样本量", "run-out", "统计模型"):
            if term.casefold() not in answer.casefold():
                failures.append(f"experiment_missing:{term}")
        table_requirements = (
            ("表1：变量定义表", "| 变量类型 | 变量 |"),
            ("表2：实验分组表", "| 组别 |"),
            ("表3：预测与证伪表", "| 检验项目 | 假设成立时的预测结果 |"),
        )
        for label, header in table_requirements:
            if label not in answer or header not in answer:
                failures.append(f"experiment_missing_table:{label}")
        if answer.count("|---") < 3:
            failures.append("experiment_markdown_tables_missing")
        if not any(term in answer for term in ("回归", "似然比", "置信区间", "生存分析", "统计模型")):
            failures.append("experiment_statistical_test_missing")
    if "研究已经证明" in answer and any(term in answer for term in ("系统推断", "候选模型", "候选假设")):
        failures.append("system_inference_presented_as_fact")
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
        "query_frame_checked": bool(frame),
        "formula_applicability_checked": True,
        "evidence_level_checked": True,
    }


def precise_refusal(skill_name: str, failures: list[str], evidence_bundle: dict[str, Any]) -> str:
    missing = evidence_bundle.get("synthesis", {}).get("missing_conditions") or []
    details = "、".join(missing[:5]) or "可核验的页码、实验条件或反向证据"
    original_query = str((evidence_bundle.get("query_frame") or {}).get("original_query") or "").strip()
    query_context = f"针对问题“{original_query}”，" if original_query else ""
    return (
        "## 直接结论\n\n"
        f"{query_context}当前证据不足以形成满足具体性与可证伪要求的科研答案。以下列出需要补充的文献或实验条件。\n\n"
        "## 当前不能确定的内容\n\n"
        f"当前正式证据缺少或无法一致核对：{details}。需要补足这些证据后才能由 {skill_name} 继续生成。\n\n"
        "## 仍需补足的证据\n\n"
        "需要补充同材料状态、同疲劳阶段的匹配实验，核对公式参数及单位，并检索能够支持或推翻主结论的直接研究。"
    )

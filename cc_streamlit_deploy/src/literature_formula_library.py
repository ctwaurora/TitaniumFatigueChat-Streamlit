"""Read-only view models for formulas extracted during real-page deep reading."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


LITERATURE_FORMULA_SOURCE = "LITERATURE_FORMULA"
SYSTEM_MODEL_REGISTRY = "SYSTEM_MODEL_REGISTRY"
FORMULA_IMAGE_REVIEW_REQUIRED = "FORMULA_IMAGE_REVIEW_REQUIRED"
FORMULA_EMPTY_MESSAGE = "当前可信文献中尚未提取到可确认公式。"

FORMULA_TYPES = [
    "全部",
    "S-N",
    "缺陷疲劳",
    "裂纹扩展",
    "应变寿命",
    "统计或拟合模型",
    "其他",
]

EVIDENCE_STATUSES = ["已确认", "待人工复核"]

FORMULA_TABLE_FIELDS = [
    "论文题目",
    "作者与年份",
    "真实页码",
    "章节",
    "原始公式",
    "标准化公式",
    "变量解释",
    "公式用途",
    "适用条件",
    "参数值与单位",
    "证据状态",
]


SYSTEM_THEORY_MODELS: List[Dict[str, str]] = [
    {
        "name": "Basquin",
        "formula": "sigma_a = sigma_f' * (2Nf)^b",
        "purpose": "高周疲劳 S-N 应力寿命关系参考。",
    },
    {
        "name": "Murakami sqrt(area)",
        "formula": "sigma_w = A * (HV + 120) / (sqrt_area)^(1/6)",
        "purpose": "缺陷尺寸与疲劳极限关系参考。",
    },
    {
        "name": "Kitagawa-Takahashi",
        "formula": "Delta_sigma_threshold = Delta_Kth / sqrt(pi * a)",
        "purpose": "裂纹或缺陷尺寸与门槛应力范围关系参考。",
    },
    {
        "name": "El Haddad",
        "formula": "Delta_sigma = Delta_Kth / sqrt(pi * (a + a0))",
        "purpose": "小裂纹修正参考。",
    },
    {
        "name": "Paris law",
        "formula": "da/dN = C * (Delta_K)^m",
        "purpose": "稳定长裂纹扩展区关系参考。",
    },
    {
        "name": "Coffin-Manson",
        "formula": "Delta_epsilon_p / 2 = epsilon_f' * (2Nf)^c",
        "purpose": "低周疲劳塑性应变寿命关系参考。",
    },
]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _manifest_index(base_dir: Path) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("paper_id") or ""): row
        for row in _read_jsonl(base_dir / "data" / "paper_manifest.jsonl")
        if row.get("paper_id")
    }


def _page_section_index(paper_dir: Path) -> Dict[int, str]:
    sections: Dict[int, str] = {}
    for row in _read_jsonl(paper_dir / "page_records.jsonl"):
        try:
            page_number = int(row.get("page_number") or 0)
        except (TypeError, ValueError):
            continue
        if page_number > 0:
            sections[page_number] = str(row.get("section_title") or "未分类")
    return sections


def _extract_formula_text(original_text: str, latex_candidate: str) -> str:
    if latex_candidate.strip():
        return latex_candidate.strip()
    lines = [" ".join(line.split()) for line in original_text.splitlines()]
    lines = [line for line in lines if line]
    equation_indexes = [
        index
        for index, line in enumerate(lines)
        if "=" in line
        or re.search(r"\b(?:da\s*/\s*dN|Delta\s*K|ΔK|sqrt|logit|log10)\b", line, re.I)
    ]
    if not equation_indexes:
        return ""
    start = max(0, equation_indexes[0] - 1)
    end = min(len(lines), equation_indexes[0] + 6)
    return "\n".join(lines[start:end])


def _formula_type(text: str) -> str:
    lower = text.lower()
    if any(term in lower for term in ("da/dn", "delta k", "δk", "crack growth", "paris")):
        return "裂纹扩展"
    if any(term in lower for term in ("strain life", "coffin", "plastic strain", "epsilon")):
        return "应变寿命"
    if any(term in lower for term in ("sqrt area", "√area", "defect", "pore", "edge distance")):
        return "缺陷疲劳"
    if any(term in lower for term in ("s-n", "basquin", "fatigue life", "number of cycles")):
        return "S-N"
    if any(term in lower for term in ("regression", "fitting", "model", "neural network", "probability")):
        return "统计或拟合模型"
    return "其他"


def _equation_number(text: str) -> str:
    match = re.search(r"(?:equation\s*)?\((\d+[a-z]?)\)", text, re.I)
    return match.group(1) if match else "未确认"


def _source_sentences(text: str, patterns: Iterable[str], limit: int = 3) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", " ".join(text.split()))
    matches = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip() and any(re.search(pattern, sentence, re.I) for pattern in patterns)
    ]
    return matches[:limit]


def _authors_and_year(metadata: Mapping[str, Any]) -> str:
    authors = metadata.get("authors") or "作者未记录"
    if isinstance(authors, list):
        authors = "; ".join(str(item) for item in authors)
    year = metadata.get("publication_date") or metadata.get("year") or "年份未记录"
    return f"{authors} ({year})"


def load_literature_formulas(base_dir: Path) -> List[Dict[str, Any]]:
    """Load only real-page equation records; never substitute registry formulas."""
    root = Path(base_dir).resolve()
    manifest = _manifest_index(root)
    deep_read_root = root / "data" / "deep_read"
    formulas: List[Dict[str, Any]] = []
    if not deep_read_root.exists():
        return formulas

    for equations_path in sorted(deep_read_root.glob("*/equations.jsonl")):
        paper_dir = equations_path.parent
        sections = _page_section_index(paper_dir)
        for index, raw in enumerate(_read_jsonl(equations_path), start=1):
            paper_id = str(raw.get("paper_id") or paper_dir.name).strip()
            try:
                page_number = int(raw.get("page_number") or 0)
            except (TypeError, ValueError):
                continue
            if not paper_id or page_number <= 0:
                continue
            metadata = manifest.get(paper_id, {})
            original_context = str(raw.get("original_text") or "").strip()
            latex_candidate = str(raw.get("latex_candidate") or "").strip()
            original_formula = _extract_formula_text(original_context, latex_candidate)
            raw_review_status = str(raw.get("review_status") or "").upper()
            confirmed = raw_review_status in {"CONFIRMED", "HUMAN_CONFIRMED", "VERIFIED"}
            evidence_status = "已确认" if confirmed else "待人工复核"
            audit_status = (
                raw_review_status or "TEXT_EXTRACTED"
                if original_formula
                else FORMULA_IMAGE_REVIEW_REQUIRED
            )
            symbol_definitions = _source_sentences(original_context, (r"\bwhere\b", r"denotes", r"defined as"))
            unit_sources = _source_sentences(
                original_context,
                (r"\b(?:MPa|GPa|Pa|Hz|kHz|mm|μm|um|m/cycle|cycles?)\b",),
            )
            condition_sources = _source_sentences(
                original_context,
                (r"\bR\s*=", r"temperature", r"condition", r"frequency", r"loading"),
            )
            normalized_formula = latex_candidate or "待人工复核，不自动补写"
            formula = {
                "formula_id": f"{paper_id}:p{page_number}:e{index}",
                "source_type": LITERATURE_FORMULA_SOURCE,
                "paper_id": paper_id,
                "paper_title": str(metadata.get("title") or "题名未记录"),
                "authors_year": _authors_and_year(metadata),
                "doi": str(metadata.get("doi") or "未记录"),
                "page_number": page_number,
                "section": sections.get(page_number, "未分类"),
                "equation_number": _equation_number(original_context),
                "original_formula": original_formula or FORMULA_IMAGE_REVIEW_REQUIRED,
                "normalized_formula": normalized_formula,
                "normalized_latex": latex_candidate,
                "context_before_after": original_context,
                "symbol_definitions": symbol_definitions or ["未从公式证据片段中结构化确认。"],
                "symbol_units": unit_sources or ["未从公式证据片段中结构化确认。"],
                "parameter_values_units": unit_sources or ["未从公式证据片段中结构化确认。"],
                "formula_type": _formula_type(original_context),
                "formula_purpose": "根据公式原文上下文分类，需人工复核用途。",
                "applicable_conditions": condition_sources or ["原文公式片段未结构化报告适用条件。"],
                "data_source": str(equations_path),
                "author_scope": "原文公式片段未结构化报告完整适用范围。",
                "author_limitations": "原文公式片段未结构化报告完整局限。",
                "evidence_status": evidence_status,
                "manual_review_status": audit_status,
                "raw_review_status": raw_review_status or "TEXT_EXTRACTED",
            }
            formulas.append(formula)
    return formulas


def filter_literature_formulas(
    formulas: Iterable[Mapping[str, Any]],
    paper_id: Optional[str] = None,
    formula_type: str = "全部",
    evidence_status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for formula in formulas:
        if formula.get("source_type") != LITERATURE_FORMULA_SOURCE:
            continue
        if paper_id and formula.get("paper_id") != paper_id:
            continue
        if formula_type != "全部" and formula.get("formula_type") != formula_type:
            continue
        if evidence_status and formula.get("evidence_status") != evidence_status:
            continue
        rows.append(dict(formula))
    return rows


def formula_to_table_row(formula: Mapping[str, Any]) -> Dict[str, Any]:
    row = {
        "论文题目": formula.get("paper_title", ""),
        "作者与年份": formula.get("authors_year", ""),
        "真实页码": formula.get("page_number", ""),
        "章节": formula.get("section", ""),
        "原始公式": formula.get("original_formula", ""),
        "标准化公式": formula.get("normalized_formula", ""),
        "变量解释": "；".join(formula.get("symbol_definitions", [])),
        "公式用途": formula.get("formula_purpose", ""),
        "适用条件": "；".join(formula.get("applicable_conditions", [])),
        "参数值与单位": "；".join(formula.get("parameter_values_units", [])),
        "证据状态": formula.get("evidence_status", ""),
    }
    return {field: row[field] for field in FORMULA_TABLE_FIELDS}


def load_system_model_registry() -> List[Dict[str, str]]:
    return [dict(model, source_type=SYSTEM_MODEL_REGISTRY) for model in SYSTEM_THEORY_MODELS]


def formula_summary(formulas: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    rows = list(formulas)
    return {
        "total": len(rows),
        "confirmed": sum(row.get("evidence_status") == "已确认" for row in rows),
        "pending_review": sum(row.get("evidence_status") == "待人工复核" for row in rows),
        "image_review_required": sum(
            row.get("manual_review_status") == FORMULA_IMAGE_REVIEW_REQUIRED
            for row in rows
        ),
    }

"""Strict evidence snippet trace.

This module extracts only claim-supporting text snippets. Background sentences
such as “Ti-6Al-4V is widely used” are excluded from core evidence.
"""
from __future__ import annotations

import csv
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
TRUSTED_EVIDENCE_PATH = DATA_DIR / "evidence" / "trusted_evidence.csv"

from skills.library_skill import get_all_papers

VALID_TYPES = [
    "pore_fatigue_life", "pore_crack_initiation", "fcgr_da_dN", "paris_walker_model",
    "surface_roughness_fatigue", "microCT_defect", "SEM_fractography", "EBSD_microstructure",
    "heat_treatment_FCGR", "HCF_VHCF_internal_crack", "titanium_fatigue_general",
]

TYPE_RULES: List[Tuple[str, List[str], str]] = [
    ("pore_crack_initiation", ["pore", "porosity", "defect", "lack-of-fusion", "lack of fusion", "crack initiation", "initiation", "起裂", "萌生"], "孔隙/缺陷特征影响疲劳裂纹起裂位置"),
    ("pore_fatigue_life", ["pore", "porosity", "defect", "lack-of-fusion", "lack of fusion", "fatigue life", "s-n", "疲劳寿命", "疲劳强度"], "孔隙/缺陷特征影响疲劳寿命或疲劳强度"),
    ("fcgr_da_dN", ["fcgr", "da/dn", "delta k", "δk", "Δk", "crack growth rate", "crack propagation", "裂纹扩展速率"], "裂纹扩展行为与 da/dN-ΔK 或 FCGR 相关"),
    ("paris_walker_model", ["paris", "walker", "c/m", "c and m", "paris law"], "Paris/Walker 参数可用于描述裂纹扩展行为"),
    ("surface_roughness_fatigue", ["roughness", "surface", "as-built", "ra", "rz", "粗糙度", "表面"], "表面状态/粗糙度影响疲劳性能"),
    ("microCT_defect", ["micro-ct", "micro ct", "x-ray ct", "computed tomography", "μct"], "micro-CT 可用于孔隙/缺陷三维表征"),
    ("SEM_fractography", ["sem", "fractograph", "fracture surface", "断口"], "SEM/断口分析可用于识别起裂位置或断裂机制"),
    ("EBSD_microstructure", ["ebsd", "electron backscatter", "grain orientation", "晶粒取向"], "EBSD 可用于组织/取向与裂纹路径分析"),
    ("heat_treatment_FCGR", ["heat treatment", "hip", "hot isostatic", "anneal", "热处理", "热等静压"], "热处理/HIP 影响缺陷状态、组织或裂纹扩展行为"),
    ("HCF_VHCF_internal_crack", ["vhcf", "very high cycle", "hcf", "internal crack", "fish-eye", "内部裂纹", "超高周"], "HCF/VHCF 中内部缺陷或内部裂纹与疲劳失效相关"),
]

BAD_BACKGROUND = [
    "widely used", "attractive method", "low waste", "biomedical", "aerospace industry",
    "review discusses", "this review", "has been widely", "本文综述", "广泛应用",
]
TRUNC_PATTERNS = [r"\b[a-zA-Z]{6,}$", r"提取\s*[A-Z]?\s*$", r"initiati$", r"treat$", r"perform$", r"further\s+t$", r"\bpo$", r"\bmanufa$"]


def _safe_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    if not isinstance(x, str):
        return [] if x is None else [str(x)]
    s = x.strip()
    if not s or s.lower() == "nan":
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            v = parser(s)
            if isinstance(v, list):
                return [str(i) for i in v if str(i).strip()]
        except Exception:
            pass
    return [s]

def _first_author(authors_raw: Any) -> str:
    authors = _safe_list(authors_raw)
    if not authors:
        return "Unknown"
    first = authors[0].strip()
    # Handle "Surname, Initial" first.
    if "," in first:
        surname = first.split(",", 1)[0].strip()
        if len(surname) > 1:
            return re.sub(r"\s+[a-z](?:,[a-z])*\b", "", surname, flags=re.I).strip() or surname
    tokens = [t for t in re.split(r"\s+", first.replace(",", " ")) if t]
    if not tokens:
        return first[:30]
    # Remove affiliation suffixes such as a,b or a,c.
    tokens = [tok for tok in tokens if not re.fullmatch(r"[a-z](?:,[a-z])+", tok.lower())]
    for token in reversed(tokens):
        clean = token.strip().rstrip(".")
        if len(clean) > 1 and clean.lower() not in {"et", "al", "and"} and not re.fullmatch(r"[A-Z]", clean):
            return clean
    return first[:30]

def _sentences(text: str) -> List[str]:
    if not text:
        return []
    # Convert JSON-like lists to individual strings first.
    parts = _safe_list(text)
    out: List[str] = []
    for part in parts:
        part = re.sub(r"\s+", " ", part).strip()
        for s in re.split(r"(?<=[。.!?])\s+", part):
            s = s.strip(" '[]\"，,;；")
            if 35 <= len(s) <= 280:
                out.append(s)
    return out


def _is_truncated(s: str) -> bool:
    if not s or len(s) < 35:
        return True
    if any(b in s.lower() for b in BAD_BACKGROUND):
        return True
    if s.count("[") or s.count("]"):
        return True
    if s.endswith((",", ";", "，", "；", ":")):
        return True
    for pat in TRUNC_PATTERNS:
        if re.search(pat, s.strip(), flags=re.I):
            return True
    return False


def _classify_and_claim(s: str) -> Tuple[str, str, str, str, str]:
    """Classify a snippet using strict, claim-oriented rules.

    Avoid substring traps such as `ra` inside `microstructure` and avoid treating
    generic `fracture surface` as surface roughness evidence.
    """
    sl = s.lower()
    if not any(k in sl for k in ["fatigue", "crack", "fcgr", "da/dn", "paris", "walker", "疲劳", "裂纹"]):
        return "", "", "", "", ""

    variable = ""
    indicator = ""
    mechanism = ""

    if any(k in sl for k in ["pore", "porosity", "defect", "lack of fusion", "lack-of-fusion", "孔隙", "缺陷"]):
        variable = "pore/defect"
    elif any(k in sl for k in ["roughness", "as-built", "粗糙度"]):
        variable = "surface_roughness"
    elif any(k in sl for k in ["heat treatment", "hip", "hot isostatic", "anneal", "热处理", "热等静压"]):
        variable = "heat_treatment"
    elif any(k in sl for k in ["microstructure", "grain", "lath", "alpha", "β", "microstructural", "组织"]):
        variable = "microstructure"

    if any(k in sl for k in ["fatigue life", "s-n", "sn curve", "cycles to failure", "nf", "疲劳寿命", "疲劳强度"]):
        indicator = "Nf/S-N"
    elif any(k in sl for k in ["da/dn", "fcgr", "crack growth rate", "delta k", "Δk", "threshold", "Δkth"]):
        indicator = "da/dN-ΔK/FCGR"
    elif "paris" in sl or "walker" in sl:
        indicator = "Paris/Walker parameters"

    if any(k in sl for k in ["initiation", "initiated", "crack origin", "起裂", "萌生"]):
        mechanism = "crack_initiation"
    elif any(k in sl for k in ["propagation", "growth", "扩展"]):
        mechanism = "crack_growth"

    # Priority rules: mechanism/model evidence should not be swallowed by generic surface terms.
    if "paris" in sl or "walker" in sl:
        return "paris_walker_model", variable, indicator or "Paris/Walker parameters", mechanism or "crack_growth", "Paris/Walker 参数可用于描述或修正裂纹扩展行为"
    if any(k in sl for k in ["da/dn", "fcgr", "crack growth rate", "delta k", "Δk", "crack growth behavior", "crack propagation", "threshold", "Δkth"]):
        return "fcgr_da_dN", variable, indicator or "da/dN-ΔK/FCGR", mechanism or "crack_growth", "裂纹扩展行为与 da/dN-ΔK 或 FCGR 相关"
    if any(k in sl for k in ["micro-ct", "micro ct", "x-ray ct", "computed tomography", "μct"]):
        return "microCT_defect", variable or "pore/defect", indicator, mechanism, "micro-CT 可用于孔隙/缺陷三维表征"
    if any(k in sl for k in ["ebsd", "electron backscatter", "grain orientation", "晶粒取向"]):
        return "EBSD_microstructure", variable or "microstructure", indicator, mechanism, "EBSD 可用于组织/取向与裂纹路径分析"
    if any(k in sl for k in ["sem", "fractograph", "fracture surface", "断口"]):
        return "SEM_fractography", variable, indicator, mechanism, "SEM/断口分析可用于识别起裂位置或断裂机制"
    if any(k in sl for k in ["heat treatment", "hip", "hot isostatic", "anneal", "热处理", "热等静压"]):
        return "heat_treatment_FCGR", variable or "heat_treatment", indicator, mechanism, "热处理/HIP 影响缺陷状态、组织或裂纹扩展行为"
    if any(k in sl for k in ["vhcf", "very high cycle", "hcf", "internal crack", "fish-eye", "内部裂纹", "超高周"]):
        return "HCF_VHCF_internal_crack", variable, indicator or "Nf/S-N", mechanism or "crack_initiation", "HCF/VHCF 中内部缺陷或内部裂纹与疲劳失效相关"
    # Surface-roughness-dominant statements should not be reclassified as pore evidence just
    # because the sentence contrasts roughness with internal defects.
    if any(k in sl for k in ["roughness", "as-built", "粗糙度"]):
        if ("dominated by the surface roughness" in sl) or ("rough" in sl and "surface" in sl) or not any(k in sl for k in ["pore", "porosity", "lack of fusion", "lack-of-fusion", "keyhole"]):
            return "surface_roughness_fatigue", "surface_roughness", indicator or "Nf/S-N", mechanism, "表面状态/粗糙度影响疲劳性能"
    if any(k in sl for k in ["pore", "porosity", "defect", "lack of fusion", "lack-of-fusion", "孔隙", "缺陷"]):
        if mechanism == "crack_initiation":
            return "pore_crack_initiation", variable, indicator, mechanism, "孔隙/缺陷特征影响疲劳裂纹起裂位置"
        return "pore_fatigue_life", variable, indicator or "Nf/S-N", mechanism, "孔隙/缺陷特征影响疲劳寿命或疲劳强度"
    if any(k in sl for k in ["roughness", "as-built", "粗糙度"]):
        return "surface_roughness_fatigue", "surface_roughness", indicator or "Nf/S-N", mechanism, "表面状态/粗糙度影响疲劳性能"
    return "titanium_fatigue_general", "", indicator, mechanism, "支持钛合金疲劳行为分析"


def run_evidence_trace() -> Dict[str, Any]:
    """Report only reviewed Stage-1 evidence.

    The former implementation below synthesized claim snippets from metadata
    fields and wrote them to the legacy evidence file.  It remains in place for
    rollback, but is unreachable until a later stage supplies page-bound text.
    """
    trusted: List[Dict[str, str]] = []
    if TRUSTED_EVIDENCE_PATH.exists():
        with TRUSTED_EVIDENCE_PATH.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            trusted = list(csv.DictReader(handle))
    type_counts: Dict[str, int] = {}
    for row in trusted:
        evidence_type = row.get("evidence_type", "trusted") or "trusted"
        type_counts[evidence_type] = type_counts.get(evidence_type, 0) + 1
    return {
        "total_snippets": len(trusted),
        "type_counts": type_counts,
        "paper_count": len({row.get("paper_id", "") for row in trusted}),
        "status": "STAGE1_TRUSTED_ONLY",
    }

    # Legacy implementation retained below for rollback/reference only.
    papers = [p for p in get_all_papers() if p.get("alloy_type") != "out_of_scope"]
    snippets: List[Dict[str, str]] = []
    paper_id_by_title: Dict[str, str] = {}
    for idx, p in enumerate(papers, 1):
        title = str(p.get("title", "")).strip()
        if not title:
            continue
        pid = f"P{idx:02d}"
        paper_id_by_title[title] = pid
        author = _first_author(p.get("authors", ""))
        year = str(p.get("year", "")).strip()[:4]
        author_year = f"{author} et al., {year}" if year else author
        texts = []
        # Prefer findings, abstract, conclusion, evidence-related fields; not title alone.
        for col in ["key_findings", "evidence_text", "crack_initiation", "crack_growth_mechanism", "mechanical_indicators", "conclusion"]:
            for sent in _sentences(str(p.get(col, ""))):
                texts.append((col, sent))
        used = 0
        for section, sent in texts:
            if _is_truncated(sent):
                continue
            ev_type, variable, indicator, mechanism, claim = _classify_and_claim(sent)
            if not ev_type or not claim:
                continue
            # For core output, require at least one concrete variable plus one fatigue indicator/mechanism.
            if not (variable or indicator or mechanism):
                continue
            snippets.append({
                "evidence_id": f"E{len(snippets)+1:04d}",
                "paper_id": pid,
                "author_year": author_year,
                "title": title,
                "evidence_type": ev_type,
                "snippet": sent,
                "source_section": section,
                "page_or_location": f"{section}_text",
                "linked_variable": variable,
                "linked_indicator": indicator,
                "linked_mechanism": mechanism,
                "linked_claim": claim,
                "confidence_level": "high" if ev_type != "titanium_fatigue_general" else "medium",
            })
            used += 1
            if used >= 5:
                break
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["evidence_id","paper_id","author_year","title","evidence_type","snippet","source_section","page_or_location","linked_variable","linked_indicator","linked_mechanism","linked_claim","confidence_level"]
    with (DATA_DIR / "evidence_snippets.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(snippets)
    type_counts: Dict[str, int] = {}
    for s in snippets:
        type_counts[s["evidence_type"]] = type_counts.get(s["evidence_type"], 0) + 1
    _write_report(snippets, type_counts, len(papers))
    return {"total_snippets": len(snippets), "type_counts": type_counts, "paper_count": len(papers)}


def _write_report(snippets: List[Dict[str,str]], type_counts: Dict[str,int], paper_count: int) -> None:
    desc = {
        "pore_fatigue_life":"孔隙/缺陷→疲劳寿命", "pore_crack_initiation":"孔隙/缺陷→裂纹起裂",
        "fcgr_da_dN":"裂纹扩展速率 da/dN-ΔK", "paris_walker_model":"Paris/Walker 模型参数",
        "surface_roughness_fatigue":"表面粗糙度→疲劳性能", "microCT_defect":"micro-CT 缺陷表征",
        "SEM_fractography":"SEM 断口分析", "EBSD_microstructure":"EBSD 微观组织",
        "heat_treatment_FCGR":"热处理→裂纹扩展", "HCF_VHCF_internal_crack":"HCF/VHCF 内部裂纹",
        "titanium_fatigue_general":"钛合金疲劳通用背景",
    }
    core = [s for s in snippets if s["evidence_type"] != "titanium_fatigue_general"]
    lines = ["# Evidence Trace Report（证据片段追溯报告）", "", "## 摘要", "", f"- **总证据片段数**: {len(snippets)}", f"- **来源文献数**: {paper_count}", "- **核心证据筛选原则**: 只保留能直接支持孔隙/缺陷、裂纹起裂、FCGR、表面粗糙度、热处理等 claim 的句子；背景句不进入 Top Evidence。", "", "## 按证据类型统计", "", "| evidence_type | 数量 | 说明 |", "|---|---:|---|"]
    for t in VALID_TYPES:
        if type_counts.get(t,0):
            lines.append(f"| {t} | {type_counts[t]} | {desc.get(t,'')} |")
    lines += ["", "## 与核心假设相关的 Top Evidence Snippets", ""]
    if not core:
        lines.append("未抽取到可通过核心规则的 evidence snippet。")
    priority = {
        "pore_crack_initiation": 0,
        "pore_fatigue_life": 1,
        "fcgr_da_dN": 2,
        "paris_walker_model": 3,
        "surface_roughness_fatigue": 4,
        "heat_treatment_FCGR": 5,
        "HCF_VHCF_internal_crack": 6,
        "microCT_defect": 7,
        "SEM_fractography": 8,
        "EBSD_microstructure": 9,
    }
    top_core = sorted(core, key=lambda x: (priority.get(x["evidence_type"], 99), x["paper_id"], x["evidence_id"]))
    for i,s in enumerate(top_core[:15],1):
        lines += [f"### {i}. {s['evidence_id']}", f"- **证据类型**: {s['evidence_type']}", f"- **文献**: {s['paper_id']} | {s['author_year']}", f"- **支持 claim**: {s['linked_claim']}", f"- **片段**: {s['snippet']}", f"- **来源章节**: {s['source_section']}", ""]
    claim_map = {
        "孔隙/缺陷降低疲劳寿命":[s["evidence_id"] for s in core if s["evidence_type"]=="pore_fatigue_life"],
        "孔隙成为裂纹起裂源":[s["evidence_id"] for s in core if s["evidence_type"]=="pore_crack_initiation"],
        "da/dN-ΔK 或 FCGR 受工艺/缺陷/组织影响":[s["evidence_id"] for s in core if s["evidence_type"]=="fcgr_da_dN"],
        "Paris/Walker 参数相关":[s["evidence_id"] for s in core if s["evidence_type"]=="paris_walker_model"],
        "表面粗糙度影响疲劳":[s["evidence_id"] for s in core if s["evidence_type"]=="surface_roughness_fatigue"],
        "micro-CT 缺陷表征":[s["evidence_id"] for s in core if s["evidence_type"]=="microCT_defect"],
    }
    lines += ["## 关键 Claim → Evidence ID 映射", "", "| Claim | Evidence IDs |", "|---|---|"]
    for c,ids in claim_map.items():
        lines.append(f"| {c} | {', '.join(ids[:8]) if ids else '（暂无对应高质量证据）'} |")
    lines += ["", "## 当前证据追溯的不足", "", "1. 本模块只进行文本级证据抽取，不做 OCR 和图表曲线识别。", "2. 若核心 claim 缺少 evidence_id，必须在 Evidence Quality Gate 中降级。", "3. 背景句和截断句被排除后，snippet 数量会减少，但可信度更高。"]
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "09_evidence_trace_report.md").write_text("\n".join(lines), encoding="utf-8")

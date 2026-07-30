"""Strict evidence quality gate."""
from __future__ import annotations
import csv, re
from pathlib import Path
from typing import Any, Dict, List
from src.stage1_store import TRUSTED_EVIDENCE_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
VALID_TYPES = {"pore_fatigue_life","pore_crack_initiation","fcgr_da_dN","paris_walker_model","surface_roughness_fatigue","microCT_defect","SEM_fractography","EBSD_microstructure","heat_treatment_FCGR","HCF_VHCF_internal_crack","titanium_fatigue_general"}
CORE_TYPES = {"pore_fatigue_life","pore_crack_initiation","fcgr_da_dN","paris_walker_model","surface_roughness_fatigue","microCT_defect","SEM_fractography","EBSD_microstructure","heat_treatment_FCGR","HCF_VHCF_internal_crack"}
BACKGROUND_PATTERNS = ["widely used","attractive method","biomedical","aerospace industry","this review discusses","has been widely","广泛应用"]
TRUNC_PATTERNS = [r"initiati$", r"perform$", r"treat$", r"further\s+t$", r"\b[a-zA-Z]{9,}$"]

def _load() -> List[Dict[str,str]]:
    p = TRUSTED_EVIDENCE_PATH
    if not p.exists(): return []
    with p.open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def _failures(s: Dict[str,str]) -> List[str]:
    f=[]
    sn=(s.get("snippet") or "").strip()
    if len(sn)<40: f.append("evidence_snippet_empty_or_too_short")
    if any(x in sn.lower() for x in BACKGROUND_PATTERNS): f.append("background_sentence_not_core_evidence")
    if any(re.search(pat, sn, re.I) for pat in TRUNC_PATTERNS) or sn.endswith((",",";","，","；",":")): f.append("truncated_sentence")
    if not (s.get("author_year") or "").strip() or "Unknown" in (s.get("author_year") or ""): f.append("author_year_empty")
    if not (s.get("paper_id") or "").strip(): f.append("paper_id_empty")
    if (s.get("evidence_type") or "") not in VALID_TYPES: f.append("invalid_evidence_type")
    if len((s.get("linked_claim") or "").strip())<15: f.append("linked_claim_missing_or_too_short")
    if (s.get("confidence_level") or "").lower() not in {"high","medium"}: f.append("low_confidence")
    if not (s.get("source_section") or "").strip(): f.append("invalid_source_section")
    return f

def _level(total:int, passed:List[Dict[str,Any]], core:List[str]) -> str:
    if total == 0: return "no_evidence"
    pass_rate = len(passed)/total
    core_types = {r.get("evidence_type") for r in passed if r.get("evidence_id") in core}
    has_basic_chain = len(core_types & {"pore_fatigue_life","pore_crack_initiation","fcgr_da_dN"}) >= 2
    has_full_chain = {"pore_fatigue_life","pore_crack_initiation","fcgr_da_dN","microCT_defect","paris_walker_model"}.issubset(core_types)
    # Strict rule: no micro-CT / Paris evidence means the final hypothesis must stay preliminary.
    if len(core) < 3 or not has_basic_chain:
        return "weakly_grounded"
    if pass_rate < 0.30 or len(core) < 10:
        return "preliminary"
    if has_full_chain and pass_rate >= 0.50 and len(core) >= 15:
        return "evidence_supported_candidate"
    return "preliminary"

def run_evidence_gate() -> Dict[str,Any]:
    snippets=_load()
    results=[]
    for s in snippets:
        fs=_failures(s)
        r=dict(s); r["failures"]=fs; r["passed"]=not fs
        results.append(r)
    passed=[r for r in results if r["passed"]]
    failed=[r for r in results if not r["passed"]]
    reasons={}
    for r in failed:
        for x in r["failures"]: reasons[x]=reasons.get(x,0)+1
    core=[r["evidence_id"] for r in passed if r.get("evidence_type") in CORE_TYPES]
    background=[r["evidence_id"] for r in passed if r.get("evidence_type")=="titanium_fatigue_general"]
    excluded=[r["evidence_id"] for r in failed]
    level=_level(len(snippets), passed, core)
    _write(results, passed, failed, reasons, core, background, excluded, level)
    return {"total_snippets":len(snippets),"passed_snippets":len(passed),"failed_snippets":len(failed),"failure_reasons":reasons,"core_evidence_ids":core,"background_evidence_ids":background,"excluded_evidence_ids":excluded,"evidence_level":level}

def _write(results, passed, failed, reasons, core, background, excluded, level):
    total=len(results); pass_rate=(len(passed)/total*100 if total else 0)
    lines=["# Evidence Quality Gate Report（证据质量门禁报告）","","> **目的**: 防止低质量证据进入最终假设。",f"> **总 Evidence Snippets**: {total}",f"> **通过**: {len(passed)} | **未通过**: {len(failed)}",f"> **Pass rate**: {pass_rate:.1f}%",f"> **Evidence Level**: {level}","","---","","## 1. 判定原则","","- pass_rate < 30% 时，最高只能判为 preliminary。","- 核心 claim 缺少 evidence_id 时，不得判 evidence_supported。","- 背景句、截断句、无 linked_claim 的 snippet 不得进入 core evidence。","","## 2. 统计摘要","","| Metric | Value |","|---|---:|",f"| Total snippets | {total} |",f"| Passed | {len(passed)} |",f"| Failed | {len(failed)} |",f"| Core evidence IDs | {len(core)} |",f"| Background evidence IDs | {len(background)} |",f"| Excluded evidence IDs | {len(excluded)} |","","## 3. 失败原因分布","","| Failure Reason | Count |","|---|---:|"]
    if reasons:
        for k,v in sorted(reasons.items(), key=lambda x:-x[1]): lines.append(f"| {k} | {v} |")
    else: lines.append("| — | 0 |")
    lines += ["", "## 4. 核心证据 Core Evidence", ""]
    if core:
        by_id={r.get("evidence_id"):r for r in passed}
        for eid in core[:30]:
            r=by_id[eid]
            lines.append(f"- **{eid}** | {r.get('paper_id')} | {r.get('author_year')} | {r.get('evidence_type')} | {r.get('linked_claim')}")
    else:
        lines.append("无通过门禁的核心证据。")
    lines += ["", "## 5. 被排除证据示例", ""]
    for r in failed[:30]:
        lines.append(f"- **{r.get('evidence_id')}** | {r.get('paper_id')} | Failures: {', '.join(r.get('failures', []))}")
    lines += ["", "## 6. 结论", ""]
    if level == "evidence_supported_candidate":
        lines.append("当前证据可作为 evidence-supported candidate，但仍需人工检查关键 evidence snippet 与主假设的强相关性。")
    elif level == "preliminary":
        lines.append("当前证据等级为 preliminary。系统已获得可追溯证据，但通过率、核心证据数量或关键证据类型覆盖仍不足以声明 evidence_supported。")
    elif level == "weakly_grounded":
        lines.append("当前证据为 weakly_grounded。核心 claim 的可追溯证据不足，最终假设必须降级为 preliminary 或更低。")
    else:
        lines.append("当前无有效证据。")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR/"12_evidence_quality_gate.md").write_text("\n".join(lines), encoding="utf-8")

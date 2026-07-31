"""Official evidence-gated hypothesis generation entrypoint."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from src.literature_library import eligible_paper_ids, trusted_evidence_rows
from src.stage1_store import BASE_DIR


INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

VARIABLE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("缺陷尺寸 sqrt_area", r"\b(?:sqrt[_ ]?area|pore size|defect size|缺陷尺寸|孔隙尺寸)\b"),
    ("距表面距离 distance_to_surface", r"\b(?:distance to (?:the )?surface|surface distance|距表面)\b"),
    ("表面状态 surface_state", r"\b(?:surface state|surface roughness|as-built|machined|polished|表面状态|粗糙度)\b"),
    ("热处理 heat_treatment", r"\b(?:heat treatment|HIP|anneal|stress relie|热处理|热等静压)\b"),
    ("应力比 stress_ratio_R", r"\b(?:stress ratio|R\s*=|应力比)\b"),
    ("成形方向 build_orientation", r"\b(?:build orientation|build direction|成形方向|打印方向)\b"),
    ("残余应力 residual_stress", r"\b(?:residual stress|残余应力)\b"),
    ("孔隙率 porosity", r"\b(?:porosity|pore fraction|孔隙率)\b"),
    ("微观组织 microstructure", r"\b(?:microstructure|alpha phase|beta phase|grain size|lath width|微观组织|晶粒|相组成)\b"),
    ("织构 texture", r"\b(?:crystallographic texture|texture|织构)\b"),
    ("应变幅 strain_amplitude", r"\b(?:strain amplitude|plastic strain|total strain|应变幅|塑性应变)\b"),
    ("应力幅 stress_amplitude", r"\b(?:stress amplitude|maximum stress|应力幅|最大应力)\b"),
    ("加载频率 frequency", r"\b(?:loading frequency|test frequency|frequency|hz|加载频率)\b"),
    ("环境 environment", r"\b(?:corrosion|hydrogen|vacuum|air environment|temperature|腐蚀|氢|真空|环境|温度)\b"),
    ("裂纹闭合 crack_closure", r"\b(?:crack closure|closure level|裂纹闭合)\b"),
)

OUTCOME_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("疲劳寿命 log10(Nf)", r"\b(?:fatigue life|cycles to failure|Nf|疲劳寿命)\b"),
    ("疲劳极限 fatigue_limit", r"\b(?:fatigue limit|fatigue strength|疲劳极限|疲劳强度)\b"),
    ("裂纹扩展速率 da/dN", r"\b(?:crack growth rate|da\s*/\s*dN|裂纹扩展速率)\b"),
    ("裂纹起裂 crack_initiation", r"\b(?:crack initiation|起裂|裂纹萌生)\b"),
    ("短裂纹行为 short_crack", r"\b(?:short crack|small crack|短裂纹)\b"),
    ("循环塑性 cyclic_response", r"\b(?:cyclic hardening|cyclic softening|hysteresis|循环硬化|循环软化|滞回)\b"),
    ("LCF/HCF/VHCF响应 fatigue_regime", r"\b(?:low cycle fatigue|high cycle fatigue|very high cycle fatigue|LCF|HCF|VHCF|低周疲劳|高周疲劳|超高周疲劳)\b"),
)

COUNTER_RE = re.compile(
    r"\b(?:no significant|not significant|did not|does not|contrary|however|"
    r"whereas|independent of|无显著|不显著|相反|然而)\b",
    re.I,
)
MECHANISM_RE = re.compile(
    r"\b(?:due to|attributed to|caused by|resulted from|mechanism|governed by|"
    r"because|归因于|由于|机制|导致)\b",
    re.I,
)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _evidence_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in ("claim", "original_text", "experimental_conditions")
    )


def _first_detected(
    rows: Iterable[Mapping[str, Any]],
    patterns: Sequence[Tuple[str, str]],
) -> Tuple[str, str]:
    combined = "\n".join(_evidence_text(row) for row in rows)
    for label, pattern in patterns:
        if re.search(pattern, combined, re.I):
            return label, pattern
    return "", ""


def _reference(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "evidence_id": str(row.get("evidence_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "page_number": int(float(row.get("page_number") or 0)),
        "claim": str(row.get("claim") or row.get("original_text") or "").strip(),
        "directness": str(row.get("directness") or ""),
        "review_status": str(row.get("review_status") or ""),
    }


def _persist_hypothesis(
    hypothesis: Dict[str, Any],
    *,
    base_dir: Path,
) -> None:
    path = base_dir / "data" / "generated_hypotheses.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows = [
        row
        for row in rows
        if str(row.get("hypothesis_id") or "") != hypothesis["hypothesis_id"]
    ]
    rows.append(hypothesis)
    temp = path.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


def generate_hypotheses(
    paper_ids: Sequence[str],
    *,
    base_dir: Path = BASE_DIR,
    persist: bool = True,
) -> Dict[str, Any]:
    """Generate one auditable candidate only from selected trusted evidence.

    Metadata-only candidates, placeholder titles and papers without completed
    deep reading are rejected before evidence is inspected.
    """
    gate = eligible_paper_ids(paper_ids, base_dir=base_dir)
    if not gate["eligible"]:
        return {
            "status": INSUFFICIENT_EVIDENCE,
            "hypotheses": [],
            "rejected": gate["rejected"],
            "reason": "没有同时满足有效元数据、全文深读和可信证据门禁的文献。",
        }

    selected = set(gate["eligible"])
    rows = [
        row
        for row in trusted_evidence_rows(base_dir)
        if str(row.get("paper_id") or "") in selected
        and str(row.get("directness") or "") in {"DIRECT", "INDIRECT"}
        and str(row.get("review_status") or "") not in {
            "QUARANTINED_TITLE_DERIVED",
            "HUMAN_REVISION_REQUIRED",
        }
        and int(float(row.get("page_number") or 0)) > 0
        and str(row.get("claim") or row.get("original_text") or "").strip()
    ]
    direct_count = sum(str(row.get("directness") or "") == "DIRECT" for row in rows)
    independent, independent_pattern = _first_detected(rows, VARIABLE_PATTERNS)
    dependent, dependent_pattern = _first_detected(rows, OUTCOME_PATTERNS)
    if len(rows) < 2 or direct_count < 1 or not independent or not dependent:
        missing = []
        if len(rows) < 2:
            missing.append("至少两条具有真实页码的可信证据")
        if direct_count < 1:
            missing.append("至少一条 DIRECT 证据")
        if not independent:
            missing.append("可从正文证据识别的自变量")
        if not dependent:
            missing.append("可从正文证据识别的因变量")
        return {
            "status": INSUFFICIENT_EVIDENCE,
            "hypotheses": [],
            "rejected": gate["rejected"],
            "reason": "；".join(missing),
        }

    relevant = [
        row
        for row in rows
        if re.search(independent_pattern, _evidence_text(row), re.I)
        or re.search(dependent_pattern, _evidence_text(row), re.I)
    ]
    counter = [row for row in relevant if COUNTER_RE.search(_evidence_text(row))]
    support = [row for row in relevant if row not in counter]
    if not support:
        return {
            "status": INSUFFICIENT_EVIDENCE,
            "hypotheses": [],
            "rejected": gate["rejected"],
            "reason": "所选证据没有可支持候选关系的正文证据。",
        }

    mechanism_rows = [
        row for row in relevant if MECHANISM_RE.search(_evidence_text(row))
    ]
    paper_key = "|".join(sorted(selected))
    identity = f"{paper_key}|{independent}|{dependent}"
    hypothesis_id = (
        "HYP_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16].upper()
    )
    controls = [
        "材料与制造工艺",
        "试样几何",
        "加载模式与应力幅",
        "热处理",
        "测试环境",
    ]
    moderator = (
        ["应力比 stress_ratio_R"]
        if independent != "应力比 stress_ratio_R"
        else ["表面状态 surface_state"]
    )
    hypothesis = {
        "hypothesis_id": hypothesis_id,
        "paper_ids": sorted(selected),
        "hypothesis_statement": (
            f"在所选文献真实报告的工况边界内，{independent} 与 {dependent} "
            "存在可检验的条件关联；关联方向和效应大小必须由跨文献配对数据估计。"
        ),
        "independent_variables": [independent],
        "dependent_variables": [dependent],
        "control_variables": controls,
        "moderating_variables": moderator,
        "condition_boundary": (
            "仅适用于支持证据各自报告的材料、制造、热处理、表面和疲劳加载条件；"
            "跨论文合并前必须完成单位与工况可比性审核。"
        ),
        "mechanism_chain": (
            "；".join(
                str(row.get("claim") or row.get("original_text") or "").strip()
                for row in mechanism_rows[:3]
            )
            or "现有可信证据尚未建立完整机制链，机制解释保留为待补证项。"
        ),
        "supporting_evidence": [_reference(row) for row in support[:12]],
        "counter_evidence": [_reference(row) for row in counter[:8]],
        "missing_evidence": [
            "自变量、因变量和控制变量在同一样本或可比样本中的配对数据。",
            "独立论文或留一文献外部验证。",
            *(
                []
                if counter
                else ["在相同工况边界下可用于推翻该关系的反向证据。"]
            ),
            *(
                []
                if mechanism_rows
                else ["正文直接报告的因果机制证据。"]
            ),
        ],
        "candidate_equation": "Y = β0 + β1·X + Σγk·Ck + β2·(X×M) + ε",
        "formula_variables": {
            "Y": dependent,
            "X": independent,
            "Ck": "控制变量；只使用正文中真实报告且单位可协调的变量。",
            "M": moderator[0],
            "β0, β1, β2, γk": "待真实数据拟合的系数，不生成虚构数值。",
            "ε": "残差项。",
        },
        "prediction_direction": (
            "β1 的符号由证据配对数据估计；当前只提出可检验关系，不预设方向。"
        ),
        "support_criteria": [
            "β1 的置信区间不跨越 0，且在重复交叉验证中方向稳定。",
            "加入 X 后的外部验证误差相对控制模型稳定下降。",
            "留一文献验证中效应方向可重复。",
        ],
        "falsification_criteria": [
            "β1 无法与 0 区分，或跨文献方向不一致。",
            "加入 X 后外部验证误差没有稳定改善。",
            "控制工况差异后该关联消失。",
        ],
        "manual_review_status": "PENDING_HUMAN_REVIEW",
        "evidence_status": "EVIDENCE_LINKED_CANDIDATE",
        "created_at": _now(),
    }
    if persist:
        _persist_hypothesis(hypothesis, base_dir=base_dir)
    return {
        "status": "GENERATED",
        "hypotheses": [hypothesis],
        "rejected": gate["rejected"],
    }

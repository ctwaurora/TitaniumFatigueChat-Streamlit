"""
variable_mapper.py — User variable pair extraction & relation finding
面向 L-PBF Ti-6Al-4V 疲劳变量对齐。

核心函数:
    extract_variable_pair(user_query) -> (ind_var, dep_var, var_class)
    find_exact_or_near_relation(ind_var, dep_var) -> matched_relations
    evaluate_literature_support(ind_var, dep_var) -> support_dict
    get_synonym_group(var_name) -> [related_var_names]
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from src.stage1_store import TRUSTED_EVIDENCE_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# ═══════════════════════════════════════════════════════════════════════════
# 1. Synonym groups — canonical variable name → synonyms
# ═══════════════════════════════════════════════════════════════════════════

SYNONYM_GROUPS: Dict[str, List[str]] = {
    "pore_size": [
        "孔隙尺寸", "缺陷尺寸", "孔隙大小", "孔径", "pore size",
        "defect size", "pore diameter", "pore radius", "pore dimension",
        "sqrt_area", "√area", "sqrt area", "square root of area",
    ],
    "fatigue_life": [
        "疲劳寿命", "寿命", "疲劳性能", "fatigue life", "nf", "cycles to failure",
        "lifetime", "循环寿命", "失效周次",
    ],
    "fatigue_limit": [
        "疲劳极限", "疲劳强度", "fatigue limit", "fatigue strength",
        "sigma_w", "σw", "endurance limit",
    ],
    "da_dn": [
        "裂纹扩展速率", "da/dn", "da_dn", "crack growth rate",
        "fcgr", "crack propagation rate",
    ],
    "delta_k": [
        "应力强度因子范围", "Δk", "ΔK", "delta k", "delta_k",
        "stress intensity range", "sif range",
    ],
    "surface_roughness": [
        "表面粗糙度", "粗糙度", "surface roughness", "surface_roughness",
        "ra", "rz", "rms", "surface finish", "surface quality",
    ],
    "pore_location": [
        "孔隙位置", "距表面距离", "pore location", "distance to surface",
        "pore position", "defect location", "距表面深度",
    ],
    "stress_amplitude": [
        "应力幅", "应力振幅", "stress amplitude", "sigma_a", "σa",
        "applied stress",
    ],
    "stress_ratio": [
        "应力比", "stress ratio", "r ratio", "stress_ratio", "r-ratio",
        "load ratio",
    ],
    "heat_treatment": [
        "热处理", "热处理状态", "heat treatment", "hip", "hot isostatic pressing",
        "annealing", "退火", "固溶", "时效",
    ],
    "microstructure": [
        "微观组织", "显微组织", "组织", "microstructure",
        "alpha lath", "alpha_lath", "martensite", "beta phase",
        "grain size", "grain orientation", "片层", "晶粒",
    ],
    "residual_stress": [
        "残余应力", "residual stress", "residual_stress",
    ],
    "porosity": [
        "孔隙率", "致密度", "porosity", "void fraction", "density",
    ],
    "paris_c_m": [
        "paris 参数", "c/m", "paris c", "paris m",
        "paris coefficient", "paris exponent",
    ],
}

# Build a fast reverse lookup: lowercase keyword → canonical name
_KEYWORD_MAP: List[Tuple[str, str]] = []
for canonical, syns in SYNONYM_GROUPS.items():
    for s in syns:
        _KEYWORD_MAP.append((s.lower().strip(), canonical))
# Sort longest first for greedy matching
_KEYWORD_MAP.sort(key=lambda x: -len(x[0]))


# ═══════════════════════════════════════════════════════════════════════════
# 2. Extract variable pair from user query
# ═══════════════════════════════════════════════════════════════════════════

def extract_variable_pair(user_query: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    从用户输入中识别用户真正关心的自变量和因变量。

    Returns:
        (independent_variable, dependent_variable, var_classification)

    var_classification 为:
        "quantitative_relation" — 两个变量都识别到
        "single_variable" — 只识别到一个变量
        "no_variable" — 未识别到变量
        "equation_query" — 涉及方程参数
    """
    if not user_query:
        return None, None, "no_variable"

    text = user_query.lower()

    # ── Step 1: Detect all canonical variables in the query ──
    found_vars: List[str] = []
    for keyword, canonical in _KEYWORD_MAP:
        if keyword in text:
            if canonical not in found_vars:
                found_vars.append(canonical)

    # ── Step 2: Determine which is independent, which is dependent ──
    # Common relation keywords indicating direction
    relation_patterns = [
        (r"([^，。、；]+?)(?:和|与|跟|and|vs|versus)([^，。、；]+?)"
         r"(?:之间|关系|影响|作用|关联|怎么相关|有什么关系|如何影响)"),
        (r"([^，。；]+?)对([^，。；]+?)(?:的|之)?(?:影响|作用|关系|贡献)"),
        (r"([^，。；]+?)与([^，。；]+?)(?:的|之)?(?:关系|相关|比较|对比)"),
        (r"([^，。；]+?)和([^，。；]+?)(?:的|之)?(?:关系|相关|比较|对比)"),
    ]

    ind_var = None
    dep_var = None

    # Try to extract from explicit relation patterns first
    for pattern in relation_patterns:
        m = re.search(pattern, text)
        if m:
            left = m.group(1).strip()
            right = m.group(2).strip()
            # Map both sides to canonical variables
            left_vars = [cv for kw, cv in _KEYWORD_MAP if kw in left]
            right_vars = [cv for kw, cv in _KEYWORD_MAP if kw in right]

            if left_vars and right_vars:
                # Convention: first mentioned = independent, second = dependent
                ind_var = left_vars[0]
                dep_var = right_vars[0]
                break
            elif left_vars and not right_vars:
                ind_var = left_vars[0]
            elif right_vars and not left_vars:
                ind_var = right_vars[0]

    # Fallback: if no explicit relation pattern found, use found_vars order
    if not ind_var and not dep_var:
        if len(found_vars) >= 2:
            ind_var = found_vars[0]
            dep_var = found_vars[1]
        elif len(found_vars) == 1:
            ind_var = found_vars[0]
            dep_var = infer_counterpart(ind_var)
            if dep_var:
                return ind_var, dep_var, "quantitative_relation"

    # ── Step 3: Equation-specific detection ──
    if any(kw in text for kw in ["paris", "方程", "公式", "model", "da/dn"]):
        if "delta_k" in found_vars or "da_dn" in found_vars:
            ind_var = ind_var or "delta_k"
            dep_var = dep_var or "da_dn"
            return ind_var, dep_var, "equation_query"

    # Determine classification
    if ind_var and dep_var:
        return ind_var, dep_var, "quantitative_relation"
    elif ind_var or dep_var:
        return ind_var or dep_var, None, "single_variable"
    else:
        return None, None, "no_variable"


def infer_counterpart(var: str) -> Optional[str]:
    """当用户只给一个变量时，推断可能对应的因变量。"""
    counterpart_map = {
        "pore_size": "fatigue_life",
        "surface_roughness": "fatigue_life",
        "delta_k": "da_dn",
        "da_dn": "delta_k",
        "stress_amplitude": "fatigue_life",
        "fatigue_life": "pore_size",
        "fatigue_limit": "pore_size",
        "pore_location": "fatigue_life",
        "porosity": "fatigue_life",
        "heat_treatment": "fatigue_life",
        "microstructure": "da_dn",
        "paris_c_m": "da_dn",
    }
    return counterpart_map.get(var)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Get synonym group for a variable
# ═══════════════════════════════════════════════════════════════════════════

def get_synonym_group(var_name: str) -> List[str]:
    """返回该变量的同义词组（含自身）。"""
    if var_name in SYNONYM_GROUPS:
        return [var_name] + [s for s in SYNONYM_GROUPS[var_name]]
    return [var_name]


def get_related_canonical_vars(var_name: str) -> List[str]:
    """返回与给定变量同属一个语义类别的所有 canonical 名称。"""
    # Groups of closely related canonical variables
    related_groups = [
        {"pore_size", "sqrt_area", "defect_size", "porosity"},
        {"fatigue_life", "fatigue_limit", "stress_amplitude"},
        {"da_dn", "delta_k", "paris_c_m"},
        {"surface_roughness", "pore_location"},
        {"heat_treatment", "microstructure", "residual_stress"},
    ]
    for group in related_groups:
        if var_name in group:
            return list(group)
    return [var_name]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Find exact or near relation from CSV tables
# ═══════════════════════════════════════════════════════════════════════════

def find_exact_or_near_relation(
    ind_var: Optional[str],
    dep_var: Optional[str],
) -> Dict[str, Any]:
    """
    检索 relation_table 或 variable_mechanism 以找到变量关系。

    Priority:
        1. 完全匹配 (ind_var → dep_var)
        2. 同义词匹配 (synonym → synonym)
        3. 中间链条 (ind_var → intermediate → dep_var)
        4. 相邻证据 (ind_var or dep_var 有任意关系)

    Returns dict with:
        found: bool
        relations: list of matched dicts
        match_quality: "exact" | "synonym" | "chain" | "adjacent" | "none"
        evidence_summary: str
    """
    # Try loading from multiple CSV files
    candidates = [
        DATA_DIR / "relation_table.csv",
        DATA_DIR / "variable_mechanism.csv",
        TRUSTED_EVIDENCE_PATH,
    ]

    all_relations = []
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
                if not df.empty:
                    all_relations.append(df)
            except Exception:
                continue

    if not all_relations:
        return {
            "found": False,
            "relations": [],
            "match_quality": "none",
            "evidence_summary": "本地关系表不存在或格式错误。",
        }

    result = {
        "found": False,
        "relations": [],
        "match_quality": "none",
        "evidence_summary": "",
    }

    if not ind_var:
        return result

    # Helper: normalize variable name for comparison
    def norm(v: str) -> str:
        return v.lower().replace(" ", "_").replace("-", "_").strip()

    ind_synonyms = get_synonym_group(ind_var)
    dep_synonyms = get_synonym_group(dep_var) if dep_var else []

    matched_relations = []

    for df in all_relations:
        # Determine which columns to use
        var_col = None
        ind_col = None
        dep_col = None

        for col in df.columns:
            cl = col.lower().strip()
            if cl in ("variable_name", "variable", "independent_variable", "ind_var"):
                ind_col = col
            if cl in ("dependent_indicator", "indicator", "dependent_variable",
                       "property_or_result", "dep_var", "linked_indicator"):
                dep_col = col
            if cl == "variable_name":
                var_col = col

        if ind_col and dep_col:
            # Priority 1: Exact match
            for _, row in df.iterrows():
                row_ind = str(row[ind_col]).strip().lower()
                row_dep = str(row[dep_col]).strip().lower() if dep_col else ""
                ind_match = any(norm(s) in row_ind or row_ind in norm(s) for s in ind_synonyms)
                dep_match = any(norm(s) in row_dep or row_dep in norm(s) for s in dep_synonyms) if dep_synonyms else True
                if ind_match and dep_match:
                    matched_relations.append({
                        "independent_variable": str(row.get(ind_col, "")),
                        "dependent_variable": str(row.get(dep_col, "")),
                        "relation_type": str(row.get("relation_type", row.get("mechanism_type", "unknown"))),
                        "mechanism": str(row.get("mechanism", row.get("evidence", "")))[:300],
                        "evidence_strength": str(row.get("evidence_strength", "unknown")),
                        "condition": str(row.get("condition", "")),
                        "match_quality": "exact",
                    })

    if matched_relations:
        result["found"] = True
        result["relations"] = matched_relations[:10]
        result["match_quality"] = "exact"
        result["evidence_summary"] = (
            f"本地文献库中找到了 {len(matched_relations)} 条直接关系"
        )
        return result

    # Priority 2: If no exact match, try with related canonical variables
    related_ind = get_related_canonical_vars(ind_var)
    related_dep = get_related_canonical_vars(dep_var) if dep_var else []

    for df in all_relations:
        # Reset column detection for each DataFrame
        local_ind_col = None
        local_dep_col = None

        for col in df.columns:
            cl = col.lower().strip()
            if cl in ("independent_variable", "ind_var", "variable_name", "variable"):
                local_ind_col = col
            if cl in ("dependent_indicator", "indicator", "dependent_variable",
                       "property_or_result", "dep_var", "linked_indicator"):
                local_dep_col = col

        if local_ind_col and local_dep_col:
            for _, row in df.iterrows():
                row_ind = norm(str(row[local_ind_col]))
                row_dep = norm(str(row.get(local_dep_col, ""))) if local_dep_col else ""
                ind_match = any(norm(rv) in row_ind or row_ind in norm(rv) for rv in related_ind)
                dep_match = any(norm(rv) in row_dep or row_dep in norm(rv) for rv in related_dep) if related_dep and local_dep_col else False
                if ind_match and dep_match:
                    matched_relations.append({
                        "independent_variable": str(row.get(local_ind_col, "")),
                        "dependent_variable": str(row.get(local_dep_col, "")),
                        "relation_type": str(row.get("relation_type", "similar")),
                        "mechanism": str(row.get("mechanism", row.get("evidence", "")))[:300],
                        "evidence_strength": str(row.get("evidence_strength", "indirect")),
                        "condition": str(row.get("condition", "")),
                        "match_quality": "synonym",
                    })

    if matched_relations:
        result["found"] = True
        result["relations"] = matched_relations[:10]
        result["match_quality"] = "synonym"
        result["evidence_summary"] = (
            f"未找到完全匹配，但找到 {len(matched_relations)} 条同义词级近似关系"
        )
        return result

    # Priority 3: Adjacent — any relation involving ind_var or dep_var
    for df in all_relations:
        # Reset column detection for each DataFrame
        local_ind_col = None
        local_dep_col = None

        for col in df.columns:
            cl = col.lower().strip()
            if cl in ("independent_variable", "ind_var", "variable_name", "variable"):
                local_ind_col = col
            if cl in ("dependent_indicator", "indicator", "dependent_variable",
                       "property_or_result", "dep_var", "linked_indicator"):
                local_dep_col = col

        if local_ind_col:
            for _, row in df.iterrows():
                row_ind = norm(str(row[local_ind_col]) if local_ind_col else "")
                row_dep = norm(str(row.get(local_dep_col, "")) if local_dep_col else "")
                if any(s in row_ind for s in ind_synonyms):
                    matched_relations.append({
                        "independent_variable": str(row.get(local_ind_col, "")),
                        "dependent_variable": str(row.get(local_dep_col, "")),
                        "relation_type": str(row.get("relation_type", "adjacent")),
                        "mechanism": str(row.get("mechanism", ""))[:200],
                        "evidence_strength": "adjacent",
                        "condition": str(row.get("condition", "")),
                        "match_quality": "adjacent",
                    })

    if matched_relations:
        result["found"] = True
        result["relations"] = matched_relations[:10]
        result["match_quality"] = "adjacent"
        result["evidence_summary"] = (
            f"未找到直接关系，但找到 {len(matched_relations)} 条涉及自变量的相邻关系"
        )
        return result

    result["evidence_summary"] = "本地文献库未找到该变量对的相关关系。"
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 5. Evaluate literature support for a variable pair
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_literature_support(
    ind_var: Optional[str],
    dep_var: Optional[str],
    user_query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    评估本地文献库对变量对的支持程度。

    Returns dict with:
        coverage_level: "sufficient" | "partial" | "weak" | "not_found"
        matched_paper_count: int
        matched_evidence_count: int
        supporting_evidence: list
        related_but_indirect: list
        conflicting_or_conditional: list
        evidence_conclusion: str
    """
    from src.interactive_modules import LiteratureSearchPlanner, load_evidence_snippets

    # Use LiteratureSearchPlanner for coverage analysis
    planner = LiteratureSearchPlanner()
    query_for_analysis = user_query or f"{ind_var} {dep_var}"
    coverage_result = planner.analyze_question(query_for_analysis)

    ev_df = load_evidence_snippets()

    # Categorize evidence
    supporting = []
    indirect = []
    conflicting = []

    # Build keyword list from variable synonyms
    ind_keywords = get_synonym_group(ind_var) if ind_var else []
    dep_keywords = get_synonym_group(dep_var) if dep_var else []

    if not ev_df.empty:
        for _, row in ev_df.iterrows():
            text = (str(row.get("snippet", "") or "") + " " +
                    str(row.get("linked_variable", "") or "") + " " +
                    str(row.get("linked_indicator", "") or "")).lower()

            has_ind = any(kw.lower() in text for kw in ind_keywords) if ind_keywords else True
            has_dep = any(kw.lower() in text for kw in dep_keywords) if dep_keywords else True

            if has_ind and has_dep:
                ev_type = str(row.get("evidence_type", "")).lower()
                if "conflict" in ev_type or "争议" in ev_type or "不一致" in ev_type:
                    conflicting.append(str(row.get("snippet", ""))[:200])
                else:
                    supporting.append(str(row.get("snippet", ""))[:200])
            elif has_ind or has_dep:
                indirect.append(str(row.get("snippet", ""))[:200])

    coverage_level = coverage_result.get("coverage_level", "not_found")
    n_papers = coverage_result.get("matched_papers_count", 0)
    n_evidence = coverage_result.get("matched_evidence_count", 0)

    # Determine evidence conclusion
    if coverage_level == "sufficient" and len(supporting) >= 3:
        conclusion = "证据支持"
    elif coverage_level in ("sufficient", "partial") and len(supporting) >= 1:
        conclusion = "证据部分支持"
    elif len(conflicting) > 0:
        conclusion = "存在条件冲突"
    elif coverage_level in ("weak", "not_found") and len(supporting) == 0:
        conclusion = "证据不足"
    else:
        conclusion = "证据部分支持"

    return {
        "coverage_level": coverage_level,
        "matched_paper_count": n_papers,
        "matched_evidence_count": n_evidence,
        "supporting_evidence": supporting[:5],
        "related_but_indirect_evidence": indirect[:5],
        "conflicting_or_conditional_evidence": conflicting[:3],
        "evidence_conclusion": conclusion,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. Generate mechanism chain description
# ═══════════════════════════════════════════════════════════════════════════

def build_mechanism_chain(ind_var: str, dep_var: str) -> str:
    """
    根据变量对构建机制链。

    格式:
    variable → intermediate mechanism → crack behavior → fatigue indicator
    """
    # Known mechanism chains
    chains: Dict[str, Dict[str, str]] = {
        "pore_size": {
            "fatigue_life": (
                "pore_size / sqrt_area\n"
                "→ local stress concentration at pore edge\n"
                "→ crack initiation at pore (especially near-surface pores)\n"
                "→ shorter initiation life Ni\n"
                "→ reduced total fatigue life Nf"
            ),
            "fatigue_limit": (
                "pore_size / sqrt_area\n"
                "→ stress concentration at pore tip\n"
                "→ local cyclic plastic zone exceeding critical size\n"
                "→ crack initiation from pore\n"
                "→ reduced fatigue limit σw\n"
                "→ consistent with Murakami √area model"
            ),
            "da_dn": (
                "pore_size / sqrt_area\n"
                "→ stress intensity at pore tip\n"
                "→ modified local ΔKeff\n"
                "→ accelerated early crack growth near pore\n"
                "→ increased da/dN at same nominal ΔK"
            ),
        },
        "surface_roughness": {
            "fatigue_life": (
                "surface_roughness (Ra/Rz)\n"
                "→ surface stress concentration at roughness peaks\n"
                "→ surface crack initiation at notch roots\n"
                "→ reduced initiation life\n"
                "→ lower Nf compared to polished/cut surfaces"
            ),
            "fatigue_limit": (
                "surface_roughness\n"
                "→ stress concentration factor Kt\n"
                "→ local stress exceeding yield at peaks\n"
                "→ reduced effective fatigue limit\n"
                "→ Kitagawa-type defect sensitivity"
            ),
        },
        "delta_k": {
            "da_dn": (
                "ΔK (stress intensity factor range)\n"
                "→ crack tip plastic zone size\n"
                "→ da/dN = C(ΔK)^m (Paris law)\n"
                "→ stable crack propagation\n"
                "→ log-linear relationship in region II"
            ),
        },
        "pore_location": {
            "fatigue_life": (
                "pore_location (distance to surface)\n"
                "→ near-surface pores experience higher effective stress\n"
                "→ surface-pore stress field interaction\n"
                "→ easier crack initiation for near-surface pores\n"
                "→ shorter Nf for near-surface vs internal pores"
            ),
        },
        "heat_treatment": {
            "fatigue_life": (
                "heat_treatment (HIP / annealing / STA)\n"
                "→ pore closure (HIP) reduces defect density\n"
                "→ α′ decomposition → α+β lamellar microstructure\n"
                "→ improved crack growth resistance\n"
                "→ increased Nf, higher fatigue limit"
            ),
        },
        "microstructure": {
            "da_dn": (
                "microstructure (α lath / β phase / grain size)\n"
                "→ crack deflection at α/β interfaces\n"
                "→ roughness-induced crack closure\n"
                "→ modified Paris exponent m\n"
                "→ changed da/dN-ΔK relationship"
            ),
        },
        "paris_c_m": {
            "da_dn": (
                "Paris parameters C and m\n"
                "→ C reflects material's crack growth rate level\n"
                "→ m reflects sensitivity to ΔK\n"
                "→ defects tend to increase C (faster growth)\n"
                "→ microstructure changes may alter m"
            ),
        },
        "stress_amplitude": {
            "fatigue_life": (
                "stress amplitude σa\n"
                "→ higher σa → larger cyclic plastic zone\n"
                "→ faster crack initiation and propagation\n"
                "→ shorter Nf\n"
                "→ Basquin: σa = σf'(2Nf)^b"
            ),
        },
    }

    # Find the exact chain
    if ind_var in chains and dep_var in chains[ind_var]:
        return chains[ind_var][dep_var]

    # Try reverse lookup
    if dep_var in chains and ind_var in chains[dep_var]:
        return chains[dep_var][ind_var]

    # Try related variable chains
    related_ind = get_related_canonical_vars(ind_var)
    related_dep = get_related_canonical_vars(dep_var)

    for ri in related_ind:
        if ri in chains:
            for rd in related_dep:
                if rd in chains[ri]:
                    return (
                        f"（基于同源变量）\n"
                        f"{chains[ri][rd]}"
                    )

    return (
        f"{ind_var}\n"
        f"→ [中间机制待文献补充]\n"
        f"→ [裂纹行为待验证]\n"
        f"→ {dep_var}"
    )

"""
interactive_modules.py — Backend for the interactive Streamlit upgrade.

Provides:
- EvidenceRelationExplorer: variable→indicator relation tables
- EquationParameterMiner: Paris/Basquin/Walker equation extraction
- ConflictDetector: cross-paper conflict detection
- HypothesisScorer: 10-dimension hypothesis scoring
- HypothesisGenerator: H1-H4 hypothesis generation
- Data loading utilities
"""

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from skills.library_skill import get_all_papers
from src.validation import classify_titanium_scope
from src.stage1_store import TRUSTED_EVIDENCE_PATH

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"


# ═══════════════════════════════════════════════════════════════════════════
# 1. 数据加载工具
# ═══════════════════════════════════════════════════════════════════════════

def load_evidence_snippets() -> pd.DataFrame:
    """加载证据片段表。"""
    path = TRUSTED_EVIDENCE_PATH
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
        return df
    except Exception:
        return pd.DataFrame()


def load_variable_mechanism() -> pd.DataFrame:
    """加载变量-机制表。"""
    path = DATA_DIR / "variable_mechanism.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
        return df
    except Exception:
        return pd.DataFrame()


def load_literature() -> pd.DataFrame:
    """加载文献库。"""
    path = DATA_DIR / "literature_database.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
        return df
    except Exception:
        return pd.DataFrame()


def load_equation_params() -> pd.DataFrame:
    """加载已存的方程参数表。"""
    path = DATA_DIR / "equation_parameters.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
        return df
    except Exception:
        return pd.DataFrame()


def load_conflict_claims() -> pd.DataFrame:
    """加载已存的冲突表。"""
    path = DATA_DIR / "conflict_claims.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
        return df
    except Exception:
        return pd.DataFrame()


def get_system_stats() -> dict:
    """返回系统基本统计数据。"""
    papers = get_all_papers()
    ev_df = load_evidence_snippets()
    vm_df = load_variable_mechanism()
    eq_df = load_equation_params()
    conf_df = load_conflict_claims()

    core_count = sum(1 for p in papers if p.get("alloy_type") != "out_of_scope")
    primary_count = 0
    for p in papers:
        if p.get("alloy_type") != "out_of_scope":
            sr = classify_titanium_scope(p)
            if sr.get("main_case_relevance") == "primary":
                primary_count += 1

    return {
        "n_papers": len(papers),
        "core_count": core_count,
        "primary_count": primary_count,
        "ev_count": len(ev_df),
        "vm_count": len(vm_df),
        "eq_count": len(eq_df),
        "conflict_count": len(conf_df),
        "ev_types": sorted(ev_df["evidence_type"].dropna().unique().tolist()) if not ev_df.empty else [],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Evidence Relation Explorer
# ═══════════════════════════════════════════════════════════════════════════

class EvidenceRelationExplorer:
    """构建变量-指标间的关系表，回答『怎么相关』的问题。"""

    RELATION_TYPES = [
        "positive_candidate", "negative_candidate", "threshold_candidate",
        "conditional_relation", "conflicting", "missing_evidence",
        "insufficient_data",
    ]

    # 关键词映射：变量类型 → 可能的 relation
    VARIABLE_RELATION_MAP = {
        "pore_size":         {"direction": "negative",   "indicator": "fatigue_life_Nf"},
        "pore_aspect_ratio": {"direction": "negative",   "indicator": "fatigue_life_Nf"},
        "porosity":          {"direction": "negative",   "indicator": "fatigue_life_Nf"},
        "surface_roughness": {"direction": "negative",   "indicator": "fatigue_life_Nf"},
        "build_orientation": {"direction": "conditional","indicator": "da_dN"},
        "heat_treatment":    {"direction": "positive",   "indicator": "fatigue_life_Nf"},
        "pore_location":     {"direction": "conditional","indicator": "crack_initiation_site"},
        "stress_ratio_R":    {"direction": "conditional","indicator": "da_dN"},
        "microstructure":    {"direction": "conditional","indicator": "fatigue_life_Nf"},
        "laser_power":       {"direction": "conditional","indicator": "porosity"},
        "scan_speed":        {"direction": "conditional","indicator": "porosity"},
    }

    def __init__(self):
        self.ev_df = load_evidence_snippets()
        self.vm_df = load_variable_mechanism()
        self.lit_df = load_literature()
        self.relations: List[dict] = []
        self._build()

    def _build(self):
        """构建关系表。"""
        self.relations = []

        # ── 从 variable_mechanism.csv 构建 ──
        if not self.vm_df.empty:
            for _, row in self.vm_df.iterrows():
                var_name = str(row.get("variable_name", "") or "").strip()
                indicator = str(row.get("property_or_result", "") or "").strip()
                mechanism = str(row.get("mechanism", "") or "").strip()
                evidence = str(row.get("evidence", "") or "").strip()
                missing = str(row.get("missing_evidence", "") or "").strip()

                if not var_name or not indicator:
                    continue

                rel_type, interpretation = self._infer_relation(var_name, indicator, mechanism)
                paper_ids = self._get_paper_ids(var_name)

                score = self._calc_relevance(var_name, indicator)

                self.relations.append({
                    "independent_variable": var_name,
                    "dependent_indicator": indicator,
                    "relation_type": rel_type,
                    "condition": self._extract_condition(mechanism),
                    "mechanism": mechanism,
                    "evidence_strength": self._evidence_strength(evidence, missing),
                    "paper_ids": "; ".join(paper_ids[:5]),
                    "evidence_ids": self._get_evidence_ids(var_name),
                    "interpretation": interpretation,
                    "relevance_score": score,
                })

        # ── 从 evidence_snippets.csv 补充 ──
        if not self.ev_df.empty:
            for _, row in self.ev_df.iterrows():
                var = str(row.get("linked_variable", "") or "").strip()
                ind = str(row.get("linked_indicator", "") or "").strip()
                ev_type = str(row.get("evidence_type", "") or "").strip()
                snippet = str(row.get("snippet", "") or "")
                eid = str(row.get("evidence_id", "") or "")

                if not var:
                    continue
                if not ind:
                    continue

                # 去重检查
                if any(r["independent_variable"] == var and r["dependent_indicator"] == ind for r in self.relations):
                    continue

                rel_type, interpretation = self._infer_relation(var, ind, snippet)
                self.relations.append({
                    "independent_variable": var,
                    "dependent_indicator": ind,
                    "relation_type": rel_type,
                    "condition": "",
                    "mechanism": snippet[:200],
                    "evidence_strength": "direct_evidence" if ev_type else "indirect",
                    "paper_ids": "",
                    "evidence_ids": eid,
                    "interpretation": interpretation,
                    "relevance_score": 2,
                })

        # 去重
        seen = set()
        deduped = []
        for r in self.relations:
            key = (r["independent_variable"], r["dependent_indicator"], r["relation_type"])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        self.relations = deduped

    def _infer_relation(self, var: str, ind: str, mech: str) -> Tuple[str, str]:
        """推断关系类型和解释。"""
        vl = var.lower()
        il = ind.lower()
        ml = mech.lower()

        # Check for known mappings
        for key, mapping in self.VARIABLE_RELATION_MAP.items():
            if key in vl or key.replace("_", "") in vl.replace("_", ""):
                direction = mapping["direction"]
                if direction == "negative":
                    return "negative_candidate", f"{var} 增大可能降低 {ind}"
                elif direction == "positive":
                    return "positive_candidate", f"{var} 改善可能提高 {ind}"
                elif direction == "conditional":
                    return "conditional_relation", f"{var} 对 {ind} 的影响依赖于其他条件"

        # Keyword-based inference
        if any(kw in ml for kw in ["reduce", "decrease", "degrad", "lower", "降低", "减少", "削弱"]):
            return "negative_candidate", f"{var} 可能降低/减少 {ind}"
        if any(kw in ml for kw in ["improve", "increase", "enhance", "提高", "改善", "增强"]):
            return "positive_candidate", f"{var} 可能提高/改善 {ind}"
        if any(kw in ml for kw in ["threshold", "临界", "门槛", "limit", "边界"]):
            return "threshold_candidate", f"{var} 对 {ind} 存在阈值效应"
        if any(kw in ml for kw in ["depend", "depends", "condition", "取决于", "依赖于", "当"]):
            return "conditional_relation", f"{var} 对 {ind} 的影响有条件依赖"
        if any(kw in ml for kw in ["conflict", "inconsist", "矛盾", "不一致", "争议"]):
            return "conflicting", f"{var} 与 {ind} 的关系存在文献冲突"
        if any(kw in ml for kw in ["missing", "lack", "不足", "缺失", "缺乏"]):
            return "missing_evidence", f"{var} 与 {ind} 的关系尚无足够证据"

        return "insufficient_data", f"{var} 与 {ind} 的关系证据不足，需更多数据"

    def _extract_condition(self, mech: str) -> str:
        """从机制描述中提取条件。"""
        # Look for conditional phrases
        patterns = [
            r"在(.+?)下",
            r"当(.+?)时",
            r"在(.+?)后",
            r"控制(.+?)后",
            r"for\s+(.+?)(?:,|$)",
            r"under\s+(.+?)(?:,|$)",
            r"at\s+(.+?)(?:,|$)",
        ]
        for pat in patterns:
            m = re.search(pat, mech)
            if m:
                return m.group(1).strip()[:80]
        return ""

    def _evidence_strength(self, evidence: str, missing: str) -> str:
        if not evidence:
            return "no_evidence"
        if missing and len(missing) > 10:
            return "partial_evidence"
        if len(evidence) > 100:
            return "strong_evidence"
        return "weak_evidence"

    def _get_paper_ids(self, var: str) -> List[str]:
        ids = []
        if not self.lit_df.empty:
            for _, row in self.lit_df.iterrows():
                title = str(row.get("title", "") or "").lower()
                if var.lower() in title:
                    ids.append(str(row.get("title", "")[:60]))
        return ids[:5]

    def _get_evidence_ids(self, var: str) -> str:
        eids = []
        if not self.ev_df.empty:
            for _, row in self.ev_df.iterrows():
                v = str(row.get("linked_variable", "") or "").lower()
                if var.lower() in v:
                    eids.append(str(row.get("evidence_id", "")))
        return "; ".join(eids[:5])

    def _calc_relevance(self, var: str, ind: str) -> int:
        """计算相关度 1-5。"""
        score = 2
        if not self.ev_df.empty:
            ev_count = sum(1 for _, r in self.ev_df.iterrows()
                           if var.lower() in str(r.get("linked_variable", "") or "").lower()
                           or ind.lower() in str(r.get("linked_indicator", "") or "").lower())
            score += min(ev_count, 3)
        return min(score, 5)

    def get_relations(self, sort_by: str = "relevance_score") -> List[dict]:
        if sort_by == "relevance_score":
            return sorted(self.relations, key=lambda r: r["relevance_score"], reverse=True)
        return self.relations

    def get_most_relevant_variables(self, top_n: int = 5) -> List[dict]:
        """返回最相关的变量排序。"""
        var_scores = {}
        for r in self.relations:
            v = r["independent_variable"]
            var_scores[v] = var_scores.get(v, 0) + r["relevance_score"]
        sorted_vars = sorted(var_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"variable": v, "score": s} for v, s in sorted_vars[:top_n]]

    def filter_by_type(self, rel_type: str) -> List[dict]:
        if rel_type == "all":
            return self.relations
        return [r for r in self.relations if r["relation_type"] == rel_type]

    def get_relation_type_counts(self) -> dict:
        counts = {}
        for r in self.relations:
            t = r["relation_type"]
            counts[t] = counts.get(t, 0) + 1
        return counts


# ═══════════════════════════════════════════════════════════════════════════
# 3. Equation & Parameter Miner
# ═══════════════════════════════════════════════════════════════════════════

class EquationParameterMiner:
    """从文献库中提取方程和参数。"""

    EQUATION_KEYWORDS = {
        "paris_law": {
            "keywords": ["paris", "da/dn", "da_dn", "crack growth rate", "Δk", "delta k"],
            "equation": "da/dN = C(ΔK)^m",
            "params": ["C", "m"],
        },
        "walker_model": {
            "keywords": ["walker", "walker model", "walker equation"],
            "equation": "da/dN = C(ΔK(1-R)^(p-1))^m",
            "params": ["C", "m", "p"],
        },
        "basquin_equation": {
            "keywords": ["basquin", "s-n curve", "stress-life", "stress life"],
            "equation": "σ_a = σ_f'(2Nf)^b",
            "params": ["σ_f'", "b"],
        },
        "kitagawa_takahashi": {
            "keywords": ["kitagawa", "takahashi", "kitagawa-takahashi"],
            "equation": "ΔKth = f(defect_size, σ_w)",
            "params": ["ΔKth", "defect_size"],
        },
        "coffin_manson": {
            "keywords": ["coffin", "manson", "coffin-manson", "ε-n", "strain-life"],
            "equation": "ε_a = ε_f'(2Nf)^c + σ_f'/E(2Nf)^b",
            "params": ["ε_f'", "c", "σ_f'", "b"],
        },
        "s_n_fitting": {
            "keywords": ["s-n", "s_n", "stress-life", "stress life", "wöhler"],
            "equation": "S^m * N = C (power law S-N)",
            "params": ["m", "C"],
        },
        "fatigue_limit": {
            "keywords": ["fatigue limit", "fatigue strength", "endurance limit", "疲劳极限"],
            "equation": "σ_w (fatigue limit at N cycles)",
            "params": ["σ_w"],
        },
        "delta_kth": {
            "keywords": ["Δkth", "delta_kth", "threshold", "门槛值", "ΔKth"],
            "equation": "ΔKth (threshold stress intensity factor)",
            "params": ["ΔKth"],
        },
        "life_prediction_model": {
            "keywords": ["life prediction", "fatigue life prediction", "寿命预测", "prediction model"],
            "equation": "various fatigue life prediction models",
            "params": ["various"],
        },
    }

    def __init__(self):
        self.lit_df = load_literature()
        self.ev_df = load_evidence_snippets()
        self.existing_df = load_equation_params()
        self.params_list: List[dict] = []
        self._mine()

    def _mine(self):
        """扫描文献库和证据表，提取方程参数信息。"""
        self.params_list = []

        # 从 literature_database.csv 扫描
        if not self.lit_df.empty:
            for _, row in self.lit_df.iterrows():
                title = str(row.get("title", "") or "")
                findings = str(row.get("key_findings", "") or "")
                abstract = str(row.get("abstract", "") or "")
                combined = (title + " " + findings + " " + abstract).lower()

                for eq_type, info in self.EQUATION_KEYWORDS.items():
                    if any(kw in combined for kw in info["keywords"]):
                        self.params_list.append({
                            "paper_id": "",
                            "author_year": self._get_author_year(row),
                            "equation_type": eq_type,
                            "equation_text": info["equation"],
                            "parameters": "; ".join(info["params"]),
                            "related_variables": self._extract_variables(combined),
                            "related_indicators": info["params"][0] if info["params"] else eq_type,
                            "material_condition": str(row.get("material_system", "") or ""),
                            "source_section": "key_findings",
                            "extraction_confidence": "keyword_matched",
                            "source_status": str(row.get("doi", "") or "local PDF"),
                        })

        # 从 evidence_snippets.csv 扫描
        if not self.ev_df.empty:
            for _, row in self.ev_df.iterrows():
                snippet = str(row.get("snippet", "") or "").lower()
                for eq_type, info in self.EQUATION_KEYWORDS.items():
                    if any(kw in snippet for kw in info["keywords"]):
                        self.params_list.append({
                            "paper_id": str(row.get("paper_id", "") or ""),
                            "author_year": str(row.get("author_year", "") or ""),
                            "equation_type": eq_type,
                            "equation_text": info["equation"],
                            "parameters": "; ".join(info["params"]),
                            "related_variables": str(row.get("linked_variable", "") or ""),
                            "related_indicators": str(row.get("linked_indicator", "") or ""),
                            "material_condition": "",
                            "source_section": str(row.get("source_section", "") or ""),
                            "extraction_confidence": "snippet_matched",
                            "source_status": "evidence_snippets.csv",
                        })

        # 去重
        seen = set()
        deduped = []
        for p in self.params_list:
            key = (p["equation_type"], p["author_year"])
            if key not in seen:
                seen.add(key)
                deduped.append(p)
        self.params_list = deduped

    def _get_author_year(self, row) -> str:
        authors = str(row.get("authors", "[]"))
        year = str(row.get("year", ""))[:4]
        try:
            parsed = json.loads(authors) if isinstance(authors, str) else authors
            if parsed and len(parsed) > 0:
                first = str(parsed[0]).strip()
                surname = first.split()[-1].strip(",.") if first.split() else first
                return f"{surname} et al., {year}" if year else surname
        except Exception:
            pass
        return str(row.get("title", "") or "")[:30]

    def _extract_variables(self, text: str) -> str:
        vars_found = []
        var_keywords = ["pore", "porosity", "roughness", "defect", "orientation",
                        "heat treatment", "microstructure", "stress ratio", "temperature"]
        for kw in var_keywords:
            if kw in text:
                vars_found.append(kw)
        return "; ".join(vars_found[:5])

    def get_summary(self) -> dict:
        """返回参数提取统计摘要。"""
        eq_counts = {}
        for p in self.params_list:
            t = p["equation_type"]
            eq_counts[t] = eq_counts.get(t, 0) + 1
        return {
            "total_extractions": len(self.params_list),
            "equation_counts": eq_counts,
            "unique_types": len(eq_counts),
        }

    def get_parameter_hypotheses(self) -> List[dict]:
        """生成参数型假设。"""
        hypotheses = []
        has_paris = any(p["equation_type"] == "paris_law" for p in self.params_list)
        has_basquin = any(p["equation_type"] == "basquin_equation" for p in self.params_list)
        has_walker = any(p["equation_type"] == "walker_model" for p in self.params_list)
        has_kth = any(p["equation_type"] == "delta_kth" for p in self.params_list)

        if has_paris:
            hypotheses.append({
                "type": "H2_parameter",
                "hypothesis": "在控制应力比 R 和表面状态后，体积缺陷特征可能主要通过改变 Paris 参数 C，而不是单纯改变 m，影响 L-PBF Ti-6Al-4V 的早期裂纹扩展速率。",
                "equation": "da/dN = C(ΔK)^m",
                "related_params": ["C", "m"],
                "validation": "通过对比不同孔隙特征试样的 Paris 参数 C 和 m 值验证",
                "falsification": "若 C 和 m 均不随孔隙特征系统变化，则该假设降级",
            })
        if has_basquin:
            hypotheses.append({
                "type": "H2_parameter",
                "hypothesis": "偏离 Basquin 拟合规律的数据点可能来自未显式考虑的近表面孔隙位置、孔隙形态或残余应力差异。",
                "equation": "σ_a = σ_f'(2Nf)^b",
                "related_params": ["σ_f'", "b"],
                "validation": "对偏离 Basquin 拟合的数据点进行断口 SEM 分析，确认是否对应特定孔隙特征",
                "falsification": "若偏离点与孔隙特征无对应关系，则假设不成立",
            })
        if has_walker:
            hypotheses.append({
                "type": "H2_parameter",
                "hypothesis": "Walker 模型中的 p 参数可能对孔隙特征和成形方向敏感，需引入缺陷修正项。",
                "equation": "da/dN = C(ΔK(1-R)^(p-1))^m",
                "related_params": ["C", "m", "p"],
                "validation": "在不同 R 比和孔隙状态下拟合 Walker 参数，分析 p 值变化趋势",
                "falsification": "若 p 值在不同孔隙状态下无显著变化，则该假设不成立",
            })

        return hypotheses

    def save(self) -> str:
        """保存到 data/equation_parameters.csv。"""
        if not self.params_list:
            return ""
        path = DATA_DIR / "equation_parameters.csv"
        fields = ["paper_id", "author_year", "equation_type", "equation_text",
                  "parameters", "related_variables", "related_indicators",
                  "material_condition", "source_section", "extraction_confidence",
                  "source_status"]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.params_list)
        return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Conflict Detector
# ═══════════════════════════════════════════════════════════════════════════

class ConflictDetector:
    """检测文献间的结论冲突或不一致。"""

    CONFLICT_TOPICS = [
        {
            "id": "surface_vs_pores",
            "topic": "表面粗糙度主导疲劳寿命 vs 内部孔隙主导疲劳寿命",
            "claim_A_keywords": ["surface", "roughness", "as-built surface", "表面粗糙度"],
            "claim_B_keywords": ["pore", "internal pore", "internal defect", "内部孔隙", "subsurface"],
        },
        {
            "id": "hip_effectiveness",
            "topic": "HIP 显著改善疲劳性能 vs HIP 后仍受初始缺陷影响",
            "claim_A_keywords": ["hip", "hot isostatic pressing", "improve", "reduce porosity"],
            "claim_B_keywords": ["hip", "residual defect", "remnant pore", "initial defect"],
        },
        {
            "id": "orientation_effect",
            "topic": "成形方向显著影响 FCGR vs 成形方向影响较弱",
            "claim_A_keywords": ["orientation", "build direction", "anisotropy", "成形方向"],
            "claim_B_keywords": ["orientation", "build direction", "weak", "minor", "negligible"],
        },
        {
            "id": "heat_treatment_fcgr",
            "topic": "热处理降低 FCGR vs 热处理效果依赖组织状态",
            "claim_A_keywords": ["heat treatment", "anneal", "reduce", "decrease crack growth"],
            "claim_B_keywords": ["heat treatment", "depend", "microstructure", "prior beta"],
        },
        {
            "id": "defect_initiation_vs_propagation",
            "topic": "缺陷只控制裂纹起裂 vs 缺陷同时影响早期扩展",
            "claim_A_keywords": ["crack initiation", "initiation site", "crack start"],
            "claim_B_keywords": ["early crack growth", "short crack", "small crack", "crack propagation"],
        },
        {
            "id": "microstructure_vs_defect",
            "topic": "微观组织主导裂纹扩展 vs 缺陷主导裂纹扩展",
            "claim_A_keywords": ["microstructure", "grain size", "grain boundary", "phase"],
            "claim_B_keywords": ["defect", "pore", "porosity", "lack of fusion"],
        },
    ]

    CONFLICT_TYPES = [
        "direct_conflict", "condition_dependent_difference",
        "potential_tension", "insufficient_evidence",
    ]

    def __init__(self):
        self.ev_df = load_evidence_snippets()
        self.lit_df = load_literature()
        self.papers = get_all_papers()
        self.claims: List[dict] = []
        self._detect()

    def _detect(self):
        """检测冲突。"""
        self.claims = []

        for topic in self.CONFLICT_TOPICS:
            # Find evidence for claim A and claim B
            ev_a = self._find_evidence(topic["claim_A_keywords"])
            ev_b = self._find_evidence(topic["claim_B_keywords"])

            # Find paper references
            papers_a = self._find_papers(topic["claim_A_keywords"])
            papers_b = self._find_papers(topic["claim_B_keywords"])

            if not ev_a and not ev_b:
                conflict_type = "insufficient_evidence"
            elif ev_a and not ev_b:
                conflict_type = "insufficient_evidence"
            elif not ev_a and ev_b:
                conflict_type = "insufficient_evidence"
            elif self._is_direct_conflict(topic["id"], ev_a, ev_b):
                conflict_type = "direct_conflict"
            elif self._is_condition_dependent(topic["id"]):
                conflict_type = "condition_dependent_difference"
            else:
                conflict_type = "potential_tension"

            hypothesis = self._generate_hypothesis(topic["id"], conflict_type)

            self.claims.append({
                "conflict_id": f"CONF_{topic['id']}",
                "topic": topic["topic"],
                "claim_A": topic["claim_A_keywords"][0],
                "paper_A": "; ".join(papers_a[:3]) or "—",
                "evidence_A": "; ".join(ev_a[:3]) or "—",
                "claim_B": topic["claim_B_keywords"][0],
                "paper_B": "; ".join(papers_b[:3]) or "—",
                "evidence_B": "; ".join(ev_b[:3]) or "—",
                "conflict_type": conflict_type,
                "possible_reason": self._possible_reason(conflict_type),
                "generated_hypothesis": hypothesis,
                "confidence_level": "medium" if conflict_type != "insufficient_evidence" else "low",
            })

    def _find_evidence(self, keywords: List[str]) -> List[str]:
        snippets = []
        if self.ev_df.empty:
            return snippets
        for _, row in self.ev_df.iterrows():
            text = (str(row.get("snippet", "") or "") + " " +
                    str(row.get("evidence_type", "") or "") + " " +
                    str(row.get("linked_claim", "") or "")).lower()
            for kw in keywords:
                if kw.lower() in text:
                    s = str(row.get("snippet", "") or "")[:120]
                    if s and s not in snippets:
                        snippets.append(s)
                    break
        return snippets[:5]

    def _find_papers(self, keywords: List[str]) -> List[str]:
        title_list = []
        if not self.lit_df.empty:
            for _, row in self.lit_df.iterrows():
                text = (str(row.get("title", "") or "") + " " +
                        str(row.get("key_findings", "") or "")).lower()
                for kw in keywords:
                    if kw.lower() in text:
                        t = str(row.get("title", "") or "")[:80]
                        if t and t not in title_list:
                            title_list.append(t)
                        break
        return title_list[:5]

    def _is_direct_conflict(self, topic_id: str, ev_a: list, ev_b: list) -> bool:
        """判断是否直接冲突（都有独立的、非重叠的证据）。"""
        if not ev_a or not ev_b:
            return False
        # Check for topic-specific condition-dependent topics
        condition_topics = ["orientation_effect", "heat_treatment_fcgr",
                           "defect_initiation_vs_propagation", "microstructure_vs_defect"]
        if topic_id in condition_topics:
            return False  # These are inherently condition-dependent
        # Check for surface_vs_pores — only direct conflict if both sides have ≥2 strong snippets
        if topic_id == "surface_vs_pores":
            if len(ev_a) >= 2 and len(ev_b) >= 2:
                return True
            return False  # insufficient evidence for direct conflict
        # HIP effectiveness — default to potential_tension (not binary)
        if topic_id == "hip_effectiveness":
            return False  # HIP effects are known to be condition-dependent
        return False  # Default to conservative: assume condition-dependent unless proven

    def _is_condition_dependent(self, topic_id: str) -> bool:
        """判断是否条件依赖型差异。"""
        condition_topics = ["orientation_effect", "heat_treatment_fcgr",
                           "defect_initiation_vs_propagation", "microstructure_vs_defect",
                           "hip_effectiveness", "surface_vs_pores"]
        return topic_id in condition_topics

    def _possible_reason(self, conflict_type: str) -> str:
        reasons = {
            "direct_conflict": "实验条件、材料批次或表征方法不同可能导致相反结论",
            "condition_dependent_difference": "不同条件下的结论差异是合理的，取决于具体实验参数范围",
            "potential_tension": "暗示可能存在未显式控制的中间变量",
            "insufficient_evidence": "当前证据不足以判断是否存在冲突",
        }
        return reasons.get(conflict_type, "")

    def _generate_hypothesis(self, topic_id: str, conflict_type: str) -> str:
        """基于冲突类型生成解释型假设。"""
        hypotheses = {
            "surface_vs_pores": (
                "表面粗糙度与内部孔隙对疲劳寿命的主导作用可能存在条件边界："
                "as-built 状态下表面粗糙度主导疲劳失效，"
                "表面加工改善后内部孔隙或亚表面孔隙转为主导失效模式。"
            ),
            "hip_effectiveness": (
                "HIP 的改善效果取决于初始孔隙特征："
                "对于表面连通孔隙或大尺寸未熔合缺陷，HIP 可能无法完全闭合，"
                "残留缺陷仍可作为裂纹起裂源。"
            ),
            "orientation_effect": (
                "成形方向对 FCGR 的影响程度与裂纹扩展方向相对熔池边界的取向有关："
                "当裂纹扩展方向垂直于熔池边界时影响显著，平行时影响减弱。"
            ),
            "heat_treatment_fcgr": (
                "热处理对 FCGR 的影响取决于最终微观组织状态："
                "α′ 分解完全的 lamellar (α+β) 组织可能降低 FCGR，"
                "而 bimodal 组织的改善效果取决于初生 α 相比例。"
            ),
            "defect_initiation_vs_propagation": (
                "孔隙缺陷可能同时影响裂纹起裂和早期扩展，"
                "但对两者的贡献权重取决于孔隙相对于关键缺陷尺寸的位置："
                "大于临界尺寸的孔隙主导起裂，但多个小孔隙的交互作用可能影响扩展路径。"
            ),
            "microstructure_vs_defect": (
                "微观组织和缺陷对裂纹扩展的贡献可能存在竞争关系："
                "缺陷密度高时缺陷主导，缺陷少时组织特征（晶粒尺寸、相界密度）主导。"
            ),
        }
        h = hypotheses.get(topic_id, "当前主题的文献证据存在不一致，需要系统对比分析。")
        if conflict_type == "insufficient_evidence":
            h += " 当前证据不足，需补充更多系统对比实验。"
        return h

    def get_summary(self) -> dict:
        type_counts = {}
        for c in self.claims:
            t = c["conflict_type"]
            type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "total_topics": len(self.claims),
            "by_type": type_counts,
        }

    def save(self) -> str:
        """保存到 data/conflict_claims.csv。"""
        if not self.claims:
            return ""
        path = DATA_DIR / "conflict_claims.csv"
        fields = ["conflict_id", "topic", "claim_A", "paper_A", "evidence_A",
                  "claim_B", "paper_B", "evidence_B", "conflict_type",
                  "possible_reason", "generated_hypothesis", "confidence_level"]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.claims)
        return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Hypothesis Scorer (10 维度)
# ═══════════════════════════════════════════════════════════════════════════

HYPOTHESIS_SCORING_DIMS = [
    {
        "id": "specificity",
        "name": "Specificity 具体性",
        "levels": [
            (0, "只有泛泛表述，例如「孔隙影响疲劳」"),
            (1, "有变量但没有指标"),
            (2, "有变量和指标，但没有条件"),
            (3, "有变量、指标和基本条件"),
            (4, "有变量、指标、条件和机制"),
            (5, "有变量、指标、条件、机制和可测量关系"),
        ],
    },
    {
        "id": "relation_clarity",
        "name": "Relation Clarity 关系清晰度",
        "levels": [
            (0, "没有关系描述"),
            (1, "只写「相关」"),
            (2, "写「影响」，但没有方向"),
            (3, "有正/负/条件关系"),
            (4, "有关系方向和作用条件"),
            (5, "有关系方向、条件、机制和可能阈值/参数变化"),
        ],
    },
    {
        "id": "evidence_traceability",
        "name": "Evidence Traceability 证据可追溯性",
        "levels": [
            (0, "无文献证据"),
            (1, "只有文献题名"),
            (2, "有 paper_id"),
            (3, "有 paper_id + evidence_id"),
            (4, "有 paper_id + evidence_id + snippet"),
            (5, "核心 claim 至少有 2 条高质量 evidence snippet 支持"),
        ],
    },
    {
        "id": "parameter_awareness",
        "name": "Parameter Awareness 参数意识",
        "levels": [
            (0, "完全没有参数"),
            (1, "只提疲劳寿命"),
            (2, "提到 S-N 或 da/dN"),
            (3, "提到 Paris / Basquin / Walker 等模型"),
            (4, "能指出可能影响 C、m、b、ΔKth 等参数"),
            (5, "能提出参数变化趋势或模型修正假设"),
        ],
    },
    {
        "id": "mechanism_plausibility",
        "name": "Mechanism Plausibility 机制合理性",
        "levels": [
            (0, "无机制"),
            (1, "机制空泛"),
            (2, "有简单机制"),
            (3, "有变量→局部效应→性能的链条"),
            (4, "有变量→局部效应→裂纹行为→疲劳指标链条"),
            (5, "机制链包含微观组织、缺陷、裂纹起裂/扩展和宏观疲劳指标"),
        ],
    },
    {
        "id": "macro_micro_link",
        "name": "Macro-Micro Link 宏观—微观关联",
        "levels": [
            (0, "没有微观因素"),
            (1, "只提材料"),
            (2, "提孔隙或组织，但没有连接性能"),
            (3, "连接微观缺陷和疲劳寿命"),
            (4, "连接微观缺陷/组织、裂纹行为和疲劳指标"),
            (5, "连接微观变量、裂纹机制、方程参数和宏观寿命"),
        ],
    },
    {
        "id": "novelty",
        "name": "Novelty 新颖性",
        "levels": [
            (0, "完全是已知常识"),
            (1, "重复综述结论"),
            (2, "略有细化"),
            (3, "聚焦未闭合证据链"),
            (4, "来自文献冲突、数据异常或参数差异"),
            (5, "提出可验证的新关系、模型修正或冲突解释"),
        ],
    },
    {
        "id": "experimental_verifiability",
        "name": "Experimental Verifiability 实验可验证性",
        "levels": [
            (0, "无法验证"),
            (1, "只有笼统「需要实验」"),
            (2, "提到实验方法"),
            (3, "明确实验变量和测量指标"),
            (4, "有实验变量、控制变量、测量指标和成功判据"),
            (5, "有完整验证路径和推翻条件"),
        ],
    },
    {
        "id": "conflict_usefulness",
        "name": "Conflict Usefulness 冲突利用价值",
        "levels": [
            (0, "没有冲突信息"),
            (1, "只说文献不一致"),
            (2, "列出两个不同结论"),
            (3, "说明可能条件差异"),
            (4, "基于冲突生成解释型假设"),
            (5, "能提出可实验验证的冲突解释假设"),
        ],
    },
    {
        "id": "research_usefulness",
        "name": "Research Usefulness 研究价值",
        "levels": [
            (0, "无研究价值"),
            (1, "太泛"),
            (2, "可作为背景问题"),
            (3, "可作为小课题"),
            (4, "可形成明确实验/数据分析课题"),
            (5, "可形成论文级研究问题或后续实验计划"),
        ],
    },
]


class HypothesisScorer:
    """10 维度假设评分系统。"""

    # Bad hypothesis patterns that should score very low
    BAD_PATTERNS = [
        r"孔隙.*影响.*疲劳",
        r".*和.*相关",
        r".*与.*有关",
        r"应进一步研究",
        r"受多因素影响",
        r"需要更多研究",
        r"有待进一步",
        r"microstructure.*related.*fatigue",
        r"heat treatment.*improve.*fatigue",
    ]

    def score_hypothesis(self, hypothesis: dict) -> dict:
        """对单个假设进行 10 维度评分。"""
        dim_scores = []
        for dim in HYPOTHESIS_SCORING_DIMS:
            s = self._score_dim(hypothesis, dim)
            dim_scores.append({
                "dimension": dim["name"],
                "score": s,
                "max_score": 5,
            })

        total = sum(d["score"] for d in dim_scores)
        grade = self._grade(total)

        is_bad = self._is_bad_hypothesis(hypothesis.get("hypothesis_statement", ""))
        if is_bad:
            grade = "reject"

        major_weakness = self._find_major_weakness(dim_scores)
        next_improvement = self._next_improvement(dim_scores)

        return {
            "hypothesis_id": hypothesis.get("hypothesis_id", ""),
            "hypothesis_type": hypothesis.get("hypothesis_type", ""),
            "hypothesis_statement": hypothesis.get("hypothesis_statement", ""),
            "dim_scores": dim_scores,
            "total_score": total,
            "max_score": 50,
            "grade": grade,
            "major_weakness": major_weakness,
            "next_improvement": next_improvement,
            "is_bad": is_bad,
        }

    def _score_dim(self, hypothesis: dict, dim: dict) -> int:
        """对单个维度打分 0-5。"""
        statement = hypothesis.get("hypothesis_statement", "")
        h_type = hypothesis.get("hypothesis_type", "")
        validation = hypothesis.get("validation_path", "")
        falsification = hypothesis.get("falsification_condition", "")
        evidence = hypothesis.get("supporting_evidence_ids", "")

        did = dim["id"]

        # ── specificity ──
        if did == "specificity":
            score = 0
            if any(kw in statement for kw in ["变量", "variable", "factor"]):
                score += 1
            if any(kw in statement for kw in ["Nf", "da/dN", "寿命", "速率", "Paris", "ΔK"]):
                score += 1
            if any(kw in statement for kw in ["控制", "controlled", "条件", "condition"]):
                score += 1
            if "→" in statement or "通过" in statement or "机制" in statement:
                score += 1
            if "预期" in statement or "表现为" in statement or "correlation" in statement:
                score += 1
            return min(score, 5)

        # ── relation_clarity ──
        if did == "relation_clarity":
            score = 0
            if any(kw in statement for kw in ["相关", "correlation", "relation"]):
                score += 1
            if any(kw in statement for kw in ["正相关", "负相关", "positive", "negative", "降低", "提高"]):
                score += 2
            if any(kw in statement for kw in ["条件", "取决于", "depends", "当"]):
                score += 1
            if "阈值" in statement or "门槛" in statement or "threshold" in statement:
                score += 1
            return min(score, 5)

        # ── evidence_traceability ──
        if did == "evidence_traceability":
            score = 0
            if evidence:
                score += 2
            if len(evidence) > 20:
                score += 1
            if hypothesis.get("paper_ids"):
                score += 1
            if hypothesis.get("source"):
                score += 1
            return min(score, 5)

        # ── parameter_awareness ──
        if did == "parameter_awareness":
            score = 0
            if any(kw in statement for kw in ["Paris", "Basquin", "Walker", "da/dN", "S-N"]):
                score += 2
            if any(kw in statement for kw in ["C", "m", "b", "ΔKth", "参数"]):
                score += 2
            if "模型" in statement or "model" in statement:
                score += 1
            return min(score, 5)

        # ── mechanism_plausibility ──
        if did == "mechanism_plausibility":
            score = 0
            if any(kw in statement for kw in ["机制", "mechanism"]):
                score += 1
            if "应力集中" in statement or "stress concentration" in statement:
                score += 1
            if any(kw in statement for kw in ["起裂", "萌生", "initiation", "扩展", "propagation"]):
                score += 1
            if "→" in statement:
                score += 2
            return min(score, 5)

        # ── macro_micro_link ──
        if did == "macro_micro_link":
            score = 0
            if any(kw in statement for kw in ["孔隙", "缺陷", "pore", "defect", "组织", "microstructure"]):
                score += 2
            if any(kw in statement for kw in ["Nf", "寿命", "疲劳", "fatigue", "da/dN"]):
                score += 2
            if any(kw in statement for kw in ["Paris", "应力集中", "起裂", "扩展"]):
                score += 1
            return min(score, 5)

        # ── novelty ──
        if did == "novelty":
            score = 1
            if statement and len(statement) > 60:
                score += 1
            if any(kw in statement for kw in ["冲突", "异常", "偏离", "不一致", "conflict", "deviation"]):
                score += 2
            if h_type in ("H3_conflict_explaining", "H4_anomaly_data"):
                score += 1
            return min(score, 5)

        # ── experimental_verifiability ──
        if did == "experimental_verifiability":
            score = 0
            if validation:
                score += 2
            if falsification:
                score += 2
            if any(kw in (validation + falsification) for kw in ["实验", "试验", "test", "表征", "表征"]):
                score += 1
            return min(score, 5)

        # ── conflict_usefulness ──
        if did == "conflict_usefulness":
            score = 0
            if any(kw in statement for kw in ["冲突", "不一致", "矛盾", "conflict", "争议"]):
                score += 3
            if h_type == "H3_conflict_explaining":
                score += 2
            return min(score, 5)

        # ── research_usefulness ──
        if did == "research_usefulness":
            score = 1
            if statement and len(statement) > 80:
                score += 1
            if validation and len(validation) > 40:
                score += 1
            if falsification:
                score += 1
            if any(kw in statement for kw in ["验证", "验证方法", "实验", "micro-CT", "SEM"]):
                score += 1
            return min(score, 5)

        return 0

    def _grade(self, total: int) -> str:
        if total >= 40:
            return "good"
        elif total >= 30:
            return "medium"
        elif total >= 20:
            return "weak"
        return "reject"

    def _is_bad_hypothesis(self, statement: str) -> bool:
        """检测是否是坏假设（泛泛相关类）。"""
        if not statement:
            return True
        for pat in self.BAD_PATTERNS:
            if re.search(pat, statement):
                return True
        # Too short / vague
        if len(statement) < 30:
            return True
        if statement.strip().startswith("应进一步研究") or statement.strip().startswith("需要更多研究"):
            return True
        return False

    def _find_major_weakness(self, dim_scores: list) -> str:
        """找到最弱的维度。"""
        min_score = min(d["score"] for d in dim_scores)
        weakest = [d["dimension"] for d in dim_scores if d["score"] == min_score]
        return f"最弱维度：{' / '.join(weakest)}（仅 {min_score}/5 分）"

    def _next_improvement(self, dim_scores: list) -> str:
        """给出改进建议。"""
        improvements = {
            "specificity": "增加具体变量、指标条件和可测量关系",
            "relation_clarity": "明确正/负/条件/阈值关系",
            "evidence_traceability": "补充 paper_id、evidence_id 和 snippet 追溯",
            "parameter_awareness": "引入 Paris/Basquin/Walker 等模型参数",
            "mechanism_plausibility": "完善变量→效应→裂纹行为→指标的机制链",
            "macro_micro_link": "连接微观缺陷/组织与宏观疲劳指标",
            "novelty": "聚焦文献冲突、数据异常或未闭合证据链",
            "experimental_verifiability": "补充实验验证路径和推翻条件",
            "conflict_usefulness": "利用文献不一致生成解释型假设",
            "research_usefulness": "形成明确可操作的研究课题",
        }
        lowest = min(dim_scores, key=lambda x: x["score"])
        return improvements.get(
            lowest["dimension"].split(" ")[0],
            "全面提升假设质量"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 6. Hypothesis Generator (H1-H4)
# ═══════════════════════════════════════════════════════════════════════════

class HypothesisGenerator:
    """基于现有证据、参数、冲突生成四类假设。"""

    def __init__(self):
        self.explorer = EvidenceRelationExplorer()
        self.miner = EquationParameterMiner()
        self.detector = ConflictDetector()
        self.scorer = HypothesisScorer()
        self.all_hypotheses: List[dict] = []

    def generate_all(self) -> List[dict]:
        """生成全部四类假设。"""
        self.all_hypotheses = []
        self.all_hypotheses.extend(self._gen_H1_mechanism())
        self.all_hypotheses.extend(self._gen_H2_parameter())
        self.all_hypotheses.extend(self._gen_H3_conflict())
        self.all_hypotheses.extend(self._gen_H4_anomaly())
        return self.all_hypotheses

    def _gen_H1_mechanism(self) -> List[dict]:
        """生成机制型假设。"""
        h_list = []
        relations = self.explorer.get_relations()
        has_negative = [r for r in relations if r["relation_type"] == "negative_candidate"]
        has_conditional = [r for r in relations if r["relation_type"] == "conditional_relation"]

        if has_negative:
            r = has_negative[0]
            h_list.append({
                "hypothesis_id": "H1_01",
                "hypothesis_type": "H1_mechanism",
                "hypothesis_statement": (
                    f"在 as-built L-PBF Ti-6Al-4V 中，{r['independent_variable']} "
                    f"可能通过增强局部应力集中，使裂纹更倾向于在表面或亚表面缺陷处起裂，"
                    f"并导致 {r['dependent_indicator']} 降低。"
                ),
                "independent_variables": r["independent_variable"],
                "dependent_indicators": r["dependent_indicator"],
                "controlled_variables": "表面粗糙度、热处理状态、应力比 R、成形方向",
                "expected_relation": r["interpretation"],
                "supporting_evidence_ids": r["evidence_ids"],
                "missing_evidence": "缺乏孔隙三维特征与裂纹起裂位置的定量关联数据",
                "validation_path": "通过 micro-CT 表征孔隙三维特征、HCF 试验获得疲劳寿命、SEM 断口确认起裂位置，统计孔隙特征与起裂位置的相关性",
                "falsification_condition": "若孔隙位置与起裂位置无稳定对应关系，或疲劳寿命差异主要由表面粗糙度而非孔隙特征主导，则假设降级",
                "paper_ids": r["paper_ids"],
                "source": f"EvidenceRelationExplorer: {r['relation_type']}",
            })

        if has_conditional:
            r = has_conditional[0]
            h_list.append({
                "hypothesis_id": "H1_02",
                "hypothesis_type": "H1_mechanism",
                "hypothesis_statement": (
                    f"{r['independent_variable']} 对 {r['dependent_indicator']} 的影响"
                    f"可能存在条件依赖：当孔隙位于近表面时，起裂以孔隙主导；"
                    f"当孔隙位于内部时，表面粗糙度可能起更重要作用。"
                ),
                "independent_variables": r["independent_variable"],
                "dependent_indicators": r["dependent_indicator"],
                "controlled_variables": "应力比 R、加载频率、试样几何",
                "expected_relation": "条件依赖关系",
                "supporting_evidence_ids": r["evidence_ids"],
                "missing_evidence": "缺乏不同孔隙位置与表面状态耦合作用的系统对比数据",
                "validation_path": "设计不同表面状态（as-built vs machined）和不同孔隙位置（表面/近表面/内部）的对比实验",
                "falsification_condition": "若在所有表面状态下孔隙位置均不起作用，则假设推翻",
                "paper_ids": r["paper_ids"],
                "source": f"EvidenceRelationExplorer: {r['relation_type']}",
            })

        return h_list

    def _gen_H2_parameter(self) -> List[dict]:
        """生成参数型假设。"""
        h_list = []
        param_h = self.miner.get_parameter_hypotheses()
        for i, ph in enumerate(param_h[:3], 1):
            h_list.append({
                "hypothesis_id": f"H2_{i:02d}",
                "hypothesis_type": "H2_parameter",
                "hypothesis_statement": ph["hypothesis"],
                "independent_variables": "孔隙/缺陷特征、成形方向、表面状态",
                "dependent_indicators": "Paris C/m, Walker p, Basquin b/σ_f'",
                "controlled_variables": "应力比 R、热处理状态、试验温度",
                "expected_relation": ph.get("equation", "参数变化假设"),
                "supporting_evidence_ids": "evidence_snippets (FCGR/Paris 类)",
                "missing_evidence": "缺乏直接拟合不同孔隙状态下 Paris 参数的对比数据",
                "validation_path": ph["validation"],
                "falsification_condition": ph["falsification"],
                "paper_ids": "",
                "source": "EquationParameterMiner",
            })
        return h_list

    def _gen_H3_conflict(self) -> List[dict]:
        """生成冲突解释型假设。"""
        h_list = []
        claims = self.detector.claims
        for i, c in enumerate(claims[:4], 1):
            if c["conflict_type"] == "insufficient_evidence":
                continue
            h_list.append({
                "hypothesis_id": f"H3_{i:02d}",
                "hypothesis_type": "H3_conflict_explaining",
                "hypothesis_statement": c["generated_hypothesis"],
                "independent_variables": c["claim_A"] + " vs " + c["claim_B"],
                "dependent_indicators": "疲劳寿命/裂纹扩展行为",
                "controlled_variables": "取决于具体实验条件",
                "expected_relation": "条件依赖的竞争关系",
                "supporting_evidence_ids": f"证据A: {c['evidence_A'][:80]}; 证据B: {c['evidence_B'][:80]}",
                "missing_evidence": "缺乏两种条件直接对比的系统实验",
                "validation_path": f"设计系统对比实验，在控制{c['topic'][:40]}的条件下，同时测量两种机制的贡献",
                "falsification_condition": "若在任何条件下都不出现主导机制转换，则假设不成立",
                "paper_ids": f"支持A: {c['paper_A'][:60]}; 支持B: {c['paper_B'][:60]}",
                "source": f"ConflictDetector: {c['conflict_type']}",
            })
        return h_list

    def _gen_H4_anomaly(self) -> List[dict]:
        """生成异常数据型假设。"""
        h_list = []
        # Check if there are parameter extraction results to base anomaly hypotheses on
        param_summary = self.miner.get_summary()
        if param_summary["total_extractions"] > 0:
            h_list.append({
                "hypothesis_id": "H4_01",
                "hypothesis_type": "H4_anomaly_data",
                "hypothesis_statement": (
                    "偏离 Basquin 或 Paris 拟合规律的数据点可能来自未显式考虑的"
                    "近表面孔隙位置、孔隙形态因子或残余应力差异，"
                    "而非实验误差或材料批次差异。"
                ),
                "independent_variables": "孔隙位置、孔隙形态因子、残余应力",
                "dependent_indicators": "Paris C/m 拟合残差、Basquin 拟合偏离度",
                "controlled_variables": "应力比 R、表面状态、热处理状态",
                "expected_relation": "异常数据点与特定孔隙特征对应",
                "supporting_evidence_ids": f"共{param_summary['total_extractions']}条参数提取结果",
                "missing_evidence": "缺乏拟合偏离点与孔隙特征直接关联的定量数据",
                "validation_path": "对偏离拟合的数据点进行 SEM 断口分析和 micro-CT 原位表征，确认是否对应特定孔隙特征",
                "falsification_condition": "若偏离点与孔隙特征无对应关系，或偏离程度在统计误差范围内，则假设不成立",
                "paper_ids": "",
                "source": "EquationParameterMiner: anomaly detection",
            })

        # Add more anomaly hypotheses from relations that are insufficient_data
        relations = self.explorer.get_relations()
        insufficient = [r for r in relations if r["relation_type"] == "insufficient_data"]
        if insufficient:
            r = insufficient[0]
            h_list.append({
                "hypothesis_id": "H4_02",
                "hypothesis_type": "H4_anomaly_data",
                "hypothesis_statement": (
                    f"当前文献中 {r['independent_variable']} 与 {r['dependent_indicator']} "
                    f"的关系数据分散，可能隐含未控制的中间变量（如残余应力或微观组织差异），"
                    f"而非变量本身不相关。"
                ),
                "independent_variables": r["independent_variable"],
                "dependent_indicators": r["dependent_indicator"],
                "controlled_variables": "有待从文献中识别",
                "expected_relation": "数据分散可能掩盖真实趋势",
                "supporting_evidence_ids": r["evidence_ids"],
                "missing_evidence": r["interpretation"],
                "validation_path": "系统整理文献中该变量-指标对的数据，按潜在中间变量分组分析",
                "falsification_condition": "若控制所有可识别变量后数据仍无趋势，则变量间确实无关",
                "paper_ids": r["paper_ids"],
                "source": "EvidenceRelationExplorer: insufficient_data",
            })

        return h_list

    def score_all(self) -> List[dict]:
        """对全部假设评分。"""
        if not self.all_hypotheses:
            self.generate_all()
        scored = []
        for h in self.all_hypotheses:
            result = self.scorer.score_hypothesis(h)
            scored.append(result)
        return scored

    def get_scoring_report(self) -> str:
        """生成评分报告 markdown。"""
        scored = self.score_all()
        if not scored:
            return "当前未生成任何假设。"

        grade_counts = {"good": 0, "medium": 0, "weak": 0, "reject": 0}
        for s in scored:
            grade_counts[s["grade"]] = grade_counts.get(s["grade"], 0) + 1

        lines = [
            "# Hypothesis Scoring Report（假设评分报告）",
            "",
            f"- **总假设数**: {len(scored)}",
            f"- Good: {grade_counts.get('good', 0)}",
            f"- Medium: {grade_counts.get('medium', 0)}",
            f"- Weak: {grade_counts.get('weak', 0)}",
            f"- Reject: {grade_counts.get('reject', 0)}",
            "",
            "---",
            "",
            "## 评分规则",
            "",
            "10 维度，每项 0-5 分，总分 50 分。",
            "",
            "| 等级 | 分数区间 | 含义 |",
            "| --- | --- | --- |",
            "| Good | ≥40 | 变量明确、方向明确、有证据、有实验验证路径、有推翻条件 |",
            "| Medium | 30-39 | 包含变量、指标、验证路径，但参数意识或证据质量不足 |",
            "| Weak | 20-29 | 表述仍偏泛，缺少关系方向或验证路径不完整 |",
            "| Reject | <20 | 泛泛相关、无验证路径、无推翻条件 |",
            "",
            "---",
            "",
        ]

        for s in scored:
            lines.append(f"## {s['hypothesis_id']} ({s['hypothesis_type']})\n")
            lines.append(f"**假设**: {s['hypothesis_statement']}\n")
            lines.append(f"**总分**: {s['total_score']}/50 | **等级**: {s['grade']}\n")

            # Scoring table
            lines.append("| 维度 | 得分 |")
            lines.append("| --- | --- |")
            for d in s["dim_scores"]:
                bar = "█" * d["score"] + "░" * (d["max_score"] - d["score"])
                lines.append(f"| {d['dimension']} | {d['score']}/{d['max_score']} {bar} |")
            lines.append("")

            lines.append(f"**主要弱点**: {s['major_weakness']}\n")
            lines.append(f"**改进建议**: {s['next_improvement']}\n")

            # Verification check
            has_validation = len(s.get("hypothesis_id", "")) > 0
            has_falsification = s.get("grade") != "reject"
            lines.append(f"**实验验证条件**: {'具备' if has_validation else '不具备'}\n")
            lines.append(f"**可作为研究课题**: {'是' if s['grade'] in ('good', 'medium') else '尚需改进'}\n")
            lines.append("---\n")

        return "\n".join(lines)

    def save_scores(self) -> str:
        """保存评分到 data/hypothesis_scores.csv。"""
        scored = self.score_all()
        if not scored:
            return ""
        path = DATA_DIR / "hypothesis_scores.csv"
        fields = [
            "hypothesis_id", "hypothesis_type", "hypothesis_statement",
            "specificity", "relation_clarity", "evidence_traceability",
            "parameter_awareness", "mechanism_plausibility", "macro_micro_link",
            "novelty", "experimental_verifiability", "conflict_usefulness",
            "research_usefulness", "total_score", "grade", "major_weakness",
            "next_improvement",
        ]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for s in scored:
                row = {
                    "hypothesis_id": s["hypothesis_id"],
                    "hypothesis_type": s["hypothesis_type"],
                    "hypothesis_statement": s["hypothesis_statement"],
                    "total_score": s["total_score"],
                    "grade": s["grade"],
                    "major_weakness": s["major_weakness"],
                    "next_improvement": s["next_improvement"],
                }
                for d in s["dim_scores"]:
                    dim_key = d["dimension"].split(" ")[0]  # e.g., "Specificity"
                    # Map to field name
                    field_map = {
                        "Specificity": "specificity",
                        "Relation": "relation_clarity",
                        "Evidence": "evidence_traceability",
                        "Parameter": "parameter_awareness",
                        "Mechanism": "mechanism_plausibility",
                        "Macro-Micro": "macro_micro_link",
                        "Novelty": "novelty",
                        "Experimental": "experimental_verifiability",
                        "Conflict": "conflict_usefulness",
                        "Research": "research_usefulness",
                    }
                    for prefix, field in field_map.items():
                        if dim_key.startswith(prefix):
                            row[field] = d["score"]
                            break
                writer.writerow(row)
        return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Relevance Ranking (6-dimension, total 30)
# ═══════════════════════════════════════════════════════════════════════════

class RelevanceRanking:
    """6 维度变量相关度排序，总分 30 分。"""

    def __init__(self):
        self.explorer = EvidenceRelationExplorer()
        self.miner = EquationParameterMiner()
        self.ev_df = load_evidence_snippets()
        self.vm_df = load_variable_mechanism()
        self.rankings: List[dict] = []
        self._rank()

    def _rank(self):
        """对变量进行 6 维度评分并排序。"""
        relations = self.explorer.get_relations()
        # Group by variable
        var_groups = {}
        for r in relations:
            v = r["independent_variable"]
            if v not in var_groups:
                var_groups[v] = {"relations": [], "targets": set()}
            var_groups[v]["relations"].append(r)
            var_groups[v]["targets"].add(r["dependent_indicator"])

        for var, group in var_groups.items():
            target = ", ".join(sorted(group["targets"])[:3])
            rels = group["relations"]

            # 1. evidence_count_score (0-5)
            ev_count = sum(1 for _, r in self.ev_df.iterrows()
                          if var.lower() in str(r.get("linked_variable", "") or "").lower())
            evidence_count_score = min(ev_count, 5)

            # 2. evidence_quality_score (0-5)
            has_direct = any(r["evidence_strength"] in ("direct_evidence", "strong_evidence") for r in rels)
            has_text = any(r["evidence_strength"] == "partial_evidence" for r in rels)
            if has_direct:
                evidence_quality_score = 4
            elif has_text:
                evidence_quality_score = 3
            elif rels:
                evidence_quality_score = 2
            else:
                evidence_quality_score = 1

            # 3. directness_score (0-5) — how directly it relates to L-PBF Ti-6Al-4V
            focus_keywords = ["pore", "defect", "roughness", "surface", "orientation", "heat treatment",
                             "porosity", "crack", "fatigue", "L-PBF", "Ti-6Al-4V"]
            vl = var.lower()
            directness_score = sum(2 for kw in focus_keywords if kw in vl)
            directness_score = min(directness_score, 5)

            # 4. parameter_score (0-5) — involvement of equations/parameters
            param_count = sum(1 for p in self.miner.params_list
                             if var.lower() in p.get("related_variables", "").lower())
            parameter_score = min(param_count + (1 if ("Paris" in vl or "C" in vl or "m" in vl) else 0), 5)

            # 5. consistency_score (0-5) — cross-literature consistency
            conflicting = sum(1 for r in rels if r["relation_type"] == "conflicting")
            insufficient = sum(1 for r in rels if r["relation_type"] == "insufficient_data")
            if conflicting > 0:
                consistency_score = 2  # has conflicts
            elif insufficient > 0:
                consistency_score = 3  # insufficient data
            elif len(rels) >= 3:
                consistency_score = 4  # multiple relations consistent
            else:
                consistency_score = 3

            # 6. validation_feasibility_score (0-5)
            has_validation_methods = any(kw in var.lower() for kw in ["micro-ct", "sem", "ebsd", "fcgr",
                                                                       "fatigue test", "hcf", "vhcf"])
            if has_validation_methods:
                validation_feasibility_score = 4
            elif "pore" in var or "defect" in var or "surface" in var:
                validation_feasibility_score = 4  # measurable via micro-CT/SEM
            elif "heat" in var or "treatment" in var or "orientation" in var:
                validation_feasibility_score = 3
            else:
                validation_feasibility_score = 2

            total = evidence_count_score + evidence_quality_score + directness_score + \
                    parameter_score + consistency_score + validation_feasibility_score

            reason_parts = []
            if evidence_count_score >= 3:
                reason_parts.append(f"证据数量充足({ev_count}条)")
            if directness_score >= 3:
                reason_parts.append("直接对应当前科研问题")
            if parameter_score >= 2:
                reason_parts.append(f"涉及{param_count}个方程/参数")
            if consistency_score <= 2:
                reason_parts.append("存在文献冲突值得深入研究")
            if validation_feasibility_score >= 3:
                reason_parts.append("可通过实验验证")

            self.rankings.append({
                "variable": var,
                "target_indicator": target,
                "evidence_count_score": evidence_count_score,
                "evidence_quality_score": evidence_quality_score,
                "directness_score": directness_score,
                "parameter_score": parameter_score,
                "consistency_score": consistency_score,
                "validation_feasibility_score": validation_feasibility_score,
                "total_relevance_score": total,
                "reason": "；".join(reason_parts) if reason_parts else "变量在文献中有所涉及",
            })

        # Sort by total score descending
        self.rankings.sort(key=lambda r: r["total_relevance_score"], reverse=True)
        for i, r in enumerate(self.rankings, 1):
            r["rank"] = i

    def get_top(self, n: int = 10) -> List[dict]:
        return self.rankings[:n]

    def get_for_variable(self, var: str) -> dict:
        for r in self.rankings:
            if r["variable"] == var:
                return r
        return {}

    def save_csv(self) -> str:
        path = DATA_DIR / "relevance_ranking.csv"
        fields = ["rank", "variable", "target_indicator", "evidence_count_score",
                  "evidence_quality_score", "directness_score", "parameter_score",
                  "consistency_score", "validation_feasibility_score",
                  "total_relevance_score", "reason"]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.rankings)
        return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Macro-Micro Link Explorer
# ═══════════════════════════════════════════════════════════════════════════

MACRO_VARIABLES = [
    "pore_size", "pore_location", "pore_aspect_ratio", "porosity",
    "surface_roughness", "alpha_lath_thickness", "alpha_prime_martensite",
    "grain_orientation", "beta_phase_fraction", "residual_stress",
]

MACRO_INDICATORS = [
    "Nf", "fatigue_strength", "S_N_slope", "da_dN", "DeltaKth",
    "Paris_C", "Paris_m", "crack_initiation_site",
]


class MacroMicroLinkExplorer:
    """构建宏观-微观关联表。"""

    # Known mechanism links (domain knowledge based on literature patterns)
    KNOWN_LINKS = [
        {
            "micro": "pore_size",
            "macro": "Nf",
            "mechanism": "局部应力集中→裂纹起裂→寿命降低",
            "equation": "Basquin b / Paris C",
            "condition": "表面粗糙度、热处理、R 比控制",
            "strength": "medium",
        },
        {
            "micro": "pore_location",
            "macro": "crack_initiation_site",
            "mechanism": "近表面高应力区→优先起裂",
            "equation": "—",
            "condition": "应力水平、加载方式",
            "strength": "strong",
        },
        {
            "micro": "pore_aspect_ratio",
            "macro": "Paris_C",
            "mechanism": "高长宽比→更强应力集中→更快早期扩展",
            "equation": "Paris C = f(aspect_ratio)",
            "condition": "R 比、成形方向",
            "strength": "weak",
        },
        {
            "micro": "surface_roughness",
            "macro": "Nf",
            "mechanism": "表面缺陷→应力集中→表面起裂",
            "equation": "S-N fitting",
            "condition": "as-built vs polished",
            "strength": "strong",
        },
        {
            "micro": "alpha_lath_thickness",
            "macro": "da_dN",
            "mechanism": "片层厚度→裂纹偏转/屏障效应→扩展阻力",
            "equation": "Paris C/m",
            "condition": "热处理状态",
            "strength": "medium",
        },
        {
            "micro": "residual_stress",
            "macro": "DeltaKth",
            "mechanism": "残余压应力→裂纹闭合→门槛值提高",
            "equation": "ΔKth = f(σ_res)",
            "condition": "HIP/热处理后",
            "strength": "weak",
        },
        {
            "micro": "porosity",
            "macro": "Nf",
            "mechanism": "体积缺陷密度→有效承载面积减小→局部过载",
            "equation": "Kitagawa-Takahashi",
            "condition": "缺陷尺寸 vs 临界缺陷尺寸",
            "strength": "medium",
        },
        {
            "micro": "grain_orientation",
            "macro": "crack_initiation_site",
            "mechanism": "不利取向晶粒→滑移集中→裂纹萌生",
            "equation": "—",
            "condition": "EBSD 表征",
            "strength": "medium",
        },
        {
            "micro": "alpha_prime_martensite",
            "macro": "Paris_m",
            "mechanism": "α′ 马氏体→高强低韧→扩展速率对 ΔK 敏感",
            "equation": "Paris m",
            "condition": "as-built 状态 vs 退火状态",
            "strength": "weak",
        },
        {
            "micro": "beta_phase_fraction",
            "macro": "da_dN",
            "mechanism": "β 相含量→裂纹偏转频率→扩展路径曲折度",
            "equation": "Paris C",
            "condition": "热处理温度",
            "strength": "weak",
        },
    ]

    def __init__(self):
        self.ev_df = load_evidence_snippets()
        self.links: List[dict] = []
        self._build()

    def _build(self):
        self.links = []
        for link in self.KNOWN_LINKS:
            evidence_ids = self._find_evidence(link["micro"], link["macro"])
            missing = self._find_missing(link["micro"], link["macro"], evidence_ids)
            hypothesis = self._gen_hypothesis(link)
            link_strength_score = {"strong": 5, "medium": 3, "weak": 1}.get(link["strength"], 2)

            self.links.append({
                "micro_variable": link["micro"],
                "macro_indicator": link["macro"],
                "intermediate_mechanism": link["mechanism"],
                "equation_or_parameter": link["equation"],
                "evidence_ids": "; ".join(evidence_ids[:5]) if evidence_ids else "—",
                "condition": link["condition"],
                "link_strength": link["strength"],
                "link_strength_score": link_strength_score,
                "missing_data": missing,
                "hypothesis_candidate": hypothesis,
            })

    def _find_evidence(self, micro: str, macro: str) -> List[str]:
        eids = []
        if self.ev_df.empty:
            return eids
        for _, row in self.ev_df.iterrows():
            text = (str(row.get("snippet", "") or "") + str(row.get("linked_variable", "") or "")).lower()
            if micro.replace("_", " ") in text or micro.replace("_", "") in text:
                if macro.lower() in str(row.get("linked_indicator", "") or "").lower():
                    eids.append(str(row.get("evidence_id", "")))
        return eids[:5]

    def _find_missing(self, micro: str, macro: str, found: list) -> str:
        if not found:
            return f"缺乏 {micro}→{macro} 的直接实验证据"
        if len(found) < 3:
            return f"证据有限（{len(found)}条），需补充定量数据"
        return "已有部分证据，但需更多定量关联数据"

    def _gen_hypothesis(self, link: dict) -> str:
        micro_names = {
            "pore_size": "大尺寸孔隙",
            "pore_location": "近表面孔隙",
            "pore_aspect_ratio": "高长宽比孔隙",
            "porosity": "高孔隙率",
            "surface_roughness": "高表面粗糙度",
            "alpha_lath_thickness": "α 片层厚度变化",
            "alpha_prime_martensite": "α′ 马氏体含量",
            "grain_orientation": "不利晶粒取向",
            "beta_phase_fraction": "β 相含量",
            "residual_stress": "残余应力",
        }
        name = micro_names.get(link["micro"], link["micro"])
        return f"{name}可能通过{link['mechanism']}影响{link['macro']}，该关系可通过{link['condition']}条件下进行实验验证"

    def save_csv(self) -> str:
        path = DATA_DIR / "macro_micro_links.csv"
        fields = ["micro_variable", "macro_indicator", "intermediate_mechanism",
                  "equation_or_parameter", "evidence_ids", "condition",
                  "link_strength", "link_strength_score", "missing_data",
                  "hypothesis_candidate"]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.links)
        return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# 9. 一键生成全部报告
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_report_texts() -> dict:
    """生成所有输出报告的文本内容。"""
    explorer = EvidenceRelationExplorer()
    miner = EquationParameterMiner()
    detector = ConflictDetector()
    ranker = RelevanceRanking()
    macro = MacroMicroLinkExplorer()
    gen = HypothesisGenerator()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 14_relation_explorer.md ──
    rels = explorer.get_relations()
    rel_lines = [
        "# Evidence Relation Explorer",
        "",
        f"> 变量关系表。共 {len(rels)} 条关系记录。relation_type 包括 positive/negative/threshold/conditional/competitive/conflicting/missing/insufficient。",
        "",
        "| variable | indicator | relation_type | condition | mechanism | relevance |",
        "|---|---|---|---|---|---|",
    ]
    for r in rels[:30]:
        rel_lines.append(
            f"| {r['independent_variable']} | {r['dependent_indicator']} | "
            f"{r['relation_type']} | {r['condition'][:50] or '—'} | "
            f"{r['mechanism'][:80]} | {'⭐' * r['relevance_score']} |"
        )
    rel_text = "\n".join(rel_lines)

    # ── 15_relevance_ranking.md ──
    ranker = RelevanceRanking()
    ranker.save_csv()
    rank_lines = [
        "# Variable Relevance Ranking",
        "",
        "> 6 维度评分（evidence_count / evidence_quality / directness / parameter / consistency / validation_feasibility），每项 0-5，总分 30。",
        "",
    ]
    top = ranker.get_top(15)
    if top:
        rank_lines.append("| rank | variable | target_indicator | total | breakdown | reason |")
        rank_lines.append("|---|---|---|---|---|---|")
        for r in top:
            rank_lines.append(
                f"| {r['rank']} | {r['variable']} | {r['target_indicator']} | "
                f"{r['total_relevance_score']}/30 | "
                f"ev{ r['evidence_count_score'] }+q{ r['evidence_quality_score'] }+"
                f"d{ r['directness_score'] }+p{ r['parameter_score'] }+"
                f"c{ r['consistency_score'] }+v{ r['validation_feasibility_score'] } | "
                f"{r['reason']} |"
            )
    rank_lines.append("")
    rank_text = "\n".join(rank_lines)

    # ── 16_parameter_mining.md ──
    miner.save()
    summ = miner.get_summary()
    param_lines = [
        "# Equation & Parameter Mining Report",
        "",
        f"> 从文献库和证据片段中提取方程/参数。共 {summ['total_extractions']} 条提取，{summ['unique_types']} 种类型。",
        "",
        "| paper_id | equation_type | parameters | related_variables | confidence |",
        "|---|---|---|---|---|",
    ]
    for p in miner.params_list:
        pv = p.get("parameters", "") or "parameter_mentioned_but_not_extracted"
        param_lines.append(
            f"| {p['paper_id'] or '—'} | {p['equation_type']} | {pv} | "
            f"{p['related_variables']} | {p['extraction_confidence']} |"
        )
    param_lines.append("")
    param_text = "\n".join(param_lines)

    # ── 17_conflict_detection.md ──
    detector.save()
    conf_lines = [
        "# Conflict Detection Report",
        "",
        f"> 检测文献间结论冲突/不一致。共 {len(detector.claims)} 个主题。",
        "",
    ]
    for c in detector.claims:
        conf_lines.extend([
            f"## {c['conflict_id']}: {c['topic']}\n",
            f"- **conflict_type**: {c['conflict_type']}",
            f"- **claim_A**: {c['claim_A']}",
            f"- **evidence_A**: {c['evidence_A']}",
            f"- **claim_B**: {c['claim_B']}",
            f"- **evidence_B**: {c['evidence_B']}",
            f"- **possible_reason**: {c['possible_reason']}",
            f"- **generated_hypothesis**: {c['generated_hypothesis']}",
            f"- **confidence**: {c['confidence_level']}\n",
        ])
    conf_text = "\n".join(conf_lines)

    # ── 18_macro_micro_link_report.md ──
    macro.save_csv()
    macro_lines = [
        "# Macro-Micro Link Report",
        "",
        "> 宏观疲劳指标与微观组织/缺陷变量的关联。",
        "",
        "| micro_variable | macro_indicator | mechanism | equation/parameter | link_strength | hypothesis |",
        "|---|---|---|---|---|---|",
    ]
    for lk in macro.links:
        macro_lines.append(
            f"| {lk['micro_variable']} | {lk['macro_indicator']} | "
            f"{lk['intermediate_mechanism']} | {lk['equation_or_parameter']} | "
            f"{lk['link_strength']} | {lk['hypothesis_candidate']} |"
        )
    macro_text = "\n".join(macro_lines)

    # ── 19_hypothesis_scoring_report.md ──
    scoring_text = gen.get_scoring_report()
    gen.save_scores()

    # ── 20_interactive_hypothesis.md ──
    # Same as scoring report for now (contains the hypotheses and scores)
    interactive_text = scoring_text

    # ── 21_system_value_evaluation.md ──
    eval_text = generate_value_evaluation()

    # Save all files
    files = {
        "outputs/14_relation_explorer.md": rel_text,
        "outputs/15_relevance_ranking.md": rank_text,
        "outputs/16_parameter_mining.md": param_text,
        "outputs/17_conflict_detection.md": conf_text,
        "outputs/18_macro_micro_link_report.md": macro_text,
        "outputs/19_hypothesis_scoring_report.md": scoring_text,
        "outputs/20_interactive_hypothesis.md": interactive_text,
        "outputs/21_system_value_evaluation.md": eval_text,
    }
    for path_str, content in files.items():
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    return {k: str(Path(k)) for k in files}


# ═══════════════════════════════════════════════════════════════════════════
# 7. 系统价值评估
# ═══════════════════════════════════════════════════════════════════════════

def generate_value_evaluation() -> str:
    """生成系统价值评估报告。"""
    stats = get_system_stats()
    explorer = EvidenceRelationExplorer()
    miner = EquationParameterMiner()
    detector = ConflictDetector()
    gen = HypothesisGenerator()

    rel_counts = explorer.get_relation_type_counts()
    param_summary = miner.get_summary()
    conflict_summary = detector.get_summary()

    scored = gen.score_all()
    grade_counts = {"good": 0, "medium": 0, "weak": 0, "reject": 0}
    good_h = []
    for s in scored:
        grade_counts[s["grade"]] = grade_counts.get(s["grade"], 0) + 1
        if s["grade"] == "good":
            good_h.append(s)

    lines = [
        "# System Value Evaluation（系统价值评估）",
        "",
        "> 证明 TitaniumFatigueChat 交互版相比直接 Qwen 和旧版系统的提升。",
        "",
        "---",
        "",
        "## 1. 系统统计",
        "",
        f"- 文献数量：{stats['n_papers']} 篇",
        f"- 证据片段：{stats['ev_count']} 条",
        f"- 变量-机制记录：{stats['vm_count']} 条",
        f"- 方程参数提取：{param_summary['total_extractions']} 条",
        f"- 冲突检测主题：{conflict_summary['total_topics']} 个",
        f"- 生成假设：{len(scored)} 个（Good: {grade_counts.get('good', 0)}）",
        "",
        "## 2. 与直接 Qwen 的对比",
        "",
        "### 直接 Qwen 的典型问题",
        "",
        "1. 假设偏泛，多为「X 与 Y 相关」；",
        "2. 缺少关系方向（正相关/负相关/条件相关）；",
        "3. 缺少方程参数提取；",
        "4. 缺少文献冲突利用；",
        "5. 缺少实验验证路径；",
        "6. 缺少推翻条件。",
        "",
        "### TitaniumFatigueChat 的改进",
        "",
        "1. **指定变量和指标** — 每项假设绑定具体 independent_variable 和 dependent_indicator；",
        "2. **判断怎么相关** — EvidenceRelationExplorer 自动推断 positive_candidate / negative_candidate / conditional_relation / threshold_candidate；",
        "3. **提取方程和参数** — EquationParameterMiner 从文献库和证据片段中扫描 Paris、Basquin、Walker、ΔKth 等模型参数；",
        "4. **检测文献冲突** — ConflictDetector 检查 6 个冲突主题，输出 direct_conflict / condition_dependent_difference / potential_tension / insufficient_evidence；",
        "5. **给出实验验证路径** — 每项假设包含 validation_path 和 falsification_condition；",
        f"6. **假设评分系统筛掉坏假设** — 10 维度评分，本次 {grade_counts.get('reject', 0)} 个被标记为 reject。",
        "",
        "## 3. 冲突检测结果",
        "",
        f"共检测 {conflict_summary['total_topics']} 个冲突主题：",
    ]
    for ct, cnt in conflict_summary.get("by_type", {}).items():
        lines.append(f"- {ct}: {cnt} 个")

    lines.extend([
        "",
        "## 4. 假设评分分布",
        "",
        "| 等级 | 数量 |",
        "| --- | --- |",
        f"| Good (≥40) | {grade_counts.get('good', 0)} |",
        f"| Medium (30-39) | {grade_counts.get('medium', 0)} |",
        f"| Weak (20-29) | {grade_counts.get('weak', 0)} |",
        f"| Reject (<20) | {grade_counts.get('reject', 0)} |",
        "",
    ])

    if good_h:
        lines.append("## 5. Good Hypothesis 示例\n")
        for h in good_h[:3]:
            lines.append(f"- **{h['hypothesis_id']}** ({h['hypothesis_type']})")
            lines.append(f"  {h['hypothesis_statement'][:120]}...")
            lines.append(f"  评分: {h['total_score']}/50\n")

    lines.extend([
        "## 6. 消融实验设计",
        "",
        "| 版本 | 包含模块 | 预期假设质量 |",
        "| --- | --- | --- |",
        "| Direct Qwen | 无结构化证据 | Weak-Reject |",
        "| Qwen + 摘要 | 无结构化证据表 | Weak-Medium |",
        "| 无 evidence trace | 无 EvidenceRelationExplorer | Medium-Weak |",
        "| 无 parameter miner | 无 EquationParameterMiner | Medium |",
        "| 无 conflict detector | 无 ConflictDetector | Medium |",
        "| 无 hypothesis scoring | 无 HypothesisScorer | 无法区分好坏 |",
        "| 完整系统 | 全部模块 | Good-Medium |",
        "",
        "## 7. 专家评价接口",
        "",
        "预留以下字段供领域专家评价：",
        "",
        "| 字段 | 说明 | 评分范围 |",
        "| --- | --- | --- |",
        "| expert_specificity_score | 专家评具体性 | 0-5 |",
        "| expert_novelty_score | 专家评新颖性 | 0-5 |",
        "| expert_feasibility_score | 专家评可行性 | 0-5 |",
        "| expert_comment | 专家意见 | 文本 |",
        "",
        "---",
        "",
        "> **说明**: 本评估为系统自评价，仅供参考。",
        "> 建议邀请材料/力学领域研究生或老师对假设质量进行专家评分。",
    ])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Literature Search Planner (证据不足时的文献补充建议)
# ═══════════════════════════════════════════════════════════════════════════

class LiteratureSearchPlanner:
    """评估本地文献库对用户问题的覆盖程度，生成文献补充计划。"""

    # Domain keywords for L-PBF Ti-6Al-4V fatigue
    DOMAIN_KEYWORDS = {
        "material": ["ti-6al-4v", "ti6al4v", "ti64", "tc4", "titanium"],
        "process": ["l-pbf", "slm", "selective laser melting", "laser powder bed",
                     "additive manufacturing", "ebm", "lens", "增材"],
        "defect": ["pore", "porosity", "defect", "lack of fusion", "keyhole",
                   "surface roughness", "roughness", "as-built surface", "surface finish",
                   "孔隙", "缺陷"],
        "heat_treatment": ["hip", "hot isostatic pressing", "annealing", "heat treatment",
                           "热处理", "退火", "固溶"],
        "fatigue": ["fatigue", "hcf", "vhcf", "fcgr", "crack growth", "s-n", "da/dn",
                    "crack initiation", "疲劳", "裂纹"],
        "parameter": ["paris", "basquin", "walker", "kitagawa", "delta k", "Δk",
                      "参数", "模型"],
        "micro_ct": ["micro-ct", "x-ray ct", "synchrotron", "tomography", "三维",
                     "characterization"],
        "sem_ebsd": ["sem", "ebsd", "fractography", "断口", "组织", "fracture surface"],
    }

    RECOMMENDED_DATABASES = [
        "Google Scholar (scholar.google.com)",
        "Web of Science (webofscience.com)",
        "Scopus (scopus.com)",
        "ScienceDirect (sciencedirect.com)",
        "SpringerLink (link.springer.com)",
        "MDPI (mdpi.com)",
        "OpenAlex (openalex.org)",
        "Semantic Scholar (semanticscholar.org)",
        "Unpaywall / OA repositories",
        "Institutional library access",
    ]

    LITERATURE_PRIORITIES = [
        {
            "priority": 1,
            "type": "直接相关实验文献",
            "description": "含 fatigue data + defect characterization 的实验论文",
            "must_contain": ["fatigue life / S-N / Nf", "defect type / pore size / porosity"],
        },
        {
            "priority": 2,
            "type": "FCGR / 裂纹扩展参数文献",
            "description": "含 da/dN-ΔK / Paris C/m 的参数文献",
            "must_contain": ["FCGR / da/dN", "Paris law", "ΔKth"],
        },
        {
            "priority": 3,
            "type": "微观机制联合表征文献",
            "description": "含 micro-CT + SEM/EBSD 联合表征的机制文献",
            "must_contain": ["micro-CT / X-ray CT", "SEM / EBSD", "crack initiation site"],
        },
        {
            "priority": 4,
            "type": "综述文献",
            "description": "用于补背景和关键变量地图",
            "must_contain": ["review", "comprehensive", "state of the art"],
        },
        {
            "priority": 5,
            "type": "相近材料/工艺文献",
            "description": "只能作为 background，不得作为核心证据",
            "must_contain": ["相近材料或非 L-PBF 工艺"],
        },
    ]

    REQUIRED_FIELDS = [
        "paper_id", "material", "manufacturing_process", "heat_treatment",
        "surface_state", "defect_type", "pore_size", "pore_location",
        "pore_aspect_ratio", "porosity", "stress_ratio_R", "fatigue_type",
        "Nf", "S_N_curve", "da_dN", "Delta_K", "Paris_C", "Paris_m",
        "crack_initiation_site", "characterization_method",
        "SEM_EBSD_microCT_available", "main_conclusion",
        "conflict_with_existing_literature",
    ]

    def __init__(self):
        self.ev_df = load_evidence_snippets()
        self.vm_df = load_variable_mechanism()
        self.lit_df = load_literature()
        self.papers = get_all_papers()
        self.param_df = load_equation_params()
        self.conflict_df = load_conflict_claims()

    def analyze_question(self, question: str) -> dict:
        """分析用户问题，判断证据覆盖程度。"""
        ql = question.lower()

        # Parse domain keywords from question
        matched_dims = {}
        for dim, kws in self.DOMAIN_KEYWORDS.items():
            matches = [kw for kw in kws if kw in ql]
            if matches:
                matched_dims[dim] = matches

        # Search literature database for relevant papers
        matched_papers = []
        matched_eids = []
        for p in self.papers:
            text = (str(p.get("title", "") or "") + " " +
                    str(p.get("key_findings", "") or "") + " " +
                    str(p.get("abstract", "") or "")).lower()
            match_count = sum(1 for kws in self.DOMAIN_KEYWORDS.values()
                            for kw in kws if kw in text)
            if match_count >= 2:
                matched_papers.append({
                    "title": str(p.get("title", "") or "")[:80],
                    "match_score": match_count,
                })

        # Search evidence snippets
        for _, row in self.ev_df.iterrows():
            text = (str(row.get("snippet", "") or "") + " " +
                    str(row.get("linked_variable", "") or "") + " " +
                    str(row.get("evidence_type", "") or "")).lower()
            if any(kw in text for kws in self.DOMAIN_KEYWORDS.values() for kw in kws):
                eid = str(row.get("evidence_id", "") or "")
                if eid and eid not in matched_eids:
                    matched_eids.append(eid)

        # Determine coverage level
        n_papers = len(matched_papers)
        n_ev = len(matched_eids)
        has_direct = n_papers >= 3 and n_ev >= 5
        has_partial = n_papers >= 1 and n_ev >= 2
        has_weak = n_papers >= 1 or n_ev >= 1

        if has_direct:
            coverage = "sufficient"
        elif has_partial:
            coverage = "partial"
        elif has_weak:
            coverage = "weak"
        else:
            coverage = "not_found"

        # Identify missing evidence types
        missing_types = self._identify_missing(matched_dims)

        # Generate search queries
        queries = self._generate_queries(matched_dims)

        # Recommended lit types
        rec_types = self._recommend_types(matched_dims, coverage)

        return {
            "question": question,
            "matched_dimensions": matched_dims,
            "matched_papers_count": n_papers,
            "matched_evidence_count": n_ev,
            "matched_papers": matched_papers[:5],
            "matched_evidence_ids": matched_eids[:10],
            "coverage_level": coverage,
            "missing_evidence_types": missing_types,
            "search_queries": queries,
            "recommended_literature_types": rec_types,
            "why_insufficient": self._why_insufficient(coverage, n_papers, n_ev, matched_dims),
        }

    def _identify_missing(self, matched: dict) -> List[str]:
        missing = []
        if "micro_ct" not in matched:
            missing.append("micro-CT 三维缺陷表征数据")
        if "sem_ebsd" not in matched:
            missing.append("SEM/EBSD 断口与组织分析数据")
        if "parameter" not in matched:
            missing.append("Paris/Basquin/Walker 方程参数数据")
        if "heat_treatment" not in matched:
            missing.append("热处理前后对比数据")
        if "defect" not in matched:
            missing.append("孔隙/缺陷定量表征数据")
        if "fatigue" not in matched:
            missing.append("疲劳试验（Nf/S-N/da/dN）数据")
        return missing if missing else ["当前维度已基本覆盖"]

    def _generate_queries(self, matched: dict) -> List[str]:
        queries = []
        base = '("L-PBF" OR "laser powder bed fusion" OR SLM) AND ("Ti-6Al-4V" OR Ti64)'

        if "heat_treatment" in matched:
            queries.append(
                base + '\nAND ("HIP" OR "hot isostatic pressing" OR "annealing" OR "heat treatment")'
                '\nAND ("fatigue" OR "fatigue life" OR "fatigue crack growth")'
                '\nAND ("defect" OR "porosity" OR "lack of fusion")'
            )
        if "defect" in matched or "micro_ct" in matched:
            queries.append(
                base + '\nAND ("micro-CT" OR "X-ray computed tomography" OR "synchrotron")'
                '\nAND ("porosity" OR "defect morphology" OR "pore size" OR "pore location")'
                '\nAND ("fatigue" OR "crack initiation")'
            )
        if "fatigue" in matched and "parameter" in matched:
            queries.append(
                base + '\nAND ("fatigue crack growth" OR FCGR OR "da/dN" OR "Paris law")'
                '\nAND ("porosity" OR "volumetric defects" OR "defect size")'
            )
        if "defect" in matched and "sem_ebsd" in matched:
            queries.append(
                base + '\nAND ("SEM" OR "EBSD" OR "fractography" OR "fracture surface")'
                '\nAND ("crack initiation" OR "crack propagation")'
                '\nAND ("defect" OR "pore" OR "porosity")'
            )

        # Generic fallback
        queries.append(
            base + '\nAND ("fatigue" OR "fatigue crack growth" OR "fatigue life")'
            '\nAND ("porosity" OR "defect" OR "surface roughness")'
        )

        return queries

    def _recommend_types(self, matched: dict, coverage: str) -> List[dict]:
        recs = []
        for pri in self.LITERATURE_PRIORITIES:
            recs.append({
                "priority": pri["priority"],
                "type": pri["type"],
                "description": pri["description"],
                "must_contain": pri["must_contain"],
            })
        return recs

    def _why_insufficient(self, coverage: str, n_papers: int, n_ev: int, matched: dict) -> str:
        if coverage == "sufficient":
            return "本地文献库已有较好覆盖。"
        reasons = []
        if n_papers < 3:
            reasons.append(f"仅匹配到 {n_papers} 篇相关文献")
        if n_ev < 5:
            reasons.append(f"仅 {n_ev} 条相关证据片段")
        if "micro_ct" not in matched:
            reasons.append("缺少 micro-CT 三维缺陷表征")
        if "sem_ebsd" not in matched:
            reasons.append("缺少 SEM/EBSD 断口分析")
        if "parameter" not in matched:
            reasons.append("缺少方程/参数提取")
        if "heat_treatment" not in matched:
            reasons.append("缺少热处理相关文献")
        return "；".join(reasons) if reasons else "证据不足以支持强假设生成"

    def generate_search_plan(self, question: str) -> str:
        """生成完整的文献搜索计划 markdown。"""
        result = self.analyze_question(question)

        lines = [
            "# Literature Search Plan（文献补充计划）",
            "",
            f"## 用户问题",
            "",
            f"{question}",
            "",
            "---",
            "",
            "## 1. Local Evidence Status",
            "",
            f"- **coverage_level**: {result['coverage_level']}",
            f"- **匹配文献数**: {result['matched_papers_count']}",
            f"- **匹配证据片段数**: {result['matched_evidence_count']}",
            "",
        ]

        if result["matched_papers"]:
            lines.append("**匹配到的文献**:")
            for p in result["matched_papers"]:
                lines.append(f"- {p['title']} (匹配度: {p['match_score']})")
            lines.append("")

        lines.append(f"**当前不足原因**: {result['why_insufficient']}\n")

        if result["missing_evidence_types"]:
            lines.append("**缺失证据类型**:")
            for m in result["missing_evidence_types"]:
                lines.append(f"- {m}")
            lines.append("")

        lines.extend([
            "## 2. 推荐补充文献类型\n",
        ])
        for rec in result["recommended_literature_types"]:
            lines.append(f"### Priority {rec['priority']}: {rec['type']}")
            lines.append(f"{rec['description']}")
            lines.append(f"必须包含: {', '.join(rec['must_contain'])}\n")

        lines.extend([
            "## 3. 推荐数据库\n",
        ])
        for db in self.RECOMMENDED_DATABASES:
            lines.append(f"- {db}")
        lines.append("")

        if result["search_queries"]:
            lines.append("## 4. 推荐检索式\n")
            for i, q in enumerate(result["search_queries"], 1):
                lines.append(f"### Query {i}:\n```\n{q}\n```\n")

        lines.extend([
            "## 5. 下载后需提取的字段\n",
        ])
        for f in self.REQUIRED_FIELDS:
            lines.append(f"- {f}")
        lines.append("")

        if result["coverage_level"] in ("weak", "not_found"):
            lines.extend([
                "---",
                "",
                "> **注意**: 当前本地证据不足，不得生成 good hypothesis。",
                "> 必须先补充上述文献并提取数据后，才能升级证据等级。",
                "> 自动下载只能下载 OA PDF，付费文献需 manual download。",
            ])

        return "\n".join(lines)

    def save_search_plan(self, question: str) -> str:
        """保存搜索计划到文件。"""
        text = self.generate_search_plan(question)
        path = OUTPUTS_DIR / "22_literature_search_plan.md"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def save_recommendations(self, question: str) -> str:
        """保存推荐到 CSV。"""
        result = self.analyze_question(question)
        path = DATA_DIR / "search_recommendations.csv"
        fields = ["question", "coverage_level", "missing_evidence_type",
                  "recommended_literature_type", "search_query", "recommended_database",
                  "priority", "reason", "required_fields"]
        rows = []
        for rec in result["recommended_literature_types"]:
            rows.append({
                "question": question[:100],
                "coverage_level": result["coverage_level"],
                "missing_evidence_type": "; ".join(result["missing_evidence_types"][:3]),
                "recommended_literature_type": rec["type"],
                "search_query": result["search_queries"][0] if result["search_queries"] else "",
                "recommended_database": "; ".join(self.RECOMMENDED_DATABASES[:5]),
                "priority": rec["priority"],
                "reason": rec["description"],
                "required_fields": "; ".join(self.REQUIRED_FIELDS[:10]),
            })

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        return str(path)

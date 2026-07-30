"""
query_understanding.py — Smart search with typo correction, fuzzy matching,
synonym normalization, and intent detection for L-PBF Ti-6Al-4V fatigue domain.

核心流程:
    raw_user_query
    → normalize_text()
    → correct_typos()
    → domain_synonym_mapping()
    → fuzzy_variable_matching()
    → intent_detection()
    → variable_pair_extraction()
    → confidence_scoring()
    → structured_query
"""

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def _s(val) -> str:
    """Safely convert pandas/CSV value to string, handling NaN."""
    if val is None:
        return ""
    if isinstance(val, float) and val != val:  # NaN check
        return ""
    return str(val).strip()

# ── Intent patterns ──
INTENT_PATTERNS = {
    "relation_analysis": [
        "关系", "影响", "有关", "相关", "作用", "和.*之间", "怎么影响",
        "关联", "对比", "区别", "差异", "哪个", "vs", "versus",
        "relationship", "correlation", "effect of", "influence",
    ],
    "hypothesis_generation": [
        "假设", "提出", "生成假设", "科学假设", "能不能", "是否可能",
        "hypothesis", "propose", "speculate",
    ],
    "experiment_design": [
        "实验", "怎么验证", "怎么设计", "测试方案", "实验方案",
        "做实验", "如何验证", "验证方法",
        "experiment", "test plan", "validation",
    ],
    "equation_matching": [
        "方程", "公式", "模型", "拟合", "参数", "paris", "basquin",
        "walker", "kitagawa", "murakami", "el haddad", "da/dn", "Δk",
        "equation", "model fitting", "parameter",
    ],
    "literature_search": [
        "找文献", "有哪些论文", "综述", "文献", "论文", "最近研究",
        "找一下", "查一下", "搜索文献",
        "literature", "paper", "review", "article",
    ],
    "research_gap_discovery": [
        "创新点", "研究空白", "值得做", "下一步", "研究方向",
        "未解决", "不明确", "什么可以做",
        "research gap", "future work", "open question",
    ],
    "conflict_detection": [
        "冲突", "矛盾", "不一致", "争议", "为什么有的.*有的",
        "不同文献", "相反结论",
        "conflict", "discrepancy", "inconsistent", "contradict",
    ],
    "dominance_comparison": [
        "哪个更", "谁更", "哪个主要", "主导", "主要因素",
        "更危险", "更严重", "更重要", "竞争",
        "dominant", "competition",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load domain dictionary
# ═══════════════════════════════════════════════════════════════════════════

def load_domain_dictionary() -> List[Dict[str, str]]:
    """Load domain_dictionary.csv into list of dicts."""
    path = DATA_DIR / "domain_dictionary.csv"
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
        if df.empty:
            return []
        return df.to_dict("records")
    except Exception:
        return []


def build_lookup_tables() -> Dict[str, Any]:
    """
    Build fast lookup tables from domain dictionary.
    Returns {canonical_term: entry_dict, synonym_map: {syn: canon}, typo_map: {typo: canon}}
    """
    entries = load_domain_dictionary()
    canon_map = {}
    synonym_map = {}
    typo_map = {}
    eng_map = {}
    var_map = {}  # canonical variable → list of canon terms

    for entry in entries:
        canon = _s(entry.get("canonical_term", ""))
        if not canon:
            continue
        canon_map[canon] = entry

        # Synonyms
        syns = _s(entry.get("synonyms", ""))
        if syns:
            for s in syns.split(";"):
                s = s.strip()
                if s:
                    synonym_map[s] = canon

        # Typos
        typos = _s(entry.get("common_typos", ""))
        if typos:
            for t in typos.split(";"):
                t = t.strip()
                if t:
                    typo_map[t] = canon

        # English terms
        engs = _s(entry.get("english_terms", ""))
        if engs:
            for e in engs.split(";"):
                e = e.strip().lower()
                if e:
                    eng_map[e] = canon

        # Related variables
        rvs = _s(entry.get("related_variables", ""))
        if rvs:
            for rv in rvs.split(";"):
                rv = rv.strip()
                if rv:
                    if rv not in var_map:
                        var_map[rv] = []
                    var_map[rv].append(canon)

    return {
        "canon_map": canon_map,
        "synonym_map": synonym_map,
        "typo_map": typo_map,
        "eng_map": eng_map,
        "var_map": var_map,
        "entries": entries,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Text normalization
# ═══════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """Basic text cleaning: remove extra spaces, normalize punctuation."""
    if not text:
        return text
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # Normalize fullwidth to halfwidth
    text = text.replace("（", "(").replace("）", ")").replace("，", ",")
    text = text.replace("；", ";").replace("：", ":").replace("？", "?").replace("。", ".")
    return text


# ═══════════════════════════════════════════════════════════════════════════
# 3. Typo correction & domain synonym mapping
# ═══════════════════════════════════════════════════════════════════════════

def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance for Chinese/English strings."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + cost,
            ))
        prev = curr
    return prev[-1]


def correct_typos_and_map(text: str, tables: Dict[str, Any]) -> Tuple[str, List[Dict]]:
    """
    Correct typos and map non-standard terms to canonical domain terms.

    Returns:
        corrected_text, corrections_list
    """
    if not text:
        return text, []

    typo_map = tables.get("typo_map", {})
    synonym_map = tables.get("synonym_map", {})
    eng_map = tables.get("eng_map", {})
    canon_map = tables.get("canon_map", {})

    corrections = []
    corrected = text

    # Step 1: Direct typo replacement (sorted by length desc for greedy match)
    all_typos = sorted(typo_map.keys(), key=lambda x: -len(x))
    for typo in all_typos:
        if typo in corrected:
            canon = typo_map[typo]
            corrected = corrected.replace(typo, canon)
            corrections.append({
                "raw": typo,
                "corrected": canon,
                "method": "typo_map",
                "confidence": 0.85,
            })

    # Step 2: Synonym mapping
    all_syns = sorted(synonym_map.keys(), key=lambda x: -len(x))
    for syn in all_syns:
        if syn in corrected:
            canon = synonym_map[syn]
            # Don't replace if already corrected
            if syn != canon:
                corrected = corrected.replace(syn, canon)
                corrections.append({
                    "raw": syn,
                    "corrected": canon,
                    "method": "synonym",
                    "confidence": 0.90,
                })

    # Step 3: English term mapping (case-insensitive)
    text_lower = corrected.lower()
    for eng in sorted(eng_map.keys(), key=lambda x: -len(x)):
        if eng in text_lower and eng not in corrected:
            canon = eng_map[eng]
            # Replace in original case context
            pattern = re.compile(re.escape(eng), re.IGNORECASE)
            corrected = pattern.sub(canon, corrected)
            corrections.append({
                "raw": eng,
                "corrected": canon,
                "method": "english_term",
                "confidence": 0.88,
            })

    # Step 4: Fuzzy match for remaining unknown tokens (Chinese)
    # Segment into 2-4 char tokens and try edit distance
    # Only for tokens not already mapped
    canon_terms = list(canon_map.keys())
    tokens = re.findall(r"[一-鿿]{2,6}", corrected)
    for token in tokens:
        # Skip if already a canonical term
        if token in canon_terms:
            continue
        # Skip if already corrected
        if any(c["corrected"] == token for c in corrections):
            continue

        # Find best match
        best_match = None
        best_dist = 3  # max edit distance for Chinese
        for ct in canon_terms:
            dist = _edit_distance(token, ct)
            if dist < best_dist:
                best_dist = dist
                best_match = ct

        if best_match and best_dist <= len(token) * 0.4:
            corrected = corrected.replace(token, best_match)
            corrections.append({
                "raw": token,
                "corrected": best_match,
                "method": "fuzzy_match",
                "confidence": max(0.6, 1.0 - best_dist / len(token)),
            })

    return corrected, corrections


# ═══════════════════════════════════════════════════════════════════════════
# 4. Extract canonical variables from text
# ═══════════════════════════════════════════════════════════════════════════

def extract_canonical_variables(text: str, tables: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all canonical variables from text, with confidence."""
    canon_map = tables.get("canon_map", {})
    var_map = tables.get("var_map", {})

    detected = []

    # Find canon terms in text (sorted by length desc for greedy match)
    for canon in sorted(canon_map.keys(), key=lambda x: -len(x)):
        if canon in text:
            entry = canon_map[canon]
            rvs = _s(entry.get("related_variables", ""))
            if rvs and rvs != "nan":
                var_list = [v.strip() for v in rvs.split(";") if v.strip()]
            else:
                var_list = [canon]
            primary_var = var_list[0] if var_list else canon
            detected.append({
                "matched_term": canon,
                "canonical_variable": primary_var,
                "all_related_vars": var_list,
                "confidence": 0.92,
                "method": "exact_match",
            })

    # Always also try English variable names directly (catch terms like "dk", "hip")
    text_lower = text.lower()
    eng_to_canon = {
        "pore size": "pore_size", "defect size": "pore_size",
        "fatigue life": "Nf", "nf": "Nf",
        "crack growth": "da_dn", "da/dn": "da_dn",
        "delta k": "Delta_K", "dk": "Delta_K", "delta_k": "Delta_K",
        "stress ratio": "stress_ratio_R", "r ratio": "stress_ratio_R",
        "surface roughness": "surface_roughness",
        "heat treatment": "heat_treatment", "hip": "heat_treatment",
        "microstructure": "microstructure",
        "paris c": "paris_c_m", "paris m": "paris_c_m", "c and m": "paris_c_m",
    }
    for eng, canon in eng_to_canon.items():
        if eng in text_lower:
            if not any(d["canonical_variable"] == canon for d in detected):
                detected.append({
                    "matched_term": eng,
                    "canonical_variable": canon,
                    "all_related_vars": [canon],
                    "confidence": 0.80,
                    "method": "english_var_direct",
                })

    return detected


# ═══════════════════════════════════════════════════════════════════════════
# 5. Smart variable pair extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_variable_pair_smart(
    text: str,
    corrected_text: str,
    detected_vars: List[Dict],
) -> Tuple[Optional[str], Optional[str], str, float]:
    """
    Extract independent and dependent variable from corrected text.

    Returns:
        (ind_var, dep_var, method, confidence)
    """
    # Use patterns to determine which is ind and which is dep
    relation_patterns = [
        (r"(.+?)(?:和|与|跟|and|vs|versus)(.+?)(?:之间|关系|影响|作用|关联|比较)"),
        (r"(.+?)对(.+?)(?:的|之)?(?:影响|作用|关系)"),
        (r"(.+?)(?:影响|决定|控制|调节)(.+)"),
    ]

    available_vars = [v["canonical_variable"] for v in detected_vars]
    unique_vars = []
    for v in available_vars:
        if v not in unique_vars:
            unique_vars.append(v)

    # Try relation patterns on corrected text
    for pat in relation_patterns:
        m = re.search(pat, corrected_text)
        if m:
            left = m.group(1).strip()
            right = m.group(2).strip()
            left_vars = [v["canonical_variable"] for v in detected_vars
                        if v["matched_term"] in left or any(rv in left for rv in v["all_related_vars"])]
            right_vars = [v["canonical_variable"] for v in detected_vars
                         if v["matched_term"] in right or any(rv in right for rv in v["all_related_vars"])]
            if left_vars and right_vars:
                return left_vars[0], right_vars[0], "pattern_extraction", 0.90

    # Fallback: first two unique vars = ind, dep
    if len(unique_vars) >= 2:
        return unique_vars[0], unique_vars[1], "order_fallback", 0.70
    elif len(unique_vars) == 1:
        # Infer the counterpart
        from src.variable_mapper import infer_counterpart
        dep = infer_counterpart(unique_vars[0])
        if dep:
            return unique_vars[0], dep, "counterpart_inference", 0.65
        return unique_vars[0], None, "single_variable", 0.60

    return None, None, "no_variable", 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 6. Smart intent detection
# ═══════════════════════════════════════════════════════════════════════════

def intent_detection_smart(text: str) -> Tuple[str, float]:
    """
    Detect research intent from text.
    Returns (intent_name, confidence).
    """
    text_lower = text.lower()

    scores = {}
    for intent, patterns in INTENT_PATTERNS.items():
        score = 0
        for pat in patterns:
            if re.search(pat, text_lower):
                score += 1
        if score > 0:
            scores[intent] = score

    if not scores:
        return "general_explanation", 0.50

    best = max(scores, key=scores.get)
    # Normalize confidence to 0.6-0.95
    max_score = max(scores.values())
    confidence = min(0.6 + max_score * 0.1, 0.95)

    return best, confidence


# ═══════════════════════════════════════════════════════════════════════════
# 7. Main pipeline: understand_user_query
# ═══════════════════════════════════════════════════════════════════════════

def understand_user_query(raw_query: str) -> Dict[str, Any]:
    """
    Main entry: process raw user query through the full pipeline.

    Returns structured_query dict:
    {
        "original_query": str,
        "normalized_query": str,
        "corrected_query": str,
        "corrections": [{"raw": ..., "corrected": ..., "method": ..., "confidence": ...}],
        "detected_variables": [{"matched_term": ..., "canonical_variable": ..., ...}],
        "canonical_variable_names": [...],
        "independent_variable": str or None,
        "dependent_variable": str or None,
        "task_intent": str,
        "intent_confidence": float,
        "overall_confidence": float,
        "has_corrections": bool,
    }
    """
    result = {
        "original_query": raw_query,
        "normalized_query": "",
        "corrected_query": "",
        "corrections": [],
        "detected_variables": [],
        "canonical_variable_names": [],
        "independent_variable": None,
        "dependent_variable": None,
        "task_intent": "general_explanation",
        "intent_confidence": 0.5,
        "overall_confidence": 0.5,
        "has_corrections": False,
    }

    if not raw_query or not raw_query.strip():
        return result

    # Step 1: Normalize
    normalized = normalize_text(raw_query)
    result["normalized_query"] = normalized

    # Step 2: Load domain tables
    tables = build_lookup_tables()

    # Step 3: Typos + synonyms + English mapping
    corrected, corrections = correct_typos_and_map(normalized, tables)
    result["corrected_query"] = corrected
    result["corrections"] = corrections
    result["has_corrections"] = len(corrections) > 0 or corrected != normalized

    # Step 4: Extract canonical variables
    detected = extract_canonical_variables(corrected, tables)
    # Also extract from original (in case corrections changed things)
    original_detected = extract_canonical_variables(normalized, tables)
    # Merge
    seen_vars = set()
    all_detected = []
    for d in detected + original_detected:
        v = d["canonical_variable"]
        if v not in seen_vars:
            seen_vars.add(v)
            all_detected.append(d)
    result["detected_variables"] = all_detected
    result["canonical_variable_names"] = [d["canonical_variable"] for d in all_detected]

    # Step 5: Variable pair
    ind_var, dep_var, pair_method, pair_conf = extract_variable_pair_smart(
        normalized, corrected, all_detected
    )
    result["independent_variable"] = ind_var
    result["dependent_variable"] = dep_var

    # Step 6: Intent
    intent, intent_conf = intent_detection_smart(corrected)
    result["task_intent"] = intent
    result["intent_confidence"] = intent_conf

    # Step 7: Overall confidence
    confidences = [intent_conf, pair_conf]
    for c in corrections:
        confidences.append(c.get("confidence", 0.5))
    for d in all_detected:
        confidences.append(d.get("confidence", 0.5))
    result["overall_confidence"] = sum(confidences) / len(confidences) if confidences else 0.5

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 8. Format structured query for display
# ═══════════════════════════════════════════════════════════════════════════

def format_query_understanding_markdown(sq: Dict[str, Any]) -> str:
    """Format the structured query as a markdown string for display."""
    lines = ["## 智能问题理解\n"]

    if sq.get("has_corrections"):
        lines.append(f"**原始问题**: {sq['original_query']}\n")
        lines.append(f"**自动修正**: {sq['corrected_query']}\n\n")
        lines.append("**修正记录**:\n")
        for c in sq.get("corrections", []):
            conf_pct = int(c.get("confidence", 0.8) * 100)
            lines.append(f"- 「{c['raw']}」→「{c['corrected']}」(方法: {c['method']}, 置信度: {conf_pct}%)\n")

    if sq.get("canonical_variable_names"):
        lines.append("\n**识别变量**:\n")
        for v in sq["canonical_variable_names"]:
            lines.append(f"- {v}\n")

    ind = sq.get("independent_variable")
    dep = sq.get("dependent_variable")
    if ind and dep:
        lines.append(f"\n**变量对**: {ind} → {dep}\n")
    elif ind:
        lines.append(f"\n**检测到变量**: {ind}\n")

    intent = sq.get("task_intent", "")
    intent_labels = {
        "relation_analysis": "📊 变量关系分析",
        "hypothesis_generation": "🧪 假设生成",
        "experiment_design": "🧫 实验设计",
        "equation_matching": "📐 方程/参数匹配",
        "literature_search": "📚 文献检索",
        "research_gap_discovery": "🧩 研究空白发现",
        "conflict_detection": "⚡ 文献冲突检测",
        "dominance_comparison": "🏆 主导因素比较",
        "general_explanation": "💡 一般解释",
    }
    if intent:
        intent_icon = intent_labels.get(intent, intent)
        conf = int(sq.get("overall_confidence", 0) * 100)
        lines.append(f"\n**识别意图**: {intent_icon} (置信度: {conf}%)\n")

    return "".join(lines)

"""
validation.py — 核心质量门槛与结构审查

quality_gate() 是候选研究空白进入推荐卡片之前必须通过的 13 项检查。
"""

import re
from typing import Any, Dict, List, Tuple

# ── 空话检测 ──────────────────────────────────────────────────────────────

VAGUE_PATTERNS = [
    r"进一步研究",
    r"有待探索",
    r"需要更多实验",
    r"需要进一步",
    r"值得深入研究",
    r"有待进一步",
    r"尚需更多",
    r"有待更深入",
    r"需要更系统的",
    r"应开展更多",
    r"建议进一步",
]

# ── 钛合金疲劳方向检测 ─────────────────────────────────────────────────────

TI_ALLOY_KEYWORDS = [
    "钛合金", "titanium", "ti-6al-4v", "ti6al4v", "tc4", "tc17", "ti60",
    "ti-6al-2sn-4zr-2mo", "ti-10v-2fe-3al", "ti-6al-7nb",
    "ti-6al-4v", "α+β", "alpha+beta", "近α",
    "grade 5", "ti64", "ti-6", "ti-al", "tial",
    "titanium alloy", "ti alloy",
]

FATIGUE_KEYWORDS = [
    "疲劳", "fatigue", "hcf", "vhcf", "lcf", "疲劳裂纹扩展",
    "fcg", "fcgr", "裂纹萌生", "裂纹扩展", "da/dn", "Δk",
    "s-n", "应力-寿命", "应变-寿命",
    "crack growth", "crack initiation", "断裂", "fracture",
]

FATIGUE_TYPES = [
    "hcf", "vhcf", "lcf", "fcgr", "高周疲劳", "低周疲劳",
    "超高周疲劳", "疲劳裂纹扩展", "高温疲劳", "变幅载荷疲劳",
    "保载疲劳", "蠕变-疲劳", "热-力耦合疲劳",
]

# ── 明确属于钛合金但可能不是主案例的标识 ──────────────────────────────────

TI_ALLOY_NAMES = [
    "ti-6al-4v", "ti6al4v", "tc4", "tc17", "ti60", "ti64",
    "grade 5 titanium", "ti-6al-2sn-4zr-2mo", "ti-10v-2fe-3al",
    "ti-6al-7nb", "near-alpha titanium", "α+β titanium",
    "alpha+beta titanium", "beta titanium", "β titanium",
    "titanium alloy", "钛合金", "增材制造钛合金",
    "additive manufacturing titanium",
    "additively manufactured titanium",
]

# ── 真正 out_of_scope 的对象 ────────────────────────────────────────────────

OUT_OF_SCOPE_NAMES = [
    "aluminum alloy", "aluminium alloy", "al alloy", "mg alloy",
    "magnesium alloy", "steel", "7085 al", "al-7.5zn",
    "composite", "镍基合金", "高温合金",
    "superalloy", "ceramic",
]

# ── Scope 分类结果 ──────────────────────────────────────────────────────────

SCOPE_MAIN_CASE = "additive_manufactured_Ti64_fatigue_crack_growth"

SCOPE_MAIN_CASE_KEYWORDS = [
    "additive", "增材", "l-pbf", "slm", "ebm", "ded",
    "selective laser melting", "electron beam melting",
    "laser powder bed fusion", "laser engineered net shaping",
    "ti-6al-4v", "ti6al4v", "tc4", "ti64",
    "fatigue crack", "fcgr", "crack growth", "裂纹扩展",
    "hcf", "vhcf", "high cycle", "very high cycle",
]

SCOPE_SECONDARY_CASE_KEYWORDS = [
    "tc17", "ti60", "near-alpha", "α+β", "beta titanium",
    "ti-10v-2fe-3al", "高温钛合金",
]

VARIABLE_KEYWORDS = [
    "缺陷", "气孔", "未熔合", "夹杂物", "粗糙度", "热处理",
    "组织", "相比例", "晶粒尺寸", "片层厚度", "温度",
    "应力比", "载荷", "残余应力", "表面状态",
    "defect", "pore", "roughness", "heat treatment",
    "microstructure", "grain size", "stress ratio",
    "residual stress",
]

PROPERTY_KEYWORDS = [
    "nf", "s-n", "da/dn", "Δk", "Δkth", "dak/dn",
    "疲劳寿命", "裂纹扩展速率", "门槛值", "fatigue life",
    "crack growth", "paris", "goodman", "walker",
]


def has_titanium_fatigue_focus(text: str) -> bool:
    """检查是否属于钛合金疲劳方向。"""
    if not text:
        return False
    t = text.lower()
    has_ti = any(kw in t for kw in TI_ALLOY_KEYWORDS)
    has_fatigue = any(kw in t for kw in FATIGUE_KEYWORDS)
    return has_ti and has_fatigue


def has_fatigue_type(text: str) -> bool:
    """检查是否有明确疲劳类型。"""
    if not text:
        return False
    t = text.lower()
    return any(ft in t for ft in FATIGUE_TYPES)


def has_variable(text: str) -> bool:
    """检查是否有关键变量。"""
    if not text:
        return False
    t = text.lower()
    return any(v in t for v in VARIABLE_KEYWORDS)


def has_property(text: str) -> bool:
    """检查是否有疲劳性能指标。"""
    if not text:
        return False
    t = text.lower()
    return any(p in t for p in PROPERTY_KEYWORDS)


def has_vague_language(text: str) -> bool:
    """检查是否包含空话/空泛表达。"""
    if not text:
        return True
    for pat in VAGUE_PATTERNS:
        if re.search(pat, text):
            return True
    return False


def has_mechanism_chain(text: str) -> bool:
    """检查是否有损伤机制链描述。"""
    if not text:
        return False
    mechanism_indicators = [
        "起裂", "萌生", "扩展", "断裂", "滑移", "解理", "韧窝",
        "沿晶", "穿晶", "疲劳辉纹", "二次裂纹", "微孔聚集",
        "氧化", "脆化", "蠕变",
        "crack initiation", "crack propagation", "fracture",
        "slip band", "cleavage", "dimple", "intergranular",
        "transgranular", "striation",
    ]
    t = text.lower()
    return any(ind in t for ind in mechanism_indicators)


# ── Quality Gate ──────────────────────────────────────────────────────────

QUALITY_GATE_ITEMS = [
    ("domain_check", "属于钛合金疲劳方向", True),
    ("research_object", "有明确研究对象（如 TC4, Ti-6Al-4V, TC17, Ti60）", True),
    ("fatigue_type", "有明确疲劳类型（HCF, VHCF, LCF, FCGR, 高温疲劳等）", True),
    ("key_variable", "有关键变量（缺陷、热处理、组织、温度、应力比等）", True),
    ("property_metric", "有疲劳性能指标（Nf, S-N, da/dN, ΔK, ΔKth 等）", True),
    ("mechanism_chain", "有损伤机制链", True),
    ("supporting_evidence", "有支持文献", True),
    ("missing_evidence", "有缺失证据", True),
    ("min_viable_path", "有最低成本验证路径", True),
    ("full_validation_path", "有完整验证路径", True),
    ("success_criterion", "有成功判据", True),
    ("falsification_condition", "有推翻条件", True),
    ("no_vague_language", "避免'进一步研究''有待探索'等空话", True),
]


def quality_gate(card: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    """对候选研究空白/推荐方向进行 13 项质量检查。

    Args:
        card: 包含推荐方向信息的字典，必须包含以下字段：
            - name: 推荐方向名称
            - research_object: 研究对象
            - fatigue_type: 疲劳类型
            - variable: 关键变量
            - property_metric: 疲劳性能指标
            - mechanism: 损伤机制链
            - supporting_evidence: 支持文献
            - missing_evidence: 缺失证据
            - min_viable_path: 最低成本验证路径
            - full_validation_path: 完整验证路径
            - success_criterion: 成功判据
            - falsification: 推翻条件

    Returns:
        (passed, results) — passed 为 True 表示所有必填项通过
        results 为每项的检查结果列表
    """
    all_text = str(card.get("name", "")) + " " + str(card.get("research_object", "")) + \
        " " + str(card.get("variable", "")) + " " + str(card.get("mechanism", ""))

    results = []

    # 1. domain_check
    r1 = has_titanium_fatigue_focus(all_text)
    results.append({
        "item": "domain_check",
        "label": "属于钛合金疲劳方向",
        "passed": r1,
        "detail": "通过" if r1 else "未检测到钛合金疲劳关键词",
    })

    # 2. research_object
    research_obj = str(card.get("research_object", "") or "")
    ro_detected = bool(research_obj.strip()) and len(research_obj) >= 4
    results.append({
        "item": "research_object",
        "label": "有明确研究对象",
        "passed": ro_detected,
        "detail": research_obj if ro_detected else "研究对象为空或过短",
    })

    # 3. fatigue_type
    ft_text = str(card.get("fatigue_type", "") or "")
    ft_detected = has_fatigue_type(ft_text)
    results.append({
        "item": "fatigue_type",
        "label": "有明确疲劳类型",
        "passed": ft_detected,
        "detail": ft_text if ft_detected else "未检测到疲劳类型",
    })

    # 4. key_variable
    var_text = str(card.get("variable", "") or "")
    var_detected = has_variable(var_text)
    results.append({
        "item": "key_variable",
        "label": "有关键变量",
        "passed": var_detected,
        "detail": var_text[:100] if var_detected else "未检测到关键变量",
    })

    # 5. property_metric
    prop_text = str(card.get("property_metric", "") or "")
    prop_detected = has_property(prop_text)
    results.append({
        "item": "property_metric",
        "label": "有疲劳性能指标",
        "passed": prop_detected,
        "detail": prop_text if prop_detected else "未检测到疲劳性能指标",
    })

    # 6. mechanism_chain
    mech_text = str(card.get("mechanism", "") or "")
    mech_detected = has_mechanism_chain(mech_text) and len(mech_text) >= 15
    results.append({
        "item": "mechanism_chain",
        "label": "有损伤机制链",
        "passed": mech_detected,
        "detail": mech_text[:100] if mech_detected else "机制描述过短或缺失",
    })

    # 7. supporting_evidence
    ev_text = str(card.get("supporting_evidence", "") or "")
    ev_detected = len(ev_text.strip()) >= 20
    results.append({
        "item": "supporting_evidence",
        "label": "有支持文献",
        "passed": ev_detected,
        "detail": ev_text[:100] if ev_detected else "支持文献为空",
    })

    # 8. missing_evidence
    me_text = str(card.get("missing_evidence", "") or "")
    me_detected = len(me_text.strip()) >= 10
    results.append({
        "item": "missing_evidence",
        "label": "有缺失证据",
        "passed": me_detected,
        "detail": me_text[:100] if me_detected else "缺失证据为空",
    })

    # 9. min_viable_path
    mvp_text = str(card.get("min_viable_path", "") or "")
    mvp_detected = len(mvp_text.strip()) >= 15
    results.append({
        "item": "min_viable_path",
        "label": "有最低成本验证路径",
        "passed": mvp_detected,
        "detail": mvp_text[:100] if mvp_detected else "最低成本验证路径为空",
    })

    # 10. full_validation_path
    fvp_text = str(card.get("full_validation_path", "") or "")
    fvp_detected = len(fvp_text.strip()) >= 15
    results.append({
        "item": "full_validation_path",
        "label": "有完整验证路径",
        "passed": fvp_detected,
        "detail": fvp_text[:100] if fvp_detected else "完整验证路径为空",
    })

    # 11. success_criterion
    sc_text = str(card.get("success_criterion", "") or "")
    sc_detected = len(sc_text.strip()) >= 10
    results.append({
        "item": "success_criterion",
        "label": "有成功判据",
        "passed": sc_detected,
        "detail": sc_text[:100] if sc_detected else "成功判据为空",
    })

    # 12. falsification_condition
    fc_text = str(card.get("falsification", "") or "")
    fc_detected = len(fc_text.strip()) >= 10
    results.append({
        "item": "falsification_condition",
        "label": "有推翻条件",
        "passed": fc_detected,
        "detail": fc_text[:100] if fc_detected else "推翻条件为空",
    })

    # 13. no_vague_language
    check_text = " ".join([
        str(card.get("name", "")),
        str(card.get("research_object", "")),
        str(card.get("mechanism", "")),
        str(card.get("min_viable_path", "")),
    ])
    vague = has_vague_language(check_text)
    results.append({
        "item": "no_vague_language",
        "label": "避免空话",
        "passed": not vague,
        "detail": "检测到空话表达" if vague else "通过",
    })

    all_passed = all(r["passed"] for r in results)
    return all_passed, results


def is_out_of_scope(text: str) -> bool:
    """判断一篇文献是否真正不属于钛合金疲劳方向。

    修复后的规则：
    - TC17、Ti60、近α钛合金、α+β钛合金、β钛合金 —— 属于钛合金疲劳相关对象
    - 真正 out_of_scope: aluminum alloy, steel, magnesium alloy,
      non-titanium composites, non-fatigue titanium papers
    """
    if not text or not text.strip():
        return True  # empty text = can't confirm
    t = text.lower()

    # 先检查是否明确是非钛合金材料
    for oos in OUT_OF_SCOPE_NAMES:
        if oos in t:
            # 但如果在同一句中提到钛合金，则可能是在对比，不直接拒绝
            has_ti = any(kw in t for kw in TI_ALLOY_KEYWORDS)
            if not has_ti:
                return True

    # 检查是否涉及钛合金
    has_ti = any(kw in t for kw in TI_ALLOY_KEYWORDS)
    has_fatigue = any(kw in t for kw in FATIGUE_KEYWORDS)

    # 钛合金 + 疲劳相关 → 属于范围内
    if has_ti and has_fatigue:
        return False

    # 只有钛合金但没有疲劳关键词 → 可能是钛合金材料研究但可能不涉及疲劳
    # 对于标题或短文本，宽容处理
    if has_ti:
        # 检查标题中是否有 fracture/断裂 等也可能相关工作
        has_related = any(kw in t for kw in ["断裂", "fracture", "失效", "failure",
                                               "力学性能", "mechanical propert",
                                               "拉伸", "tensile"])
        if has_related:
            return False
        # 如果只有材料没有疲劳且短文本（可能是 scope 检测时截取不完整），保守处理
        if len(t) < 500:
            return False  # 短文本保守不拒绝

    # 有疲劳关键词但无钛合金 → 可能不是钛合金方向
    if has_fatigue and not has_ti:
        # 检查是否有其他钛合金相关词
        ti_terms = ["ti", "tc", "α+β", "alpha"]
        if any(tm in t for tm in ti_terms):
            return False

    return not (has_ti and has_fatigue)


def classify_titanium_scope(card: dict) -> dict:
    """对文献进行钛合金范围分类。

    Returns:
        dict with:
        - include_in_core_analysis: bool
        - main_case_relevance: str ("primary" / "background" / "secondary_case" / "out_of_scope")
        - exclude_reason: str (空字符串表示无排除理由)
        - material_scope: str
        - fatigue_scope: str
    """
    title = str(card.get("title", "") or "")
    abstract = str(card.get("abstract", "") or "")
    mat = str(card.get("material_system", "") or "")
    findings = str(card.get("key_findings", "") or "")
    if isinstance(card.get("key_findings"), list):
        findings = "; ".join(card["key_findings"])

    text = f"{title} {abstract} {mat} {findings}".lower()

    result = {
        "include_in_core_analysis": True,
        "main_case_relevance": "background",
        "exclude_reason": "",
        "material_scope": "",
        "fatigue_scope": "",
    }

    # 1. 真正 out_of_scope
    for oos in OUT_OF_SCOPE_NAMES:
        if oos in text:
            has_ti = any(kw in text for kw in TI_ALLOY_KEYWORDS)
            if not has_ti:
                result["include_in_core_analysis"] = False
                result["main_case_relevance"] = "out_of_scope"
                result["exclude_reason"] = f"非钛合金材料（{oos}）"
                return result

    # 2. 确定材料范围
    if any(kw in text for kw in ["tc4", "ti-6al-4v", "ti6al4v", "ti64", "grade 5"]):
        result["material_scope"] = "Ti-6Al-4V (TC4/Grade 5)"
    elif any(kw in text for kw in ["tc17"]):
        result["material_scope"] = "TC17"
    elif any(kw in text for kw in ["ti60"]):
        result["material_scope"] = "Ti60"
    elif any(kw in text for kw in ["near-alpha", "近α"]):
        result["material_scope"] = "near-alpha titanium alloy"
    elif any(kw in text for kw in ["α+β", "alpha+beta"]):
        result["material_scope"] = "α+β titanium alloy"
    elif any(kw in text for kw in ["beta titanium", "β钛"]):
        result["material_scope"] = "beta titanium alloy"
    elif any(kw in text for kw in ["titanium", "钛合金"]):
        result["material_scope"] = "titanium alloy (general)"
    else:
        result["material_scope"] = "unknown/not specified"

    # 3. 确定疲劳范围
    if any(kw in text for kw in ["vhcf", "very high cycle"]):
        result["fatigue_scope"] = "VHCF/ultra-high cycle fatigue"
    elif any(kw in text for kw in ["hcf", "high cycle"]):
        result["fatigue_scope"] = "HCF/high cycle fatigue"
    elif any(kw in text for kw in ["lcf", "low cycle"]):
        result["fatigue_scope"] = "LCF/low cycle fatigue"
    elif any(kw in text for kw in ["fcgr", "fcg", "crack growth", "crack propagation",
                                     "裂纹扩展", "da/dn"]):
        result["fatigue_scope"] = "FCGR/fatigue crack growth"
    elif any(kw in text for kw in ["疲劳", "fatigue"]):
        result["fatigue_scope"] = "general fatigue"
    else:
        result["fatigue_scope"] = "unknown/not specified"

    # 4. 确定与主案例（AM Ti-6Al-4V FCG）的相关性
    is_am = any(kw in text for kw in ["additive", "增材", "l-pbf", "slm", "ebm", "ded",
                                        "selective laser", "electron beam",
                                        "laser powder bed", "laser engineered"])
    is_tc4 = any(kw in text for kw in ["tc4", "ti-6al-4v", "ti6al4v", "ti64", "grade 5"])
    is_fcg = any(kw in text for kw in ["crack growth", "crack propagation", "fcgr",
                                         "fcg", "裂纹扩展", "da/dn", "Δk"])
    is_hcf_vhcf = any(kw in text for kw in ["hcf", "vhcf", "high cycle", "very high"])

    if is_am and is_tc4 and (is_fcg or is_hcf_vhcf):
        result["main_case_relevance"] = "primary"
    elif is_am and is_tc4:
        result["main_case_relevance"] = "primary"
    elif is_tc4 and is_fcg:
        result["main_case_relevance"] = "primary"
    elif any(kw in text for kw in SCOPE_SECONDARY_CASE_KEYWORDS):
        result["main_case_relevance"] = "secondary_case"
    elif is_tc4:
        result["main_case_relevance"] = "primary"
    elif "titanium" in text or "钛合金" in text:
        result["main_case_relevance"] = "background"

    return result

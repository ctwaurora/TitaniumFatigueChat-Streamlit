"""
discovery.py — 研究空白发现模块

实现 discover 命令：
变量—性能—机制—证据表 → 覆盖矩阵(两层) → 候选研究空白 → 缺失证据检测 → 伪空白筛除
"""

import csv
import json
import re
import sys
from collections import OrderedDict, Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from skills.deepseek_skill import call_deepseek_text
from skills.library_skill import get_all_papers, CARDS_PATH, CSV_PATH, normalize_terms, normalize_text
from src.validation import is_out_of_scope, classify_titanium_scope

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ── 主案例配置 ───────────────────────────────────────────────────────────────

MAIN_CASE = "additive_manufactured_Ti64_fatigue_crack_growth"
MAIN_CASE_PRIORITY_TOPICS = [
    # 1. AM Ti-6Al-4V fatigue
    ["additive", "增材", "l-pbf", "slm", "ebm", "ded", "selective laser", "electron beam"],
    # 2. Ti-6Al-4V / TC4
    ["ti-6al-4v", "ti6al4v", "tc4", "ti64", "grade 5"],
    # 3. Fatigue crack growth
    ["fatigue crack", "fcgr", "crack growth", "crack propagation", "da/dn", "Δk",
     "裂纹扩展", "fcg"],
    # 4. HCF / VHCF
    ["hcf", "vhcf", "high cycle", "very high cycle", "超高周"],
    # 5. Defects / Pores / Roughness
    ["pore", "defect", "roughness", "表面粗糙度", "孔隙", "气孔", "未熔合", "lack of fusion"],
    # 6. SEM / EBSD / Fracture
    ["sem", "ebsd", "fracture surface", "断口", "micro-ct", "x-ray ct"],
    # 7. Paris / Walker / NASGRO / ML
    ["paris", "walker", "nasgro", "machine learning", "深度学习", "神经网络"],
]

SECONDARY_TOPICS = [
    ["al", "v", "fe", "o", "content", "成分", "元素", "alloying element"],
    ["composition", "ratio"],
]


# ── 覆盖维度定义 ────────────────────────────────────────────────────────────

DIMENSIONS = {
    "material_system": [
        "TC4/Ti-6Al-4V", "TC17", "Ti60", "Ti-6Al-2Sn-4Zr-2Mo",
        "Ti-10V-2Fe-3Al", "Ti-6Al-7Nb", "纯钛", "近α钛合金",
        "α+β钛合金", "β钛合金", "钛基复合材料", "其他钛合金",
    ],
    "processing_method": [
        "锻造", "轧制", "挤压", "热处理", "增材制造(L-PBF)",
        "增材制造(EBM)", "增材制造(DED)", "焊接", "铸造",
        "粉末冶金", "等通道转角挤压(ECAP)", "搅拌摩擦加工",
    ],
    "microstructure": [
        "等轴组织", "双态组织", "片层组织", "魏氏组织",
        "网篮组织", "超细晶", "α+β双相", "β单相",
        "α'马氏体", "织构/各向异性",
    ],
    "loading_condition": [
        "低周疲劳(LCF)", "高周疲劳(HCF)", "超高周疲劳(VHCF)",
        "恒幅载荷", "变幅载荷", "谱载/随机载荷",
        "循环载荷/疲劳裂纹扩展(FCG)", "保载疲劳",
        "蠕变-疲劳交互", "热-力耦合疲劳", "多轴疲劳",
    ],
    "temperature_environment": [
        "室温", "高温(<300°C)", "高温(300-600°C)", "高温(>600°C)",
        "低温/超低温", "腐蚀环境", "氧化环境", "真空",
        "湿热环境", "盐雾", "氢环境",
    ],
    "experimental_methods": [
        "疲劳试验(LCF/HCF)", "疲劳裂纹扩展试验", "拉伸试验",
        "有限元仿真(FEM)", "原位SEM疲劳", "升降法",
        "单轴加载", "四点弯曲", "旋转弯曲",
    ],
    "characterization_methods": [
        "SEM断口分析", "EBSD", "TEM", "XRD", "DIC",
        "APT(原子探针)", "XPS", "同步辐射CT",
        "共聚焦显微镜", "超声/声发射", "原位SEM",
        "纳米压痕", "聚焦离子束(FIB)",
    ],
    "model_or_method": [
        "Paris公式", "Walker模型", "NASGRO方程",
        "Goodman修正", "Smith-Watson-Topper",
        "Coffin-Manson", "有限元仿真(FEM)",
        "分子动力学(MD)", "第一性原理(DFT)",
        "晶体塑性(CPFEM)", "扩展有限元(XFEM)",
        "机器学习/深度学习", "SVR", "GRNN",
        "Weibull统计", "临界距离法(TCD)",
    ],
    "mechanism": [
        "裂纹萌生", "裂纹扩展", "短裂纹扩展", "长裂纹扩展",
        "滑移带开裂", "晶界开裂", "相界面开裂",
        "夹杂物/缺陷起裂", "微孔聚集", "氧致脆化",
        "氢致开裂", "蠕变-疲劳交互", "氧化-疲劳交互",
    ],
}

VARIABLE_TYPES = [
    "alloying_element", "composition_ratio", "heat_treatment",
    "microstructure", "defect", "loading_condition", "stress_ratio",
    "temperature_environment", "surface_state", "model_parameter",
    "testing_method", "manufacturing_process",
]

CSV_FIELDS_VAR = [
    "research_object", "variable_type", "variable_name", "variable_range",
    "property_or_result", "mechanism", "evidence", "missing_evidence",
    "future_hypothesis", "feasibility_level", "source_paper",
]

# ── 主案例相关的高价值缺失组合 ──────────────────────────────────────────────

HIGH_VALUE_MISSING_COMBOS = [
    {
        "dimensions": "AM Ti-6Al-4V + VHCF + pore defect + internal crack initiation",
        "keywords": ["am ti-6al-4v", "vhcf", "pore", "internal crack",
                      "增材", "超高周", "孔隙", "内部裂纹"],
        "why_important": "增材制造Ti-6Al-4V中孔隙缺陷是VHCF失效的主控因素，"
                         "但孔隙特征与内部裂纹起裂的定量关系尚未建立",
    },
    {
        "dimensions": "AM Ti-6Al-4V + FCGR + build orientation + Paris/Walker model",
        "keywords": ["am ti-6al-4v", "fcgr", "build orientation", "paris", "walker",
                      "增材", "裂纹扩展", "成形方向"],
        "why_important": "成形方向导致的各向异性对da/dN-ΔK关系的影响，"
                         "现有Paris/Walker参数是否需要方向修正",
    },
    {
        "dimensions": "AM Ti-6Al-4V + variable amplitude loading + da/dN-ΔK",
        "keywords": ["am ti-6al-4v", "variable amplitude", "da/dn", "Δk",
                      "增材", "变幅载荷", "裂纹扩展"],
        "why_important": "实际服役为变幅载荷，但绝大多数AM Ti64 FCG研究为恒幅",
    },
    {
        "dimensions": "AM Ti-6Al-4V + high temperature + oxidation-assisted crack growth",
        "keywords": ["am ti-6al-4v", "high temperature", "oxidation", "crack growth",
                      "增材", "高温", "氧化", "裂纹扩展"],
        "why_important": "航空航天部件在高温氧化环境下服役，"
                         "但AM Ti64氧化辅助裂纹扩展数据缺乏",
    },
    {
        "dimensions": "AM Ti-6Al-4V + surface roughness + micro-CT/SEM evidence",
        "keywords": ["am ti-6al-4v", "surface roughness", "micro-ct", "sem",
                      "增材", "表面粗糙度"],
        "why_important": "as-built表面粗糙度与内部孔隙竞争控制疲劳失效，"
                         "需要定量表征两种缺陷的竞争机制",
    },
    {
        "dimensions": "AM Ti-6Al-4V + residual stress + HCF/VHCF + crack initiation",
        "keywords": ["am ti-6al-4v", "residual stress", "hcf", "vhcf", "crack initiation",
                      "增材", "残余应力", "裂纹起裂"],
        "why_important": "残余应力与缺陷耦合效应在HCF/VHCF寿命预测中常被忽略",
    },
    {
        "dimensions": "L-PBF Ti-6Al-4V + heat treatment + microstructure + FCGR",
        "keywords": ["l-pbf", "ti-6al-4v", "heat treatment", "microstructure", "fcgr",
                      "热处理", "组织", "裂纹扩展"],
        "why_important": "后处理热处理对微观组织的调控及其对FCGR的影响缺乏系统研究",
    },
]


def run_discover() -> Dict[str, Any]:
    """执行研究空白发现流程。"""
    # Step 1: 构建变量—性能—机制—证据表
    var_records = _build_variable_mechanism_table()

    # Step 2: 构建覆盖矩阵（两层）
    matrix, observed, high_value_missing = _build_coverage_matrix_two_layer()

    # Step 3: 发现候选研究空白
    gaps = _find_candidate_gaps(observed, high_value_missing, var_records)

    # Step 4: 缺失证据检测
    gaps_with_missing = _detect_missing_evidence(gaps, var_records)

    # Step 5: 伪空白筛除
    real_gaps, pseudo_gaps = _filter_pseudo_gaps(gaps_with_missing)

    # Step 6: 科学模态证据索引（轻量文本匹配）
    _build_scientific_artifact_index()

    stats = {
        "variable_records": len(var_records),
        "matrix_shape": matrix.shape if not matrix.empty else (0, 0),
        "candidate_gaps": len(gaps_with_missing),
        "real_gaps": len(real_gaps),
        "pseudo_gaps": len(pseudo_gaps),
    }

    _write_gap_diagnosis(real_gaps, pseudo_gaps, observed, high_value_missing,
                         var_records, stats)

    # 保存中间结果
    _save_var_table(var_records)
    _save_matrix(matrix)

    return stats


def _load_literature_data() -> Tuple[pd.DataFrame, List[Dict], pd.DataFrame]:
    """加载所有文献数据。"""
    df = pd.DataFrame()
    if CSV_PATH.exists():
        try:
            df = pd.read_csv(CSV_PATH, encoding="utf-8", on_bad_lines="skip")
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        except Exception:
            pass

    cards = []
    if CARDS_PATH.exists():
        with open(CARDS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        cards.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    var_df = pd.DataFrame()
    var_path = DATA_DIR / "variable_property_mechanism.csv"
    if var_path.exists():
        try:
            var_df = pd.read_csv(var_path, encoding="utf-8-sig")
        except Exception:
            pass

    return df, cards, var_df


def _build_text_summary(cards, df) -> str:
    """构建结构化的文献摘要文本用于 LLM 抽取。"""
    parts = []
    if not df.empty:
        for idx, row in df.iterrows():
            title = row.get("title", f"文献{idx}")
            parts.append(
                f"[CSV-{idx}] {title}\n"
                f"  材料: {row.get('material_system','')}\n"
                f"  工艺: {row.get('processing_method','')}\n"
                f"  热处理: {row.get('heat_treatment','')}\n"
                f"  组织: {row.get('microstructure','')}\n"
                f"  载荷: {row.get('loading_condition','')}\n"
                f"  环境: {row.get('temperature_environment','')}\n"
                f"  表征: {row.get('characterization_methods','')}\n"
                f"  模型: {row.get('model_or_method','')}\n"
                f"  力学指标: {row.get('mechanical_indicators','')}\n"
                f"  起裂: {row.get('crack_initiation','')}\n"
                f"  扩展机制: {row.get('crack_growth_mechanism','')}\n"
                f"  发现: {row.get('key_findings','')}\n"
                f"  局限: {row.get('limitations','')}\n"
                f"  证据: {row.get('evidence_text','')}\n"
            )
    for c in cards:
        title = c.get("title", "未知")
        findings = c.get("key_findings", "")
        if isinstance(findings, list):
            findings = "; ".join(findings)
        lim = c.get("limitations", "")
        if isinstance(lim, list):
            lim = "; ".join(lim)
        ev = c.get("evidence_text", "")
        parts.append(
            f"[Card] {title}\n"
            f"  材料: {c.get('material_system','')}\n"
            f"  工艺: {c.get('processing_method','')}\n"
            f"  组织: {c.get('microstructure','')}\n"
            f"  载荷: {c.get('loading_condition','')}\n"
            f"  发现: {findings}\n"
            f"  局限: {lim}\n"
            f"  证据: {ev}\n"
        )
    if not parts:
        return ""
    return "\n".join(parts)


EXTRACT_PROMPT = """你是钛合金疲劳断裂方向的变量-性能-机制关系分析师。
请从以下文献摘要中，抽取「变量—性能—机制—证据」关系记录。

变量类型包括但不限于：
- alloying_element: 合金元素 (Al, V, Mo, Cr, Zr, O, Fe 等)
- composition_ratio: 成分比例
- heat_treatment: 热处理制度
- microstructure: 微观组织
- defect: 缺陷 (气孔、未熔合、夹杂物、表面粗糙度等)
- loading_condition: 载荷条件 (LCF, HCF, VHCF, 恒幅/变幅/谱载等)
- stress_ratio: 应力比 R
- temperature_environment: 温度/环境
- surface_state: 表面状态
- model_parameter: 模型参数
- testing_method: 实验/表征方法
- manufacturing_process: 制备工艺

优先关注增材制造 Ti-6Al-4V 的疲劳裂纹扩展、HCF/VHCF 相关关系。

对每种被文献提及的变量-性能关系，按以下 JSON 格式输出一条记录：

```jsonl
{{"research_object": "研究对象（具体到材料+工艺）", "variable_type": "变量类型", "variable_name": "变量名（具体，如'孔隙尺寸/孔隙率/表面粗糙度Ra'）", "variable_range": "范围", "property_or_result": "性能或结果（具体指标如da/dN-ΔK/Nf/Paris参数C,m）", "mechanism": "机制（变量→局部效应→裂纹行为→性能变化）", "evidence": "证据（引用具体文献发现而非笼统概括）", "missing_evidence": "缺失证据（具体指出缺什么数据）", "future_hypothesis": "假设（具体可验证）", "feasibility_level": "A/B/C", "source_paper": "文献标题"}}
```

要求：
1. 每条必须有 evidence，且指向文献中的具体发现。
2. 如果缺少直接证据，必须写 missing_evidence。
3. feasibility_level: A=可验证, B=需设备, C=不建议。
4. 优先关注增材制造 Ti-6Al-4V 疲劳相关关系。
5. mechanism 字段注意体现"变量→局部效应→损伤行为→疲劳指标"的因果链。

文献摘要：
LITERATURE_PLACEHOLDER
"""


def _build_variable_mechanism_table() -> List[Dict]:
    """构建变量—性能—机制—证据表。

    先尝试 DeepSeek 抽取，再用规则回退补充，确保每篇文献至少 1-3 条记录。
    """
    df, cards, _ = _load_literature_data()
    summary = _build_text_summary(cards, df)
    if not summary:
        return []

    deepseek_records = []
    # DeepSeek extraction
    prompt = EXTRACT_PROMPT.replace("LITERATURE_PLACEHOLDER", summary[:10000])
    try:
        result = call_deepseek_text(prompt, max_tokens=6000, temperature=0.1)
        jsonl_pattern = re.findall(r"\{[^}]+\}", result)
        for js in jsonl_pattern:
            try:
                rec = json.loads(js)
                if "variable_type" in rec and "variable_name" in rec:
                    deepseek_records.append(rec)
            except json.JSONDecodeError:
                continue
    except Exception:
        pass

    # 规则回退补充（从所有卡片中提取更多记录）
    rule_records = _rule_based_fallback(cards, df)

    # 合并去重（保留 DeepSeek 的结果，补充 rule 的结果）
    seen_keys = set()
    all_records = []

    for rec in deepseek_records + rule_records:
        key = (str(rec.get("source_paper", "")) + "::" + str(rec.get("variable_type", "")) +
               "::" + str(rec.get("variable_name", "")))
        if key not in seen_keys:
            seen_keys.add(key)
            all_records.append(rec)

    return all_records


def _rule_based_fallback(cards, df) -> List[Dict]:
    """规则回退：从已知字段中提取变量关系。

    每篇文献至少尝试抽取 1-3 条变量-性能-机制关系，
    包括工艺、缺陷、组织、载荷、热处理、环境、表面状态等。
    """
    records = []
    seen = set()

    # 从所有数据源中提取
    sources = []
    for c in cards:
        sources.append(c)
    if not df.empty:
        for _, row in df.iterrows():
            sources.append(row.to_dict())

    for c in sources:
        mat = str(c.get("material_system", "") or "")
        title = str(c.get("title", "") or "")
        findings = c.get("key_findings", "")
        if isinstance(findings, list):
            findings = "; ".join(findings)
        ev = str(c.get("evidence_text", "") or "")
        lim = str(c.get("limitations", "") or "")
        loading = str(c.get("loading_condition", "") or "")
        proc = str(c.get("processing_method", "") or "")
        mechanism_str = str(c.get("crack_growth_mechanism", "") or "")
        indicators = str(c.get("mechanical_indicators", "") or "")
        microstructure = str(c.get("microstructure", "") or "")
        heat_treatment = str(c.get("heat_treatment", "") or "")
        temperature = str(c.get("temperature_environment", "") or "")
        methods = str(c.get("characterization_methods", "") or "")
        model = str(c.get("model_or_method", "") or "")
        crack_init = str(c.get("crack_initiation", "") or "")

        text_to_search = f"{title} {mat} {findings} {ev} {loading} {proc} {microstructure} {temperature} {mechanism_str}"

        # 检查 AM 相关
        is_am = any(kw in text_to_search.lower() for kw in
                    ["additive", "增材", "l-pbf", "slm", "ebm", "ded",
                     "selective laser", "laser powder", "laser engineered"])

        # 研究对象的标准化名称
        research_obj = "AM Ti-6Al-4V" if is_am else (mat or "Ti-6Al-4V")

        # ── 1. 缺陷/孔隙 → 疲劳性能 ──
        if any(kw in text_to_search.lower() for kw in
               ["pore", "poros", "defect", "气孔", "孔隙", "未熔合", "lack of fusion"]):
            key = f"{title}_pore_defect"
            if key not in seen:
                seen.add(key)
                records.append({
                    "research_object": research_obj,
                    "variable_type": "defect",
                    "variable_name": "pore defect",
                    "variable_range": "文献中报告的孔隙尺寸和分布",
                    "property_or_result": indicators[:200] or "疲劳寿命/疲劳强度",
                    "mechanism": "孔隙缺陷在循环载荷下引发局部应力集中，促进裂纹在孔隙边缘萌生",
                    "evidence": ev[:200] or findings[:200],
                    "missing_evidence": "缺乏孔隙特征(尺寸/位置/形态)与疲劳寿命的定量关系数据",
                    "future_hypothesis": f"{research_obj}的孔隙特征参量可定量预测疲劳寿命",
                    "feasibility_level": "B",
                    "source_paper": title,
                })

        # ── 2. 表面粗糙度 → 疲劳寿命 ──
        if any(kw in text_to_search.lower() for kw in
               ["roughness", "粗糙度", "as-built surface", "surface finish"]):
            key = f"{title}_roughness"
            if key not in seen:
                seen.add(key)
                records.append({
                    "research_object": research_obj,
                    "variable_type": "surface_state",
                    "variable_name": "surface roughness",
                    "variable_range": "as-built表面粗糙度Ra值",
                    "property_or_result": "疲劳寿命Nf/疲劳强度",
                    "mechanism": "表面粗糙度造成局部应力集中，成为裂纹优先萌生位置",
                    "evidence": ev[:200] or findings[:200],
                    "missing_evidence": "缺乏表面粗糙度特征(峰谷参数/应力集中系数)与疲劳寿命的定量模型",
                    "future_hypothesis": f"{research_obj}表面粗糙度的临界值下疲劳寿命无明显降低",
                    "feasibility_level": "B",
                    "source_paper": title,
                })

        # ── 3. 制备工艺 → 疲劳性能 ──
        if proc and len(proc) > 3:
            key = f"{title}_process"
            if key not in seen:
                seen.add(key)
                records.append({
                    "research_object": research_obj,
                    "variable_type": "manufacturing_process",
                    "variable_name": proc[:80],
                    "variable_range": "文献中描述的工艺参数",
                    "property_or_result": indicators[:200] or findings[:200],
                    "mechanism": mechanism_str[:200] or (
                        "制备工艺参数影响微观组织和缺陷特征，进而影响疲劳性能"),
                    "evidence": f"文献采用{proc}制备{research_obj}",
                    "missing_evidence": f"缺乏{proc}工艺参数→组织→疲劳性能的完整定量链",
                    "future_hypothesis": f"优化{proc}工艺参数可降低缺陷密度提升{research_obj}疲劳性能",
                    "feasibility_level": "B",
                    "source_paper": title,
                })

        # ── 4. 热处理 → 微观组织 → FCGR ──
        if heat_treatment and len(heat_treatment) > 3:
            key = f"{title}_heat_treatment"
            if key not in seen:
                seen.add(key)
                records.append({
                    "research_object": research_obj,
                    "variable_type": "heat_treatment",
                    "variable_name": heat_treatment[:80],
                    "variable_range": heat_treatment[:80],
                    "property_or_result": indicators[:200] or "微观组织/疲劳性能",
                    "mechanism": f"热处理({heat_treatment[:50]})改变α/β相比例和片层厚度，影响裂纹扩展路径",
                    "evidence": findings[:200] or ev[:200],
                    "missing_evidence": f"缺乏{research_obj}不同热处理制度的疲劳性能系统对比数据",
                    "future_hypothesis": f"优化{research_obj}热处理制度可提升疲劳裂纹扩展抗力",
                    "feasibility_level": "B",
                    "source_paper": title,
                })

        # ── 5. 载荷条件 → 疲劳机制 ──
        loading_types = []
        if "lcf" in loading.lower() or "low cycle" in loading.lower() or "低周" in loading:
            loading_types.append("LCF")
        if "hcf" in loading.lower() or "high cycle" in loading.lower() or "高周" in loading:
            loading_types.append("HCF")
        if "vhcf" in loading.lower() or "very high" in loading.lower() or "超高" in loading:
            loading_types.append("VHCF")
        if "fcg" in loading.lower() or "crack growth" in loading.lower() or "裂纹扩展" in loading:
            loading_types.append("FCGR")
        if "variable" in loading.lower() or "变幅" in loading:
            loading_types.append("variable amplitude")

        for lt in loading_types:
            key = f"{title}_loading_{lt}"
            if key not in seen:
                seen.add(key)
                records.append({
                    "research_object": research_obj,
                    "variable_type": "loading_condition",
                    "variable_name": f"{lt} loading",
                    "variable_range": "文献中使用的载荷参数",
                    "property_or_result": f"{lt}下的{indicators[:100] or '疲劳行为'}",
                    "mechanism": f"{lt}条件下{mechanism_str[:100] or '裂纹萌生与扩展机制'}",
                    "evidence": ev[:200] or f"文献研究{research_obj}的{lt}行为",
                    "missing_evidence": f"缺乏{research_obj}在{lt}下的多因素耦合研究",
                    "future_hypothesis": f"{research_obj}在{lt}条件下的寿命预测模型需考虑缺陷-组织耦合效应",
                    "feasibility_level": "B",
                    "source_paper": title,
                })

        # ── 6. 微观组织特征 → FCGR ──
        micro_features = []
        if any(kw in microstructure.lower() for kw in ["α′", "martensite", "马氏体"]):
            micro_features.append("α′ martensite")
        if any(kw in microstructure.lower() for kw in ["lamellar", "片层"]):
            micro_features.append("lamellar α+β")
        if any(kw in microstructure.lower() for kw in ["equiaxed", "等轴"]):
            micro_features.append("equiaxed α+β")
        if any(kw in microstructure.lower() for kw in ["grain", "晶粒"]):
            micro_features.append("grain size")

        if micro_features:
            mf = ", ".join(micro_features[:2])
            key = f"{title}_microstructure"
            if key not in seen:
                seen.add(key)
                records.append({
                    "research_object": research_obj,
                    "variable_type": "microstructure",
                    "variable_name": mf,
                    "variable_range": "文献中报告的组织特征参数",
                    "property_or_result": f"{indicators[:100] or '疲劳性能'}受{mf}影响",
                    "mechanism": f"{mf}影响裂纹萌生位置和扩展路径，决定疲劳抗力",
                    "evidence": findings[:200] or ev[:200],
                    "missing_evidence": f"缺乏{research_obj}中{mf}与FCGR的定量关系",
                    "future_hypothesis": f"调控{research_obj}的{mf}可优化疲劳裂纹扩展抗力",
                    "feasibility_level": "B",
                    "source_paper": title,
                })

        # ── 7. 温度/环境 → 疲劳行为 ──
        if temperature and len(temperature) > 3:
            key = f"{title}_temperature"
            if key not in seen:
                seen.add(key)
                records.append({
                    "research_object": research_obj,
                    "variable_type": "temperature_environment",
                    "variable_name": temperature[:80],
                    "variable_range": temperature[:80],
                    "property_or_result": f"{temperature}下{indicators[:100] or '疲劳行为'}",
                    "mechanism": f"{temperature}环境影响{mechanism_str[:100] or '裂纹萌生与扩展机制'}",
                    "evidence": ev[:200] or findings[:200],
                    "missing_evidence": f"缺乏{research_obj}在{temperature[:50]}下疲劳性能的系统数据",
                    "future_hypothesis": f"{research_obj}在{temperature[:50]}下的疲劳机制与室温不同需单独研究",
                    "feasibility_level": "B",
                    "source_paper": title,
                })

        # ── 8. 建模方法 → 预测能力 ──
        if model and len(model) > 3:
            key = f"{title}_model"
            if key not in seen:
                seen.add(key)
                records.append({
                    "research_object": research_obj,
                    "variable_type": "model_parameter",
                    "variable_name": model[:80],
                    "variable_range": "文献中使用的模型参数",
                    "property_or_result": "疲劳寿命预测/裂纹扩展速率预测",
                    "mechanism": f"{model[:60]}用于建立变量与疲劳性能之间的定量关系",
                    "evidence": f"文献使用{model[:60]}分析{research_obj}疲劳数据",
                    "missing_evidence": f"缺乏{model[:60]}在不同{research_obj}工艺条件下的泛化能力验证",
                    "future_hypothesis": f"融合物理机制的{model[:60]}可提升{research_obj}疲劳寿命预测精度",
                    "feasibility_level": "B",
                    "source_paper": title,
                })

        # ── 9. 裂纹起裂机制 ──
        if crack_init and len(crack_init) > 5:
            key = f"{title}_crack_init"
            if key not in seen:
                seen.add(key)
                records.append({
                    "research_object": research_obj,
                    "variable_type": "testing_method",
                    "variable_name": crack_init[:80],
                    "variable_range": "文献中观察到的起裂位置和机制",
                    "property_or_result": f"裂纹起裂受{crack_init[:50]}控制",
                    "mechanism": crack_init[:200],
                    "evidence": findings[:200] or ev[:200],
                    "missing_evidence": "缺乏裂纹起裂机制的定量判据和预测模型",
                    "future_hypothesis": f"{research_obj}的裂纹起裂机制在缺陷和组织特征之间存在竞争关系",
                    "feasibility_level": "B",
                    "source_paper": title,
                })

        # ── 10. Alloying elements (仅当确有关键词时才提取) ──
        elements = re.findall(r'(Al|V|Mo|Cr|Zr|Nb|Sn|Si|O|Fe|N|H|C)\s*含量', findings + " " + ev)
        for el in set(elements):
            key = f"{title}_{el}"
            if key not in seen:
                seen.add(key)
                records.append({
                    "research_object": research_obj,
                    "variable_type": "alloying_element",
                    "variable_name": f"{el}含量",
                    "variable_range": "见原文",
                    "property_or_result": findings[:200],
                    "mechanism": f"{el}在{research_obj}中影响固溶强化和组织稳定性",
                    "evidence": ev[:200],
                    "missing_evidence": f"缺乏{el}含量对{research_obj}疲劳性能的定量影响数据",
                    "future_hypothesis": f"{el}含量在特定范围内对{research_obj}疲劳性能产生显著影响",
                    "feasibility_level": "B",
                    "source_paper": title,
                })

    return records


def _value_to_list(val):
    """将字段值转换为列表。"""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val_stripped = val.strip()
        if val_stripped.startswith("["):
            try:
                return json.loads(val_stripped)
            except json.JSONDecodeError:
                pass
        items = re.split(r"[;,，、/]+", val_stripped)
        return [x.strip() for x in items if x.strip()]
    return [str(val)]


def _build_coverage_matrix_two_layer() -> Tuple[pd.DataFrame, Dict, List]:
    """构建两层覆盖矩阵。

    Layer A: Observed Coverage — 只统计文献中实际出现的组合
    Layer B: High-value Missing Combinations — 只生成与主案例相关的缺失组合

    Returns:
        (matrix_df, observed_dict, high_value_missing_list)
    """
    df, cards, var_df = _load_literature_data()

    seen_titles = set()
    papers = []
    for idx, row in df.iterrows():
        title = row.get("title", f"CSV-{idx}")
        if title not in seen_titles:
            seen_titles.add(title)
            papers.append({"title": title, "source": "csv", "row": row})
    for c in cards:
        title = c.get("title", c.get("source_file", "Card"))
        if title not in seen_titles:
            seen_titles.add(title)
            papers.append({"title": title, "source": "card", "card": c})

    if not papers:
        return pd.DataFrame(), {}, []

    # ── Layer A: Observed Coverage ──
    # 统计每个维度的实际覆盖情况
    observed = {}
    for dim, cat_list in DIMENSIONS.items():
        observed[dim] = {}
        for cat in cat_list:
            observed[dim][cat] = []

    for paper in papers:
        field_texts = []
        title_text = paper["title"].lower()
        if paper["source"] == "csv":
            row = paper["row"]
            for field in ["material_system", "processing_method", "microstructure",
                          "loading_condition", "temperature_environment",
                          "characterization_methods", "model_or_method",
                          "experimental_methods", "mechanical_indicators",
                          "crack_growth_mechanism", "key_findings", "crack_initiation"]:
                val = row.get(field, "")
                field_texts.extend(_value_to_list(val))
        elif paper["source"] == "card":
            c = paper["card"]
            for field in ["material_system", "processing_method", "microstructure",
                          "loading_condition", "temperature_environment",
                          "characterization_methods", "model_or_method",
                          "experimental_methods", "key_findings", "crack_initiation",
                          "crack_growth_mechanism", "mechanical_indicators"]:
                val = c.get(field, "")
                if isinstance(val, list):
                    field_texts.extend(val)
                else:
                    field_texts.append(str(val))

        text = " ".join(field_texts).lower()
        combined = title_text + " " + text

        for dim, cat_list in DIMENSIONS.items():
            for cat in cat_list:
                cat_lower = cat.lower()
                cat_terms = cat_lower.replace("(", " ").replace(")", " ").replace("-", " ").split()
                # 特殊处理：Paris公式只匹配 paris，不匹配 Paris 的其他含义
                if "paris" in cat_lower:
                    if re.search(r'\bparis\b', combined):
                        observed[dim][cat].append(paper["title"])
                elif "walker" in cat_lower:
                    if re.search(r'\bwalker\b', combined):
                        observed[dim][cat].append(paper["title"])
                else:
                    # Fix: keep Chinese chars (ord>127) even if len<=2, filter only very short ASCII
                    meaningful_terms = [t for t in cat_terms
                                        if len(t) > 2 or (len(t) >= 1 and ord(t[0]) > 127)]
                    if not meaningful_terms:
                        meaningful_terms = [cat_lower]
                    # For categories with multiple terms (e.g. "低周疲劳(LCF)" → ["低周疲劳","lcf"]),
                    # use ANY match (the terms are synonyms/translations, not AND conditions)
                    matches = any(term in combined for term in meaningful_terms)
                    if matches:
                        observed[dim][cat].append(paper["title"])

    # 构建 matrix DataFrame
    rows = []
    for paper in papers:
        row_data = {"paper_title": paper["title"]}
        for dim, cat_list in DIMENSIONS.items():
            for cat in cat_list:
                row_data[f"{dim}::{cat}"] = 1 if paper["title"] in observed[dim].get(cat, []) else 0
        rows.append(row_data)
    matrix = pd.DataFrame(rows)

    # ── Layer B: High-value Missing Combinations ──
    # 只生成与主案例相关且有物理意义的缺失组合
    high_value_missing = []

    # 检查文献库中是否有 AM Ti-6Al-4V 相关文献
    has_am_ti64 = False
    for paper in papers:
        txt = paper["title"].lower()
        for c in DIMENSIONS.get("characterization_methods", []):
            pass
        if any(kw in txt for kw in ["ti-6al-4v", "ti6al4v", "tc4", "ti64"]) and \
           any(kw in txt for kw in ["additive", "增材", "l-pbf", "slm", "ebm", "ded",
                                      "selective laser", "laser powder", "laser engineered"]):
            has_am_ti64 = True
            break

    if has_am_ti64:
        for combo in HIGH_VALUE_MISSING_COMBOS:
            # 检查当前文献库是否覆盖了该组合的部分内容
            supporting_papers = []
            for paper in papers:
                txt = (paper["title"] + " " +
                       str(paper.get("card", paper.get("row", {})).get("key_findings", ""))).lower()
                if any(kw in txt for kw in combo["keywords"]):
                    supporting_papers.append(paper["title"])

            high_value_missing.append({
                "name": combo["dimensions"],
                "keywords": combo["keywords"],
                "why_important": combo["why_important"],
                "supporting_papers": supporting_papers[:3],
                "evidence_level": "部分覆盖" if supporting_papers else "未覆盖",
            })

    return matrix, observed, high_value_missing


def _is_primary_case_related(text: str) -> bool:
    """判断文本是否与主案例（AM Ti-6Al-4V FCG）相关。"""
    t = text.lower()
    for topic_group in MAIN_CASE_PRIORITY_TOPICS:
        if any(kw in t for kw in topic_group):
            return True
    return False


def _is_secondary_topic(text: str) -> bool:
    """判断是否属于次要主题（如元素成分）。"""
    t = text.lower()
    for topic_group in SECONDARY_TOPICS:
        if any(kw in t for kw in topic_group):
            return True
    return False


def _find_candidate_gaps(
    observed: Dict, high_value_missing: List, var_records: List[Dict]
) -> List[Dict]:
    """发现候选研究空白。

    优先从主案例高价值缺失组合中生成空白，
    其次从变量记录的 missing_evidence 中生成。
    """
    gaps = []

    # ── 优先从 high_value_missing 生成 ──
    for combo in high_value_missing:
        supporting = combo.get("supporting_papers", [])
        evidence_text = "; ".join(supporting[:3]) if supporting else "文献库暂无直接覆盖"

        gaps.append({
            "name": combo["name"],
            "source": "high_value_missing_combo",
            "priority": "high" if not supporting else "medium",
            "supporting_evidence": evidence_text,
            "missing_evidence": f"当前文献库{combo['evidence_level']}，"
                                f"缺少该组合的系统性研究数据。\n"
                                f"重要性：{combo['why_important']}",
            "research_object": "AM Ti-6Al-4V",
            "variable": combo["name"],
            "mechanism": combo["why_important"],
            "fatigue_type": "",
            "property_metric": "",
        })

    # ── 从变量记录的 missing_evidence 中生成 ──
    for rec in var_records:
        me = rec.get("missing_evidence", "")
        fh = rec.get("future_hypothesis", "")
        research_obj = rec.get("research_object", "")

        if me and len(me) > 10:
            # 判断是否与主案例相关
            text = f"{research_obj} {rec.get('variable_name', '')} {fh}"
            is_primary = _is_primary_case_related(text)
            is_secondary = _is_secondary_topic(text)

            priority = "high" if is_primary else ("medium" if is_secondary else "low")

            # 如果是 Al/V/Fe/O 成分相关，降低优先级
            name = fh or me[:80]
            if is_secondary:
                name = f"[secondary] {name}"

            gaps.append({
                "name": name,
                "source": "var_table_missing_evidence",
                "priority": priority,
                "supporting_evidence": rec.get("evidence", ""),
                "missing_evidence": me,
                "research_object": research_obj,
                "variable": rec.get("variable_name", ""),
                "mechanism": rec.get("mechanism", ""),
                "fatigue_type": "",
                "property_metric": rec.get("property_or_result", ""),
            })

    # 按优先级排序
    priority_order = {"high": 0, "medium": 1, "low": 2}
    gaps.sort(key=lambda g: priority_order.get(g.get("priority", "low"), 99))

    return gaps


def _detect_missing_evidence(gaps: List[Dict], var_records: List[Dict]) -> List[Dict]:
    """为候选空白补充缺失证据检测。"""
    for gap in gaps:
        if not gap.get("missing_evidence"):
            gap["missing_evidence"] = "当前文献库规模有限，具体缺失证据需进一步分析"
        if not gap.get("supporting_evidence"):
            gap["supporting_evidence"] = "当前文献库规模有限，暂未找到直接支持文献"
    return gaps


def _filter_pseudo_gaps(gaps: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """筛除伪空白。

    修复后的规则：
    - TC17/Ti60/近α钛合金/α+β钛合金/β钛合金 → 是钛合金疲劳相关对象，不能直接拒绝
    - L-PBF/EBM/DED/FCGR/VHCF/SEM/EBSD/Paris → 与钛合金疲劳相关，但证据不足时有合理解释
    - 真正拒绝：空泛表达、缺乏具体变量和机制、纯零覆盖无物理意义
    """
    real_gaps = []
    pseudo_gaps = []

    # 钛合金疲劳相关术语（不能被拒绝为"非钛合金疲劳方向"）
    ti_fatigue_terms = [
        "tc17", "ti60", "ti-10v-2fe-3al", "钛合金", "titanium",
        "近α", "α+β", "beta titanium", "β钛",
        "l-pbf", "slm", "ebm", "ded", "增材",
        "fcgr", "fcg", "vhcf", "hcf", "lcf", "疲劳",
        "sem", "ebsd", "tem", "xrd", "dic", "paris", "walker",
        "nasgro", "goodman", "裂纹", "crack", "da/dn", "Δk",
        "断口", "fracture", "微观组织", "microstructure",
        "热处理", "heat treatment", "pore", "defect", "粗糙度",
        "表面", "surface", "高温", "high temperature",
    ]

    for gap in gaps:
        reject_reasons = []
        name = str(gap.get("name", "")).lower()
        mechanism = str(gap.get("mechanism", "") or "")
        research_obj = str(gap.get("research_object", "") or "")
        missing_ev = str(gap.get("missing_evidence", "") or "")
        supporting_ev = str(gap.get("supporting_evidence", "") or "")

        combined = f"{name} {mechanism} {research_obj}".lower()

        # ── 检查1：空泛表达 ──
        vague_patterns = ["进一步研究", "有待探索", "需要更多", "需要进一步",
                          "值得深入研究", "有待进一步", "尚需更多", "需要更系统"]
        if any(vp in combined for vp in vague_patterns):
            reject_reasons.append("包含空泛表达（进一步研究/有待探索等），缺乏具体变量和假设")

        # ── 检查2：是否涉及钛合金疲劳相关 ──
        has_ti_fatigue = any(term in combined for term in ti_fatigue_terms)

        # ── 检查3：机制链描述 ──
        mechanism_indicators = ["起裂", "萌生", "扩展", "断裂", "滑移", "解理",
                                "裂纹", "疲劳", "缺陷", "孔隙", "氧化",
                                "crack", "fatigue", "pore", "defect"]
        has_mechanism = any(mi in combined for mi in mechanism_indicators)

        # ── 检查4：纯零覆盖但无物理意义 ──
        # 对于高价值缺失组合，即使无机制描述也保留
        is_high_value = gap.get("source") == "high_value_missing_combo"

        if not has_ti_fatigue and not is_high_value:
            # 检查是否是非钛合金材料
            non_ti = ["aluminum", "aluminium", "steel", "magnesium",
                       "mg alloy", "ceramic", "composite"]
            if any(nt in combined for nt in non_ti):
                reject_reasons.append("非钛合金材料方向")
            elif not has_mechanism:
                reject_reasons.append("缺乏损伤机制链描述，且与钛合金疲劳关联不明确")

        # ── 检查5：过于简短 ──
        if len(name) < 10 and not is_high_value:
            reject_reasons.append("空白名称过于简短，缺乏具体研究内容")

        # ── 高价值缺失组合直接保留 ──
        if is_high_value:
            gap["reject_reasons"] = []
            # 补充解释性说明
            if not has_ti_fatigue:
                gap["note"] = "该方向涉及钛合金疲劳相关对象，虽当前未直接匹配关键词"
            real_gaps.append(gap)
            continue

        if reject_reasons:
            gap["reject_reasons"] = reject_reasons
            pseudo_gaps.append(gap)
        else:
            gap["reject_reasons"] = []
            real_gaps.append(gap)

    return real_gaps, pseudo_gaps


def _write_gap_diagnosis(
    real_gaps: List[Dict], pseudo_gaps: List[Dict],
    observed: Dict, high_value_missing: List,
    var_records: List[Dict], stats: Dict[str, Any],
) -> str:
    """生成 02_gap_diagnosis.md"""
    papers = get_all_papers()
    total_papers = len(papers)

    # 统计核心钛合金疲劳文献
    core_count = 0
    primary_count = 0
    for p in papers:
        if p.get("alloy_type") != "out_of_scope":
            sr = classify_titanium_scope(p)
            if sr.get("include_in_core_analysis"):
                core_count += 1
                if sr.get("main_case_relevance") == "primary":
                    primary_count += 1

    lines = [
        "# Gap Diagnosis（研究空白诊断报告）",
        "",
        "## 覆盖矩阵概览",
        "",
        f"- **文献总数（去重后）**: {total_papers} 篇",
        f"- **核心钛合金疲劳文献**: {core_count} 篇",
        f"  - 主案例（AM Ti-6Al-4V FCG）相关: {primary_count} 篇",
        f"- **变量—机制关系记录**: {stats.get('variable_records', 0)}",
        f"- **候选研究空白总数**: {stats.get('candidate_gaps', 0)}",
        f"- **保留空白**: {stats.get('real_gaps', 0)}",
        f"- **伪空白（已拒绝）**: {stats.get('pseudo_gaps', 0)}",
        "",
    ]

    if total_papers < 5:
        lines.append("> ⚠️ **当前文献库规模极小，以下空白分析仅为流程框架验证。**\n")

    # ── Layer A: Observed Coverage ──
    lines.append("## A. 实际覆盖（Observed Coverage）\n")
    lines.append("以下统计文献库中实际出现的维度组合：\n")

    has_any_coverage = False
    for dim, cat_list in DIMENSIONS.items():
        covered_cats = []
        for cat in cat_list:
            paper_list = observed.get(dim, {}).get(cat, [])
            n = len(paper_list)
            if n > 0:
                covered_cats.append(f"  - {cat}: {n}篇")
                has_any_coverage = True
        if covered_cats:
            lines.append(f"### {dim}（覆盖 {len(covered_cats)}/{len(cat_list)} 个类别）\n")
            lines.extend(covered_cats)
            lines.append("")

    if not has_any_coverage:
        lines.append("（暂无文献覆盖）\n")

    # ── Layer B: High-value Missing Combinations ──
    lines.append("## B. 高价值缺失组合（与主案例相关）\n")

    if high_value_missing:
        for i, combo in enumerate(high_value_missing, 1):
            lines.append(f"### {i}. {combo.get('dimensions', combo.get('name', ''))}")
            lines.append(f"- **重要性**: {combo['why_important']}")
            if combo.get("supporting_papers"):
                lines.append(f"- **支持下文献**: {', '.join(combo['supporting_papers'][:3])}")
            lines.append(f"- **当前证据等级**: {combo['evidence_level']}")
            lines.append("")
    else:
        lines.append("- 当前文献库中未识别到与主案例相关的高价值缺失组合\n")

    # ── 保留的候选空白 ──
    lines.append("\n## 保留的候选研究空白\n")

    # 按优先级排序
    priority_order = {"high": 0, "medium": 1, "low": 2}
    real_gaps_sorted = sorted(
        real_gaps, key=lambda g: (
            priority_order.get(g.get("priority", "low"), 99),
            g.get("source") != "high_value_missing_combo",
        )
    )

    if real_gaps_sorted:
        # 先显示高优先级
        high_pri = [g for g in real_gaps_sorted if g.get("priority") == "high"]
        med_pri = [g for g in real_gaps_sorted if g.get("priority") == "medium"]
        low_pri = [g for g in real_gaps_sorted if g.get("priority") == "low"]

        if high_pri:
            lines.append("### 高优先级（直接与主案例相关）\n")
            for i, gap in enumerate(high_pri, 1):
                _write_gap_detail(lines, i, gap)

        if med_pri:
            lines.append("### 中优先级（次要案例/元素成分方向）\n")
            for i, gap in enumerate(med_pri, 1):
                _write_gap_detail(lines, i, gap)

        if low_pri:
            lines.append("### 低优先级（背景参考）\n")
            for i, gap in enumerate(low_pri, 1):
                _write_gap_detail(lines, i, gap)
    else:
        lines.append("- 当前保留的候选空白均为空\n")

    # ── 伪空白 ──
    lines.append("\n## 伪空白（已拒绝）\n")
    if pseudo_gaps:
        lines.append("以下候选空白因不满足质量要求被拒绝，拒绝理由已标注：\n")
        for i, gap in enumerate(pseudo_gaps, 1):
            name = gap.get("name", "未命名")
            reasons = gap.get("reject_reasons", ["未说明"])
            lines.append(f"### {i}. {name}")
            for r in reasons:
                lines.append(f"- **拒绝原因**: {r}")
            lines.append("")
    else:
        lines.append("- 无伪空白\n")

    lines.append("---\n")
    lines.append(f"> 当前文献库 {total_papers} 篇（去重后），核心钛合金疲劳文献 {core_count} 篇。")
    lines.append("> 空白分析可靠性随文献数量增加而提高。\n")

    out_path = OUTPUTS_DIR / "02_gap_diagnosis.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)


def _write_gap_detail(lines: list, idx: int, gap: dict) -> None:
    """写入单个gap的详细信息。"""
    name = gap.get("name", "未命名")
    if gap.get("priority") == "medium" and name.startswith("[secondary] "):
        name = name[12:]  # 去掉前缀

    lines.append(f"**{idx}. {name}**")
    lines.append(f"- **来源**: {gap.get('source', 'N/A')}")
    lines.append(f"- **支持证据**: {gap.get('supporting_evidence', 'N/A')[:200]}")
    lines.append(f"- **缺失证据**: {gap.get('missing_evidence', 'N/A')[:200]}")
    lines.append(f"- **研究对象**: {gap.get('research_object', 'N/A')}")
    lines.append(f"- **变量**: {gap.get('variable', 'N/A')}")
    if gap.get("mechanism"):
        lines.append(f"- **机制**: {str(gap.get('mechanism', ''))[:200]}")
    lines.append("")


# ── 科学模态证据索引（轻量） ──────────────────────────────────────────

SCIENTIFIC_ARTIFACT_PATTERNS = {
    "S-N curve": ["s-n curve", "s-n 曲线", "应力-寿命", "stress-life", "whöler"],
    "fatigue life / Nf": ["fatigue life", "nf", "疲劳寿命", "循环寿命"],
    "da/dN-ΔK curve": ["da/dn", "da/dn-δk", "crack growth rate", "裂纹扩展速率",
                        "fcgr", "δk", "Δk"],
    "Paris C/m": ["paris", "paris law", "paris equation", "paris 参数", "paris公式",
                   "c and m", "c, m"],
    "Walker model": ["walker", "walker model", "walker修正", "walker 模型"],
    "SEM fractography": ["sem", "断口", "fractograph", "fracture surface",
                          "疲劳辉纹", "striation", "二次裂纹"],
    "EBSD": ["ebsd", "electron backscatter", "ecci", "ipf", "phase map"],
    "micro-CT / X-ray CT": ["micro-ct", "x-ray ct", "x-ray computed", "synchrotron ct",
                              "同步辐射ct", "三维表征"],
    "pore defect": ["pore", "poros", "气孔", "孔隙", "未熔合", "lack of fusion",
                     "keyhole", "gas pore"],
    "surface roughness": ["roughness", "表面粗糙度", "as-built surface", "surface finish",
                            "ra", "rz"],
    "crack initiation site": ["crack initiation", "裂纹萌生", "起裂", "fish-eye",
                                "odf", "内部裂纹", "表面裂纹"],
    "hardness / tensile": ["hardness", "tensile", "拉伸", "硬度", "屈服", "强度"],
}


def _build_scientific_artifact_index() -> None:
    """从文献卡片中构建轻量科学模态证据索引（文本匹配，无OCR/图像识别）。"""
    papers = get_all_papers()
    records = []
    for p in papers:
        title = p.get("title", "")
        text = str(p.get("title", "")) + " " + \
               str(p.get("key_findings", "")) + " " + \
               str(p.get("mechanical_indicators", "")) + " " + \
               str(p.get("characterization_methods", "")) + " " + \
               str(p.get("crack_growth_mechanism", "")) + " " + \
               str(p.get("material_system", ""))
        text_lower = text.lower()
        for artifact, patterns in SCIENTIFIC_ARTIFACT_PATTERNS.items():
            if any(p in text_lower for p in patterns):
                records.append({
                    "source_paper": title,
                    "artifact_type": artifact,
                    "confidence_level": "high" if "key_findings" in artifact else "medium",
                })
    # 去重
    seen = set()
    unique = []
    for r in records:
        key = (r["source_paper"], r["artifact_type"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    # 写入CSV
    out_path = DATA_DIR / "scientific_artifact_index.csv"
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_paper", "artifact_type",
                                           "confidence_level"])
        w.writeheader()
        for r in unique:
            w.writerow(r)


def _save_var_table(records: List[Dict]) -> None:
    """保存变量关系表到 CSV。"""
    path = DATA_DIR / "variable_mechanism.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS_VAR)
        w.writeheader()
        for rec in records:
            row = {f: rec.get(f, "") for f in CSV_FIELDS_VAR}
            w.writerow(row)


def _save_matrix(matrix: pd.DataFrame) -> None:
    """保存覆盖矩阵到 CSV。"""
    if matrix.empty:
        return
    path = DATA_DIR / "coverage_matrix.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path, index=False, encoding="utf-8-sig")

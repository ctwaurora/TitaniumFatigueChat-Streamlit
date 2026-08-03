"""Topic routing and query expansion for titanium-fatigue research."""

from __future__ import annotations

import re
from typing import Iterable


TOPIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "MICROSTRUCTURE": ("microstructure", "微观组织", "显微组织", "alpha lath", "α片层", "α′", "martensite", "马氏体", "grain", "晶粒", "织构", "texture", "phase", "相组成"),
    "RESIDUAL_STRESS": ("residual stress", "残余应力", "应力释放", "stress relief"),
    "HEAT_TREATMENT": ("heat treatment", "热处理", "anneal", "退火", "aging", "时效"),
    "HIP": ("hip", "hot isostatic", "热等静压"),
    "SURFACE_CONDITION": ("surface roughness", "表面粗糙", "machining", "机加工", "polish", "抛光", "shot peen", "喷丸", "as-built surface"),
    "BUILD_ORIENTATION": ("build orientation", "build direction", "建造方向", "成形方向", "anisotropy", "各向异性"),
    "PROCESS_PARAMETER": ("process parameter", "工艺参数", "laser power", "激光功率", "scan speed", "扫描速度", "energy density", "能量密度", "layer thickness", "层厚"),
    "DEFECT": ("pore", "porosity", "孔隙", "孔洞", "defect size", "缺陷尺寸", "lack of fusion", "未熔合", "√area", "sqrt area", "sqrt-area"),
    "LOADING_CONDITION": ("stress ratio", "应力比", "frequency", "频率", "loading", "载荷", "应力幅", "stress amplitude"),
    "FATIGUE_LOADING": ("stress ratio", "应力比", "frequency", "频率", "loading", "载荷", "应力幅", "stress amplitude"),
    "ENVIRONMENT": ("environment", "环境", "temperature", "温度", "corrosion", "腐蚀", "vacuum", "真空"),
    "CRACK_INITIATION": ("crack initiation", "裂纹起裂", "crack origin", "起裂位置", "nucleation"),
    "SHORT_CRACK": ("short crack", "small crack", "短裂纹", "小裂纹"),
    "CRACK_PROPAGATION": ("crack growth", "crack propagation", "裂纹扩展", "da/dn", "delta k", "δk", "Δk", "crack closure", "裂纹闭合"),
    "HCF_VHCF": ("hcf", "vhcf", "lcf", "高周", "超高周", "低周", "fatigue regime"),
    "HCF": ("hcf", "high cycle fatigue", "高周疲劳"),
    "VHCF": ("vhcf", "very high cycle fatigue", "超高周疲劳"),
    "LCF": ("lcf", "low cycle fatigue", "低周疲劳"),
    "FATIGUE_LIFE": ("fatigue life", "疲劳寿命", "fatigue limit", "疲劳极限", "s-n", "寿命分散"),
    "FATIGUE_LIMIT": ("fatigue limit", "endurance limit", "疲劳极限"),
    "FORMULA_MODEL": ("formula", "equation", "model", "公式", "方程", "模型", "paris", "basquin", "murakami"),
    "LITERATURE_CONFLICT": ("conflict", "contradict", "冲突", "相反", "为什么存在", "不一致"),
    "RESEARCH_GAP": ("research gap", "研究空白", "尚未解决", "缺少哪些"),
    "EXPERIMENT_DESIGN": ("experiment design", "实验设计", "验证方案", "如何验证", "设计一个"),
}


TOPIC_EXPANSIONS: dict[str, str] = {
    "MICROSTRUCTURE": "microstructure alpha lath alpha prime martensite beta phase grain texture EBSD",
    "RESIDUAL_STRESS": "residual stress redistribution stress relief crack driving force",
    "HEAT_TREATMENT": "heat treatment annealing aging stress relief microstructure fatigue",
    "HIP": "hot isostatic pressing HIP microstructure fatigue",
    "SURFACE_CONDITION": "surface roughness machining polishing shot peening surface fatigue",
    "BUILD_ORIENTATION": "build orientation build direction texture anisotropy fatigue",
    "PROCESS_PARAMETER": "laser power scan speed energy density layer thickness process fatigue",
    "DEFECT": "pore porosity lack of fusion defect sqrt area fatigue",
    "FATIGUE_LOADING": "stress ratio frequency load amplitude crack closure fatigue",
    "LOADING_CONDITION": "stress ratio frequency loading mode amplitude mean stress fatigue",
    "ENVIRONMENT": "temperature environment corrosion vacuum fatigue crack growth",
    "CRACK_INITIATION": "crack initiation crack origin nucleation fatigue",
    "SHORT_CRACK": "short crack small crack microstructurally short crack growth",
    "CRACK_PROPAGATION": "fatigue crack growth da/dN Delta K Delta Keff crack closure",
    "HCF_VHCF": "LCF HCF VHCF fatigue regime initiation run-out",
    "HCF": "high cycle fatigue HCF S-N crack initiation",
    "VHCF": "very high cycle fatigue VHCF ultrasonic internal initiation run-out",
    "LCF": "low cycle fatigue LCF strain life Coffin Manson cyclic plasticity",
    "FATIGUE_LIFE": "fatigue life Nf fatigue limit S-N scatter",
    "FATIGUE_LIMIT": "fatigue limit endurance limit stress amplitude run-out",
    "FORMULA_MODEL": "equation formula model Paris Basquin parameters units applicability",
    "LITERATURE_CONFLICT": "contradictory no significant effect condition dependent comparison",
    "RESEARCH_GAP": "research gap unresolved matched-condition evidence",
    "EXPERIMENT_DESIGN": "experiment validation control variable falsification",
}


PORE_PATTERN = re.compile(r"孔隙|孔洞|气孔|缺陷尺寸|近表面缺陷|√area|sqrt[-_\s]*area|\bpore\b|porosity", re.I)


def identify_topics(text: str) -> list[str]:
    lowered = str(text or "").casefold()
    return [
        topic
        for topic, patterns in TOPIC_PATTERNS.items()
        if any(pattern.casefold() in lowered for pattern in patterns)
    ]


def query_mentions_pores(text: str) -> bool:
    return bool(PORE_PATTERN.search(str(text or "")))


def expand_topic_query(question: str, topics: Iterable[str] | None = None) -> str:
    """Expand only detected topics; never inject defect terms by default."""
    selected = list(topics or identify_topics(question))
    expansions = [TOPIC_EXPANSIONS[item] for item in selected if item in TOPIC_EXPANSIONS]
    if not query_mentions_pores(question):
        expansions = [
            re.sub(r"\b(?:pore|porosity|defect|sqrt area)\b", "", value, flags=re.I)
            for value in expansions
            if value != TOPIC_EXPANSIONS["DEFECT"]
        ]
    return " ".join([str(question).strip(), *expansions]).strip()


def document_topics(text: str, conditions: dict | None = None) -> list[str]:
    condition_text = " ".join(
        str(value) for value in (conditions or {}).values() if value not in (None, "", "NOT_REPORTED")
    )
    return identify_topics(f"{text} {condition_text}")


def pore_is_dominant(text: str) -> bool:
    value = str(text or "")
    pore_hits = len(PORE_PATTERN.findall(value))
    return pore_hits >= 2 and pore_hits * 35 >= max(1, len(value))

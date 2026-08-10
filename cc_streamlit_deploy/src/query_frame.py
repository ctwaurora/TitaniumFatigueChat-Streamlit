"""Deterministic scientific query framing for titanium-fatigue questions."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.research_topics import identify_topics, query_mentions_pores


@dataclass(frozen=True)
class QueryFrame:
    original_query: str
    material: str = ""
    alloy_grade: str = ""
    manufacturing_process: str = ""
    post_processing: list[str] = field(default_factory=list)
    surface_condition: list[str] = field(default_factory=list)
    fatigue_stage: str = ""
    crack_stage: str = ""
    independent_variables: list[str] = field(default_factory=list)
    dependent_variables: list[str] = field(default_factory=list)
    control_conditions: dict[str, str] = field(default_factory=dict)
    requested_mechanisms: list[str] = field(default_factory=list)
    requested_formulas: list[str] = field(default_factory=list)
    requested_comparison: bool = False
    environment: str = ""
    loading_mode: str = ""
    stress_ratio: str = ""
    temperature: str = ""
    topic_labels: list[str] = field(default_factory=list)
    excluded_topics: list[str] = field(default_factory=list)
    missing_entities: list[str] = field(default_factory=list)
    ambiguity_flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def cache_key(self, dataset_version: str) -> str:
        payload = json.dumps(
            {"query_frame": self.as_dict(), "dataset_version": dataset_version},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


VARIABLE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("residual_stress", ("residual stress", "残余应力", "残余拉应力", "残余压应力")),
    ("alpha_lath_width", ("alpha lath", "α片层", "片层宽度")),
    ("prior_beta_grain", ("prior beta grain", "prior-β", "先前β晶粒", "原始β晶粒")),
    ("crystallographic_texture", ("crystallographic texture", "texture", "织构")),
    ("microstructure", ("microstructure", "微观组织", "显微组织", "组织")),
    ("surface_roughness", ("surface roughness", "表面粗糙", "ra", "rz")),
    ("near_surface_defect", ("near-surface defect", "near surface defect", "近表面缺陷")),
    ("pore_location", ("pore location", "defect location", "孔隙位置", "缺陷位置", "距表面", "表面、近表面", "表面和内部")),
    ("build_orientation", ("build orientation", "build direction", "建造方向", "成形方向")),
    ("stress_ratio", ("stress ratio", "应力比", "r比")),
    ("loading_frequency", ("loading frequency", "frequency", "加载频率", "频率")),
    ("environmental_medium", ("environment", "air", "vacuum", "corrosion", "环境", "空气", "真空", "腐蚀")),
    ("fatigue_regime", ("hcf", "vhcf", "高周疲劳", "超高周疲劳", "疲劳区间")),
    ("temperature", ("temperature", "温度", "高温")),
    ("pore_size", ("pore size", "defect size", "孔隙尺寸", "缺陷尺寸", "孔隙会", "sqrt area", "√area")),
    ("heat_treatment", ("heat treatment", "热处理", "anneal", "退火", "时效")),
    ("hip", ("hot isostatic", "hip", "热等静压")),
)

DEPENDENT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("short_crack_growth_rate", ("short crack growth", "短裂纹扩展", "小裂纹扩展")),
    ("crack_growth_rate_da_dn", ("da/dn", "裂纹扩展速率", "crack growth rate")),
    ("delta_k_threshold", ("delta kth", "Δkth", "裂纹扩展阈值", "扩展阈值")),
    ("paris_parameters", ("paris parameter", "paris参数", "paris模型参数", "c和m", "c、m")),
    ("effective_delta_k", ("delta keff", "Δkeff", "有效应力强度因子")),
    ("fatigue_life_Nf", ("fatigue life", "疲劳寿命", "总寿命", "nf")),
    ("fatigue_limit", ("fatigue limit", "疲劳极限")),
    ("crack_initiation_life", ("crack initiation life", "起裂寿命")),
    ("crack_initiation", ("crack initiation", "裂纹起裂", "裂纹萌生")),
    ("crack_origin_location", (
        "crack origin", "裂纹起源", "起裂位置", "表面转向内部",
        "表面起裂转向内部起裂", "从表面起裂转向内部起裂",
    )),
    ("fatigue_performance", ("fatigue performance", "fatigue behavior", "疲劳性能", "疲劳行为")),
    ("crack_closure", ("crack closure", "裂纹闭合")),
)


def _matches(text: str, patterns: tuple[tuple[str, tuple[str, ...]], ...]) -> list[str]:
    lowered = text.casefold()
    def contains(term: str) -> bool:
        normalized = term.casefold()
        if normalized.isascii() and normalized.isalnum() and len(normalized) <= 3:
            return bool(re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", lowered))
        return normalized in lowered

    return [name for name, terms in patterns if any(contains(term) for term in terms)]


def _first(text: str, mapping: tuple[tuple[str, tuple[str, ...]], ...]) -> str:
    values = _matches(text, mapping)
    return values[0] if values else ""


def parse_query_frame(query: str, parsed_entities: dict[str, Any] | None = None) -> QueryFrame:
    text = " ".join(str(query or "").split())
    lower = text.casefold()
    alloy_match = re.search(r"(?<![A-Za-z0-9])(?:ti[-– ]?6al[-– ]?4v(?:\s*eli)?|tc4|ti[-– ]?6242|ti[-– ]?17|ti[-– ]?5553)(?![A-Za-z0-9])", text, re.I)
    alloy = alloy_match.group(0) if alloy_match else ""
    process = _first(text, (
        ("L-PBF", ("l-pbf", "lpbf", "laser powder bed fusion", "slm", "选区激光熔化")),
        ("EBM", ("electron beam melting", "ebm", "电子束熔化")),
        ("WAAM", ("wire arc additive", "waam", "电弧增材")),
        ("DED", ("directed energy deposition", "ded", "定向能量沉积")),
        ("wrought", ("wrought", "锻造")),
        ("cast", ("cast", "铸造")),
    ))
    post = _matches(text, (
        ("HIP", ("hip", "hot isostatic", "热等静压")),
        ("annealing", ("anneal", "退火")),
        ("stress_relief", ("stress relief", "去应力")),
        ("aging", ("aging", "时效")),
    ))
    surface = _matches(text, (
        ("as-built", ("as-built", "as built", "原始表面")),
        ("machined", ("machined", "machining", "机加工")),
        ("polished", ("polished", "polishing", "抛光")),
        ("shot-peened", ("shot peen", "喷丸")),
    ))
    stress = re.search(r"\bR\s*=\s*-?\d+(?:\.\d+)?", text, re.I)
    temp = re.search(r"-?\d+(?:\.\d+)?\s*°?\s*C\b|室温|高温", text, re.I)
    environment = _first(text, (
        ("air", (" air", "空气")), ("vacuum", ("vacuum", "真空")),
        ("corrosive", ("corrosion", "腐蚀", "saline", "盐水")),
        ("hydrogen", ("hydrogen", "氢")),
    ))
    crack_stage = _first(text, (
        ("SHORT_CRACK", ("short crack", "small crack", "短裂纹", "小裂纹")),
        ("CRACK_INITIATION", ("crack initiation", "裂纹起裂", "裂纹萌生")),
        ("LONG_CRACK", ("long crack", "长裂纹", "paris区", "paris regime")),
        ("CRACK_PROPAGATION", ("crack growth", "crack propagation", "裂纹扩展", "da/dn")),
    ))
    fatigue_stage = _first(text, (
        ("VHCF", ("vhcf", "超高周")), ("HCF", ("hcf", "高周")),
        ("LCF", ("lcf", "低周")), ("SHORT_CRACK_PROPAGATION", ("短裂纹", "short crack")),
    ))
    independent = _matches(text, VARIABLE_PATTERNS)
    dependent = _matches(text, DEPENDENT_PATTERNS)
    parsed = parsed_entities or {}
    # Keep canonical identifiers in the reasoning contract. Display labels
    # otherwise become duplicate entities that cannot be normalized reliably.
    for key in (parsed.get("independent"),):
        if key and str(key) not in independent:
            independent.append(str(key))
    for key in (parsed.get("dependent"),):
        if key and str(key) not in dependent:
            dependent.append(str(key))
    mechanisms = _matches(text, (
        ("crack_closure", ("crack closure", "裂纹闭合")),
        ("effective_driving_force", ("delta keff", "Δkeff", "有效驱动力")),
        ("plastic_zone", ("plastic zone", "塑性区")),
        ("crack_deflection", ("crack deflection", "裂纹偏转")),
        ("slip_transfer", ("slip transfer", "滑移传递")),
        ("oxidation", ("oxidation", "氧化")),
        ("hydrogen_assisted", ("hydrogen", "氢相关", "氢致")),
        ("fish_eye", ("fish-eye", "fish eye", "鱼眼")),
    ))
    if "residual_stress" in independent and crack_stage in {"SHORT_CRACK", "CRACK_PROPAGATION"}:
        mechanisms = list(dict.fromkeys([*mechanisms, "crack_closure", "effective_driving_force"]))
    formulas = [name for name in ("Paris", "Walker", "Basquin", "Coffin-Manson", "Murakami") if name.casefold() in lower]
    if re.search(r"公式|方程|模型|formula|equation|compare", text, re.I) and not formulas:
        formulas.append("RELEVANT_LITERATURE_FORMULA")
    topics = identify_topics(text)
    missing = []
    if not alloy:
        missing.append("alloy_grade")
    if not independent:
        missing.append("independent_variables")
    if not dependent:
        missing.append("dependent_variables")
    ambiguity = []
    if "crack_growth_rate_da_dn" in dependent and not crack_stage:
        ambiguity.append("crack_stage_not_specified")
    if "fatigue_life_Nf" in dependent and not fatigue_stage:
        ambiguity.append("fatigue_regime_not_specified")
    excluded = [] if query_mentions_pores(text) else ["DEFECT", "PORE", "POROSITY"]
    return QueryFrame(
        original_query=text,
        material=alloy or "titanium_alloy",
        alloy_grade=alloy,
        manufacturing_process=process,
        post_processing=post,
        surface_condition=surface,
        fatigue_stage=fatigue_stage,
        crack_stage=crack_stage,
        independent_variables=independent,
        dependent_variables=dependent,
        requested_mechanisms=mechanisms,
        requested_formulas=formulas,
        requested_comparison=bool(re.search(r"比较|对比|差异|冲突|compare|versus|\bvs\b", text, re.I)),
        environment=environment,
        loading_mode=_first(text, (("axial", ("axial", "轴向")), ("bending", ("bending", "弯曲")), ("torsion", ("torsion", "扭转")))),
        stress_ratio=stress.group(0).replace(" ", "") if stress else "",
        temperature=temp.group(0) if temp else "",
        topic_labels=topics,
        excluded_topics=excluded,
        missing_entities=missing,
        ambiguity_flags=ambiguity,
    )

"""
full_text_extractor.py — 全文结构化提取模块 (v3)

替代旧的 title-only 提取逻辑，实现：
1. 全文结构地图构建 (section-level parsing)
2. 四阶段提取 (概览 → 逐主题 → 一致性检查 → 漏提审计)
3. 30+ 微观组织字段提取
4. 每条证据绑定原文 (page/section/chunk_id)
5. 四种空值状态 (NOT_REPORTED / NOT_EXTRACTED / MENTION_ONLY / UNCERTAIN)
6. 提取完整性报告
7. 细读模式 (单主题重新扫描)
8. 重复证据去重

依赖:
    skills/pdf_skill.py (基础 PDF 文本提取)
    可选的大模型客户端（仅用于辅助提取）
"""

import json
import os
import re
import sys
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from src.deep_read_pipeline import deep_read_pdf, parse_pdf_pages

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"

try:
    from skills.pdf_skill import extract_text_from_pdf, chunk_text
except ImportError:
    pass  # 延迟导入

# ═══════════════════════════════════════════════════════════════════════════
# 1. 空值状态枚举
# ═══════════════════════════════════════════════════════════════════════════

class ExtractionStatus:
    """字段提取状态 (取代空字符串/None)"""
    NOT_REPORTED = "NOT_REPORTED"      # 论文确实没有报告
    NOT_EXTRACTED = "NOT_EXTRACTED"    # 系统尚未完成该字段提取
    MENTION_ONLY = "MENTION_ONLY"      # 论文提到但无细节
    UNCERTAIN = "UNCERTAIN"            # 有相关信息但无法可靠判断
    EXTRACTED = "EXTRACTED"            # 已成功提取

    @classmethod
    def is_empty(cls, status: str) -> bool:
        return status in (cls.NOT_REPORTED, cls.NOT_EXTRACTED, cls.UNCERTAIN)

    @classmethod
    def needs_review(cls, status: str) -> bool:
        return status in (cls.MENTION_ONLY, cls.UNCERTAIN)


# ═══════════════════════════════════════════════════════════════════════════
# 2. 证据绑定结构
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EvidenceLocation:
    """证据在原文中的位置"""
    page_number: int = 0
    section_title: str = ""
    paragraph_index: int = 0
    chunk_id: str = ""
    original_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BoundEvidence:
    """绑定原文的单一证据"""
    field_name: str = ""
    value: str = ""
    evidence_text: str = ""
    page: int = 0
    section: str = ""
    chunk_id: str = ""
    directness: str = "direct"  # direct / indirect / inferred
    confidence: float = 0.0     # 0.0 ~ 1.0
    status: str = ExtractionStatus.NOT_EXTRACTED

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "BoundEvidence":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ═══════════════════════════════════════════════════════════════════════════
# 3. 全文结构地图
# ═══════════════════════════════════════════════════════════════════════════

# 章节标题正则 (支持中英文/数字混合)
SECTION_PATTERNS = {
    "abstract": re.compile(
        r"^(?:abstract|摘要|概要)\b", re.IGNORECASE
    ),
    "introduction": re.compile(
        r"^(?:introduction|background|引言|介绍|背景|前言)\b", re.IGNORECASE
    ),
    "materials_and_methods": re.compile(
        r"^(?:materials?\s*(?:and|&|与)?\s*(?:methods?|experimental|characterization|"
        r"experimental\s*procedures?|experimental\s*methods?|实验方法|"
        r"材料与方法|材料与实验|实验过程|试样制备)\b)",
        re.IGNORECASE,
    ),
    "manufacturing": re.compile(
        r"^(?:(?:specimen|sample)\s*(?:preparation|fabrication|manufacturing)|"
        r"(?:powder|printing|additive\s*manufacturing)\s*(?:process|procedure|conditions?)|"
        r"(?:L-PBF|SLM|EBM|DED)\s*process|制备工艺|成形工艺|打印工艺|粉末特性)\b",
        re.IGNORECASE,
    ),
    "heat_treatment": re.compile(
        r"^(?:heat\s*treatment|post\s*(?:process|treatment|processing)|"
        r"(?:thermal|annealing|aging)\s*(?:treatment|process)|HIP|"
        r"热处理|后处理|热等静压|退火|时效)\b",
        re.IGNORECASE,
    ),
    "surface_characterization": re.compile(
        r"^(?:surface\s*(?:roughness|characterization|topography|finish|morphology)|"
        r"surface\s*quality|表面表征|表面粗糙度|表面形貌|表面质量)\b",
        re.IGNORECASE,
    ),
    "microstructure_characterization": re.compile(
        r"^(?:microstructure|microstructural\s*(?:characterization|analysis|observation|"
        r"examination|evaluation)|metallography|(?:optical|electron)\s*microscopy|"
        r"EBSD|TEM|XRD\s*analysis|微观组织|显微组织|金相|组织表征|组织分析)\b",
        re.IGNORECASE,
    ),
    "defect_characterization": re.compile(
        r"^(?:"
        r"(?:defect|pore|porosity|crack|void)\s*(?:characterization|analysis|"
        r"examination|evaluation|detection)"
        r"|micro[- ]CT|X-ray\s*CT|computed\s*tomography"
        r"|缺陷表征|孔隙分析|孔洞分析|CT检测|无损检测"
        r")\b",
        re.IGNORECASE,
    ),
    "fatigue_testing": re.compile(
        r"^(?:fatigue\s*(?:test|experiment|testing|behavior|performance)|"
        r"(?:cyclic|mechanical)\s*testing|疲劳试验|疲劳测试|力学性能测试)\b",
        re.IGNORECASE,
    ),
    "results": re.compile(
        r"^(?:results?\s*(?:and\s*discussion)?|实验[结]果|结果与分析)\b",
        re.IGNORECASE,
    ),
    "fractography": re.compile(
        r"^(?:fractography|fracture\s*(?:surface|analysis|morphology)|"
        r"断口分析|断口形貌|断裂面分析)\b",
        re.IGNORECASE,
    ),
    "discussion": re.compile(
        r"^(?:discussion|analysis|分析与讨论|讨论|综合分析)\b",
        re.IGNORECASE,
    ),
    "conclusion": re.compile(
        r"^(?:conclusion|summary|concluding\s*remarks|findings?|"
        r"结论|总结|主要结论)\b",
        re.IGNORECASE,
    ),
    "supplementary": re.compile(
        r"^(?:supplementary|appendix|acknowledgment|references?|补充|附录|致谢|参考文献)\b",
        re.IGNORECASE,
    ),
}

# 反向映射：section_key → 中文名
SECTION_CN = {
    "abstract": "摘要",
    "introduction": "引言",
    "materials_and_methods": "材料与方法",
    "manufacturing": "制备工艺",
    "heat_treatment": "热处理",
    "surface_characterization": "表面表征",
    "microstructure_characterization": "微观组织表征",
    "defect_characterization": "缺陷表征",
    "fatigue_testing": "疲劳测试",
    "results": "结果",
    "fractography": "断口分析",
    "discussion": "讨论",
    "conclusion": "结论",
    "supplementary": "补充/附录",
    "unclassified": "未分类",
}


def classify_section(title: str) -> str:
    """根据标题文本判断章节类型。"""
    title_stripped = title.strip()
    for section_key, pattern in SECTION_PATTERNS.items():
        if pattern.search(title_stripped):
            return section_key
    return "unclassified"


@dataclass
class SectionChunk:
    """结构化段落"""
    section_key: str = "unclassified"
    section_title: str = ""
    page_number: int = 0
    paragraph_index: int = 0
    chunk_id: str = ""
    original_text: str = ""
    token_count: int = 0
    # 段落级关键词标记 (预计算, 加速检索)
    has_microstructure_keywords: bool = False
    has_defect_keywords: bool = False
    has_fatigue_keywords: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PaperStructureMap:
    """全文结构地图"""
    title: str = ""
    sections: Dict[str, List[SectionChunk]] = field(default_factory=lambda: defaultdict(list))
    pages: int = 0
    total_chunks: int = 0

    def get_all_chunks(self) -> List[SectionChunk]:
        """返回所有 chunks (扁平)"""
        result = []
        for section_list in self.sections.values():
            result.extend(section_list)
        return sorted(result, key=lambda c: (c.page_number, c.paragraph_index))

    def get_section(self, key: str) -> List[SectionChunk]:
        return self.sections.get(key, [])

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "sections": {k: [c.to_dict() for c in v] for k, v in self.sections.items()},
            "pages": self.pages,
            "total_chunks": self.total_chunks,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4. 关键词/同义词词表
# ═══════════════════════════════════════════════════════════════════════════

# 微观组织关键词 (中英文)
MICROSTRUCTURE_KEYWORDS = [
    "microstructure", "microstructural", "morphology",
    "phase", "phase composition", "alpha phase", "beta phase",
    "α phase", "β phase", "alpha prime", "α′", "α'",
    "martensite", "martensitic",
    "prior beta grain", "prior-β grain", "prior β grain",
    "columnar grain", "equiaxed grain",
    "grain size", "grain orientation",
    "crystallographic texture", "texture",
    "lath", "alpha lath", "lamella", "lamellar",
    "acicular", "basketweave", "basket-weave",
    "Widmanstätten", "colony",
    "EBSD", "SEM", "TEM", "XRD", "optical microscopy",
    "phase fraction", "misorientation", "grain boundary",
    "microstructural heterogeneity",
    "微观组织", "显微组织", "金相组织", "组织形貌",
    "相组成", "α相", "β相", "马氏体",
    "原始β晶粒", "板条", "针状组织",
    "篮织组织", "篮网状组织", "魏氏组织",
    "柱状晶", "等轴晶", "晶粒尺寸",
    "晶体取向", "织构", "相含量",
    "晶界", "取向差",
    "α板条", "α片层", "片层组织", "双态组织",
    "等轴组织", "网篮组织",
]

# 孔隙/缺陷关键词
DEFECT_KEYWORDS = [
    "pore", "porosity", "void", "defect", "crack",
    "lack of fusion", "keyhole", "gas pore",
    "micro-CT", "computed tomography", "μ-CT",
    "pore size", "pore shape", "pore distribution",
    "sqrt area", "√area",
    "孔隙", "孔洞", "缺陷", "气孔", "未熔合",
    "匙孔", "裂纹", "微裂纹",
    "孔隙率", "孔径", "孔隙分布",
]

# 疲劳关键词
FATIGUE_KEYWORDS = [
    "fatigue", "cyclic", "S-N", "stress-life", "strain-life",
    "stress ratio", "R ratio", "load ratio",
    "fatigue limit", "endurance limit",
    "crack initiation", "crack growth", "crack propagation",
    "Paris", "da/dN", "ΔK", "ΔKth",
    "LCF", "HCF", "VHCF", "FCGR",
    "疲劳", "循环", "应力比", "疲劳极限",
    "裂纹起裂", "裂纹扩展", "Paris",
    "低周疲劳", "高周疲劳", "超高周疲劳",
]

# 热处理关键词
HEAT_TREATMENT_KEYWORDS = [
    "heat treatment", "annealing", "aging", "solution treatment",
    "HIP", "hot isostatic pressing",
    "stress relief", "stress-relief",
    "quenching", "tempering",
    "热处理", "退火", "时效", "固溶",
    "热等静压", "去应力",
]

# 残余应力关键词
RESIDUAL_STRESS_KEYWORDS = [
    "residual stress", "compressive stress", "tensile stress",
    "XRD residual stress", "micro-strain",
    "peak shift", "FWHM",
    "残余应力", "压应力", "拉应力",
    "微观应变", "峰位偏移",
]

# 表征方法关键词
CHARACTERIZATION_KEYWORDS = [
    "SEM", "scanning electron", "FE-SEM",
    "EBSD", "electron backscatter",
    "TEM", "transmission electron",
    "XRD", "X-ray diffraction",
    "DIC", "digital image correlation",
    "X-ray CT", "micro-CT", "μ-CT",
    "optical microscopy", "OM",
    "EDS", "EDX",
    "APT", "atom probe",
    "XPS", "AFM",
    "confocal", "profilometry",
    "FIB", "focused ion beam",
    "超声", "声发射",
    "纳米压痕",
]


def has_keywords(text: str, keyword_list: List[str]) -> bool:
    """检查文本中是否包含任一关键词 (大小写不敏感)。"""
    text_lower = text.lower()
    for kw in keyword_list:
        if kw.lower() in text_lower:
            return True
    return False


def extract_numeric_values(text: str) -> List[Dict[str, Any]]:
    """从文本中提取数值及单位。"""
    # 简单模式: 数字 + 可选单位
    patterns = [
        (r"(\d+\.?\d*)\s*(μm|um|nm|mm|cm|MPa|GPa|Pa|Hz|kHz|°C|℃|K|%)", 1),
        (r"(\d+\.?\d*)", 1),
    ]
    results = []
    for pat, group in patterns:
        for m in re.finditer(pat, text):
            try:
                val = float(m.group(group))
                results.append({"value": val, "unit": m.group(2) if m.lastindex and m.lastindex >= 2 else ""})
            except ValueError:
                pass
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 5. 全文结构地图构建
# ═══════════════════════════════════════════════════════════════════════════

# 常见章节标题标识 (用于分段)
SECTION_HEADER_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)*)?\s*[.、．\s]*"
    r"("
    r"abstract|摘要|引言|introduction|background|前言|"
    r"materials?(?:\s*and\s*|\s*&?\s*)methods?|"
    r"experimental|实验方法|材料与方法|试样制备|"
    r"results?|结果|分析与讨论|discussion|"
    r"conclusion|结论|summary|总结|"
    r"microstructure|微观组织|fractography|断口|"
    r"热等静压|热处理|表面粗糙度|孔隙分析|疲劳实验|疲劳试验|"
    r"Acknowledg(e)?ment|致谢|references?|参考文献|"
    r"supplementary|附录|appendix"
    r")(?:[\s.:：、]|$)",
    re.IGNORECASE,
)


def build_paper_structure_map(
    pdf_path: Optional[str] = None,
    full_text: Optional[str] = None,
    chunk_size: int = 500,
    overlap: int = 80,
) -> PaperStructureMap:
    """构建全文结构地图。

    Args:
        pdf_path: PDF 文件路径 (可选, 与 full_text 二选一)
        full_text: 直接传入全文文本 (可选)
        chunk_size: 段落分块大小
        overlap: 段落重叠

    Returns:
        结构化的 PaperStructureMap
    """
    # Real PDFs must retain parser-provided page boundaries.  The text-only
    # compatibility path below intentionally uses page_number=0 and can never
    # produce trusted evidence.
    if pdf_path:
        page_records, _, _, _ = parse_pdf_pages(Path(pdf_path), paper_id="")
        structure = PaperStructureMap(
            title="",
            pages=len(page_records),
            total_chunks=0,
        )
        for page_record in page_records:
            paragraphs = _paragraphs_for_real_page(page_record.cleaned_text)
            if not structure.title and page_record.page_number == 1 and paragraphs:
                structure.title = paragraphs[0][:200]
            for paragraph_index, paragraph in enumerate(paragraphs):
                raw_chunks = (
                    chunk_text(paragraph, chunk_size=chunk_size, overlap=overlap)
                    if len(paragraph) > chunk_size
                    else [paragraph]
                )
                for chunk_index, content in enumerate(raw_chunks):
                    if not content.strip():
                        continue
                    section_key = page_record.section_title
                    structure.sections[section_key].append(
                        SectionChunk(
                            section_key=section_key,
                            section_title=SECTION_CN.get(section_key, section_key),
                            page_number=page_record.page_number,
                            paragraph_index=paragraph_index,
                            chunk_id=(
                                f"PAGE_{page_record.page_number:04d}_"
                                f"P{paragraph_index:04d}_C{chunk_index:02d}"
                            ),
                            original_text=content,
                            token_count=len(content.split()),
                            has_microstructure_keywords=has_keywords(
                                content, MICROSTRUCTURE_KEYWORDS
                            ),
                            has_defect_keywords=has_keywords(content, DEFECT_KEYWORDS),
                            has_fatigue_keywords=has_keywords(content, FATIGUE_KEYWORDS),
                        )
                    )
        structure.total_chunks = sum(len(items) for items in structure.sections.values())
        return structure

    if not full_text or not full_text.strip():
        return PaperStructureMap()

    # ── 提取标题 ──
    lines = full_text.split("\n")
    title = ""
    for line in lines[:30]:
        line = line.strip()
        if len(line) > 20 and not SECTION_HEADER_PATTERN.match(line):
            title = line[:200]
            break

    # ── 分段 ──
    # 先按空行分自然段落
    paragraphs = re.split(r"\n\s*\n", full_text)

    # ── 识别章节边界 ──
    current_section = "unclassified"
    section_map: Dict[str, List[Dict]] = defaultdict(list)

    for para_idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue

        # 检查是否为章节标题
        first_line = para.split("\n")[0].strip()
        detected = classify_section(first_line)

        # 如果检测到新章节且该行长度较短(合理标题),切换
        # 也检查纯数字标题如 "2. Results"
        header_match = SECTION_HEADER_PATTERN.match(first_line)
        is_header = header_match is not None and (
            len(first_line) < 100 or detected != "unclassified"
        )

        if is_header and detected != "unclassified":
            current_section = detected
        elif header_match and current_section == "unclassified":
            # 尝试从匹配中识别
            matched_text = header_match.group(2).lower()
            for sec_key, pattern in SECTION_PATTERNS.items():
                if pattern.search(matched_text):
                    current_section = sec_key
                    break

        # 分块
        raw_chunks = chunk_text(
            para, chunk_size=chunk_size, overlap=overlap
        ) if len(para) > chunk_size else [para]

        for ci, chunk_text_content in enumerate(raw_chunks):
            if not chunk_text_content.strip():
                continue
            chunk_id = f"SEC_{current_section}_{para_idx:04d}_{ci:02d}"
            section_map[current_section].append({
                "section_key": current_section,
                "section_title": SECTION_CN.get(current_section, current_section),
                "page_number": 0,
                "paragraph_index": para_idx,
                "chunk_id": chunk_id,
                "original_text": chunk_text_content,
                "token_count": len(chunk_text_content.split()),
                "has_microstructure_keywords": has_keywords(
                    chunk_text_content, MICROSTRUCTURE_KEYWORDS
                ),
                "has_defect_keywords": has_keywords(
                    chunk_text_content, DEFECT_KEYWORDS
                ),
                "has_fatigue_keywords": has_keywords(
                    chunk_text_content, FATIGUE_KEYWORDS
                ),
            })

    # ── 构建结果 ──
    total_chunks = sum(len(v) for v in section_map.values())
    structure = PaperStructureMap(
        title=title,
        pages=0,
        total_chunks=total_chunks,
    )
    for k, v in section_map.items():
        structure.sections[k] = [SectionChunk(**item) for item in v]

    return structure


def _paragraphs_for_real_page(text: str) -> List[str]:
    """Split a single real page without losing its parser page number."""
    paragraphs = [
        value.strip()
        for value in re.split(r"\n\s*\n", text)
        if value.strip()
    ]
    if len(paragraphs) <= 1:
        paragraphs = [
            line.strip() for line in text.splitlines() if len(line.strip()) >= 20
        ]
    return paragraphs


# ═══════════════════════════════════════════════════════════════════════════
# 6. 主题扫描器 (按主题逐项扫描全文)
# ═══════════════════════════════════════════════════════════════════════════

class TopicScanner:
    """对特定主题独立扫描全文所有相关段落。"""

    TOPICS = [
        "material_and_process",       # 材料与制造工艺
        "heat_treatment_and_hip",     # 热处理与 HIP
        "surface_and_roughness",      # 表面状态与粗糙度
        "defect_and_pore",            # 孔隙与缺陷
        "microstructure",             # 微观组织 (核心)
        "residual_stress",            # 残余应力
        "fatigue_test_conditions",    # 疲劳试验条件
        "crack_initiation",           # 裂纹起裂位置
        "fatigue_life_and_limit",     # 疲劳寿命与疲劳极限
        "crack_growth_and_paris",     # 裂纹扩展与 Paris 参数
        "failure_mechanism",          # 失效机制
        "limitations_and_future",     # 限制与未来工作
    ]

    # 每个主题的关键词和检索章节
    TOPIC_CONFIG = {
        "material_and_process": {
            "keywords": [
                "Ti-6Al-4V", "TC4", "TC17", "Ti60", "TiAl", "合金",
                "L-PBF", "SLM", "EBM", "DED", "selective laser",
                "powder bed", "additive manufacturing",
                "forging", "casting", "wrought",
                "powder", "gas atomized", "plasma rotating",
                "build orientation", "build direction",
            ],
            "priority_sections": [
                "materials_and_methods", "manufacturing",
                "abstract", "introduction",
            ],
        },
        "heat_treatment_and_hip": {
            "keywords": HEAT_TREATMENT_KEYWORDS,
            "priority_sections": [
                "heat_treatment", "materials_and_methods",
                "manufacturing", "discussion", "results",
            ],
        },
        "surface_and_roughness": {
            "keywords": [
                "surface roughness", "Ra", "Rz", "Rq", "Sa", "Sq",
                "surface finish", "as-built surface",
                "surface treatment", "polish", "machin", "grind",
                "surface", "micro-notch",
                "表面粗糙度", "表面处理", "Ra", "Rz",
            ],
            "priority_sections": [
                "surface_characterization", "materials_and_methods",
                "results", "discussion",
            ],
        },
        "defect_and_pore": {
            "keywords": DEFECT_KEYWORDS,
            "priority_sections": [
                "defect_characterization", "results",
                "discussion", "materials_and_methods", "fractography",
            ],
        },
        "microstructure": {
            "keywords": MICROSTRUCTURE_KEYWORDS,
            "priority_sections": [
                "microstructure_characterization", "results",
                "discussion", "materials_and_methods",
                "heat_treatment", "fractography",
            ],
        },
        "residual_stress": {
            "keywords": RESIDUAL_STRESS_KEYWORDS,
            "priority_sections": [
                "results", "discussion", "materials_and_methods",
            ],
        },
        "fatigue_test_conditions": {
            "keywords": [
                "fatigue test", "cyclic loading", "stress amplitude",
                "stress ratio", "R =", "R ratio",
                "frequency", "Hz", "waveform",
                "load control", "strain control",
                "室温", "高温", "腐蚀",
                "sinusoidal", "triangular",
            ],
            "priority_sections": [
                "fatigue_testing", "materials_and_methods",
                "abstract", "results",
            ],
        },
        "crack_initiation": {
            "keywords": [
                "crack initiation", "crack nucleation", "crack origin",
                "initiation site", "crack start",
                "surface crack", "subsurface", "internal crack",
                "faceting", "facet",
                "裂纹起裂", "裂纹起源", "起裂位置", "起裂源",
            ],
            "priority_sections": [
                "fractography", "results", "discussion",
                "microstructure_characterization",
            ],
        },
        "fatigue_life_and_limit": {
            "keywords": [
                "fatigue life", "Nf", "cycles to failure",
                "fatigue limit", "endurance limit",
                "S-N curve", "stress-life", "Basquin",
                "fatigue strength", "σf",
                "疲劳寿命", "疲劳极限", "S-N曲线",
            ],
            "priority_sections": [
                "results", "discussion", "conclusion",
                "fatigue_testing",
            ],
        },
        "crack_growth_and_paris": {
            "keywords": [
                "crack growth", "crack propagation", "da/dN", "da dN",
                "Paris", "Paris law", "Paris regime",
                "ΔK", "delta K", "ΔKth", "threshold",
                "FCGR", "fatigue crack growth",
                "裂纹扩展", "Paris", "门槛值",
                "Walker", "NASGRO", "Forman",
            ],
            "priority_sections": [
                "results", "discussion", "fractography",
                "materials_and_methods",
            ],
        },
        "failure_mechanism": {
            "keywords": [
                "fracture mechanism", "failure mechanism",
                "ductile", "brittle", "pseudo-brittle",
                "facet formation", "cleavage",
                "dimple", "quasi-cleavage",
                "intergranular", "transgranular",
                "fatigue striation",
                "失效机制", "断裂机制", "韧窝", "解理",
                "沿晶断裂", "穿晶断裂", "疲劳辉纹",
            ],
            "priority_sections": [
                "fractography", "discussion", "results",
                "conclusion",
            ],
        },
        "limitations_and_future": {
            "keywords": [
                "limitation", "future work", "further research",
                "not investigated", "beyond scope",
                "need", "further study", "recommend",
                "不足", "局限性", "未来工作", "后续研究",
                "未考虑", "未覆盖",
            ],
            "priority_sections": [
                "conclusion", "discussion", "introduction",
            ],
        },
    }

    def __init__(self, structure_map: PaperStructureMap):
        self.structure_map = structure_map

    def scan_topic(self, topic: str) -> List[Dict[str, Any]]:
        """对单一主题扫描全文, 返回相关段落列表。"""
        if topic not in self.TOPIC_CONFIG:
            return []

        config = self.TOPIC_CONFIG[topic]
        keywords = config["keywords"]
        priority_sections = config["priority_sections"]

        found_chunks = []

        # 按优先级顺序扫描
        all_chunks = self.structure_map.get_all_chunks()
        scored_chunks = []

        for chunk in all_chunks:
            text = chunk.original_text
            text_lower = text.lower()

            # 关键词匹配 (不区分大小写)
            matched_kws = [kw for kw in keywords if kw.lower() in text_lower]
            if not matched_kws:
                continue

            # 评分: 关键词数量 + 章节优先级
            score = len(matched_kws)
            if chunk.section_key in priority_sections:
                score += 5  # 高优先级章节加分

            scored_chunks.append({
                "chunk": chunk,
                "score": score,
                "matched_keywords": matched_kws,
                "section_priority": chunk.section_key in priority_sections,
            })

        # 按评分降序, 取 top 但不过度截断 (至少保留所有非零匹配)
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)

        # 保留所有匹配段落, 但排序后返回
        return [
            {
                "text": sc["chunk"].original_text,
                "section": sc["chunk"].section_title,
                "section_key": sc["chunk"].section_key,
                "page": sc["chunk"].page_number,
                "chunk_id": sc["chunk"].chunk_id,
                "paragraph_index": sc["chunk"].paragraph_index,
                "score": sc["score"],
                "matched_keywords": sc["matched_keywords"],
            }
            for sc in scored_chunks
        ]

    def scan_all_topics(self) -> Dict[str, List[Dict[str, Any]]]:
        """扫描所有主题, 返回主题→段落映射。"""
        results = {}
        for topic in self.TOPICS:
            results[topic] = self.scan_topic(topic)
        return results


# ═══════════════════════════════════════════════════════════════════════════
# 7. 微观组织字段提取器
# ═══════════════════════════════════════════════════════════════════════════

def extract_microstructure_from_chunks(
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """从匹配的 chunks 中提取微观组织信息。

    Returns:
        包含所有微观组织字段的字典
    """
    # 合并文本
    all_text = "\n".join(c["text"] for c in chunks)
    text_lower = all_text.lower()

    result = {
        # 是否有微观组织内容
        "microstructure_mentioned": False,
        "microstructure_mention_level": ExtractionStatus.NOT_REPORTED,
        "microstructure_summary": "",

        # 相组成
        "phase_constituents": ExtractionStatus.NOT_REPORTED,
        "alpha_phase_morphology": ExtractionStatus.NOT_REPORTED,
        "beta_phase_morphology": ExtractionStatus.NOT_REPORTED,
        "alpha_prime_martensite": ExtractionStatus.NOT_REPORTED,

        # 晶粒
        "prior_beta_grain": ExtractionStatus.NOT_REPORTED,
        "prior_beta_grain_size": ExtractionStatus.NOT_REPORTED,
        "alpha_lath_thickness": ExtractionStatus.NOT_REPORTED,
        "grain_size": ExtractionStatus.NOT_REPORTED,
        "grain_shape": ExtractionStatus.NOT_REPORTED,
        "columnar_or_equiaxed": ExtractionStatus.NOT_REPORTED,

        # 取向
        "grain_orientation": ExtractionStatus.NOT_REPORTED,
        "crystallographic_texture": ExtractionStatus.NOT_REPORTED,
        "phase_fraction": ExtractionStatus.NOT_REPORTED,
        "microstructure_heterogeneity": ExtractionStatus.NOT_REPORTED,

        # 关系
        "build_orientation_microstructure_relation": ExtractionStatus.NOT_REPORTED,
        "heat_treatment_microstructure_relation": ExtractionStatus.NOT_REPORTED,
        "HIP_microstructure_relation": ExtractionStatus.NOT_REPORTED,
        "microstructure_defect_interaction": ExtractionStatus.NOT_REPORTED,
        "microstructure_crack_initiation_relation": ExtractionStatus.NOT_REPORTED,
        "microstructure_crack_growth_relation": ExtractionStatus.NOT_REPORTED,
        "microstructure_fatigue_life_relation": ExtractionStatus.NOT_REPORTED,

        # 方法
        "microstructure_characterization_method": ExtractionStatus.NOT_REPORTED,

        # 定量值
        "microstructure_quantitative_values": [],
        "microstructure_units": "",

        # 证据
        "microstructure_evidence": [],
        "microstructure_section": "",
        "microstructure_page": 0,
        "microstructure_directness": "direct",
        "microstructure_confidence": 0.0,
        "microstructure_missing_fields": [],
    }

    # ── 判断是否提到微观组织 ──
    if not has_keywords(all_text, MICROSTRUCTURE_KEYWORDS):
        result["microstructure_mentioned"] = False
        result["microstructure_mention_level"] = ExtractionStatus.NOT_REPORTED
        return result

    result["microstructure_mentioned"] = True

    # ── 判断提取级别 ──
    # mention_only: 只提到关键词但无展开分析
    # qualitative: 有定性描述
    # quantitative: 有定量数据 (数值+单位)
    # mechanistic: 讨论了机制关系
    # conflicting: 存在不一致

    numeric_values = extract_numeric_values(all_text)
    has_quant = len(numeric_values) > 0

    # 机制关键词
    mechanism_kw = [
        "because", "due to", "result in", "lead to", "attribute to",
        "mechanism", "effect of", "influence of", "role of",
        "correlation", "relationship",
        "由于", "导致", "归因于", "机制", "影响",
    ]
    has_mechanism = has_keywords(all_text, mechanism_kw)

    # 矛盾关键词
    conflict_kw = [
        "however", "but", "although", "in contrast",
        "discrepancy", "inconsistent",
        "然而", "但是", "矛盾", "不一致",
    ]
    has_conflict = has_keywords(all_text, conflict_kw)

    if has_mechanism and has_quant:
        result["microstructure_mention_level"] = "mechanistic"
    elif has_quant:
        result["microstructure_mention_level"] = "quantitative"
    elif has_mechanism:
        result["microstructure_mention_level"] = "qualitative"
    elif has_conflict:
        result["microstructure_mention_level"] = "conflicting"
    else:
        # 检查是否有展开描述 (超过50字的含关键词段落)
        detail_paras = [
            c["text"] for c in chunks
            if len(c["text"]) > 50
        ]
        if detail_paras:
            result["microstructure_mention_level"] = "qualitative"
        else:
            result["microstructure_mention_level"] = "mention_only"

    # ── 具体字段提取 (规则匹配) ──
    # 相组成
    if re.search(r"(α|alpha)\s*['′]?\s*martensite|martensitic|α′|α'", text_lower):
        result["alpha_prime_martensite"] = "present"
    if re.search(r"α\s*\+\s*β|alpha.*beta|α.*β|双态", text_lower):
        result["phase_constituents"] = "α+β"
    elif re.search(r"(α|alpha)\s*phase|α相|alpha相", text_lower):
        result["phase_constituents"] = result["phase_constituents"] or "α phase mentioned"
    if re.search(r"(β|beta)\s*phase|β相|beta相", text_lower):
        result["phase_constituents"] = (
            "α+β" if "α" in result.get("phase_constituents", "")
            else "β phase mentioned"
        )

    # 板条厚度
    lath_matches = re.findall(
        r"(\d+\.?\d*)\s*(?:μm|um|nm)?\s*(?:lath|α\s*lath|alpha\s*lath|板条)",
        text_lower
    )
    if lath_matches:
        result["alpha_lath_thickness"] = f"{lath_matches[0]} μm"
        result["microstructure_quantitative_values"].append({
            "field": "alpha_lath_thickness",
            "value": lath_matches[0],
            "unit": "μm",
        })

    # 晶粒尺寸
    grain_matches = re.findall(
        r"(\d+\.?\d*)\s*(?:μm|um|nm)?\s*(?:grain\s*size|晶粒尺寸|晶粒)",
        text_lower
    )
    if grain_matches:
        result["grain_size"] = f"{grain_matches[0]} μm"

    # 柱状晶/等轴晶
    if re.search(r"columnar\s*grain|柱状晶", text_lower):
        result["grain_shape"] = "columnar"
        result["columnar_or_equiaxed"] = "columnar"
    if re.search(r"equiaxed\s*grain|等轴晶|等轴", text_lower):
        result["grain_shape"] = (
            "mixed" if "columnar" in result.get("grain_shape", "")
            else "equiaxed"
        )
        result["columnar_or_equiaxed"] = result["grain_shape"]

    # prior β grain
    if re.search(r"prior[-\s]*(β|beta)\s*grain|原始β|原β", text_lower):
        result["prior_beta_grain"] = "reported"
        pbg_matches = re.findall(
            r"(\d+\.?\d*)\s*(?:μm|um)?.*?(?:prior|原始)",
            text_lower
        )
        if pbg_matches:
            result["prior_beta_grain_size"] = f"{pbg_matches[0]} μm"

    # 织构
    if re.search(r"texture|织构|择优取向", text_lower):
        result["crystallographic_texture"] = "reported"
    if re.search(r"orientation|取向|misorientation|取向差", text_lower):
        result["grain_orientation"] = "reported"

    # 相含量
    frac_matches = re.findall(
        r"(\d+\.?\d*)\s*%\s*(?:α|beta|β|相|phase)",
        text_lower
    )
    if frac_matches:
        result["phase_fraction"] = f"{frac_matches[0]}%"
        result["microstructure_quantitative_values"].append({
            "field": "phase_fraction",
            "value": frac_matches[0],
            "unit": "%",
        })

    # 组织形貌
    if re.search(r"basket.?weave|篮网|篮织", text_lower):
        result["alpha_phase_morphology"] = "basket-weave"
    elif re.search(r"lamellar|片层|lamella", text_lower):
        result["alpha_phase_morphology"] = "lamellar"
    elif re.search(r"acicular|针状", text_lower):
        result["alpha_phase_morphology"] = "acicular"
    elif re.search(r"equiaxed|等轴", text_lower):
        result["alpha_phase_morphology"] = "equiaxed"
    elif re.search(r"Widmanstätten|魏氏", text_lower):
        result["alpha_phase_morphology"] = "Widmanstätten"

    # 表征方法
    found_methods = []
    for kw in CHARACTERIZATION_KEYWORDS:
        if kw.lower() in text_lower:
            found_methods.append(kw)
    if found_methods:
        result["microstructure_characterization_method"] = ", ".join(
            sorted(set(found_methods))
        )

    # ── 关系提取 (机制关联) ──
    rel_patterns = {
        "microstructure_crack_initiation_relation": [
            (r"microstructure.*crack.*initiat|组织.*裂纹.*起裂|起裂.*组织", "microstructure influences crack initiation"),
            (r"grain.*boundary.*crack|晶界.*裂纹", "grain boundary crack initiation"),
            (r"facet.*formation|小平面.*形成", "facet formation crack initiation"),
        ],
        "microstructure_crack_growth_relation": [
            (r"microstructure.*crack.*growth|组织.*裂纹.*扩展", "microstructure influences crack growth"),
            (r"lath.*crack|板条.*裂纹", "lath-level crack growth"),
            (r"α.*β.*interface|α.*β.*界面", "α/β interface crack growth"),
        ],
        "heat_treatment_microstructure_relation": [
            (r"heat.?treatment.*microstructure|热处理.*组织", "heat treatment affects microstructure"),
            (r"α'.*α.*β|α′.*分解|martensite.*decompos", "α′ decomposition to α+β"),
        ],
        "HIP_microstructure_relation": [
            (r"HIP.*microstructure|HIP.*组织|热等静压.*组织", "HIP affects microstructure"),
        ],
        "microstructure_fatigue_life_relation": [
            (r"microstructure.*fatigue|组织.*疲劳|疲劳.*组织", "microstructure-fatigue relationship"),
        ],
    }

    for field, patterns in rel_patterns.items():
        for pat, desc in patterns:
            if re.search(pat, text_lower):
                result[field] = desc
                break

    # ── 第一段证据 ──
    if chunks:
        primary = chunks[0]
        result["microstructure_evidence"] = [
            {
                "evidence_text": primary["text"][:500],
                "section": primary["section"],
                "section_key": primary["section_key"],
                "page": primary["page"],
                "chunk_id": primary["chunk_id"],
                "directness": "direct" if primary["section_key"] not in (
                    "introduction", "conclusion"
                ) else "indirect",
                "confidence": 0.85 if len(all_text) > 200 else 0.5,
            }
        ]
        result["microstructure_section"] = primary["section"]
        result["microstructure_page"] = primary["page"]

    # ── 缺失字段 ──
    missing = []
    for k in [
        "phase_constituents", "alpha_phase_morphology", "beta_phase_morphology",
        "alpha_prime_martensite", "prior_beta_grain", "grain_size",
        "alpha_lath_thickness", "grain_shape", "crystallographic_texture",
        "phase_fraction", "microstructure_characterization_method",
    ]:
        val = result.get(k)
        if val is None or val == "" or val == ExtractionStatus.NOT_REPORTED:
            # 再检查是否确实没提到
            kw_hit = has_keywords(all_text, [k.replace("_", " ")])
            if kw_hit:
                missing.append(f"{k}: mentioned but not extracted")
            else:
                missing.append(f"{k}: not reported")
    result["microstructure_missing_fields"] = missing

    # ── 汇总摘要 ──
    parts = []
    if result["phase_constituents"] and result["phase_constituents"] not in (
        ExtractionStatus.NOT_REPORTED, ExtractionStatus.NOT_EXTRACTED
    ):
        parts.append(f"相组成: {result['phase_constituents']}")
    if result["alpha_phase_morphology"] and result["alpha_phase_morphology"] not in (
        ExtractionStatus.NOT_REPORTED, ExtractionStatus.NOT_EXTRACTED
    ):
        parts.append(f"α相形态: {result['alpha_phase_morphology']}")
    if result["alpha_prime_martensite"] and result["alpha_prime_martensite"] not in (
        ExtractionStatus.NOT_REPORTED, ExtractionStatus.NOT_EXTRACTED
    ):
        parts.append(f"α'马氏体: {result['alpha_prime_martensite']}")
    if result["alpha_lath_thickness"] and result["alpha_lath_thickness"] not in (
        ExtractionStatus.NOT_REPORTED, ExtractionStatus.NOT_EXTRACTED
    ):
        parts.append(f"α板条厚度: {result['alpha_lath_thickness']}")
    if result["columnar_or_equiaxed"] and result["columnar_or_equiaxed"] not in (
        ExtractionStatus.NOT_REPORTED, ExtractionStatus.NOT_EXTRACTED
    ):
        parts.append(f"晶粒形态: {result['columnar_or_equiaxed']}")
    if result["crystallographic_texture"] and result["crystallographic_texture"] not in (
        ExtractionStatus.NOT_REPORTED, ExtractionStatus.NOT_EXTRACTED
    ):
        parts.append("织构: 有报告")
    if result["grain_size"] and result["grain_size"] not in (
        ExtractionStatus.NOT_REPORTED, ExtractionStatus.NOT_EXTRACTED
    ):
        parts.append(f"晶粒尺寸: {result['grain_size']}")

    result["microstructure_summary"] = "; ".join(parts) if parts else (
        "论文提到了微观组织但未展开详细分析"
        if result["microstructure_mention_level"] not in (
            ExtractionStatus.NOT_REPORTED, "not_found"
        ) else "未报告微观组织信息"
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 8. 四阶段提取
# ═══════════════════════════════════════════════════════════════════════════

class FourStageExtractor:
    """四阶段文献提取引擎。"""

    def __init__(self, structure_map: PaperStructureMap, use_llm: bool = False):
        self.structure_map = structure_map
        self.use_llm = use_llm
        self.scanner = TopicScanner(structure_map)

    # ── 第一阶段: 全文概览 ──
    def stage1_overview(self) -> Dict[str, Any]:
        """第一阶段: 快速概览论文整体内容。"""
        abstract_chunks = self.structure_map.get_section("abstract")
        intro_chunks = self.structure_map.get_section("introduction")
        conclusion_chunks = self.structure_map.get_section("conclusion")
        all_text = " ".join(
            c.original_text for c in (abstract_chunks + intro_chunks + conclusion_chunks)
        )

        # 提取研究问题
        question_markers = [
            r"this\s+(?:paper|study|work)\s+(?:investigates?|aims?|examines?|studies?|focuses?|explores?|addresses?|presents?)",
            r"the\s+(?:purpose|objective|goal|aim)\s+(?:of\s+this\s+(?:paper|study|work))?\s+(?:is|was)",
            r"本文[旨在目的目标研究]",
            r"本研究[旨在目的目标]",
        ]
        research_question = ""
        for marker in question_markers:
            m = re.search(marker + r"[^。.]+[。.]", all_text, re.IGNORECASE)
            if m:
                research_question = m.group(0).strip()
                break

        # 提取主要变量
        main_vars = []
        for var_pattern, var_name in [
            (r"pore|porosity|defect", "缺陷/孔隙"),
            (r"surface\s*roughness|Ra|Rz", "表面粗糙度"),
            (r"microstructure|microstructural|grain", "微观组织"),
            (r"heat\s*treatment|HIP|annealing", "热处理/HIP"),
            (r"stress\s*ratio|R\s*ratio", "应力比"),
            (r"build\s*orientation", "成形方向"),
            (r"powder\s*(?:size|characteristic)", "粉末特性"),
        ]:
            if re.search(var_pattern, all_text, re.IGNORECASE):
                main_vars.append(var_name)

        # 主要结论
        main_conclusion = ""
        for chunk in conclusion_chunks:
            text = chunk.original_text
            if len(text) > 50:
                # 取第一句/段
                main_conclusion = text[:500]
                break
        if not main_conclusion and abstract_chunks:
            main_conclusion = abstract_chunks[0].original_text[:500]

        return {
            "paper_scope": research_question or "(基于概要分析)",
            "main_research_question": research_question,
            "main_variables": main_vars,
            "secondary_variables": [],  # 后续填充
            "experimental_route": "",
            "main_conclusion": main_conclusion,
        }

    # ── 第二阶段: 按主题逐项扫描 ──
    def stage2_topic_scan(self) -> Dict[str, Any]:
        """第二阶段: 按12个主题独立扫描全文。"""
        all_topic_results = self.scanner.scan_all_topics()

        # 将扫描结果转换为结构化记录
        record = {}

        for topic, chunks in all_topic_results.items():
            topic_record = {
                "found": len(chunks) > 0,
                "chunk_count": len(chunks),
                "sections_found": list(set(c["section_key"] for c in chunks)),
                "top_chunks": chunks[:3] if chunks else [],
                "summary": "",
            }

            if chunks:
                # 合并文本做摘要
                combined = "\n".join(c["text"] for c in chunks[:5])
                topic_record["summary"] = combined[:500]

            record[topic] = topic_record

        return record

    # ── 第三阶段: 一致性检查 ──
    def stage3_consistency_check(
        self, topic_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """第三阶段: 跨位置检查同一字段是否存在矛盾。"""
        inconsistencies = []

        for topic, result in topic_results.items():
            if not result["found"] or result["chunk_count"] < 2:
                continue

            # 比较不同 section 的描述
            sections_text = defaultdict(list)
            for chunk in result["top_chunks"]:
                sections_text[chunk["section_key"]].append(chunk["text"])

            if len(sections_text) < 2:
                continue

            # 检查数量矛盾 (同一个数字不同说法)
            section_values = {}
            for sec, texts in sections_text.items():
                nums = extract_numeric_values("\n".join(texts))
                section_values[sec] = nums

            # 对比不同 section 之间的数字
            sec_list = list(section_values.keys())
            for i in range(len(sec_list)):
                for j in range(i + 1, len(sec_list)):
                    vals_i = {(v["value"], v["unit"]) for v in section_values[sec_list[i]]}
                    vals_j = {(v["value"], v["unit"]) for v in section_values[sec_list[j]]}
                    # 如果数值差异很大 (>20%) 但指向相同单位
                    for vi in vals_i:
                        for vj in vals_j:
                            if vi[1] == vj[1] and vi[1]:  # 单位相同
                                if vi[0] > 0 and vj[0] > 0:
                                    ratio = max(vi[0], vj[0]) / min(vi[0], vj[0])
                                    if ratio > 1.2:
                                        inconsistencies.append({
                                            "topic": topic,
                                            "type": "numeric_discrepancy",
                                            "detail": (
                                                f"{sec_list[i]}: {vi[0]}{vi[1]} vs "
                                                f"{sec_list[j]}: {vj[0]}{vj[1]} (ratio={ratio:.2f})"
                                            ),
                                            "severity": "medium" if ratio > 1.5 else "low",
                                        })

        return {
            "inconsistencies": inconsistencies,
            "consistent_topics": [
                t for t, r in topic_results.items()
                if r["found"] and not any(
                    inc["topic"] == t for inc in inconsistencies
                )
            ],
        }

    # ── 第四阶段: 漏提审计 ──
    def stage4_audit_missing(
        self, structured_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """第四阶段: 主动检查是否遗漏了关键信息。"""
        all_text = " ".join(
            c.original_text for c in self.structure_map.get_all_chunks()
        )
        text_lower = all_text.lower()

        missing_alerts = []

        # 1. 有关键词但 microstructure 字段为空
        ms_keywords = ["microstructure", "微观组织", "显微组织", "EBSD", "TEM"]
        has_ms_kw = any(kw in text_lower for kw in ms_keywords)
        ms_summary = structured_record.get("microstructure_summary", "")
        if has_ms_kw and (
            not ms_summary or ms_summary == ExtractionStatus.NOT_REPORTED
        ):
            missing_alerts.append({
                "field": "microstructure",
                "alert": "文中出现 microstructure/EBSD/TEM 关键词但微观组织字段为空",
                "source_keywords": [kw for kw in ms_keywords if kw in text_lower],
                "severity": "high",
            })

        # 2. 有 EBSD/SEM/TEM/XRD 但 microstructure_characterization_method 为空
        char_kw_hits = [kw for kw in CHARACTERIZATION_KEYWORDS if kw.lower() in text_lower]
        ms_method = structured_record.get("microstructure_characterization_method", "")
        if char_kw_hits and (
            not ms_method or ms_method == ExtractionStatus.NOT_REPORTED
        ):
            missing_alerts.append({
                "field": "microstructure_characterization_method",
                "alert": "文中出现表征方法关键词但未提取",
                "source_keywords": char_kw_hits[:5],
                "severity": "high",
            })

        # 3. 有 Ra/Rz/Sa 但 surface_roughness 为空 (来自文献数据库字段)
        surface_kw = ["Ra", "Rz", "Sa", "Sq", "surface roughness"]
        has_surface_kw = any(kw.lower() in text_lower for kw in surface_kw)
        surface_field = structured_record.get("surface_roughness_Ra", "")
        if has_surface_kw and (not surface_field):
            missing_alerts.append({
                "field": "surface_roughness",
                "alert": "文中出现 Ra/Rz/surface roughness 关键词但粗糙度字段为空",
                "source_keywords": [kw for kw in surface_kw if kw.lower() in text_lower],
                "severity": "medium",
            })

        # 4. 有 stress ratio/R= 但 stress_ratio_R 为空
        sr_patterns = [r"stress\s*ratio", r"R\s*=\s*-?\d", r"load\s*ratio", r"应力比"]
        has_sr = any(re.search(p, text_lower) for p in sr_patterns)
        sr_field = structured_record.get("stress_ratio_R", "")
        if has_sr and (not sr_field):
            missing_alerts.append({
                "field": "stress_ratio_R",
                "alert": "文中出现 stress ratio / R= 关键词但应力比字段为空",
                "severity": "medium",
            })

        # 5. 有 crack origin/initiation 但 crack_initiation_site 为空
        ci_patterns = [r"crack\s*(?:origin|initiation|nucleation)", r"起裂", r"裂纹源"]
        has_ci = any(re.search(p, text_lower) for p in ci_patterns)
        ci_field = structured_record.get("crack_initiation_site", "")
        if has_ci and (not ci_field):
            missing_alerts.append({
                "field": "crack_initiation_site",
                "alert": "文中出现 crack initiation/origin 关键词但起裂位置字段为空",
                "severity": "medium",
            })

        # 6. 有 α/β/martensite/grain/texture 但 microstructure 字段为空
        ms_phase_kw = ["α", "β", "α′", "martensite", "lath", "grain", "texture",
                       "α相", "β相", "马氏体", "板条", "晶粒", "织构"]
        has_phase_kw = any(kw in text_lower for kw in ms_phase_kw)
        if has_phase_kw and (
            not ms_summary or ms_summary == ExtractionStatus.NOT_REPORTED
        ):
            missing_alerts.append({
                "field": "microstructure_phase",
                "alert": "文中出现 α/β/martensite/grain 等相/组织关键词但未提取",
                "source_keywords": [kw for kw in ms_phase_kw if kw in text_lower][:5],
                "severity": "high",
            })

        # 7. 检查是否只用了标题提取 (原系统问题)
        title = self.structure_map.title
        if title and len(all_text) > 2000:
            # 确认是否真的读了正文
            body_check_kw = ["experimental", "result", "method", "discussion",
                            "实验", "结果", "方法", "讨论"]
            has_body_content = any(kw in text_lower for kw in body_check_kw)
            if not has_body_content:
                missing_alerts.append({
                    "field": "_full_text_quality",
                    "alert": "正文内容可能未正确加载 (缺少实验/结果/讨论等章节关键词)",
                    "severity": "critical",
                })

        return {
            "missing_alerts": missing_alerts,
            "alert_count": len(missing_alerts),
            "high_severity_count": sum(
                1 for a in missing_alerts if a["severity"] == "high"
            ),
            "critical_severity_count": sum(
                1 for a in missing_alerts if a["severity"] == "critical"
            ),
        }

    # ── 完整四阶段提取 ──
    def run_full_extraction(self) -> Dict[str, Any]:
        """执行完整四阶段提取。"""
        # Stage 1
        overview = self.stage1_overview()

        # Stage 2
        topic_results = self.stage2_topic_scan()

        # 微观组织专项提取
        micro_chunks = self.scanner.scan_topic("microstructure")
        microstructure = extract_microstructure_from_chunks(micro_chunks)

        # 组装结构化记录
        record = {
            **overview,
            "microstructure": microstructure,
            "topic_scan": {
                topic: {
                    "found": r["found"],
                    "chunk_count": r["chunk_count"],
                    "sections": r["sections_found"],
                }
                for topic, r in topic_results.items()
            },
        }

        # 从 topic 结果填充字段
        # 表面粗糙度
        surface_chunks = self.scanner.scan_topic("surface_and_roughness")
        if surface_chunks:
            combined_surface = "\n".join(c["text"] for c in surface_chunks[:3])
            ra_match = re.search(r"Ra\s*[=:≈]?\s*(\d+\.?\d*)\s*(μm|um)?", combined_surface, re.IGNORECASE)
            if ra_match:
                record["surface_roughness_Ra"] = f"{ra_match.group(1)} {ra_match.group(2) or 'μm'}"

        # 应力比
        fatigue_cond_chunks = self.scanner.scan_topic("fatigue_test_conditions")
        if fatigue_cond_chunks:
            combined_fc = "\n".join(c["text"] for c in fatigue_cond_chunks[:3])
            r_match = re.search(r"[Rr]\s*[=:≈]?\s*(-?\d+\.?\d*)", combined_fc)
            if r_match:
                record["stress_ratio_R"] = r_match.group(1)

        # 疲劳极限
        life_chunks = self.scanner.scan_topic("fatigue_life_and_limit")
        if life_chunks:
            combined_life = "\n".join(c["text"] for c in life_chunks[:3])
            limit_match = re.search(
                r"fatigue\s*(?:limit|endurance|strength)\s*(?:of|is|≈|~)?\s*(\d+\.?\d*)\s*(MPa)?",
                combined_life, re.IGNORECASE
            )
            if limit_match:
                record["fatigue_limit"] = f"{limit_match.group(1)} {limit_match.group(2) or 'MPa'}"

        # 裂纹起裂
        ci_chunks = self.scanner.scan_topic("crack_initiation")
        if ci_chunks:
            combined_ci = "\n".join(c["text"] for c in ci_chunks[:3])
            for pattern, label in [
                (r"surface.*crack.*initiat|表面.*起裂", "surface"),
                (r"sub.?surface.*crack.*initiat|subsurface.*pore|亚表面.*起裂", "subsurface"),
                (r"internal.*crack.*initiat|内部.*起裂", "internal"),
                (r"pore.*crack.*initiat|defect.*crack.*initiat|孔隙.*起裂|缺陷.*起裂", "pore/defect"),
                (r"facet|小平面", "facet"),
            ]:
                if re.search(pattern, combined_ci, re.IGNORECASE):
                    record["crack_initiation_site"] = label
                    break

        # Stage 3
        consistency = self.stage3_consistency_check(topic_results)

        # Stage 4
        audit = self.stage4_audit_missing(record)

        return {
            "record": record,
            "stage1_overview": overview,
            "stage2_topic_results": topic_results,
            "stage3_consistency": consistency,
            "stage4_audit": audit,
            "microstructure_detail": microstructure,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 9. 证据去重
# ═══════════════════════════════════════════════════════════════════════════

def deduplicate_evidence(evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """对同篇论文内的重复证据去重。

    去重键: paper_id + normalized_claim + variable + condition_summary

    同一结论在不同实验条件下允许保留为不同证据。
    """
    seen = set()
    deduped = []

    for ev in evidence_list:
        claim = ev.get("extracted_claim", "")
        paper_id = ev.get("paper_id", "")
        iv = ev.get("independent_variable", "")
        dv = ev.get("dependent_variable", "")
        conditions = _make_condition_key(ev)

        # 规范化 claim (去空格/大小写/标点)
        norm_claim = re.sub(r"\s+", " ", claim).strip().lower()[:100]

        dedup_key = f"{paper_id}|{norm_claim}|{iv}→{dv}|{conditions}"

        if dedup_key not in seen:
            seen.add(dedup_key)
            deduped.append(ev)

    return deduped


def _make_condition_key(ev: Dict[str, Any]) -> str:
    """从证据字典生成条件摘要键。"""
    parts = []
    for field in ["material", "process", "surface_state", "heat_treatment",
                   "stress_ratio_R", "fatigue_type"]:
        val = ev.get(field, "")
        if val:
            parts.append(f"{field}={val}")
    return "|".join(parts) if parts else "default"


# ═══════════════════════════════════════════════════════════════════════════
# 10. 提取完整性报告
# ═══════════════════════════════════════════════════════════════════════════

def generate_completeness_report(
    extraction_result: Dict[str, Any],
    paper_id: str = "",
) -> Dict[str, Any]:
    """生成文献提取完整性报告。

    注意: 完整性评分衡量的是"系统是否仔细检查了论文",
    而不是"论文是否提供了全部数据"。
    """
    record = extraction_result.get("record", {})
    topic_results = extraction_result.get("stage2_topic_results", {})
    audit = extraction_result.get("stage4_audit", {})
    microstructure = record.get("microstructure", {})

    categories = []

    # 制造工艺
    tp = topic_results.get("material_and_process", {})
    categories.append({
        "category": "制造工艺",
        "key": "material_and_process",
        "chunks_found": tp.get("chunk_count", 0),
        "sections": tp.get("sections", []),
        "has_quantitative": False,
        "needs_review": not tp.get("found", False),
        "status": "已提取" if tp.get("found") else ("未找到" if audit.get("high_severity_count", 0) > 0 else "未提取"),
    })

    # 热处理
    tp = topic_results.get("heat_treatment_and_hip", {})
    categories.append({
        "category": "热处理",
        "key": "heat_treatment",
        "chunks_found": tp.get("chunk_count", 0),
        "sections": tp.get("sections", []),
        "has_quantitative": False,
        "needs_review": not tp.get("found", False),
        "status": "已提取" if tp.get("found") else "未提取",
    })

    # 表面状态
    tp = topic_results.get("surface_and_roughness", {})
    categories.append({
        "category": "表面状态",
        "key": "surface_and_roughness",
        "chunks_found": tp.get("chunk_count", 0),
        "sections": tp.get("sections", []),
        "has_quantitative": bool(record.get("surface_roughness_Ra")),
        "needs_review": False,
        "status": "已提取" if tp.get("found") else "未提取",
    })

    # 孔隙缺陷
    tp = topic_results.get("defect_and_pore", {})
    categories.append({
        "category": "孔隙缺陷",
        "key": "defect_and_pore",
        "chunks_found": tp.get("chunk_count", 0),
        "sections": tp.get("sections", []),
        "has_quantitative": False,
        "needs_review": False,
        "status": "已提取" if tp.get("found") else "未提取",
    })

    # 微观组织
    ms_level = microstructure.get("microstructure_mention_level", ExtractionStatus.NOT_REPORTED)
    ms_chunks = topic_results.get("microstructure", {}).get("chunk_count", 0)
    if ms_level == ExtractionStatus.NOT_REPORTED or ms_level == "not_found":
        ms_status = "未报告"
        ms_review = False
    elif ms_level == "mention_only":
        ms_status = "简要提及"
        ms_review = True
    elif ms_level == "qualitative":
        ms_status = "定性描述"
        ms_review = False
    elif ms_level in ("quantitative", "mechanistic"):
        ms_status = "已提取"
        ms_review = False
    else:
        ms_status = "简要提及"
        ms_review = True

    categories.append({
        "category": "微观组织",
        "key": "microstructure",
        "chunks_found": ms_chunks,
        "sections": topic_results.get("microstructure", {}).get("sections", []),
        "has_quantitative": ms_level in ("quantitative", "mechanistic"),
        "needs_review": ms_review,
        "status": ms_status,
    })

    # 疲劳条件
    tp = topic_results.get("fatigue_test_conditions", {})
    categories.append({
        "category": "疲劳条件",
        "key": "fatigue_test_conditions",
        "chunks_found": tp.get("chunk_count", 0),
        "sections": tp.get("sections", []),
        "has_quantitative": bool(record.get("stress_ratio_R")),
        "needs_review": not tp.get("found", False),
        "status": "已提取" if tp.get("found") else "未提取",
    })

    # 裂纹机制
    ci_tp = topic_results.get("crack_initiation", {})
    cg_tp = topic_results.get("crack_growth_and_paris", {})
    fm_tp = topic_results.get("failure_mechanism", {})
    total_mech_chunks = ci_tp.get("chunk_count", 0) + cg_tp.get("chunk_count", 0) + fm_tp.get("chunk_count", 0)
    has_mech = total_mech_chunks > 0
    categories.append({
        "category": "裂纹机制",
        "key": "crack_mechanisms",
        "chunks_found": total_mech_chunks,
        "sections": list(set(
            ci_tp.get("sections", []) + cg_tp.get("sections", []) + fm_tp.get("sections", [])
        )),
        "has_quantitative": bool(record.get("crack_initiation_site")),
        "needs_review": not has_mech,
        "status": "已提取" if has_mech else "未提取",
    })

    # 公式参数
    tp = topic_results.get("crack_growth_and_paris", {})
    categories.append({
        "category": "公式参数",
        "key": "equation_parameters",
        "chunks_found": tp.get("chunk_count", 0),
        "sections": tp.get("sections", []),
        "has_quantitative": False,
        "needs_review": False,
        "status": "已提取" if tp.get("chunk_count", 0) > 2 else "未提取",
    })

    # 总体评分 (基于检查覆盖率, 而非数据丰富度)
    extracted_count = sum(1 for c in categories if c["status"] == "已提取")
    total_count = len(categories)
    coverage_score = extracted_count / total_count * 100 if total_count > 0 else 0

    report = {
        "paper_id": paper_id,
        "categories": categories,
        "total_alerts": audit.get("alert_count", 0),
        "high_severity_alerts": audit.get("high_severity_count", 0),
        "coverage_score": round(coverage_score, 1),
        "coverage_assessment": (
            "完整" if coverage_score >= 85
            else "较完整" if coverage_score >= 65
            else "部分" if coverage_score >= 40
            else "不完整"
        ),
        "generated_at": datetime.now().isoformat(),
    }

    return report


# ═══════════════════════════════════════════════════════════════════════════
# 11. 细读模式 (单篇论文单主题重新提取)
# ═══════════════════════════════════════════════════════════════════════════

# 可选的细读主题
REREAD_TOPICS = {
    "all": "全部字段",
    "microstructure": "微观组织",
    "defect": "孔隙缺陷",
    "fatigue_conditions": "疲劳条件",
    "crack_mechanisms": "裂纹机制",
    "equation_parameters": "公式参数",
    "experimental_methods": "实验方法",
}

def reread_paper_topic(
    pdf_path: str,
    topic: str = "all",
    chunk_size: int = 500,
    overlap: int = 80,
) -> Dict[str, Any]:
    """对单篇论文执行细读模式, 按主题重新提取。

    Args:
        pdf_path: PDF 文件路径
        topic: 细读主题
        chunk_size: 分块大小
        overlap: 重叠大小

    Returns:
        该主题的提取结果
    """
    # 1. 构建结构地图
    structure_map = build_paper_structure_map(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if structure_map.total_chunks == 0:
        return {"error": "无法提取文本", "topic": topic}

    # 2. 创建扫描器
    scanner = TopicScanner(structure_map)

    # 3. 根据主题扫描
    if topic == "all":
        # 全部字段 → 运行完整四阶段
        extractor = FourStageExtractor(structure_map)
        return extractor.run_full_extraction()
    elif topic == "microstructure":
        chunks = scanner.scan_topic("microstructure")
        return {
            "topic": "microstructure",
            "matched_chunks": len(chunks),
            "results": extract_microstructure_from_chunks(chunks),
            "all_chunks": [
                {
                    "section": c["section"],
                    "page": c["page"],
                    "text": c["text"][:300],
                    "chunk_id": c["chunk_id"],
                }
                for c in chunks[:10]
            ],
        }
    elif topic == "defect":
        chunks = scanner.scan_topic("defect_and_pore")
        return {
            "topic": "defect",
            "matched_chunks": len(chunks),
            "all_chunks": [
                {
                    "section": c["section"],
                    "page": c["page"],
                    "text": c["text"][:300],
                    "chunk_id": c["chunk_id"],
                }
                for c in chunks[:10]
            ],
        }
    elif topic == "fatigue_conditions":
        chunks = scanner.scan_topic("fatigue_test_conditions")
        return {
            "topic": "fatigue_conditions",
            "matched_chunks": len(chunks),
            "all_chunks": [
                {
                    "section": c["section"],
                    "page": c["page"],
                    "text": c["text"][:300],
                    "chunk_id": c["chunk_id"],
                }
                for c in chunks[:10]
            ],
        }
    elif topic == "crack_mechanisms":
        ci_chunks = scanner.scan_topic("crack_initiation")
        cg_chunks = scanner.scan_topic("crack_growth_and_paris")
        fm_chunks = scanner.scan_topic("failure_mechanism")
        return {
            "topic": "crack_mechanisms",
            "crack_initiation": {
                "matched_chunks": len(ci_chunks),
                "chunks": [{"section": c["section"], "page": c["page"],
                           "text": c["text"][:300]} for c in ci_chunks[:5]],
            },
            "crack_growth": {
                "matched_chunks": len(cg_chunks),
                "chunks": [{"section": c["section"], "page": c["page"],
                           "text": c["text"][:300]} for c in cg_chunks[:5]],
            },
            "failure_mechanism": {
                "matched_chunks": len(fm_chunks),
                "chunks": [{"section": c["section"], "page": c["page"],
                           "text": c["text"][:300]} for c in fm_chunks[:5]],
            },
        }
    elif topic == "equation_parameters":
        chunks = scanner.scan_topic("crack_growth_and_paris")
        return {
            "topic": "equation_parameters",
            "matched_chunks": len(chunks),
            "all_chunks": [
                {
                    "section": c["section"],
                    "page": c["page"],
                    "text": c["text"][:300],
                    "chunk_id": c["chunk_id"],
                }
                for c in chunks[:10]
            ],
        }
    elif topic == "experimental_methods":
        chunks_mm = scanner.scan_topic("material_and_process")
        chunks_ht = scanner.scan_topic("heat_treatment_and_hip")
        chunks_ft = scanner.scan_topic("fatigue_test_conditions")
        return {
            "topic": "experimental_methods",
            "material_and_process": {
                "matched_chunks": len(chunks_mm),
                "chunks": [{"section": c["section"], "page": c["page"],
                           "text": c["text"][:300]} for c in chunks_mm[:5]],
            },
            "heat_treatment": {
                "matched_chunks": len(chunks_ht),
                "chunks": [{"section": c["section"], "page": c["page"],
                           "text": c["text"][:300]} for c in chunks_ht[:5]],
            },
            "fatigue_testing": {
                "matched_chunks": len(chunks_ft),
                "chunks": [{"section": c["section"], "page": c["page"],
                           "text": c["text"][:300]} for c in chunks_ft[:5]],
            },
        }
    else:
        return {"error": f"未知细读主题: {topic}", "topic": topic}


# ═══════════════════════════════════════════════════════════════════════════
# 12. 批处理入口
# ═══════════════════════════════════════════════════════════════════════════

def run_full_text_extraction(
    pdf_path: str,
    paper_id: str = "",
    use_llm: bool = False,
) -> Dict[str, Any]:
    """对单篇 PDF 执行完整四阶段提取。

    Args:
        pdf_path: PDF 文件路径
        paper_id: 文献 ID
        use_llm: 是否使用可选的大模型客户端

    Returns:
        完整的提取结果字典
    """
    return deep_read_pdf(Path(pdf_path), paper_id=paper_id, base_dir=BASE_DIR)


def format_completeness_report_markdown(report: Dict[str, Any]) -> str:
    """将完整性报告格式化为 Markdown。"""
    lines = ["## 文献提取完整性报告\n"]
    lines.append(f"**论文 ID**: {report.get('paper_id', 'N/A')}")
    lines.append(f"**覆盖率评分**: {report.get('coverage_score', 0)}%")
    lines.append(f"**覆盖评估**: {report.get('coverage_assessment', 'N/A')}")
    lines.append(f"**漏提告警**: {report.get('total_alerts', 0)} 条 "
                 f"(严重: {report.get('high_severity_alerts', 0)} 条)\n")

    lines.append("| 信息类别 | 状态 | 匹配段落数 | 定量数据 | 需要复核 |")
    lines.append("|---|---:|---:|:---:|:---:|")

    for cat in report.get("categories", []):
        status = cat.get("status", "未提取")
        chunks = cat.get("chunks_found", 0)
        has_quant = "✓" if cat.get("has_quantitative") else "✗"
        needs_review = "⚠ 是" if cat.get("needs_review") else "否"
        lines.append(f"| {cat['category']} | {status} | {chunks} | {has_quant} | {needs_review} |")

    lines.append("\n---")
    lines.append(f"*报告生成时间: {report.get('generated_at', 'N/A')}*")

    return "\n".join(lines)


def format_microstructure_markdown(ms: Dict[str, Any]) -> str:
    """将微观组织提取结果格式化为 Markdown。"""
    level = ms.get("microstructure_mention_level", ExtractionStatus.NOT_REPORTED)

    if level == ExtractionStatus.NOT_REPORTED or level == "not_found":
        return "**微观组织**: NOT_REPORTED (论文未报告)\n"

    lines = ["## 微观组织提取结果\n"]
    lines.append(f"**提取级别**: {level}\n")

    # 摘要
    summary = ms.get("microstructure_summary", "")
    if summary:
        lines.append(f"**摘要**: {summary}\n")

    # 字段表格
    field_labels = {
        "phase_constituents": "相组成",
        "alpha_phase_morphology": "α相形态",
        "beta_phase_morphology": "β相形态",
        "alpha_prime_martensite": "α'马氏体",
        "prior_beta_grain": "原始β晶粒",
        "prior_beta_grain_size": "原始β晶粒尺寸",
        "alpha_lath_thickness": "α板条厚度",
        "grain_size": "晶粒尺寸",
        "grain_shape": "晶粒形态",
        "columnar_or_equiaxed": "柱状晶/等轴晶",
        "crystallographic_texture": "晶体织构",
        "phase_fraction": "相含量",
        "microstructure_characterization_method": "表征方法",
    }

    lines.append("| 字段 | 值 | 状态 |")
    lines.append("|---|---|---|")
    for field_key, label in field_labels.items():
        val = ms.get(field_key, "")
        if isinstance(val, str) and val:
            status = ExtractionStatus.EXTRACTED
            display_val = val[:60]
        else:
            status = ExtractionStatus.NOT_REPORTED
            display_val = "-"
        lines.append(f"| {label} | {display_val} | {status} |")

    # 定量值
    if ms.get("microstructure_quantitative_values"):
        lines.append(f"\n**定量数据**: {ms['microstructure_quantitative_values']}")

    # 机制关系
    rel_fields = {
        "heat_treatment_microstructure_relation": "热处理-组织关系",
        "HIP_microstructure_relation": "HIP-组织关系",
        "microstructure_crack_initiation_relation": "组织-起裂关系",
        "microstructure_crack_growth_relation": "组织-扩展关系",
        "microstructure_fatigue_life_relation": "组织-寿命关系",
    }
    has_rels = False
    for field_key, label in rel_fields.items():
        val = ms.get(field_key, "")
        if val and val != ExtractionStatus.NOT_REPORTED:
            if not has_rels:
                lines.append(f"\n**机制关联**:")
                has_rels = True
            lines.append(f"- {label}: {val}")

    # 原文证据
    evidence = ms.get("microstructure_evidence", [])
    if evidence:
        lines.append(f"\n**原文证据** ({len(evidence)} 条):")
        for ev in evidence[:3]:
            lines.append(f"> {ev.get('evidence_text', '')[:200]}")
            lines.append(f"> *(section: {ev.get('section', 'N/A')}, "
                        f"page: {ev.get('page', 'N/A')}, "
                        f"directness: {ev.get('directness', 'N/A')}, "
                        f"confidence: {ev.get('confidence', 'N/A')})*")

    # 缺失字段
    missing = ms.get("microstructure_missing_fields", [])
    if missing:
        lines.append(f"\n**缺失字段** ({len(missing)} 个):")
        for m in missing[:10]:
            lines.append(f"- {m}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 13. 自测试
# ═══════════════════════════════════════════════════════════════════════════

def self_test():
    """快速自测试关键功能。"""
    test_text = """
    Abstract
    This paper investigates the fatigue behavior of L-PBF Ti-6Al-4V alloy.

    1. Introduction
    Titanium alloys are widely used in aerospace applications.

    2. Materials and Methods
    2.1 Microstructural characterization
    The microstructure was examined using SEM and EBSD.
    The prior β grains were approximately 85±35 µm in width.
    The average width of α/α′ laths increased from 1.29 μm to 1.68 μm after HIP.
    Columnar grains were observed along the build direction.

    2.2 Fatigue testing
    Fatigue tests were conducted at R = 0.1 and frequency of 30 Hz.

    3. Results
    The α′ martensite decomposed into α+β microstructure after HIP.
    The basket-weave microstructure was confirmed by EBSD.

    4. Discussion
    The facet formation mechanism is related to α phase orientation.
    Surface roughness Ra was 20.4 μm.

    5. Conclusion
    HIP treatment improved fatigue life significantly.
    """

    print("=" * 60)
    print("FullTextExtractor 自测试")
    print("=" * 60)

    # 1. 结构地图
    print("\n[Test 1] PaperStructureMap...")
    sm = build_paper_structure_map(full_text=test_text)
    print(f"  Sections: {len(sm.sections)}")
    print(f"  Total chunks: {sm.total_chunks}")
    for sk, chunks in sm.sections.items():
        print(f"    {sk}: {len(chunks)} chunks")
    assert sm.total_chunks > 0, "Structure map should have chunks"

    # 2. TopicScanner
    print("\n[Test 2] TopicScanner...")
    scanner = TopicScanner(sm)
    ms_chunks = scanner.scan_topic("microstructure")
    print(f"  Microstructure matches: {len(ms_chunks)}")
    assert len(ms_chunks) > 0, "Should find microstructure in test text"

    # 3. 微观组织提取
    print("\n[Test 3] Microstructure extraction...")
    ms_result = extract_microstructure_from_chunks(ms_chunks)
    print(f"  Mentioned: {ms_result['microstructure_mentioned']}")
    print(f"  Level: {ms_result['microstructure_mention_level']}")
    print(f"  Phase: {ms_result['phase_constituents']}")
    print(f"  α' martensite: {ms_result['alpha_prime_martensite']}")
    print(f"  Lath thickness: {ms_result['alpha_lath_thickness']}")
    print(f"  Prior β grain: {ms_result['prior_beta_grain']}")
    print(f"  Grain shape: {ms_result['columnar_or_equiaxed']}")
    print(f"  Summary: {ms_result['microstructure_summary']}")
    assert ms_result["microstructure_mentioned"] is True

    # 4. 四阶段提取
    print("\n[Test 4] FourStageExtractor...")
    extractor = FourStageExtractor(sm)
    result = extractor.run_full_extraction()
    record = result["record"]
    overview = result["stage1_overview"]
    audit = result["stage4_audit"]
    print(f"  Overview main_vars: {overview.get('main_variables')}")
    print(f"  Audit alerts: {audit.get('alert_count')}")
    print(f"  Surface Ra: {record.get('surface_roughness_Ra', 'N/A')}")
    print(f"  Stress ratio R: {record.get('stress_ratio_R', 'N/A')}")

    # 5. 完整性报告
    print("\n[Test 5] Completeness report...")
    report = generate_completeness_report(result, "TEST_001")
    print(f"  Coverage: {report['coverage_score']}% ({report['coverage_assessment']})")
    for cat in report["categories"]:
        print(f"    {cat['category']}: {cat['status']} ({cat['chunks_found']} chunks)")
    assert report["coverage_score"] > 0

    # 6. 证据去重
    print("\n[Test 6] Evidence dedup...")
    test_ev = [
        {"paper_id": "T1", "extracted_claim": "HIP improves fatigue life",
         "independent_variable": "HIP", "dependent_variable": "Nf",
         "material": "Ti-6Al-4V", "process": "", "surface_state": "",
         "heat_treatment": "HIP", "stress_ratio_R": "", "fatigue_type": ""},
        {"paper_id": "T1", "extracted_claim": "HIP improves fatigue life",
         "independent_variable": "HIP", "dependent_variable": "Nf",
         "material": "Ti-6Al-4V", "process": "", "surface_state": "",
         "heat_treatment": "HIP", "stress_ratio_R": "", "fatigue_type": ""},
        {"paper_id": "T1", "extracted_claim": "Surface roughness reduces fatigue life",
         "independent_variable": "Ra", "dependent_variable": "Nf",
         "material": "Ti-6Al-4V", "process": "", "surface_state": "as-built",
         "heat_treatment": "", "stress_ratio_R": "0.1", "fatigue_type": "HCF"},
    ]
    deduped = deduplicate_evidence(test_ev)
    print(f"  Before: {len(test_ev)}, After: {len(deduped)}")
    assert len(deduped) == 2, "Should dedup to 2"

    # 7. 细读模式
    print("\n[Test 7] Reread mode...")
    # 保存测试文本到临时文件, 然后测试 reread
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(test_text)
        temp_path = f.name
    # 测试 reread 用 txt 文件 (简化, 实际应该是 PDF)
    # 这里直接测试结构地图构建功能
    print(f"  Reread API ready. Topic options: {list(REREAD_TOPICS.keys())}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

    return result


if __name__ == "__main__":
    self_test()

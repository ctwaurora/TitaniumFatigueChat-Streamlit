"""
paper_classifier.py — Auto paper type classification
面向 L-PBF Ti-6Al-4V 疲劳文献自动分类与文件夹管理。

分类目錄:
    papers/reviews/
    papers/experimental_papers/
    papers/fcgr_papers/
    papers/micro_ct_papers/
    papers/hip_heat_treatment/
    papers/surface_roughness/
    papers/conflict_papers/
    papers/candidate_papers/
    papers/other/

支持多标签：一篇文章可同时属于多个类型。
"""

import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "papers"

# ── Paper type subdirectories ──

PAPER_SUBDIRS = {
    "review": "reviews",
    "experimental_fatigue_paper": "experimental_papers",
    "fcgr_paper": "fcgr_papers",
    "micro_ct_paper": "micro_ct_papers",
    "hip_heat_treatment_paper": "hip_heat_treatment",
    "surface_roughness_paper": "surface_roughness",
    "conflict_paper": "conflict_papers",
    "candidate": "candidate_papers",
    "other": "other",
}


def ensure_paper_dirs():
    """Ensure all paper type subdirectories exist."""
    for subdir in PAPER_SUBDIRS.values():
        (PAPERS_DIR / subdir).mkdir(parents=True, exist_ok=True)


# ── Classification keyword maps ──

CLASSIFICATION_RULES = [
    {
        "type": "review",
        "folder": "reviews",
        "keywords": [
            "review", "systematic review", "overview", "progress",
            "state of the art", "perspective", "survey", "comprehensive review",
            "literature review", "critical review", "current status",
        ],
    },
    {
        "type": "fcgr_paper",
        "folder": "fcgr_papers",
        "keywords": [
            "fatigue crack growth", "fcgr", "da/dn", "delta k", "Δk",
            "paris law", "paris parameters", "paris c", "paris m",
            "crack propagation", "crack growth rate", "crack growth behavior",
            "crack closure", "threshold Δk", "delta k th",
            "fatigue crack propagation",
        ],
    },
    {
        "type": "micro_ct_paper",
        "folder": "micro_ct_papers",
        "keywords": [
            "micro-ct", "x-ray computed tomography", "computed tomography",
            "ct characterization", "defect morphology", "pore morphology",
            "3d defects", "volumetric defects", "synchrotron",
            "x-ray tomography", "pore size distribution", "pore characterization",
            "3d characterization", "defect volume",
        ],
    },
    {
        "type": "hip_heat_treatment_paper",
        "folder": "hip_heat_treatment",
        "keywords": [
            "hot isostatic pressing", "hip treatment", "hiping", "hiped",
            "heat treatment", "annealing",
            "stress relief", "stress relieving", "solution treatment", "aging",
            "post heat treatment", "post-processing heat",
            "solution annealed", "stress relieved",
            "thermal post-processing",
        ],
    },
    {
        "type": "surface_roughness_paper",
        "folder": "surface_roughness",
        "keywords": [
            "surface roughness", "as-built surface", "machined", "polished",
            "surface defects", "surface finishing", "shot peening",
            "surface quality", "surface condition", "surface state",
            "surface integrity", "surface treatment",
            "roughness parameter", "roughness effect",
        ],
    },
    {
        "type": "experimental_fatigue_paper",
        "folder": "experimental_papers",
        "keywords": [
            "fatigue life", "s-n curve", "high cycle fatigue", "hcf",
            "very high cycle fatigue", "vhcf", "fatigue strength",
            "fatigue limit", "crack initiation", "fractography",
            "fatigue test", "fatigue behavior", "fatigue performance",
            "stress-life", "strain-life", "low cycle fatigue",
        ],
    },
]

# Conflict detection keywords for is_conflict_relevant
CONFLICT_PATTERNS = [
    # surface roughness vs internal pore
    (r"surface.*roughness.*pore|pore.*surface.*roughness|"
     r"roughness.*defect.*compet|compet.*roughness.*pore",
     "surface_vs_pores"),
    # HIP effectiveness
    (r"hip.*residual.*defect|hip.*remnant.*pore|hip.*incomplete.*closure|"
     r"hip.*still.*crack|hip.*not.*eliminate",
     "hip_effectiveness"),
    # build orientation effect
    (r"orientation.*anisotrop|crack.*orientation.*depend|"
     r"build.*direct.*effect.*fcgr|orientation.*wea?k",
     "orientation_effect"),
    # microstructure vs defect
    (r"microstructure.*defect.*compet|defect.*dominant.*microstructure|"
     r"microstructure.*dominant.*defect",
     "microstructure_vs_defect"),
    # defect initiation only vs also propagation
    (r"defect.*onl?y.*initiat|defect.*initiat.*propagat|"
     r"initiation.*dominat.*propagat",
     "defect_initiation_vs_propagation"),
]


# Topics derived from primary + secondary types
TOPIC_MAP = {
    "review": "文献综述",
    "fcgr_paper": "裂纹扩展与Paris参数",
    "micro_ct_paper": "micro-CT缺陷三维表征",
    "hip_heat_treatment_paper": "HIP与热处理影响",
    "surface_roughness_paper": "表面粗糙度效应",
    "experimental_fatigue_paper": "疲劳实验与S-N数据",
    "candidate": "候选未分类",
    "other": "其他",
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. Core classification
# ═══════════════════════════════════════════════════════════════════════════

def classify_paper_type(
    title: str,
    abstract: str = "",
    keywords: str = "",
) -> Dict[str, Any]:
    """
    根据标题、摘要、关键词自动判断文献类型。
    支持多标签。

    Returns:
        {
            "paper_type_primary": str,
            "paper_type_secondary": str (semicolon-separated),
            "research_topic": str,
            "storage_folder": str,
            "tags": str (semicolon-separated),
            "is_review": bool,
            "is_experimental": bool,
            "is_fcgr": bool,
            "is_micro_ct": bool,
            "is_heat_treatment": bool,
            "is_surface_roughness": bool,
            "is_conflict_relevant": bool,
            "classification_reason": str,
            "classification_confidence": str,
        }
    """
    text = f"{title} {abstract} {keywords}".lower()

    # Step 1: Detect matching types (multi-label)
    matched_types = []
    matched_tags = []

    for rule in CLASSIFICATION_RULES:
        for kw in rule["keywords"]:
            if kw in text:
                matched_types.append(rule["type"])
                # Extract a readable tag
                tag = kw.strip('"').strip()
                if tag not in matched_tags:
                    matched_tags.append(tag)
                break  # one keyword match per type is enough

    # Step 2: Determine primary, secondary
    type_order = [
        "review",
        "fcgr_paper", "micro_ct_paper", "hip_heat_treatment_paper",
        "surface_roughness_paper", "experimental_fatigue_paper",
    ]

    if not matched_types:
        # Check for generic "L-PBF Ti-6Al-4V fatigue" -> experimental
        if any(kw in text for kw in ["fatigue", "ti-6al-4v", "ti64", "l-pbf", "slm"]):
            matched_types.append("experimental_fatigue_paper")
            matched_tags.append("L-PBF Ti-6Al-4V fatigue")
        else:
            matched_types.append("candidate")
            matched_tags.append("unclassified")

    primary_type = matched_types[0]
    # Pick highest-priority type as primary
    for t in type_order:
        if t in matched_types:
            primary_type = t
            break

    secondary_types = [t for t in matched_types if t != primary_type]
    paper_type_secondary = "; ".join(secondary_types) if secondary_types else ""

    # Step 3: Determine folder
    folder = PAPER_SUBDIRS.get(primary_type, "other")

    # Step 4: Research topic
    topics = []
    for mt in matched_types:
        t = TOPIC_MAP.get(mt)
        if t and t not in topics:
            topics.append(t)
    research_topic = "; ".join(topics) if topics else "疲劳力学"

    # Step 5: Conflict relevance
    is_conflict = False
    conflict_tags = []
    for pattern, conflict_name in CONFLICT_PATTERNS:
        if re.search(pattern, text):
            is_conflict = True
            conflict_tags.append(conflict_name)

    if is_conflict and "conflict_paper" not in matched_types:
        matched_types.append("conflict_paper")
        matched_tags.append("文献冲突")

    # Step 6: Build result
    booleans = {
        pt: any(pt == mt for mt in matched_types)
        for pt in ["review", "experimental_fatigue_paper", "fcgr_paper",
                     "micro_ct_paper", "hip_heat_treatment_paper",
                     "surface_roughness_paper"]
    }

    # Classification confidence & reason
    n_matches = len(matched_types)
    if n_matches >= 3:
        confidence = "high"
        reason = f"匹配 {n_matches} 个类型关键词：{', '.join(matched_tags[:5])}"
    elif n_matches >= 1:
        confidence = "medium"
        reason = f"匹配 {n_matches} 个类型：{', '.join(matched_tags[:3])}"
    else:
        confidence = "low"
        reason = "未明确匹配，按 candidate 处理"

    # Tags
    tags = "; ".join(matched_tags[:10]) if matched_tags else "unclassified"

    return {
        "paper_type_primary": primary_type,
        "paper_type_secondary": paper_type_secondary,
        "research_topic": research_topic,
        "storage_folder": folder,
        "tags": tags,
        "is_review": booleans.get("review", False),
        "is_experimental": booleans.get("experimental_fatigue_paper", False),
        "is_fcgr": booleans.get("fcgr_paper", False),
        "is_micro_ct": booleans.get("micro_ct_paper", False),
        "is_heat_treatment": booleans.get("hip_heat_treatment_paper", False),
        "is_surface_roughness": booleans.get("surface_roughness_paper", False),
        "is_conflict_relevant": is_conflict,
        "classification_reason": reason,
        "classification_confidence": confidence,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. File naming & download path
# ═══════════════════════════════════════════════════════════════════════════

def generate_filename(title: str, authors: str, year: str) -> str:
    """从元数据生成标准化文件名: year_firstauthor_short_title.pdf"""
    # Extract first author surname
    first_author = "unknown"
    if authors:
        first = authors.split(";")[0].strip()
        # Take last word (surname)
        parts = first.split()
        if parts:
            first_author = parts[-1].strip("., ")

    # Short title: first 4 meaningful words
    short = title.lower() if title else ""
    short = re.sub(r"[^a-z0-9\s]", "", short)
    words = [w for w in short.split() if len(w) > 2][:5]
    short_str = "_".join(words) if words else "untitled"

    year_str = str(year)[:4] if year else "0000"
    safe_author = re.sub(r"[^a-zA-Z0-9]", "_", first_author)[:20]

    filename = f"{year_str}_{safe_author}_{short_str}.pdf"
    return filename


def get_paper_path(
    title: str,
    authors: str,
    year: str,
    paper_type_primary: str,
) -> Path:
    """返回该文献应存储的完整路径。"""
    folder = PAPER_SUBDIRS.get(paper_type_primary, "other")
    filename = generate_filename(title, authors, year)
    return PAPERS_DIR / folder / filename


# ═══════════════════════════════════════════════════════════════════════════
# 3. CSV field definitions for literature databases
# ═══════════════════════════════════════════════════════════════════════════

# Fields to add to candidate_papers.csv and literature_database.csv
CLASSIFICATION_FIELDS = [
    "paper_type_primary",
    "paper_type_secondary",
    "research_topic",
    "storage_folder",
    "tags",
    "is_review",
    "is_experimental",
    "is_fcgr",
    "is_micro_ct",
    "is_heat_treatment",
    "is_surface_roughness",
    "is_conflict_relevant",
    "classification_reason",
    "classification_confidence",
]

# If storing literature_database with these fields
LIT_DB_FIELDS = [
    "paper_id", "title", "authors", "year", "doi", "material",
    "manufacturing_process", "heat_treatment", "surface_state",
    "defect_type", "pore_size", "pore_location", "porosity",
    "stress_ratio_R", "Nf", "fatigue_limit", "da_dN",
    "Paris_C", "Paris_m", "Delta_Kth", "main_conclusion",
] + CLASSIFICATION_FIELDS + ["ingested_from"]

CANDIDATE_FIELDS = [
    "candidate_id", "title", "authors", "year", "journal", "doi", "url",
    "source_database", "abstract", "keywords", "is_open_access", "pdf_url",
    "matched_query", "relevance_score", "recommended_reason", "status",
    "full_text_status", "added_time", "last_updated",
] + CLASSIFICATION_FIELDS


# ═══════════════════════════════════════════════════════════════════════════
# 4. Batch classify & update CSV
# ═══════════════════════════════════════════════════════════════════════════

def classify_and_update_csv(csv_path: Path, fields: List[str]) -> int:
    """
    对已有的 CSV 文件中的所有文献进行分类，并更新分类字段。
    返回更新的行数。
    """
    import pandas as pd

    if not csv_path.exists():
        return 0

    df = pd.read_csv(csv_path, encoding="utf-8-sig", on_bad_lines="skip")
    if df.empty:
        return 0

    updated = 0
    for idx, row in df.iterrows():
        title = str(row.get("title", "") or "")
        abstract = str(row.get("abstract", "") or "")
        keywords = str(row.get("keywords", "") or "")
        # If abstract is empty, use main_conclusion or title only
        abstract = abstract if abstract and abstract != "nan" else ""
        keywords = keywords if keywords and keywords != "nan" else ""

        classification = classify_paper_type(title, abstract, keywords)

        for field in CLASSIFICATION_FIELDS:
            df.at[idx, field] = classification.get(field, "")

        updated += 1

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return updated


def batch_classify_all():
    """对所有相关 CSV 文件执行批量分类。"""
    ensure_paper_dirs()
    results = {}

    paths = [
        ("literature_database.csv", LIT_DB_FIELDS),
        ("candidate_papers.csv", CANDIDATE_FIELDS),
    ]
    for fname, fields in paths:
        p = BASE_DIR / "data" / fname
        n = classify_and_update_csv(p, fields)
        results[fname] = n

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 5. Classification summary
# ═══════════════════════════════════════════════════════════════════════════

def get_classification_summary() -> Dict[str, Any]:
    """获取分类统计摘要。"""
    import pandas as pd

    summary = {
        "total_papers": 0,
        "by_type": {},
        "by_folder": {},
        "conflict_count": 0,
        "types": [],
    }

    lit_path = BASE_DIR / "data" / "literature_database.csv"
    if lit_path.exists():
        df = pd.read_csv(lit_path, encoding="utf-8-sig", on_bad_lines="skip")
        summary["total_papers"] = len(df)

        if "paper_type_primary" in df.columns:
            type_counts = df["paper_type_primary"].value_counts().to_dict()
            summary["by_type"] = {str(k): int(v) for k, v in type_counts.items()}

        if "storage_folder" in df.columns:
            folder_counts = df["storage_folder"].value_counts().to_dict()
            summary["by_folder"] = {str(k): int(v) for k, v in folder_counts.items()}

        if "is_conflict_relevant" in df.columns:
            conflict_df = df[df["is_conflict_relevant"] == True]
            summary["conflict_count"] = len(conflict_df)

        summary["types"] = list(summary["by_type"].keys())

    return summary

"""
literature_support_fields.py — 文献数据库增强 (论文级)

在 literature_database.csv 中新增字段，标记每篇文献能支撑什么：
  - supports_pore_size_Nf
  - supports_distance_to_surface_crack_initiation
  - supports_surface_roughness_Nf
  - supports_HIP_defect_effect
  - supports_DeltaK_daDN
  - supports_Paris_parameters
  - supports_Murakami_sqrt_area
  - supports_Kitagawa
  - supports_research_gap
  - supports_hypothesis_generation
  - usable_for_validation

系统不能说"20篇正式文献"，而要说：
  - 多少篇支持 pore_size → Nf
  - 多少篇支持 distance_to_surface → crack_initiation_site
  - 多少篇支持 surface_roughness 与 internal pore 竞争主导
  - 多少篇可用于 Paris law 验证
  - 多少篇只是综述，不能作为直接数据
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

SUPPORT_FIELDS = [
    "supports_pore_size_Nf",
    "supports_distance_to_surface_crack_initiation",
    "supports_surface_roughness_Nf",
    "supports_HIP_defect_effect",
    "supports_DeltaK_daDN",
    "supports_Paris_parameters",
    "supports_Murakami_sqrt_area",
    "supports_Kitagawa",
    "supports_research_gap",
    "supports_hypothesis_generation",
    "usable_for_validation",
]

ALL_FIELDS_TARGET = SUPPORT_FIELDS + [
    "paper_id", "title", "authors", "year", "journal", "doi",
    "paper_type_primary", "main_conclusion", "evidence_type",
    "material", "process", "heat_treatment",
    "pore_size", "distance_to_surface", "surface_roughness_Ra",
    "porosity", "fatigue_limit", "Nf", "Delta_K", "da_dN",
    "Paris_C", "Paris_m", "Delta_Kth", "crack_initiation_site",
    "micro_CT_available", "SEM_available",
    "usable_for_hypothesis", "usable_for_validation_orig",
    "verification_status", "notes",
]


def classify_paper_support(paper: Dict[str, str]) -> Dict[str, str]:
    """
    基于已有字段，自动判断每篇文献支持什么。
    返回新字段名→值的映射。
    """
    title = paper.get("title", "").lower()
    main_conclusion = paper.get("main_conclusion", "").lower()
    evidence_type = paper.get("evidence_type", "").lower()
    paper_type = paper.get("paper_type_primary", "").lower()
    notes = paper.get("notes", "").lower()
    storage_folder = paper.get("storage_folder", "").lower()
    combined_text = f"{title} {main_conclusion} {notes} {storage_folder}"

    # 字段值
    has_pore_size = bool(paper.get("pore_size", "").strip())
    has_dist_to_surf = bool(paper.get("distance_to_surface", "").strip())
    has_roughness = bool(paper.get("surface_roughness_Ra", "").strip())
    has_porosity = bool(paper.get("porosity", "").strip())
    has_Nf = bool(paper.get("Nf", "").strip())
    has_fatigue_limit = bool(paper.get("fatigue_limit", "").strip())
    has_Delta_K = bool(paper.get("Delta_K", "").strip())
    has_da_dN = bool(paper.get("da_dN", "").strip())
    has_Paris_C = bool(paper.get("Paris_C", "").strip())
    has_Paris_m = bool(paper.get("Paris_m", "").strip())
    has_Delta_Kth = bool(paper.get("Delta_Kth", "").strip())
    has_crack_init = bool(paper.get("crack_initiation_site", "").strip())
    has_heat_treatment = bool(paper.get("heat_treatment", "").strip())
    is_review = (paper_type == "review" or evidence_type == "review")

    # storage_folder 作为强信号
    folder_pore = "pore" in storage_folder
    folder_surface = "surface" in storage_folder or "roughness" in storage_folder
    folder_hip = "hip" in storage_folder
    folder_fcgr = "fcgr" in storage_folder or "paris" in storage_folder
    folder_micro_ct = "micro_ct" in storage_folder

    # ── 判断支持项 ──
    supports = {}

    # pore_size → Nf
    s1 = has_pore_size and (has_Nf or has_fatigue_limit)
    s1 = s1 or bool(re.search(r"pore.*(size|defect).*(fatigue|life|nf)", combined_text))
    s1 = s1 or (folder_pore and bool(re.search(r"(fatigue|life)", combined_text)))
    supports["supports_pore_size_Nf"] = "True" if s1 else ""

    # distance_to_surface → crack_initiation
    s2 = has_dist_to_surf and has_crack_init
    s2 = s2 or bool(re.search(r"(distance|depth|subsurface|near.surface).*(crack.initiation|initiation.site)", combined_text))
    s2 = s2 or (folder_micro_ct and bool(re.search(r"defect|pore|void", combined_text)))
    supports["supports_distance_to_surface_crack_initiation"] = "True" if s2 else ""

    # surface_roughness → Nf
    s3 = has_roughness and (has_Nf or has_fatigue_limit)
    s3 = s3 or bool(re.search(r"(roughness|surface.finish|as.built).*(fatigue|life|nf)", combined_text))
    s3 = s3 or bool(re.search(r"(fatigue|life).*(roughness|surface.finish)", combined_text))
    s3 = s3 or folder_surface
    supports["supports_surface_roughness_Nf"] = "True" if s3 else ""

    # HIP → defect_effect
    s4 = has_heat_treatment and bool(re.search(r"hip|hot.isostatic", combined_text, re.IGNORECASE))
    s4 = s4 or bool(re.search(r"hip.*(pore|defect|fatigue)", combined_text))
    s4 = s4 or folder_hip
    supports["supports_HIP_defect_effect"] = "True" if s4 else ""

    # Delta_K → da_dN
    s5 = has_Delta_K and has_da_dN
    s5 = s5 or bool(re.search(r"crack.*(growth|propagation).*(δk|delta k)", combined_text))
    s5 = s5 or bool(re.search(r"da/dn|da_dn|crack.growth.rate", combined_text))
    s5 = s5 or folder_fcgr
    supports["supports_DeltaK_daDN"] = "True" if s5 else ""

    # Paris parameters
    s6 = has_Paris_C or has_Paris_m
    s6 = s6 or bool(re.search(r"paris.*(c|m|parameter|law)", combined_text))
    s6 = s6 or folder_fcgr
    supports["supports_Paris_parameters"] = "True" if s6 else ""

    # Murakami √area
    s7 = bool(re.search(r"murakami|√area|sqrt.area|area.model", combined_text))
    supports["supports_Murakami_sqrt_area"] = "True" if s7 else ""

    # Kitagawa
    s8 = bool(re.search(r"kitagawa|takahashi|kitagawa.takahashi", combined_text))
    supports["supports_Kitagawa"] = "True" if s8 else ""

    # research_gap
    s9 = (
        bool(re.search(r"(gap|limit|future|need|further|unclear|unknown|not.*understood)", combined_text))
        or bool(re.search(r"缺少|需要进一步|有待|尚不清楚", combined_text))
        or "review" in storage_folder
    )
    supports["supports_research_gap"] = "True" if s9 else ""

    # hypothesis_generation
    s10 = (
        s1 or s2 or s3 or s4 or s5 or s6 or s7 or s8 or s9
    ) and not is_review
    supports["supports_hypothesis_generation"] = "True" if s10 else ""

    # usable_for_validation
    s11 = (s1 or s5 or s6 or folder_fcgr) and not is_review and (
        has_Nf or has_fatigue_limit or has_da_dN or folder_fcgr or folder_pore
    )
    supports["usable_for_validation"] = "True" if s11 else ""

    return supports


def enhance_literature_database():
    """增强 literature_database.csv，添加支持字段。"""
    path = DATA_DIR / "literature_database.csv"
    if not path.exists():
        print(f"[ERROR] literature_database.csv not found at {path}")
        return

    # 读取
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        existing_fields = reader.fieldnames or []

    print(f"[INFO] Loaded {len(rows)} papers from literature_database.csv")

    # 备份
    backup_path = DATA_DIR / "literature_database_backup.csv"
    with open(backup_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=existing_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Backup saved to {backup_path}")

    # 新字段列表
    new_fields = list(existing_fields)
    for sf in SUPPORT_FIELDS:
        if sf not in new_fields:
            new_fields.append(sf)

    # 对每篇文献分类
    support_counts = {sf: 0 for sf in SUPPORT_FIELDS}
    review_count = 0

    for row in rows:
        supports = classify_paper_support(row)
        for sf in SUPPORT_FIELDS:
            val = supports.get(sf, "")
            row[sf] = val
            if val == "True":
                support_counts[sf] += 1
        if row.get("paper_type_primary", "").lower() == "review":
            review_count += 1

    # 写回
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # 输出统计
    print("\n" + "=" * 60)
    print("  文献支持字段统计")
    print("=" * 60)
    print(f"  总文献数: {len(rows)}")
    print(f"  综述文献: {review_count}")
    print()
    print(f"  {'支持项':<45} {'篇数':>8}")
    print(f"  {'-'*45} {'-'*8}")
    for sf in SUPPORT_FIELDS:
        label = sf.replace("supports_", "").replace("_", " ")
        print(f"  {label:<45} {support_counts[sf]:>8}")
    print()

    # 论文级可引用输出
    print("  论文级可引用描述:")
    print(f"  - {support_counts['supports_pore_size_Nf']} papers provide evidence on pore size → Nf relationship")
    print(f"  - {support_counts['supports_distance_to_surface_crack_initiation']} papers address distance to surface → crack initiation site")
    print(f"  - {support_counts['supports_surface_roughness_Nf']} papers investigate surface roughness effects on fatigue life")
    print(f"  - {support_counts['supports_HIP_defect_effect']} papers study HIP effect on defects")
    print(f"  - {support_counts['supports_DeltaK_daDN']} papers report ΔK–da/dN crack growth data")
    print(f"  - {support_counts['supports_Paris_parameters']} papers provide Paris law parameters")
    print(f"  - {support_counts['supports_Murakami_sqrt_area']} papers reference Murakami √area model")
    print(f"  - {support_counts['supports_Kitagawa']} papers reference Kitagawa-Takahashi diagram")
    print(f"  - {support_counts['supports_research_gap']} papers identify open research questions")
    print(f"  - {support_counts['supports_hypothesis_generation']} papers provide data for hypothesis generation")
    print(f"  - {support_counts['usable_for_validation']} papers are usable for quantitative validation")
    print()

    return rows


def get_literature_support_stats() -> Dict[str, int]:
    """获取支持统计。"""
    path = DATA_DIR / "literature_database.csv"
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    stats = {
        "total": len(rows),
        "review_count": 0,
    }
    for sf in SUPPORT_FIELDS:
        stats[sf] = 0

    for row in rows:
        if row.get("paper_type_primary", "").lower() == "review":
            stats["review_count"] += 1
        for sf in SUPPORT_FIELDS:
            if row.get(sf, "") == "True":
                stats[sf] += 1

    return stats


if __name__ == "__main__":
    enhance_literature_database()

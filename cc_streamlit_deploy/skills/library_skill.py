"""
library_skill.py — 文献库管理

负责文献卡片的持久化、查重、搜索以及术语归一化。
"""

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from . import dedup_skill


DATA_DIR = Path("data")
CARDS_PATH = DATA_DIR / "literature_cards.jsonl"
CSV_PATH = DATA_DIR / "literature_database.csv"

DEFAULT_FIELDS = [
    "title", "authors", "year", "journal", "doi",
    "alloy_type", "material_system", "processing_method", "heat_treatment",
    "microstructure", "loading_condition", "stress_ratio_R",
    "temperature_environment", "experimental_methods",
    "characterization_methods", "mechanical_indicators",
    "crack_initiation", "crack_growth_mechanism", "model_or_method",
    "key_findings", "limitations", "possible_innovation",
    "evidence_text", "source_file", "file_hash", "source_type",
    # 新增字段
    "dedup_key", "corpus_roles", "source_folders",
]


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(exist_ok=True)


def _normalize_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def normalize_text(text: str) -> str:
    """去空格、转小写、删除标点"""
    if not text:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── 术语归一化映射表 ────────────────────────────────────────────────────────

TERM_NORMALIZATION_MAP = [
    # Ti-6Al-4V 系列
    (r"ti-?6al-?4v", "Ti-6Al-4V"),
    (r"ti6al4v", "Ti-6Al-4V"),
    (r"\bti64\b", "Ti-6Al-4V"),
    (r"\btc4\b", "Ti-6Al-4V"),
    (r"grade\s*5\s*titanium", "Ti-6Al-4V"),
    (r"ti-?6-?4", "Ti-6Al-4V"),

    # 增材制造工艺
    (r"selective\s*laser\s*melting", "L-PBF"),
    (r"\bslm\b", "L-PBF"),
    (r"laser\s*powder\s*bed\s*fusion", "L-PBF"),
    (r"\blpbf\b", "L-PBF"),
    (r"\bl-pbf\b", "L-PBF"),

    (r"electron\s*beam\s*melting", "EBM"),
    (r"\bebm\b", "EBM"),

    (r"laser\s*engineered\s*net\s*shaping", "DED"),
    (r"\blens\b", "DED"),
    (r"\bded\b", "DED"),
    (r"directed\s*energy\s*deposition", "DED"),

    # 疲劳类型
    (r"very\s*high\s*cycle\s*fatigue", "VHCF"),
    (r"\bvhcf\b", "VHCF"),
    (r"超高周疲劳", "VHCF"),

    (r"high\s*cycle\s*fatigue", "HCF"),
    (r"\bhcf\b", "HCF"),
    (r"高周疲劳", "HCF"),

    (r"low\s*cycle\s*fatigue", "LCF"),
    (r"\blcf\b", "LCF"),
    (r"低周疲劳", "LCF"),

    # 疲劳裂纹扩展
    (r"fatigue\s*crack\s*growth\s*rate", "FCGR"),
    (r"fatigue\s*crack\s*propagation", "FCGR"),
    (r"fatigue\s*crack\s*growth", "FCGR"),
    (r"\bfcg\b", "FCGR"),
    (r"\bfcgr\b", "FCGR"),
    (r"裂纹扩展速率", "FCGR"),

    # 力学指标
    (r"crack\s*growth\s*rate", "da/dN"),
    (r"\bda/dn\b", "da/dN"),
    (r"\bdadn\b", "da/dN"),

    (r"stress\s*intensity\s*factor\s*range", "ΔK"),
    (r"delta\s*k", "ΔK"),
    (r"\bΔk\b", "ΔK"),
    (r"\bΔK\b", "ΔK"),

    # 模型
    (r"paris\s*law", "Paris"),
    (r"paris\s*equation", "Paris"),
    (r"paris\s*formula", "Paris"),
    (r"paris\s*参数", "Paris"),
    (r"\bparis\s+model\b", "Paris"),

    (r"walker\s*equation", "Walker model"),
    (r"walker\s*model", "Walker model"),
    (r"walker\s*修正", "Walker model"),

    (r"\bnasgro\b", "NASGRO"),

    # 表征方法
    (r"\bsem\b", "SEM"),
    (r"scanning\s*electron\s*microscop", "SEM"),
    (r"sem\s*fractograph", "SEM"),
    (r"断口\s*(分析|形貌|扫描)", "SEM"),

    (r"\bebsd\b", "EBSD"),
    (r"electron\s*backscatter\s*diffraction", "EBSD"),

    (r"micro\s*-?\s*ct", "X-ray CT"),
    (r"x-ray\s*ct", "X-ray CT"),
    (r"x-ray\s*computed\s*tomograph", "X-ray CT"),
    (r"computed\s*tomograph", "X-ray CT"),
    (r"同步辐射\s*(ct|成像)", "X-ray CT"),

    # 材料
    (r"ti-6al-2sn-4zr-2mo", "Ti-6Al-2Sn-4Zr-2Mo"),
    (r"ti-10v-2fe-3al", "Ti-10V-2Fe-3Al"),
    (r"ti-6al-7nb", "Ti-6Al-7Nb"),
    (r"tc17", "TC17"),
    (r"\bti60\b", "Ti60"),

    # 制造缺陷
    (r"\b孔隙\b", "pore defect"),
    (r"\b气孔\b", "pore defect"),
    (r"未熔合", "lack of fusion"),
    (r"lack\s*of\s*fusion", "lack of fusion"),
]


def normalize_terms(text: str) -> str:
    """对文本中的专业术语进行归一化。

    将同义词、缩写变体统一为标准术语。
    """
    if not text:
        return ""
    result = text
    for pattern, replacement in TERM_NORMALIZATION_MAP:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def get_paper_identity(card: dict) -> Tuple[str, str, str]:
    """获取文献唯一标识：(doi_normalized, title_normalized, dedup_key)

    去重规则：
    1. DOI 相同视为同一篇；
    2. DOI 为空则用 title(规范化) + year 判断。
    """
    doi = str(card.get("doi", "") or "").strip().lower()
    title = normalize_text(card.get("title", "") or "")
    year = str(card.get("year", "") or card.get("publication_year", "") or "")

    # 构建 dedup_key
    if doi:
        dedup_key = f"doi::{doi}"
    elif title and year:
        dedup_key = f"title::{title}::{year}"
    elif title:
        dedup_key = f"title::{title}"
    else:
        dedup_key = ""

    return (doi, title, dedup_key)


def _count_completeness(card: dict) -> int:
    """统计非空字段数量，用于判断哪条记录更完整"""
    important_fields = [
        "title", "authors", "year", "journal", "doi",
        "alloy_type", "material_system", "processing_method", "heat_treatment",
        "microstructure", "loading_condition", "model_or_method",
        "key_findings", "limitations", "possible_innovation",
    ]
    count = 0
    for f in important_fields:
        val = card.get(f)
        if val and str(val).strip() not in ("", "未说明", "None"):
            count += 1
    return count


def _build_corpus_roles(card: dict) -> list:
    """根据来源文件夹构建语料角色标记。"""
    roles = set()
    source_files = card.get("source_folders", [])
    if isinstance(source_files, str):
        source_files = [source_files]
    for sf in source_files:
        if "early" in sf.lower():
            roles.add("early")
        elif "followup" in sf.lower():
            roles.add("followup")
        else:
            roles.add("core")
    # 如果没有明确标记，默认为 core
    if not roles:
        roles.add("core")
    return sorted(roles)


def cleanup_library_duplicates() -> int:
    """清理重复文献，合并 corpus_roles 和 source_folders。"""
    from src.formal_pdf_protection import validate_formal_pdf_locks

    validate_formal_pdf_locks(Path.cwd())
    papers = get_all_papers()
    if not papers:
        return 0

    groups: Dict[tuple, list] = {}
    for p in papers:
        doi, title, dedup_key = get_paper_identity(p)

        if dedup_key and dedup_key.startswith("doi::"):
            key = dedup_key
        elif dedup_key:
            key = dedup_key
        else:
            # 没有 dedup_key 的，用 file_hash 分组
            key = ("hash", p.get("file_hash", id(p)))

        groups.setdefault(key, []).append(p)

    deduped = []
    removed = 0
    for key, group in groups.items():
        if len(group) == 1:
            deduped.append(group[0])
        else:
            # 保留字段最完整的那条，合并 source_folders 和 corpus_roles
            group.sort(key=_count_completeness, reverse=True)
            best = group[0]

            # 合并 source_folders
            all_folders = set()
            all_roles = set()
            for g in group:
                sf = g.get("source_folders", [])
                if isinstance(sf, str):
                    sf = [sf]
                for s in sf:
                    all_folders.add(s)
                cr = g.get("corpus_roles", [])
                if isinstance(cr, str):
                    cr = [cr]
                for c in cr:
                    all_roles.add(c)
                # 记录 source_file
                src = g.get("source_file", "")
                if src and Path(src).exists():
                    all_folders.add(str(Path(src).parent.name))

            best["source_folders"] = sorted(all_folders) if all_folders else []
            best["corpus_roles"] = sorted(all_roles) if all_roles else _build_corpus_roles(best)

            # 如果没有 role，自动构建
            if not best.get("corpus_roles"):
                best["corpus_roles"] = _build_corpus_roles(best)

            # 设置 dedup_key
            if isinstance(key, str) and key.startswith("doi::"):
                best["dedup_key"] = key
            elif isinstance(key, str) and key.startswith("title::"):
                best["dedup_key"] = key
            else:
                best["dedup_key"] = f"hash::{best.get('file_hash', '')}"

            deduped.append(best)
            removed += len(group) - 1

    deduped.sort(key=lambda p: normalize_text(p.get("title", "") or ""))

    with open(CARDS_PATH, "w", encoding="utf-8") as f:
        for item in deduped:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_csv(deduped)
    return removed


# 搜索权重配置
_SEARCH_WEIGHTS = {
    "title": 10,
    "material_system": 5,
    "model_or_method": 5,
    "limitations": 5,
    "possible_innovation": 5,
}

_SEARCH_FIELDS = [
    "title", "authors", "year", "journal", "doi",
    "alloy_type", "material_system", "processing_method", "heat_treatment",
    "microstructure", "loading_condition", "model_or_method",
    "key_findings", "limitations", "possible_innovation",
    "evidence_text", "source_file",
]


def search_papers(query: str) -> List[Dict[str, Any]]:
    """搜索文献，返回按匹配分数排序的结果。"""
    papers = get_all_papers()
    q = (query or "").strip()
    if not q:
        return papers

    tokens = q.split()
    if not tokens:
        return papers

    scored: List[Tuple[int, Dict[str, Any]]] = []
    for paper in papers:
        score = 0
        for token in tokens:
            for field in _SEARCH_FIELDS:
                value = str(paper.get(field, "") or "")
                if not value:
                    continue
                if token.lower() in value.lower():
                    score += _SEARCH_WEIGHTS.get(field, 1)
        if score > 0:
            scored.append((score, paper))

    scored.sort(key=lambda x: (-x[0], normalize_text(x[1].get("title", "") or "")))
    return [p for _, p in scored]


def get_all_papers() -> List[Dict[str, Any]]:
    if not CARDS_PATH.exists():
        return []
    papers = []
    with open(CARDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                papers.append(json.loads(line))
            except Exception:
                continue
    return papers


def save_literature_card(card: Dict[str, Any], file_hash: str) -> None:
    """保存或更新文献卡片。

    如果已存在相同 dedup_key 的卡片，则合并字段。
    """
    _ensure_data_dir()

    card["file_hash"] = file_hash

    # 构建 dedup_key
    doi, title, dedup_key = get_paper_identity(card)
    card["dedup_key"] = dedup_key

    # 确定 source_folders
    src = str(card.get("source_file", "") or "")
    folder = ""
    if src:
        p = Path(src)
        folder = p.parent.name if p.parent.name != "." else ""
    card["source_folders"] = [folder] if folder else []

    # 确定 corpus_roles
    card["corpus_roles"] = _build_corpus_roles(card)

    existing = get_all_papers()

    # 尝试 dedup_key 匹配
    replaced = False
    for i, p in enumerate(existing):
        existing_dedup = p.get("dedup_key", "")
        if not existing_dedup:
            _, _, edk = get_paper_identity(p)
            existing_dedup = edk

        if existing_dedup and dedup_key and existing_dedup == dedup_key:
            # 合并字段：保留非空字段多的那个
            existing_complete = _count_completeness(p)
            new_complete = _count_completeness(card)
            if new_complete > existing_complete:
                # 保留 source_folders 合并
                old_folders = p.get("source_folders", [])
                if isinstance(old_folders, str):
                    old_folders = [old_folders]
                card_folders = card.get("source_folders", [])
                if isinstance(card_folders, str):
                    card_folders = [card_folders]
                merged_folders = list(set(old_folders + card_folders))
                card["source_folders"] = merged_folders
                card["corpus_roles"] = _build_corpus_roles(card)

                existing[i] = card
            else:
                # 更新 source_folders 在原卡片上
                old_folders = p.get("source_folders", [])
                if isinstance(old_folders, str):
                    old_folders = [old_folders]
                card_folders = card.get("source_folders", [])
                if isinstance(card_folders, str):
                    card_folders = [card_folders]
                merged_folders = list(set(old_folders + card_folders))
                p["source_folders"] = merged_folders
                p["corpus_roles"] = _build_corpus_roles(p)
                # 更新 dedup_key
                p["dedup_key"] = dedup_key
                existing[i] = p
            replaced = True
            break

    if replaced:
        with open(CARDS_PATH, "w", encoding="utf-8") as f:
            for item in existing:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        save_csv(existing)
    else:
        existing.append(card)
        with open(CARDS_PATH, "w", encoding="utf-8") as f:
            for item in existing:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        save_csv(existing)

    dedup_skill.add_to_index(
        file_hash=file_hash,
        file_name=str(card.get("source_file", "")),
        saved_path=str(card.get("source_file", "")),
    )


def save_csv(papers: List[Dict[str, Any]]) -> None:
    """将文献卡片持久化为 CSV。"""
    _ensure_data_dir()

    fields = list(DEFAULT_FIELDS)
    for paper in papers:
        for key in paper.keys():
            if key not in fields:
                fields.append(key)

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for paper in papers:
            row = {field: _normalize_value(paper.get(field, "")) for field in fields}
            writer.writerow(row)

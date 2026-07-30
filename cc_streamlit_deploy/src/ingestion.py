"""
ingestion.py — 文献库构建模块

实现 ingest 命令：读取 PDF → 文献卡片抽取 → 去重 → 文献数据库生成
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── 复用现有 skills ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from skills.pdf_skill import extract_text_from_pdf
from skills.card_skill import extract_literature_card
from skills.library_skill import (
    get_all_papers,
    save_literature_card,
    save_csv,
    CSV_PATH,
    CARDS_PATH,
    get_paper_identity,
    normalize_text,
    cleanup_library_duplicates,
    normalize_terms,
)
from skills.dedup_skill import check_duplicate, add_to_index
from skills.search_skill import add_search_result_to_library
from src.validation import is_out_of_scope, classify_titanium_scope
from src.stage1_store import discover_pdf_files
from src.deep_read_pipeline import deep_read_pdf

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"


def run_ingest() -> Dict[str, Any]:
    """执行文献库构建流程。

    Returns:
        统计信息字典，包含详细的去重和分类统计
    """
    stats = {
        "total_pdfs": 0,
        "processed": 0,
        "out_of_scope": 0,
        "failed": 0,
        "skipped_duplicates": 0,
        "cards_before": len(get_all_papers()),
        "duplicates_removed": 0,
        "deep_read_completed": 0,
        "deep_read_failed": 0,
    }

    # 扫描所有文献目录
    pdf_files = _scan_paper_dirs()

    if not pdf_files:
        return _add_online_only_results(stats)

    stats["total_pdfs"] = len(pdf_files)

    for pdf_path in pdf_files:
        try:
            result = _process_single_pdf(pdf_path)
            if result.get("deep_read_complete"):
                stats["deep_read_completed"] += 1
            elif result.get("deep_read_status") == "FAILED":
                stats["deep_read_failed"] += 1
            if result["status"] == "out_of_scope":
                stats["out_of_scope"] += 1
            elif result["status"] == "processed":
                stats["processed"] += 1
            elif result["status"] == "duplicate":
                stats["skipped_duplicates"] += 1
        except Exception as e:
            stats["failed"] += 1

    # 去重后清理
    removed = cleanup_library_duplicates()
    stats["duplicates_removed"] = removed

    stats["evidence_status"] = "STAGE2_PAGE_BOUND_ONLY"

    stats["cards_after"] = len(get_all_papers())
    return stats


def _scan_paper_dirs() -> List[Path]:
    """Recursively discover PDFs, including every categorized subdirectory."""
    pdf_files = discover_pdf_files(base_dir=BASE_DIR)
    for path in sorted(BASE_DIR.glob("*.pdf")):
        resolved = path.resolve()
        if resolved not in pdf_files:
            pdf_files.append(resolved)
    return sorted(set(pdf_files), key=lambda path: str(path).lower())


def _get_source_folder(pdf_path: Path) -> str:
    """获取 PDF 的来源文件夹名。"""
    parent = pdf_path.parent.name
    if parent in ("papers", "early_papers", "followup_papers", "data", "uploaded_papers"):
        return parent
    return "papers"


def _process_single_pdf(pdf_path: Path) -> Dict[str, str]:
    """处理单个 PDF 文件。"""
    # 检查是否已处理（基于文件哈希）
    file_hash = _compute_file_hash(pdf_path)
    source_folder = _get_source_folder(pdf_path)
    deep_read = deep_read_pdf(pdf_path, base_dir=BASE_DIR)

    existing = get_all_papers()
    for p in existing:
        if p.get("file_hash") == file_hash:
            # 更新 source_folders 如果在不同目录中有重复
            fold = p.get("source_folders", [])
            if isinstance(fold, str):
                fold = [fold]
            if source_folder not in fold:
                fold.append(source_folder)
                p["source_folders"] = list(set(fold))
                # 更新 corpus_roles
                from skills.library_skill import _build_corpus_roles
                p["corpus_roles"] = _build_corpus_roles(p)
                save_csv(existing)
                with open(CARDS_PATH, "w", encoding="utf-8") as f:
                    for item in existing:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            return {
                "status": "duplicate",
                "path": str(pdf_path),
                "deep_read_status": str(deep_read.get("status", "")),
                "deep_read_complete": bool(deep_read.get("deep_read_complete")),
            }

    # 抽取文本
    pdf_text = extract_text_from_pdf(str(pdf_path))
    if not pdf_text.strip():
        return {
            "status": "failed",
            "reason": "empty_text",
            "deep_read_status": str(deep_read.get("status", "")),
            "deep_read_complete": bool(deep_read.get("deep_read_complete")),
        }

    # 判断范围（先做初步判断，再做二次校验避免误判钛合金材料）
    is_oos = is_out_of_scope(pdf_text[:2000])
    if is_oos:
        # 二次校验：TC17/Ti60/近α/α+β/β钛合金 不属于 out_of_scope
        t_check = pdf_text[:2000].lower()
        ti_safe = ["tc17", "ti60", "ti-6al-2sn-4zr-2mo", "ti-10v-2fe-3al",
                   "ti-6al-7nb", "near-alpha", "α+β", "alpha+beta",
                   "beta titanium", "β钛", "近α钛合金",
                   "alpha+beta titanium", "β钛合金"]
        if any(kw in t_check for kw in ti_safe):
            is_oos = False

    # 抽取文献卡片
    try:
        card_data = extract_literature_card(pdf_text)
    except Exception:
        # fallback: minimal card
        card_data = {
            "title": pdf_path.stem,
            "key_findings": [pdf_text[:500]],
        }

    # 构建完整卡片
    card = _build_full_card(card_data, pdf_path, file_hash, source_folder, pdf_text[:1000])

    # 如果 out_of_scope，标记
    if is_oos:
        card["alloy_type"] = "out_of_scope"

    # 对卡片内容进行术语归一化
    card = _apply_term_normalization(card)

    save_literature_card(card, file_hash)

    if is_oos:
        return {
            "status": "out_of_scope",
            "path": str(pdf_path),
            "deep_read_status": str(deep_read.get("status", "")),
            "deep_read_complete": bool(deep_read.get("deep_read_complete")),
        }
    return {
        "status": "processed",
        "path": str(pdf_path),
        "deep_read_status": str(deep_read.get("status", "")),
        "deep_read_complete": bool(deep_read.get("deep_read_complete")),
    }


def _apply_term_normalization(card: Dict[str, Any]) -> Dict[str, Any]:
    """对卡片中的文本字段进行术语归一化。"""
    text_fields = [
        "material_system", "processing_method", "microstructure",
        "loading_condition", "characterization_methods",
        "model_or_method", "experimental_methods",
        "crack_initiation", "crack_growth_mechanism",
        "mechanical_indicators", "temperature_environment",
    ]
    for field in text_fields:
        val = card.get(field, "")
        if val and isinstance(val, str):
            card[field] = normalize_terms(val)
    return card


def _build_full_card(
    card_data: Dict[str, Any], pdf_path: Path, file_hash: str,
    source_folder: str, evidence_text_fallback: str = ""
) -> Dict[str, Any]:
    """将抽取的卡片数据与标准字段合并。"""
    # 确定 corpus_roles
    roles = []
    if source_folder == "early_papers":
        roles = ["early"]
    elif source_folder == "followup_papers":
        roles = ["followup"]
    else:
        roles = ["core"]

    fields = {
        "title": card_data.get("title", pdf_path.stem),
        "authors": card_data.get("authors", []),
        "year": card_data.get("publication_year", ""),
        "journal": card_data.get("journal", ""),
        "doi": card_data.get("doi", ""),
        "abstract": card_data.get("abstract", ""),
        "key_findings": card_data.get("key_findings", []),
        "methods": card_data.get("methods", ""),
        "conclusion": card_data.get("conclusion", ""),
        "keywords": card_data.get("keywords", []),
        "material_system": card_data.get("material_system", ""),
        "processing_method": card_data.get("processing_method", ""),
        "heat_treatment": card_data.get("heat_treatment", ""),
        "microstructure": card_data.get("microstructure", ""),
        "loading_condition": card_data.get("loading_condition", ""),
        "stress_ratio_R": card_data.get("stress_ratio_R", ""),
        "temperature_environment": card_data.get("temperature_environment", ""),
        "experimental_methods": card_data.get("experimental_methods", ""),
        "characterization_methods": card_data.get("characterization_methods", ""),
        "model_or_method": card_data.get("model_or_method", ""),
        "mechanical_indicators": card_data.get("mechanical_indicators", ""),
        "crack_initiation": card_data.get("crack_initiation", ""),
        "crack_growth_mechanism": card_data.get("crack_growth_mechanism", ""),
        "limitations": card_data.get("limitations", ""),
        "possible_innovation": card_data.get("possible_innovation", ""),
        "evidence_text": card_data.get("evidence_text", evidence_text_fallback),
        "source_file": str(pdf_path),
        "file_hash": file_hash,
        "source_type": "PDF",
        "source_folders": [source_folder],
        "corpus_roles": roles,
    }
    # dedup_key 在 save_literature_card 中自动设置
    return fields


def _add_online_only_results(stats: Dict[str, Any]) -> Dict[str, Any]:
    """处理只有在线元数据的情况。"""
    existing = get_all_papers()
    online_count = sum(1 for p in existing if p.get("source_type") == "在线检索元数据")
    stats["total_pdfs"] = 0
    stats["online_metadata"] = online_count
    stats["cards_after"] = len(existing)
    return stats


def _compute_file_hash(file_path: Path) -> str:
    """计算文件的 MD5 哈希。"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ── 在线检索辅助 ──────────────────────────────────────────────────────────

def search_and_add_papers(query: str, max_results: int = 10) -> int:
    """搜索在线文献并加入文献库。"""
    from skills.search_skill import online_literature_search
    papers = online_literature_search(query, max_results=max_results)
    added = 0
    for p in papers:
        if add_search_result_to_library(p):
            added += 1
    return added


# ── Evidence Map ──────────────────────────────────────────────────────────

def _generate_evidence_map(stats: Dict[str, Any]) -> None:
    """此为占位，实际在 write_evidence_map 中完成。"""
    pass


def write_evidence_map(stats: Dict[str, Any]) -> str:
    """生成 01_evidence_map.md。

    修复后：
    - 明确区分原始 PDF 数量、去重后文献数量、核心钛合金疲劳文献数量、out_of_scope 数量
    - 文献数量与 gap_diagnosis 保持一致
    - 使用 classify_titanium_scope 进行范围分类
    """
    papers = get_all_papers()

    total = len(papers)
    out_scope_count = sum(1 for p in papers if p.get("alloy_type") == "out_of_scope")
    in_scope = [p for p in papers if p.get("alloy_type") != "out_of_scope"]

    # 使用 classify_titanium_scope 对每篇文献分类
    scope_results = []
    for p in in_scope:
        sr = classify_titanium_scope(p)
        p["_scope"] = sr
        scope_results.append(sr)

    core_ti_fatigue = [p for p in in_scope if p.get("_scope", {}).get("include_in_core_analysis", True)]
    primary_count = sum(1 for p in in_scope if p.get("_scope", {}).get("main_case_relevance") == "primary")
    secondary_count = sum(1 for p in in_scope if p.get("_scope", {}).get("main_case_relevance") == "secondary_case")
    background_count = sum(1 for p in in_scope if p.get("_scope", {}).get("main_case_relevance") == "background")

    evidence_level = "不足" if total < 10 else ("部分覆盖" if total < 30 else "基本覆盖")

    total_pdfs = stats.get("total_pdfs", 0)
    deduped = stats.get("duplicates_removed", 0)

    lines = [
        "# Evidence Map（文献证据地图）",
        "",
        "## 文献库统计",
        "",
        f"- **原始 PDF 数量**: {total_pdfs} 个",
        f"- **去重后文献总数**: {total} 篇",
        f"- **核心钛合金疲劳文献**: {len(core_ti_fatigue)} 篇",
        f"  - 主案例（AM Ti-6Al-4V FCG）相关: {primary_count} 篇",
        f"  - 次要案例相关: {secondary_count} 篇",
        f"  - 背景/参考: {background_count} 篇",
        f"- **out_of_scope**: {out_scope_count} 篇（标记为非钛合金疲劳方向）",
        f"- **证据充分性**: {evidence_level}",
        "",
        "### 剔除文献",
        "",
    ]

    if out_scope_count > 0:
        lines.append("以下文献因不涉及钛合金疲劳而被剔除：\n")
        for p in papers:
            if p.get("alloy_type") == "out_of_scope":
                title = p.get("title", "未知")
                src = Path(p.get("source_file", "")).name
                lines.append(f"- {title}")
                lines.append(f"  - 来源: {src}")
                lines.append(f"  - 原因: 非钛合金疲劳方向（涉及铝合金/钢/复合材料等）")
                lines.append("")
    else:
        lines.append("- 无剔除文献\n")

    # 覆盖维度统计
    lines.extend([
        "## 覆盖维度",
        "",
    ])

    materials = set()
    processes = set()
    microstructures = set()
    loadings = set()
    environments = set()
    methods = set()
    models = set()
    indicators = set()
    mechanisms = set()

    for p in core_ti_fatigue:
        for val in [p.get("material_system", ""), p.get("title", "")]:
            if val:
                if any(kw in val.lower() for kw in
                       ["ti-6al-4v", "ti6al4v", "tc4", "tc17", "ti60", "ti-", "titanium"]):
                    materials.add(val.strip()[:60])
        for f_name, f_set in [
            ("processing_method", processes), ("microstructure", microstructures),
            ("loading_condition", loadings), ("temperature_environment", environments),
            ("experimental_methods", methods), ("model_or_method", models),
            ("mechanical_indicators", indicators), ("crack_initiation", mechanisms),
            ("crack_growth_mechanism", mechanisms),
        ]:
            val = p.get(f_name, "")
            if val:
                val_str = str(val).strip()[:80]
                if val_str:
                    f_set.add(val_str)

    lines.extend([
        f"### 材料体系 ({len(materials)})",
        *[f"- {m}" for m in sorted(materials)],
        "",
        f"### 制备工艺 ({len(processes)})",
        *[f"- {p}" for p in sorted(processes)],
        "",
        f"### 微观组织 ({len(microstructures)})",
        *[f"- {m}" for m in sorted(microstructures)],
        "",
        f"### 疲劳载荷类型 ({len(loadings)})",
        *[f"- {l}" for l in sorted(loadings)],
        "",
        f"### 温度/环境 ({len(environments)})",
        *[f"- {e}" for e in sorted(environments)],
        "",
        f"### 实验方法 ({len(methods)})",
        *[f"- {m}" for m in sorted(methods)],
        "",
        f"### 表征方法 ({len(methods)})",
        *[f"- {m}" for m in sorted(methods)],
        "",
        f"### 疲劳指标 ({len(indicators)})",
        *[f"- {i}" for i in sorted(indicators)],
        "",
        f"### 模型/方法 ({len(models)})",
        *[f"- {m}" for m in sorted(models)],
        "",
        f"### 机制 ({len(mechanisms)})",
        *[f"- {m}" for m in sorted(mechanisms)],
        "",
    ])
    # ── 科学模态证据索引 ──
    artifact_path = DATA_DIR / "scientific_artifact_index.csv"
    if artifact_path.exists():
        try:
            import csv as csv_mod
            with open(artifact_path, "r", encoding="utf-8-sig") as f:
                reader = csv_mod.DictReader(f)
                artifact_counts = {}
                for row in reader:
                    at = row.get("artifact_type", "unknown")
                    artifact_counts[at] = artifact_counts.get(at, 0) + 1
            lines.extend([
                "## 科学模态证据索引",
                "",
                "以下从文献文本中索引到的科学证据类型（轻量文本匹配，非图像识别）：",
                "",
            ])
            for at, cnt in sorted(artifact_counts.items()):
                lines.append(f"- **{at}**: {cnt} 篇文献涉及")
            lines.append("")
            lines.append("> 索引方式：从文献卡片文本（key_findings / mechanical_indicators / "
                         "characterization_methods 等字段）中匹配关键词。")
            lines.append("> 如需精确识别图表中的曲线和数据，需引入 OCR/图表解析技术。")
            lines.append("")
        except Exception:
            pass

    lines.extend([
        "---",
        "",
        "## 当前证据评估",
        "",
    ])

    if total < 5:
        lines.append(f"> ⚠️ **当前文献库规模极小（仅 {total} 篇），以下仅为流程框架验证。**")
        lines.append("> 所有覆盖分析、研究空白检测和推荐方向都将标注'低覆盖度/高不确定性'。")
        lines.append("> 要形成有说服力的科学假设推荐，建议收集 30—100 篇钛合金疲劳相关文献。")
    elif total < 30:
        lines.append(f"> ⚠️ **文献库中等规模（{total} 篇），部分覆盖分析有初步参考价值。**")
    else:
        lines.append(f"> ✅ **文献库基本充实（{total} 篇），覆盖分析具有较高参考价值。**")

    out_path = OUTPUTS_DIR / "01_evidence_map.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)

"""
pdf_upload_handler.py — PDF 拖拽上传与自动入库处理模块

10 阶段流水线:
  Stage  1: save_uploaded_pdf
  Stage  2: extract_pdf_text
  Stage  3: extract_pdf_metadata
  Stage  4: classify_paper_type
  Stage  5: move_pdf_to_storage_folder
  Stage  6: write_candidate_papers_csv
  Stage  7: write_literature_database_csv
  Stage  8: chunk_pdf_text
  Stage  9: update_rag_index
  Stage 10: extract_structured_evidence

每个阶段独立 try/except，不阻断后续阶段。

标准数据结构 uploaded_paper_record（所有函数共用）:
"""

import csv
import json
import os
import re
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.stage1_store import (
    CANONICAL_PDF_DIR,
    RAG_CHUNKS_DIR,
    RAG_INDEX_PATH,
    register_pdf_bytes,
    update_paper_rag_status,
)
from src.deep_read_pipeline import deep_read_pdf

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = CANONICAL_PDF_DIR
CANDIDATE_DIR = CANONICAL_PDF_DIR
LOG_DIR = BASE_DIR / "logs"

CANDIDATES_CSV = DATA_DIR / "candidate_papers.csv"
LIT_DB_CSV = DATA_DIR / "literature_database.csv"
EVIDENCE_CSV = DATA_DIR / "evidence_snippets.csv"
VARIABLE_RELATION_CSV = DATA_DIR / "variable_relation_dataset.csv"
EQUATION_PARAM_CSV = DATA_DIR / "equation_parameter_dataset.csv"
PAPER_INDEX_JSON = RAG_INDEX_PATH
ERROR_LOG = LOG_DIR / "pdf_upload_errors.log"

# ── 分类目标文件夹 ──
TYPE_FOLDER_MAP = {
    "review": "reviews",
    "pore_fatigue_life": "pore_fatigue_life",
    "micro_ct_defects": "micro_ct_defects",
    "surface_roughness": "surface_roughness",
    "hip_heat_treatment": "hip_heat_treatment",
    "fcgr_paris_law": "fcgr_paris_law",
    "defect_tolerance_models": "defect_tolerance_models",
    "ai_materials_fatigue": "ai_materials_fatigue",
    "other": "other",
}

EXISTING_TO_NEW_TYPE = {
    "review": "review",
    "fcgr_paper": "fcgr_paris_law",
    "micro_ct_paper": "micro_ct_defects",
    "hip_heat_treatment_paper": "hip_heat_treatment",
    "surface_roughness_paper": "surface_roughness",
    "experimental_fatigue_paper": "pore_fatigue_life",
    "defect_tolerance_models": "defect_tolerance_models",
    "ai_materials_fatigue": "ai_materials_fatigue",
    "candidate": "other",
    "other": "other",
}

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)

FATIGUE_TERMS = [
    "fatigue", "ti-6al-4v", "ti64", "titanium", "l-pbf", "slm",
    "additive manufacturing", "pore", "crack", "defect",
]


# ═══════════════════════════════════════════════════════════════════════════
# 0. 工具函数
# ═══════════════════════════════════════════════════════════════════════════

def safe_get(record: Dict[str, Any], key: str, default: Any = "") -> Any:
    """安全获取字典字段，避免 KeyError。"""
    if not isinstance(record, dict):
        return default
    return record.get(key, default)


def make_record() -> Dict[str, Any]:
    """创建标准的 uploaded_paper_record — 始终返回 dict。"""
    return {
        # ── 基本信息 ──
        "filename": "",
        "title": "",
        "authors": "",
        "year": "",
        "doi": "",
        "abstract": "",
        "paper_type_primary": "",
        "paper_type_secondary": "",
        "storage_folder": "",
        "local_path": "",
        "source_path": "",
        "canonical_pdf_path": "",
        "file_hash_sha256": "",
        "pdf_valid": False,
        "real_page_count": 0,
        "file_size": 0,
        "ingest_status": "not_started",
        "duplicate_status": "UNKNOWN",
        "duplicate_of": "",
        "data_version": "stage1.0",
        "full_text": "",
        "candidate_id": None,
        "paper_id": None,
        "is_duplicate": False,
        "needs_manual_classification": False,

        # ── Stage 状态（两级入库） ──
        "save_status": "not_started",              # saved / failed
        "text_extraction_status": "not_started",   # success / partial / failed / scanned_pdf
        "metadata_status": "not_started",          # verified_by_doi / user_uploaded_metadata_pending / metadata_uncertain
        "classification_status": "not_started",    # success / uncertain / failed
        "candidate_csv_status": "not_started",     # added_to_candidate / duplicate / failed
        "lit_db_status": "not_started",            # ingested / not_ingested / failed
        "chunk_status": "not_started",             # success / failed
        "rag_index_status": "not_started",         # indexed_candidate / indexed_formal / failed / not_started
        "evidence_extraction_status": "not_started", # success / partial / failed / not_started

        # ── 详细状态 ──
        "n_chunks": 0,
        "rag_method": "",
        "rag_scope": "",
        "evidence_count": 0,
        "deep_read_status": "not_started",
        "deep_read_complete": False,
        "page_record_path": "",
        "page_coverage_ratio": 0.0,
        "variable_relations_count": 0,
        "equations_count": 0,

        # ── 错误 ──
        "stage_errors": {},  # {stage_name: error_message}
        "error_message": "",
    }


CHUNK_RECORD_KEYS = [
    "chunk_id", "paper_id", "candidate_id", "title",
    "paper_type_primary", "source_path", "chunk_index", "text",
    "token_count", "created_time",
]

CHUNKS_DIR = RAG_CHUNKS_DIR


def ensure_dirs():
    """确保所有需要的目录存在。"""
    CANONICAL_PDF_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_INDEX_JSON.parent.mkdir(parents=True, exist_ok=True)


def _log_error(filename: str, stage: str, error: str, tb: str = ""):
    """记录处理错误到日志文件。"""
    ensure_dirs()
    try:
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {filename} | {stage}\n")
            f.write(f"  Error: {error}\n")
            if tb:
                for line in tb.split("\n")[-6:]:
                    f.write(f"  {line}\n")
            f.write("\n")
    except Exception:
        pass


def _load_csv(path: Path) -> List[Dict[str, str]]:
    """加载 CSV 为 dict 列表。"""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _next_candidate_id(existing: List[Dict[str, str]]) -> str:
    """生成下一个候选 ID。"""
    max_num = 0
    for e in existing:
        cid = e.get("candidate_id", "")
        if cid.startswith("CAND_"):
            try:
                num = int(cid.split("_")[1])
                if num > max_num:
                    max_num = num
            except (IndexError, ValueError):
                pass
    return f"CAND_{max_num + 1:04d}"


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: save_uploaded_pdf
# ═══════════════════════════════════════════════════════════════════════════

def stage_save_pdf(record: Dict[str, Any], pdf_content: bytes) -> Dict[str, Any]:
    """Stage-1 compatibility wrapper: validate and register a canonical PDF."""
    try:
        ensure_dirs()
        original_filename = safe_get(record, "filename", "unknown.pdf")
        registered = register_pdf_bytes(
            pdf_content,
            original_filename,
            source_path=safe_get(record, "source_path", original_filename),
            source_type="USER_UPLOAD",
            metadata_override=dict(record.get("_metadata_override") or {}),
            base_dir=Path(safe_get(record, "_base_dir", BASE_DIR)),
        )
        if not registered.get("pdf_valid"):
            record.update(registered)
            record["save_status"] = "failed"
            record["ingest_status"] = "REJECTED_INVALID_PDF"
            record["stage_errors"]["save_pdf"] = registered.get(
                "error", "invalid PDF"
            )
            return record
        record.update(registered)
        record["is_duplicate"] = registered.get("status") == "DUPLICATE"
        record["save_status"] = "success"
    except Exception as e:
        tb = traceback.format_exc()
        _log_error(safe_get(record, "filename"), "stage_save_pdf", str(e), tb)
        record["save_status"] = "failed"
        record["stage_errors"]["save_pdf"] = str(e)
    return record


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: extract_pdf_text
# ═══════════════════════════════════════════════════════════════════════════

def stage_extract_text(record: Dict[str, Any]) -> Dict[str, Any]:
    """抽取 PDF 文本。如果是扫描版 PDF，标记 scanned_or_no_text。"""
    local_path = safe_get(record, "local_path", "")
    if not local_path or not Path(local_path).exists():
        record["text_extraction_status"] = "failed"
        record["stage_errors"]["extract_text"] = "PDF file not found"
        return record

    try:
        import fitz
    except ImportError:
        record["text_extraction_status"] = "failed"
        record["stage_errors"]["extract_text"] = "PyMuPDF not installed"
        _log_error(safe_get(record, "filename"), "stage_extract_text", "PyMuPDF not installed")
        return record

    try:
        doc = fitz.open(local_path)
    except Exception as e:
        record["text_extraction_status"] = "failed"
        record["stage_errors"]["extract_text"] = f"Cannot open PDF: {e}"
        return record

    full_text = ""
    try:
        for page in doc:
            text = page.get_text()
            full_text += text + "\n"
    except Exception as e:
        doc.close()
        record["text_extraction_status"] = "partial"
        record["stage_errors"]["extract_text"] = f"Partial extraction: {e}"
        if full_text:
            record["full_text"] = full_text.strip()
        return record
    finally:
        doc.close()

    full_text = full_text.strip()
    if len(full_text) > 500:
        record["full_text"] = full_text
        record["text_extraction_status"] = "success"
    elif len(full_text) > 50:
        record["full_text"] = full_text
        record["text_extraction_status"] = "partial"
    else:
        record["full_text"] = ""
        record["text_extraction_status"] = "scanned_or_no_text"

    return record


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: extract_pdf_metadata
# ═══════════════════════════════════════════════════════════════════════════

def stage_extract_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    """从 PDF 文本识别元数据。尝试 Crossref/OpenAlex 补全。"""
    full_text = safe_get(record, "full_text", "")
    first_page_text = full_text[:3000] if full_text else ""

    title = ""
    authors = ""
    year = ""
    doi = ""
    abstract = ""
    keywords = ""
    metadata_status = "user_uploaded_metadata_pending"

    try:
        # ── Title from first page ──
        lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]
        for line in lines[:20]:
            if re.match(r"^\d+$", line):
                continue
            if len(line) < 10:
                continue
            if re.match(r"^(abstract|keywords|introduction|doi|corresponding)", line, re.IGNORECASE):
                break
            if 20 <= len(line) <= 300:
                title = line[:300]
                break

        # ── DOI ──
        doi_matches = DOI_PATTERN.findall(full_text)
        if doi_matches:
            doi = doi_matches[0][:200]

        # ── Year ──
        year_match = re.search(r"(19|20)\d{2}", full_text)
        if year_match:
            year = year_match.group()

        # ── Abstract ──
        abs_match = re.search(
            r"Abstract[:\s]*(.+?)(?:\n\s*(?:Keywords|Introduction|1\.\s*|\Z))",
            full_text,
            re.IGNORECASE | re.DOTALL,
        )
        if abs_match:
            abstract = abs_match.group(1).strip()[:1000]

        # ── Keywords ──
        kw_match = re.search(
            r"Keywords[:\s]*(.+?)(?:\n\s*(?:Introduction|1\.\s*|\Z))",
            full_text,
            re.IGNORECASE | re.DOTALL,
        )
        if kw_match:
            keywords = kw_match.group(1).strip()[:500]

        # ── Try Crossref via DOI ──
        if doi:
            crossref_data = _fetch_crossref_metadata(doi)
            if crossref_data:
                if crossref_data.get("title"):
                    title = crossref_data["title"]
                if crossref_data.get("authors"):
                    authors = crossref_data["authors"]
                if crossref_data.get("year"):
                    year = crossref_data["year"]
                if crossref_data.get("abstract"):
                    abstract = crossref_data["abstract"]
                metadata_status = "verified_by_doi"

        # ── Try OpenAlex if no authors yet ──
        if doi and not authors:
            oa_data = _fetch_openalex_metadata(doi)
            if oa_data:
                if oa_data.get("authors"):
                    authors = oa_data["authors"]
                if oa_data.get("year"):
                    year = oa_data["year"]
                if oa_data.get("title"):
                    title = oa_data["title"]
                metadata_status = "verified_by_doi"

        # ── Check completeness ──
        identifiers = sum([bool(title), bool(authors), bool(doi), bool(year)])
        if identifiers < 2:
            metadata_status = "metadata_uncertain"

    except Exception as e:
        tb = traceback.format_exc()
        _log_error(safe_get(record, "filename"), "stage_extract_metadata", str(e), tb)
        metadata_status = "metadata_uncertain"
        record["stage_errors"]["extract_metadata"] = str(e)

    record["title"] = title
    record["authors"] = authors
    record["year"] = year
    record["doi"] = doi
    record["abstract"] = abstract
    record["metadata_status"] = metadata_status
    return record


def _fetch_crossref_metadata(doi: str) -> Optional[Dict[str, str]]:
    """通过 Crossref API 获取文献元数据。"""
    try:
        import requests
        url = f"https://api.crossref.org/works/{doi}"
        headers = {"User-Agent": "TitaniumFatigueChat/1.0 (mailto:research@example.com)"}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        message = data.get("message", {})

        result = {}
        titles = message.get("title", [])
        if titles:
            result["title"] = titles[0][:300]

        authors_list = message.get("author", [])
        if authors_list:
            author_names = []
            for a in authors_list:
                given = a.get("given", "")
                family = a.get("family", "")
                if given and family:
                    author_names.append(f"{family}, {given}")
                elif family:
                    author_names.append(family)
            result["authors"] = "; ".join(author_names)

        date_parts = message.get("published-print", {}).get("date-parts", [[]])
        if not date_parts[0]:
            date_parts = message.get("published-online", {}).get("date-parts", [[]])
        if date_parts[0]:
            result["year"] = str(date_parts[0][0])

        abstract = message.get("abstract", "")
        if abstract:
            abstract = re.sub(r"<[^>]+>", "", abstract)
            result["abstract"] = abstract[:1000]

        return result
    except Exception:
        return None


def _fetch_openalex_metadata(doi: str) -> Optional[Dict[str, str]]:
    """通过 OpenAlex API 获取文献元数据。"""
    try:
        import requests
        doi_encoded = doi.replace("/", "%2F").replace("(", "%28").replace(")", "%29")
        url = f"https://api.openalex.org/works/doi:{doi_encoded}"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()

        result = {}
        title = data.get("title", "")
        if title:
            result["title"] = title[:300]

        authorships = data.get("authorships", [])
        if authorships:
            author_names = []
            for a in authorships:
                name = (a.get("author") or {}).get("display_name", "")
                if name:
                    author_names.append(name)
            result["authors"] = "; ".join(author_names)

        pub_year = data.get("publication_year")
        if pub_year:
            result["year"] = str(pub_year)

        return result
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4: classify_paper_type
# ═══════════════════════════════════════════════════════════════════════════

def stage_classify(record: Dict[str, Any]) -> Dict[str, Any]:
    """根据 title、abstract、keywords、text_excerpt 分类。"""
    title = safe_get(record, "title", "")
    abstract = safe_get(record, "abstract", "")
    full_text = safe_get(record, "full_text", "")
    first_page = full_text[:2000] if full_text else ""

    try:
        from src.paper_classifier import classify_paper_type as base_classify
        base_result = base_classify(title, abstract)

        orig_primary = base_result.get("paper_type_primary", "other")
        new_primary = EXISTING_TO_NEW_TYPE.get(orig_primary, "other")

        orig_secondary = base_result.get("paper_type_secondary", "")
        new_secondary_parts = []
        if orig_secondary:
            for st in orig_secondary.split(";"):
                st = st.strip()
                mapped = EXISTING_TO_NEW_TYPE.get(st, "")
                if mapped and mapped != new_primary:
                    new_secondary_parts.append(mapped)
        new_secondary = "; ".join(new_secondary_parts)

        storage_folder = TYPE_FOLDER_MAP.get(new_primary, "other")

        record["paper_type_primary"] = new_primary
        record["paper_type_secondary"] = new_secondary
        record["storage_folder"] = storage_folder
        if new_primary == "other":
            record["classification_status"] = "uncertain"
            record["needs_manual_classification"] = True
        else:
            record["classification_status"] = "success"

    except ImportError:
        # Fallback keyword classification
        record = _fallback_classify(record, title, first_page)
        if record["paper_type_primary"]:
            record["classification_status"] = "success"
        else:
            record["classification_status"] = "failed"
            record["stage_errors"]["classify"] = "classification module unavailable"
    except Exception as e:
        tb = traceback.format_exc()
        _log_error(safe_get(record, "filename"), "stage_classify", str(e), tb)
        record["classification_status"] = "failed"
        record["stage_errors"]["classify"] = str(e)

    return record


def _fallback_classify(record: Dict[str, Any], title: str, text_excerpt: str) -> Dict[str, Any]:
    """回退分类（关键词匹配）。"""
    text = f"{title} {text_excerpt}".lower()
    rules = [
        ("review", ["review", "overview", "survey", "state of the art"]),
        ("pore_fatigue_life", ["pore", "porosity", "fatigue life", "nf", "cycles to failure"]),
        ("micro_ct_defects", ["micro-ct", "computed tomography", "x-ray", "microtomography"]),
        ("surface_roughness", ["surface roughness", "as-built surface", "surface finishing"]),
        ("hip_heat_treatment", ["hip", "hot isostatic pressing", "heat treatment", "annealing"]),
        ("fcgr_paris_law", ["crack growth", "fcgr", "da/dn", "paris law", "delta k"]),
        ("defect_tolerance_models", ["murakami", "kitagawa", "defect tolerance", "fatigue limit"]),
        ("ai_materials_fatigue", ["machine learning", "neural network", "data-driven", "artificial intelligence"]),
    ]

    matched = []
    for ptype, keywords in rules:
        for kw in keywords:
            if kw in text:
                matched.append(ptype)
                break

    if not matched:
        if any(kw in text for kw in ["fatigue", "ti-6al-4v", "ti64", "l-pbf", "slm"]):
            matched.append("pore_fatigue_life")
        else:
            matched.append("other")

    primary = matched[0]
    secondary = "; ".join(t for t in matched[1:])
    record["paper_type_primary"] = primary
    record["paper_type_secondary"] = secondary
    record["storage_folder"] = TYPE_FOLDER_MAP.get(primary, "other")
    if primary == "other":
        record["needs_manual_classification"] = True
    return record


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5: move_pdf_to_storage_folder
# ═══════════════════════════════════════════════════════════════════════════

def stage_move_pdf(record: Dict[str, Any]) -> Dict[str, Any]:
    """根据 paper_type_primary 移动到对应文件夹。"""
    local_path = safe_get(record, "local_path", "")
    if not local_path or not Path(local_path).exists():
        record["move_status"] = "failed"
        record["stage_errors"]["move_pdf"] = "source PDF not found"
        return record

    src = Path(local_path)
    folder = safe_get(record, "storage_folder", "candidate_papers")

    try:
        target_dir = PAPERS_DIR / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / src.name

        if target.exists():
            stem = target.stem
            suffix = target.suffix
            target = target_dir / f"{stem}_{int(datetime.now().timestamp())}{suffix}"

        shutil.move(str(src), str(target))
        record["local_path"] = str(target)
        record["move_status"] = "success"
    except Exception as e:
        tb = traceback.format_exc()
        _log_error(safe_get(record, "filename"), "stage_move_pdf", str(e), tb)
        record["move_status"] = "failed"
        record["stage_errors"]["move_pdf"] = str(e)

    return record


# ═══════════════════════════════════════════════════════════════════════════
# Stage 6: write_candidate_papers_csv
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_candidate_fields(fieldnames: List[str]) -> List[str]:
    """确保 candidate_papers.csv 包含所有必要字段。"""
    required = [
        "candidate_id", "title_original", "title_verified", "authors", "year",
        "journal", "doi", "url", "source_database", "abstract", "keywords",
        "is_verified", "verification_status", "is_open_access", "pdf_url",
        "paper_type_primary", "paper_type_secondary", "research_topic",
        "storage_folder", "tags", "relevance_score", "classification_reason",
        "classification_confidence", "download_status", "local_path", "status",
        "notes",
    ]
    for f in required:
        if f not in fieldnames:
            fieldnames.append(f)
    return fieldnames


def stage_write_candidate_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    """写入 candidate_papers.csv。无论是否正式入库，只要 PDF 上传成功都写入。"""
    try:
        ensure_dirs()
        existing = _load_csv(CANDIDATES_CSV)
        candidate_id = _next_candidate_id(existing)

        record["candidate_id"] = candidate_id

        meta_status = safe_get(record, "metadata_status", "user_uploaded_metadata_pending")
        if meta_status == "verified_by_doi":
            verification_status = "verified_by_doi"
        elif meta_status == "metadata_uncertain":
            verification_status = "metadata_uncertain"
        else:
            verification_status = "user_uploaded_metadata_pending"

        status = "candidate"
        notes_parts = []
        if safe_get(record, "is_duplicate"):
            notes_parts.append("重复文献")
            status = "duplicate"
        if meta_status == "metadata_uncertain":
            notes_parts.append("元数据不完整，需人工校正")
        notes = "; ".join(notes_parts)

        row = {
            "candidate_id": candidate_id,
            "title_original": safe_get(record, "title", ""),
            "title_verified": "",
            "authors": safe_get(record, "authors", ""),
            "year": safe_get(record, "year", ""),
            "journal": "",
            "doi": safe_get(record, "doi", ""),
            "url": "",
            "source_database": "user_upload",
            "abstract": safe_get(record, "abstract", ""),
            "keywords": "",
            "is_verified": "False",
            "verification_status": verification_status,
            "is_open_access": "False",
            "pdf_url": "",
            "paper_type_primary": safe_get(record, "paper_type_primary", "other"),
            "paper_type_secondary": safe_get(record, "paper_type_secondary", ""),
            "research_topic": "",
            "storage_folder": safe_get(record, "storage_folder", "candidate_papers"),
            "tags": "",
            "relevance_score": "",
            "classification_reason": "",
            "classification_confidence": "",
            "download_status": "uploaded",
            "local_path": safe_get(record, "local_path", ""),
            "status": status,
            "notes": notes,
        }

        # Merge with existing field structure
        all_rows = existing + [row]
        fieldnames = list(all_rows[0].keys()) if all_rows else list(row.keys())
        fieldnames = _ensure_candidate_fields(fieldnames)

        with open(CANDIDATES_CSV, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

        if safe_get(record, "is_duplicate"):
            record["candidate_csv_status"] = "duplicate"
        else:
            record["candidate_csv_status"] = "success"

    except Exception as e:
        tb = traceback.format_exc()
        _log_error(safe_get(record, "filename"), "stage_write_candidate_csv", str(e), tb)
        record["candidate_csv_status"] = "failed"
        record["stage_errors"]["candidate_csv"] = str(e)

    return record


# ═══════════════════════════════════════════════════════════════════════════
# Stage 7: write_literature_database_csv (条件入库)
# ═══════════════════════════════════════════════════════════════════════════

def _check_relevance(record: Dict[str, Any]) -> float:
    """检查与 L-PBF Ti-6Al-4V fatigue 的相关性。"""
    title = safe_get(record, "title", "")
    abstract = safe_get(record, "abstract", "")
    full_text = safe_get(record, "full_text", "")
    text = f"{title} {abstract} {full_text[:2000]}".lower()
    relevance = sum(1 for t in FATIGUE_TERMS if t in text)
    return min(relevance / 5.0, 1.0)


def _check_formal_conditions(record: Dict[str, Any]) -> Tuple[bool, str]:
    """检查是否满足正式入库条件。

    Returns:
        (can_ingest: bool, reason: str)
    """
    # Condition 1: paper_type_primary != other
    primary_type = safe_get(record, "paper_type_primary", "other")
    if primary_type == "other":
        return False, "文献类型为 other，不满足入库条件"

    # Condition 2: title / DOI / authors at least 2 identifiable
    title = safe_get(record, "title", "")
    doi = safe_get(record, "doi", "")
    authors = safe_get(record, "authors", "")
    identifiers = sum([bool(title), bool(doi), bool(authors)])
    if identifiers < 2:
        return False, f"可识别标识符不足 ({identifiers}/3)"

    # Condition 3: relevance >= 0.70
    relevance = _check_relevance(record)
    if relevance < 0.70:
        return False, f"相关度不足 ({relevance:.2f} < 0.70)"

    # Condition 4: not duplicate
    if safe_get(record, "is_duplicate"):
        return False, "重复文献"

    return True, f"满足入库条件 (relevance={relevance:.2f}, identifiers={identifiers})"


def stage_write_lit_db(record: Dict[str, Any]) -> Dict[str, Any]:
    """满足正式入库条件才写入 literature_database.csv。

    other 类型不是错误 → not_ingested + 提示人工分类。
    不满足条件 → not_ingested，不清除候选记录。
    """
    try:
        can_ingest, reason = _check_formal_conditions(record)

        if not can_ingest:
            # other 类型 → 标记需人工分类，但不算失败
            record["lit_db_status"] = "not_ingested"
            record["stage_errors"]["lit_db"] = reason
            if safe_get(record, "paper_type_primary", "") == "other":
                record["needs_manual_classification"] = True
            return record

        existing = _load_csv(LIT_DB_CSV)

        # Generate paper_id
        max_num = 0
        for e in existing:
            pid = e.get("paper_id", "")
            if pid.startswith("P"):
                try:
                    num = int(pid[1:])
                    if num > max_num:
                        max_num = num
                except ValueError:
                    pass
        paper_id = f"P{max_num + 1:04d}"
        record["paper_id"] = paper_id

        primary_type = safe_get(record, "paper_type_primary", "other")

        row = {
            "paper_id": paper_id,
            "title": safe_get(record, "title", ""),
            "authors": safe_get(record, "authors", ""),
            "year": safe_get(record, "year", ""),
            "doi": safe_get(record, "doi", ""),
            "material": "",
            "manufacturing_process": "",
            "heat_treatment": "",
            "surface_state": "",
            "defect_type": "",
            "pore_size": "",
            "pore_location": "",
            "porosity": "",
            "stress_ratio_R": "",
            "Nf": "",
            "fatigue_limit": "",
            "da_dN": "",
            "Paris_C": "",
            "Paris_m": "",
            "Delta_Kth": "",
            "main_conclusion": "",
            "paper_type_primary": primary_type,
            "paper_type_secondary": safe_get(record, "paper_type_secondary", ""),
            "research_topic": "",
            "storage_folder": safe_get(record, "storage_folder", ""),
            "tags": "",
            "is_review": str(primary_type == "review"),
            "is_experimental": str(primary_type == "pore_fatigue_life"),
            "is_fcgr": str(primary_type == "fcgr_paris_law"),
            "is_micro_ct": str(primary_type == "micro_ct_defects"),
            "is_heat_treatment": str(primary_type == "hip_heat_treatment"),
            "is_surface_roughness": str(primary_type == "surface_roughness"),
            "is_conflict_relevant": "",
            "classification_reason": "",
            "classification_confidence": "",
            "ingested_from": f"user_upload:{safe_get(record, 'candidate_id', '?')}",
        }

        all_rows = existing + [row]
        fieldnames = list(all_rows[0].keys()) if all_rows else list(row.keys())

        with open(LIT_DB_CSV, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

        record["lit_db_status"] = "success"

    except Exception as e:
        tb = traceback.format_exc()
        _log_error(safe_get(record, "filename"), "stage_write_lit_db", str(e), tb)
        record["lit_db_status"] = "failed"
        record["stage_errors"]["lit_db"] = str(e)

    return record


# ═══════════════════════════════════════════════════════════════════════════
# Stage 8: chunk_pdf_text
# ═══════════════════════════════════════════════════════════════════════════

def stage_chunk_text(record: Dict[str, Any]) -> Dict[str, Any]:
    """对全文进行 chunk。每个 chunk 是 dict，打包成 List[Dict]。"""
    full_text = safe_get(record, "full_text", "")
    paper_id = safe_get(record, "paper_id", "") or safe_get(record, "candidate_id", "")
    title = safe_get(record, "title", "")
    primary_type = safe_get(record, "paper_type_primary", "")
    source_path = safe_get(record, "local_path", "")

    if not full_text or len(full_text) < 100:
        record["chunk_status"] = "failed"
        record["stage_errors"]["chunk"] = "no text to chunk"
        return record

    try:
        words = full_text.split()
        chunk_texts = []
        chunk_size = 1000
        overlap = 100
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_texts.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = end - overlap

        # Build chunk records as List[Dict]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chunk_records = []
        for i, text in enumerate(chunk_texts):
            token_count = len(text.split())
            chunk_records.append({
                "chunk_id": f"CHUNK_{paper_id}_{i:04d}" if paper_id else f"CHUNK_unknown_{i:04d}",
                "paper_id": paper_id if paper_id else "",
                "candidate_id": safe_get(record, "candidate_id", ""),
                "title": title,
                "paper_type_primary": primary_type,
                "source_path": source_path,
                "chunk_index": str(i),
                "text": text,
                "token_count": str(token_count),
                "created_time": now,
            })

        if not paper_id:
            record["chunk_status"] = "failed"
            record["stage_errors"]["chunk"] = "no paper_id for chunk file"
            return record

        # Save as JSON list of dicts
        chunk_path = CHUNKS_DIR / f"{paper_id}.json"
        with open(str(chunk_path), "w", encoding="utf-8") as f:
            json.dump(chunk_records, f, ensure_ascii=False, indent=2)

        record["n_chunks"] = len(chunk_records)
        record["chunk_status"] = "success"

    except Exception as e:
        tb = traceback.format_exc()
        _log_error(safe_get(record, "filename"), "stage_chunk_text", str(e), tb)
        record["chunk_status"] = "failed"
        record["stage_errors"]["chunk"] = str(e)

    return record


# ═══════════════════════════════════════════════════════════════════════════
# Stage 9: update_rag_index
# ═══════════════════════════════════════════════════════════════════════════

def _load_paper_index() -> Dict[str, Any]:
    """安全加载 paper_index.json，保证返回 dict 而非 list。"""
    if not PAPER_INDEX_JSON.exists():
        return {}
    try:
        with open(PAPER_INDEX_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        # CRITICAL: ensure it's a dict, not a list
        if not isinstance(data, dict):
            _log_error("paper_index.json", "load", f"index is {type(data).__name__}, resetting to dict")
            return {}
        return data
    except (json.JSONDecodeError, Exception) as e:
        _log_error("paper_index.json", "load", str(e))
        return {}


def stage_update_rag(record: Dict[str, Any]) -> Dict[str, Any]:
    """把 chunk 写入 RAG 索引 (paper_index.json)。

    不依赖正式文献库状态。只要 chunk 成功，就写入索引。
    rag_scope = candidate（候选库）或 formal（正式库）。
    """
    paper_id = safe_get(record, "paper_id", "") or safe_get(record, "candidate_id", "")
    if not paper_id:
        record["rag_index_status"] = "failed"
        record["stage_errors"]["rag"] = "no paper_id/candidate_id"
        return record

    if safe_get(record, "chunk_status") != "success":
        record["rag_index_status"] = "not_started"
        record["stage_errors"]["rag"] = "text chunking not completed"
        return record

    n_chunks = safe_get(record, "n_chunks", 0)
    title = safe_get(record, "title", "") or "Untitled"
    local_path = safe_get(record, "local_path", "")
    lit_db_status = safe_get(record, "lit_db_status", "")

    try:
        # CRITICAL: _load_paper_index always returns a dict, never a list
        index = _load_paper_index()
        if not isinstance(index, dict):
            index = {}

        # Determine rag_scope
        has_formal_id = bool(safe_get(record, "paper_id", ""))
        rag_scope = "formal" if (has_formal_id and lit_db_status == "ingested") else "candidate"
        record["rag_scope"] = rag_scope

        index[paper_id] = {
            "title": title,
            "path": local_path,
            "n_chunks": n_chunks,
            "rag_scope": rag_scope,
            "paper_type_primary": safe_get(record, "paper_type_primary", ""),
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(PAPER_INDEX_JSON, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

        status = "indexed_candidate" if rag_scope == "candidate" else "indexed_formal"
        record["rag_index_status"] = status
        record["rag_method"] = "paper_index_json"
        update_paper_rag_status(paper_id, status.upper())

    except Exception as e:
        tb = traceback.format_exc()
        _log_error(safe_get(record, "filename"), "stage_update_rag", str(e), tb)
        record["rag_index_status"] = "failed"
        record["stage_errors"]["rag"] = str(e)

    return record


# ═══════════════════════════════════════════════════════════════════════════
# Stage 10: extract_structured_evidence
# ═══════════════════════════════════════════════════════════════════════════

def stage_extract_evidence(record: Dict[str, Any]) -> Dict[str, Any]:
    """从文本中抽取 evidence snippets、变量关系、方程/模型。"""
    full_text = safe_get(record, "full_text", "")
    if not full_text or len(full_text) < 100:
        record["evidence_extraction_status"] = "skipped"
        record["stage_errors"]["evidence"] = "no text available"
        return record

    try:
        title = safe_get(record, "title", "")
        year = safe_get(record, "year", "")
        primary_type = safe_get(record, "paper_type_primary", "other")
        paper_id = safe_get(record, "paper_id", "") or safe_get(record, "candidate_id", "unknown")

        # ── evidence_snippets.csv ──
        evidence_rows = _load_csv(EVIDENCE_CSV)
        sentences = re.split(r"(?<=[.!?])\s+", full_text)

        evidence_patterns = [
            r"(?:pore|porosity|defect).*(?:size|diameter|area|volume).*(?:fatigue|life|Nf|cycle)",
            r"(?:surface roughness|Ra|surface finish).*(?:fatigue|life|Nf)",
            r"(?:HIP|hot isostatic pressing).*(?:fatigue|defect|pore|life)",
            r"(?:da/dN|da_dN|crack growth).*(?:Paris|C|m|ΔK)",
            r"(?:fatigue limit|fatigue strength).*(?:pore|defect|roughness)",
            r"(?:crack initiation).*(?:pore|defect|surface)",
            r"(?:Murakami|Kitagawa).*(?:area|defect|fatigue limit)",
            r"(?:machine learning|neural network|prediction).*(?:fatigue|life)",
        ]

        direct_ev = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 30 or len(sent) > 2000:
                continue
            for pat in evidence_patterns:
                if re.search(pat, sent, re.IGNORECASE):
                    direct_ev.append(sent)
                    break

        new_ev_count = 0
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for sent in direct_ev[:10]:
            new_id = f"EVID_{len(evidence_rows) + new_ev_count + 1:04d}"
            evidence_rows.append({
                "evidence_id": new_id,
                "paper_id": paper_id,
                "title": title,
                "year": year,
                "paper_type": primary_type,
                "original_sentence": sent[:500],
                "extracted_claim": "",
                "independent_variable": "",
                "dependent_variable": "",
                "moderating_variables": "",
                "controlled_variables": "",
                "mechanism": "",
                "fatigue_indicator": "",
                "equation_or_model": "",
                "parameter_values": "",
                "experimental_method": "",
                "characterization_method": "",
                "evidence_type": "insufficient_evidence",
                "evidence_strength": "low",
                "direct_or_indirect": "unknown",
                "is_conflicting": "",
                "notes": "自动抽取，需人工校正",
                "creation_time": now,
                "source_field": "",
                "usable_for_validation": "",
            })
            new_ev_count += 1

        if new_ev_count > 0:
            fieldnames = list(evidence_rows[0].keys()) if evidence_rows else []
            with open(EVIDENCE_CSV, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(evidence_rows)

        record["evidence_count"] = new_ev_count

        # ── variable_relation_dataset.csv ──
        vr_rows = _load_csv(VARIABLE_RELATION_CSV)
        vr_new = 0
        for sent in direct_ev[:5]:
            vars_found = _extract_variable_pair(sent)
            if vars_found:
                iv, dv = vars_found
                vr_id = f"VR_{len(vr_rows) + vr_new + 1:04d}"
                vr_rows.append({
                    "variable_relation_id": vr_id,
                    "independent_variable": iv,
                    "dependent_variable": dv,
                    "moderating_variables": "",
                    "controlled_variables": "",
                    "mechanism": "",
                    "fatigue_indicator": "",
                    "equation_or_model": "",
                    "supporting_evidence_ids": "",
                    "evidence_count": "1",
                    "direction": "",
                    "notes": "自动抽取，需人工校正",
                })
                vr_new += 1

        if vr_new > 0:
            fieldnames = list(vr_rows[0].keys()) if vr_rows else []
            with open(VARIABLE_RELATION_CSV, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(vr_rows)

        record["variable_relations_count"] = vr_new

        # ── equation_parameter_dataset.csv ──
        eq_rows = _load_csv(EQUATION_PARAM_CSV)
        eq_new = 0
        eq_patterns = [
            (r"(?:Paris|da/dN|da_dN)\s*[:=].*?C.*?m", "Paris law"),
            (r"(?:Basquin|S-N|stress[- ]life).*?(?:σ|sigma|stress)", "Basquin equation"),
            (r"(?:Coffin|Manson|strain[- ]life).*?(?:ε|epsilon|strain)", "Coffin-Manson"),
        ]
        for line in full_text.split("\n"):
            for pat, eq_name in eq_patterns:
                if re.search(pat, line, re.IGNORECASE):
                    eq_id = f"EQ_{len(eq_rows) + eq_new + 1:04d}"
                    eq_rows.append({
                        "equation_parameter_id": eq_id,
                        "equation_or_model": eq_name,
                        "paper_id": paper_id,
                        "fatigue_indicator": "",
                        "parameter_values": line[:300],
                        "condition": "",
                        "evidence_id": "",
                        "notes": "自动抽取，需人工校正",
                    })
                    eq_new += 1
                    break

        if eq_new > 0:
            fieldnames = list(eq_rows[0].keys()) if eq_rows else []
            with open(EQUATION_PARAM_CSV, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(eq_rows)

        record["equations_count"] = eq_new

        if new_ev_count > 0:
            record["evidence_extraction_status"] = "success"
        else:
            record["evidence_extraction_status"] = "needs_human_correction"

    except Exception as e:
        tb = traceback.format_exc()
        _log_error(safe_get(record, "filename"), "stage_extract_evidence", str(e), tb)
        record["evidence_extraction_status"] = "failed"
        record["stage_errors"]["evidence"] = str(e)

    return record


def _extract_variable_pair(sentence: str) -> Optional[Tuple[str, str]]:
    """从句子中简单提取变量对。"""
    sentence_lower = sentence.lower()
    iv_keywords = {
        "pore size": "pore_size", "porosity": "porosity", "defect size": "pore_size",
        "surface roughness": "surface_roughness_Ra", "ra": "surface_roughness_Ra",
        "hip": "HIP_temperature", "heat treatment": "heat_treatment_temperature",
        "stress ratio": "stress_ratio_R", "stress amplitude": "stress_amplitude",
        "build orientation": "build_orientation",
        "pore diameter": "pore_size", "defect volume": "porosity",
    }
    dv_keywords = {
        "fatigue life": "Nf", "cycles to failure": "Nf", "fatigue limit": "fatigue_limit",
        "crack growth rate": "da_dN", "da/dn": "da_dN", "crack initiation": "crack_initiation",
        "fatigue strength": "fatigue_limit", "s-n curve": "S_N_curve",
    }
    found_iv = found_dv = None
    for kw, var in iv_keywords.items():
        if kw in sentence_lower:
            found_iv = var
            break
    for kw, var in dv_keywords.items():
        if kw in sentence_lower:
            found_dv = var
            break
    if found_iv and found_dv:
        return (found_iv, found_dv)
    if found_iv and any(kw in sentence_lower for kw in ["fatigue", "life", "failure", "crack"]):
        return (found_iv, "Nf")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 主流程: process_uploaded_pdf
# ═══════════════════════════════════════════════════════════════════════════

def process_uploaded_pdf(
    pdf_content: Union[bytes, bytearray, str, Path],
    original_filename: Optional[str] = None,
    *,
    base_dir: Path = BASE_DIR,
    metadata_override: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """处理上传 PDF 的 10 阶段流水线。

    核心原则:
      - other 类型不是错误，只是需要人工分类
      - 候选入库与正式入库解耦
      - RAG 不依赖正式文献库
      - 所有返回值都是 dict

    Args:
        pdf_content: PDF bytes. A path is accepted only as a backward-compatible
            adapter and is immediately converted to bytes.
        original_filename: 原始文件名。bytes 输入时必须提供；路径输入时可省略。

    Returns:
        uploaded_paper_record (始终是 dict，不会返回 list)
    """
    record = make_record()
    record["_base_dir"] = str(Path(base_dir).resolve())
    record["_metadata_override"] = dict(metadata_override or {})
    if isinstance(pdf_content, (str, Path)):
        source_path = Path(pdf_content)
        try:
            raw_content = source_path.read_bytes()
        except OSError as exc:
            record["save_status"] = "failed"
            record["ingest_status"] = "REJECTED_UNREADABLE_FILE"
            record["stage_errors"]["save_pdf"] = str(exc)
            record["error_message"] = f"[save_pdf] {exc}"
            return record
        original_filename = original_filename or source_path.name
        record["source_path"] = str(source_path.resolve())
        pdf_bytes = raw_content
    else:
        pdf_bytes = bytes(pdf_content)
        if not original_filename:
            raise TypeError(
                "original_filename is required when process_uploaded_pdf receives bytes"
            )
        record["source_path"] = original_filename

    record["filename"] = str(original_filename)

    # Stage 1: validate, hash, count real pages, deduplicate, and save canonically
    record = stage_save_pdf(record, pdf_bytes)
    if safe_get(record, "save_status") != "success":
        record["error_message"] = "; ".join(
            f"[{name}] {message}"
            for name, message in record.get("stage_errors", {}).items()
            if message
        )
        return record

    # Stage 2: Extract text
    record = stage_extract_text(record)

    # Stage 3 identity metadata was extracted during canonical registration.
    record["metadata_status"] = (
        "verified_by_doi" if safe_get(record, "doi") else "metadata_uncertain"
    )
    if not safe_get(record, "year"):
        publication_date = safe_get(record, "publication_date", "")
        year_match = re.search(r"(19|20)\d{2}", publication_date)
        if year_match:
            record["year"] = year_match.group(0)

    # Stage 4: Classify
    record = stage_classify(record)

    # Stage 2 deep read: real pages -> sequential scan -> category sweeps ->
    # omission audit -> provenance-safe evidence.
    deep_read = deep_read_pdf(
        safe_get(record, "canonical_pdf_path", safe_get(record, "local_path", "")),
        paper_id=safe_get(record, "paper_id", ""),
        title=safe_get(record, "title", ""),
        base_dir=Path(base_dir),
    )
    record["deep_read_status"] = deep_read.get("status", "FAILED")
    record["deep_read_complete"] = bool(deep_read.get("deep_read_complete"))
    record["page_record_path"] = deep_read.get("page_record_path", "")
    record["page_coverage_ratio"] = deep_read.get("page_coverage_ratio", 0.0)
    record["evidence_count"] = int(deep_read.get("evidence_count", 0))
    if deep_read.get("status") == "FAILED":
        record["stage_errors"]["deep_read"] = deep_read.get(
            "error", "Stage-2 deep read failed"
        )

    # Stage 5: no move. paper/pdfs is the only canonical PDF root.
    record["move_status"] = "canonical_path_preserved"
    record["storage_folder"] = "paper/pdfs"

    # Isolated test/migration roots use the same canonical contract without
    # touching the repository's legacy CSV compatibility outputs.
    if Path(base_dir).resolve() != BASE_DIR.resolve():
        record["candidate_csv_status"] = "not_applicable_isolated_root"
        record["lit_db_status"] = "not_applicable_isolated_root"
        record["chunk_status"] = "not_started"
        record["rag_index_status"] = "not_started"
        record["evidence_extraction_status"] = record["deep_read_status"]
        record["error_message"] = ""
        record.pop("_base_dir", None)
        record.pop("_metadata_override", None)
        return record

    # Stage 6/7: keep legacy metadata outputs for compatibility, but exact
    # duplicates do not create another legacy primary record.
    if not safe_get(record, "is_duplicate"):
        record = stage_write_candidate_csv(record)
        if safe_get(record, "candidate_csv_status") == "success":
            record["candidate_csv_status"] = "added_to_candidate"

        record = stage_write_lit_db(record)
    else:
        record["candidate_csv_status"] = "duplicate"
        record["lit_db_status"] = "duplicate"

    # Stage 8: Chunk text (if text available)
    if safe_get(record, "text_extraction_status") in ("success", "partial"):
        record = stage_chunk_text(record)

    # Stage 9: Update RAG index (even for candidate-only papers)
    if safe_get(record, "chunk_status") == "success":
        record = stage_update_rag(record)

    # Stage 10 now reflects the page-bound Stage-2 pipeline.  The former
    # title/folder-derived extractor remains disabled.
    record["evidence_extraction_status"] = record["deep_read_status"]

    # Compile error_message (non-blocking warnings only for real errors)
    real_errors = []
    for stage_name, err_msg in record.get("stage_errors", {}).items():
        if err_msg and stage_name not in ("lit_db",):
            # lit_db "error" is just not_ingested — not a real error
            real_errors.append(f"[{stage_name}] {err_msg}")
    record["error_message"] = "; ".join(real_errors)
    record.pop("_base_dir", None)
    record.pop("_metadata_override", None)

    return record


# ═══════════════════════════════════════════════════════════════════════════
# 检索功能
# ═══════════════════════════════════════════════════════════════════════════

def search_rag_chunks(paper_id: str, query: str = "", top_k: int = 5) -> List[Dict[str, Any]]:
    """从 RAG 索引检索指定 paper 的 chunk 信息。

    使用新的 chunk 格式 (List[Dict])。
    兼容旧的字符串列表格式。

    Returns:
        [{"chunk_id": str, "chunk_index": str, "snippet": str, "score": float}, ...]
    """
    results = []
    if not PAPER_INDEX_JSON.exists():
        return results

    index = _load_paper_index()
    if not isinstance(index, dict) or paper_id not in index:
        return results

    chunk_path = CHUNKS_DIR / f"{paper_id}.json"
    if not chunk_path.exists():
        return results

    try:
        with open(str(chunk_path), "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception:
        return results

    # Ensure chunks is a list
    if not isinstance(chunks, list):
        return results

    if not query:
        for i, chunk in enumerate(chunks[:top_k]):
            if isinstance(chunk, dict):
                chunk_id = chunk.get("chunk_id", f"CHUNK_{paper_id}_{i:04d}")
                text = chunk.get("text", "")
            elif isinstance(chunk, str):
                chunk_id = f"CHUNK_{paper_id}_{i:04d}"
                text = chunk
            else:
                continue
            results.append({
                "chunk_id": chunk_id,
                "chunk_index": str(i),
                "snippet": text[:300],
                "score": 1.0,
            })
        return results

    # Keyword matching scoring
    query_lower = query.lower()
    query_words = set(query_lower.split())

    scored = []
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id", f"CHUNK_{paper_id}_{i:04d}")
        elif isinstance(chunk, str):
            text = chunk
            chunk_id = f"CHUNK_{paper_id}_{i:04d}"
        else:
            continue

        chunk_lower = text.lower()
        match_count = sum(1 for w in query_words if w in chunk_lower)
        score = match_count / max(len(query_words), 1)
        if score > 0:
            scored.append((score, i, chunk_id, text))

    scored.sort(key=lambda x: -x[0])

    for score, idx, chunk_id, text in scored[:top_k]:
        results.append({
            "chunk_id": chunk_id,
            "chunk_index": str(idx),
            "snippet": text[:300],
            "score": round(score, 4),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 人工分类与入库管理
# ═══════════════════════════════════════════════════════════════════════════

VALID_PAPER_TYPES = [
    "review", "pore_fatigue_life", "micro_ct_defects",
    "surface_roughness", "hip_heat_treatment", "fcgr_paris_law",
    "defect_tolerance_models", "ai_materials_fatigue", "other",
]


def reclassify_paper(
    candidate_id: str,
    new_type: str,
    update_csv: bool = True,
    move_pdf_file: bool = True,
    re_rag: bool = True,
) -> Dict[str, Any]:
    """人工修改文献分类。

    Args:
        candidate_id: candidate_papers.csv 中的 candidate_id
        new_type: 目标分类（必须属于 VALID_PAPER_TYPES）
        update_csv: 是否更新 CSV 记录
        move_pdf_file: 是否移动 PDF 到新文件夹
        re_rag: 是否重建 RAG 索引 metadata

    Returns:
        {"success": bool, "message": str, "paper_id": str or None}
    """
    result = {"success": False, "message": "", "paper_id": None}

    if new_type not in VALID_PAPER_TYPES:
        result["message"] = f"无效分类: {new_type}"
        return result

    try:
        # 1. Update candidate_papers.csv
        candidates = _load_csv(CANDIDATES_CSV)
        target_row = None
        for row in candidates:
            if row.get("candidate_id", "") == candidate_id:
                target_row = row
                break

        if target_row is None:
            result["message"] = f"未找到 candidate_id={candidate_id}"
            return result

        old_type = target_row.get("paper_type_primary", "other")
        new_folder = TYPE_FOLDER_MAP.get(new_type, "other")
        old_folder = target_row.get("storage_folder", "candidate_papers")

        if update_csv:
            target_row["paper_type_primary"] = new_type
            target_row["storage_folder"] = new_folder
            target_row["classification_confidence"] = "manual"
            target_row["notes"] = (target_row.get("notes", "")
                                   + f"; reclassified from {old_type} to {new_type}")

            fieldnames = list(candidates[0].keys()) if candidates else []
            with open(CANDIDATES_CSV, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(candidates)

        # 2. Move PDF
        local_path = target_row.get("local_path", "")
        if move_pdf_file and local_path and Path(local_path).exists():
            src = Path(local_path)
            target_dir = PAPERS_DIR / new_folder
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / src.name
            if target_path.exists():
                stem = target_path.stem
                target_path = target_dir / f"{stem}_{int(datetime.now().timestamp())}{target_path.suffix}"
            shutil.move(str(src), str(target_path))

            # Update local_path in CSV
            target_row["local_path"] = str(target_path)

            if update_csv:
                with open(CANDIDATES_CSV, "w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(candidates)

        # 3. Try ingest to formal DB if now qualifies
        paper_id = None
        record = {
            "paper_type_primary": new_type,
            "paper_type_secondary": target_row.get("paper_type_secondary", ""),
            "title": target_row.get("title_original", ""),
            "doi": target_row.get("doi", ""),
            "authors": target_row.get("authors", ""),
            "abstract": target_row.get("abstract", ""),
            "full_text": "",
            "candidate_id": candidate_id,
            "is_duplicate": False,
            "storage_folder": new_folder,
            "local_path": target_row.get("local_path", ""),
        }

        can_ingest, _ = _check_formal_conditions(record)
        if can_ingest:
            existing_lit = _load_csv(LIT_DB_CSV)
            max_num = 0
            for e in existing_lit:
                pid = e.get("paper_id", "")
                if pid.startswith("P"):
                    try:
                        num = int(pid[1:])
                        if num > max_num:
                            max_num = num
                    except ValueError:
                        pass
            paper_id = f"P{max_num + 1:04d}"
            row = {
                "paper_id": paper_id,
                "title": record["title"],
                "authors": record["authors"],
                "year": target_row.get("year", ""),
                "doi": record["doi"],
                "material": "", "manufacturing_process": "", "heat_treatment": "",
                "surface_state": "", "defect_type": "", "pore_size": "",
                "pore_location": "", "porosity": "", "stress_ratio_R": "",
                "Nf": "", "fatigue_limit": "", "da_dN": "",
                "Paris_C": "", "Paris_m": "", "Delta_Kth": "",
                "main_conclusion": "",
                "paper_type_primary": new_type,
                "paper_type_secondary": target_row.get("paper_type_secondary", ""),
                "research_topic": "",
                "storage_folder": new_folder,
                "tags": "",
                "is_review": str(new_type == "review"),
                "is_experimental": str(new_type == "pore_fatigue_life"),
                "is_fcgr": str(new_type == "fcgr_paris_law"),
                "is_micro_ct": str(new_type == "micro_ct_defects"),
                "is_heat_treatment": str(new_type == "hip_heat_treatment"),
                "is_surface_roughness": str(new_type == "surface_roughness"),
                "is_conflict_relevant": "",
                "classification_reason": "",
                "classification_confidence": "manual",
                "ingested_from": f"user_upload:{candidate_id}",
            }
            all_rows = existing_lit + [row]
            fieldnames = list(all_rows[0].keys()) if all_rows else list(row.keys())
            with open(LIT_DB_CSV, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(all_rows)

            # Update candidate status
            target_row["status"] = "ingested"
            with open(CANDIDATES_CSV, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(candidates)

        # 4. Rebuild RAG metadata
        if re_rag and new_type != "other":
            rag_index = _load_paper_index()
            if isinstance(rag_index, dict):
                rag_index[candidate_id] = rag_index.get(candidate_id, {})
                rag_index[candidate_id]["paper_type_primary"] = new_type
                rag_index[candidate_id]["rag_scope"] = "formal" if paper_id else "candidate"
                with open(PAPER_INDEX_JSON, "w", encoding="utf-8") as f:
                    json.dump(rag_index, f, ensure_ascii=False, indent=2)

        result["success"] = True
        result["message"] = (f"文献 {candidate_id} 已从 {old_type} 重新分类为 {new_type}"
                             + (f"，已进入正式库" if paper_id else "，仍在候选库"))
        result["paper_id"] = paper_id

    except Exception as e:
        _log_error(candidate_id, "reclassify_paper", str(e), traceback.format_exc())
        result["message"] = f"重分类失败: {e}"

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 删除文献（同步清理）
# ═══════════════════════════════════════════════════════════════════════════

DELETE_OPERATION_LOG = LOG_DIR / "library_operations.log"


def _log_operation(op: str, detail: str):
    """记录库操作日志。"""
    ensure_dirs()
    try:
        with open(DELETE_OPERATION_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {op}: {detail}\n")
    except Exception:
        pass


def delete_paper_record(
    paper_id: str = "",
    candidate_id: str = "",
    delete_pdf: bool = False,
    delete_chunks: bool = True,
    delete_evidence: bool = True,
) -> Dict[str, Any]:
    """删除文献记录，自动同步清理关联数据。

    Args:
        paper_id: literature_database.csv 中的 paper_id
        candidate_id: candidate_papers.csv 中的 candidate_id
        delete_pdf: 是否同时删除本地 PDF 文件
        delete_chunks: 是否删除 chunks JSON
        delete_evidence: 是否删除 evidence snippets / 变量关系 / 方程参数

    Returns:
        {"success": bool, "message": str, "deleted": [str]}
    """
    deleted = []
    errors = []

    try:
        # 1. Remove from candidate_papers.csv
        if candidate_id:
            candidates = _load_csv(CANDIDATES_CSV)
            removed = [r for r in candidates if r.get("candidate_id", "") != candidate_id]
            diff = len(candidates) - len(removed)
            if diff > 0:
                fieldnames = list(candidates[0].keys()) if candidates else []
                with open(CANDIDATES_CSV, "w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(removed)
                deleted.append(f"candidate_csv:{candidate_id}")
                _log_operation("DELETE_CANDIDATE", f"{candidate_id}")

        # 2. Remove from literature_database.csv
        if paper_id:
            lit_rows = _load_csv(LIT_DB_CSV)
            removed = [r for r in lit_rows if r.get("paper_id", "") != paper_id]
            diff = len(lit_rows) - len(removed)
            if diff > 0:
                fieldnames = list(lit_rows[0].keys()) if lit_rows else []
                with open(LIT_DB_CSV, "w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(removed)
                deleted.append(f"lit_db:{paper_id}")
                _log_operation("DELETE_LIT", f"{paper_id}")

        # 3. Delete PDF file
        pid = paper_id or candidate_id or ""
        if delete_pdf and pid:
            rag_index = _load_paper_index()
            if isinstance(rag_index, dict) and pid in rag_index:
                pdf_path = rag_index[pid].get("path", "")
                if pdf_path and Path(pdf_path).exists():
                    Path(pdf_path).unlink()
                    deleted.append(f"pdf:{pdf_path}")
                    _log_operation("DELETE_PDF", pdf_path)

        # 4. Delete chunks
        if delete_chunks and pid:
            chunk_path = CHUNKS_DIR / f"{pid}.json"
            if chunk_path.exists():
                chunk_path.unlink()
                deleted.append(f"chunks:{pid}.json")

            # Remove from paper_index.json
            rag_index = _load_paper_index()
            if isinstance(rag_index, dict) and pid in rag_index:
                del rag_index[pid]
                with open(PAPER_INDEX_JSON, "w", encoding="utf-8") as f:
                    json.dump(rag_index, f, ensure_ascii=False, indent=2)
                deleted.append(f"rag_index:{pid}")

        # 5. Delete evidence snippets
        if delete_evidence and pid:
            ev_rows = _load_csv(EVIDENCE_CSV)
            remaining = [r for r in ev_rows if r.get("paper_id", "") != pid]
            diff = len(ev_rows) - len(remaining)
            if diff > 0:
                fieldnames = list(ev_rows[0].keys()) if ev_rows else []
                with open(EVIDENCE_CSV, "w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(remaining)
                deleted.append(f"evidence:{pid}")

            # Variable relations
            vr_rows = _load_csv(VARIABLE_RELATION_CSV)
            remaining = [r for r in vr_rows if r.get("paper_id", "") != pid]
            if len(vr_rows) != len(remaining):
                fieldnames = list(vr_rows[0].keys()) if vr_rows else []
                with open(VARIABLE_RELATION_CSV, "w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(remaining)
                deleted.append(f"variable_relations:{pid}")

            # Equation parameters
            eq_rows = _load_csv(EQUATION_PARAM_CSV)
            remaining = [r for r in eq_rows if r.get("paper_id", "") != pid]
            if len(eq_rows) != len(remaining):
                fieldnames = list(eq_rows[0].keys()) if eq_rows else []
                with open(EQUATION_PARAM_CSV, "w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(remaining)
                deleted.append(f"equation_params:{pid}")

            # Hypothesis dataset — mark as source_deleted instead of removing
            hyp_path = DATA_DIR / "hypothesis_dataset.csv"
            if hyp_path.exists() and delete_evidence:
                hyp_rows = _load_csv(hyp_path)
                modified = False
                for hr in hyp_rows:
                    src = hr.get("source_paper_ids", hr.get("source_scope", ""))
                    if pid in src:
                        hr["source_deleted"] = "True"
                        modified = True
                if modified:
                    fieldnames = list(hyp_rows[0].keys()) if hyp_rows else []
                    if "source_deleted" not in fieldnames:
                        fieldnames.append("source_deleted")
                    with open(hyp_path, "w", encoding="utf-8-sig", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                        writer.writeheader()
                        writer.writerows(hyp_rows)
                    deleted.append(f"hypothesis_marked:{pid}")

            # Research gap dataset — mark as source_deleted
            gap_path = DATA_DIR / "research_gap_dataset.csv"
            if gap_path.exists() and delete_evidence:
                gap_rows = _load_csv(gap_path)
                modified = False
                for gr in gap_rows:
                    src = gr.get("source_paper_ids", gr.get("source_scope", ""))
                    if pid in src:
                        gr["source_deleted"] = "True"
                        modified = True
                if modified:
                    fieldnames = list(gap_rows[0].keys()) if gap_rows else []
                    if "source_deleted" not in fieldnames:
                        fieldnames.append("source_deleted")
                    with open(gap_path, "w", encoding="utf-8-sig", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                        writer.writeheader()
                        writer.writerows(gap_rows)
                    deleted.append(f"gaps_marked:{pid}")

        msg = f"已删除 {len(deleted)} 项: {', '.join(deleted)}"
        if errors:
            msg += f"；错误: {'; '.join(errors)}"
        return {"success": True, "message": msg, "deleted": deleted}

    except Exception as e:
        tb = traceback.format_exc()
        _log_error(pid or "unknown", "delete_paper_record", str(e), tb)
        return {"success": False, "message": str(e), "deleted": deleted}


# ═══════════════════════════════════════════════════════════════════════════
# 重建 RAG 索引
# ═══════════════════════════════════════════════════════════════════════════

def rebuild_rag_index() -> Dict[str, Any]:
    """读取所有已入库文献，重新抽取文本、chunk、构建索引。

    不依赖 FAISS，使用 paper_index.json + chunks JSON。
    """
    stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0, "errors": []}

    try:
        # Collect all paper paths
        paper_paths = {}  # {id: {"path": str, "title": str, "type": str}}

        # From candidate_papers.csv
        candidates = _load_csv(CANDIDATES_CSV)
        for row in candidates:
            cid = row.get("candidate_id", "")
            path = row.get("local_path", "")
            if cid and path and Path(path).exists():
                paper_paths[cid] = {
                    "path": path,
                    "title": row.get("title_original", row.get("title", "")),
                    "type": row.get("paper_type_primary", "other"),
                }

        # From literature_database.csv
        lit_rows = _load_csv(LIT_DB_CSV)
        for row in lit_rows:
            pid = row.get("paper_id", "")
            path = row.get("local_path", "")
            if pid and path and Path(path).exists():
                paper_paths[pid] = {
                    "path": path,
                    "title": row.get("title", ""),
                    "type": row.get("paper_type_primary", "other"),
                }

        stats["total"] = len(paper_paths)

        # Reset index
        index = {}

        for pid, info in paper_paths.items():
            try:
                pdf_path = Path(info["path"])
                import fitz
                doc = fitz.open(str(pdf_path))
                full_text = ""
                for page in doc:
                    full_text += page.get_text() + "\n"
                doc.close()

                full_text = full_text.strip()
                if not full_text or len(full_text) < 100:
                    stats["skipped"] += 1
                    continue

                # Chunk
                words = full_text.split()
                chunk_texts = []
                chunk_size, overlap = 1000, 100
                start = 0
                while start < len(words):
                    end = min(start + chunk_size, len(words))
                    chunk_texts.append(" ".join(words[start:end]))
                    if end == len(words):
                        break
                    start = end - overlap

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                chunk_records = []
                for i, text in enumerate(chunk_texts):
                    chunk_records.append({
                        "chunk_id": f"CHUNK_{pid}_{i:04d}",
                        "paper_id": pid,
                        "candidate_id": pid if pid.startswith("CAND") else "",
                        "title": info["title"],
                        "paper_type_primary": info["type"],
                        "source_path": info["path"],
                        "chunk_index": str(i),
                        "text": text,
                        "token_count": str(len(text.split())),
                        "created_time": now,
                    })

                chunk_path = CHUNKS_DIR / f"{pid}.json"
                with open(str(chunk_path), "w", encoding="utf-8") as f:
                    json.dump(chunk_records, f, ensure_ascii=False, indent=2)

                index[pid] = {
                    "title": info["title"],
                    "path": info["path"],
                    "n_chunks": len(chunk_records),
                    "rag_scope": "formal" if pid.startswith("P") else "candidate",
                    "paper_type_primary": info["type"],
                    "updated": now,
                }
                stats["success"] += 1

            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append(f"{pid}: {e}")
                continue

        # Write index
        with open(PAPER_INDEX_JSON, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    except Exception as e:
        _log_error("rebuild_rag_index", "global", str(e), traceback.format_exc())
        stats["errors"].append(f"global: {e}")

    return stats

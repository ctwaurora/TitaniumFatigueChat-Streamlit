"""
data_cache.py — 统一的数据加载缓存层

使用 @st.cache_data 缓存所有 CSV 和 RAG 加载，
避免每次页面渲染重复读取。
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def _csv_path(name: str) -> Path:
    return DATA_DIR / name


# ── 一次性缓存所有 CSV 读取 ─────────────────────────────────────────────

@st.cache_data(show_spinner="加载文献库...")
def load_literature_database() -> pd.DataFrame:
    path = _csv_path("literature_database.csv")
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    return pd.DataFrame()


@st.cache_data(show_spinner="加载候选文献...")
def load_candidate_papers() -> pd.DataFrame:
    path = _csv_path("candidate_papers.csv")
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    return pd.DataFrame()


@st.cache_data(show_spinner="加载证据片段...")
def load_evidence_snippets() -> pd.DataFrame:
    """Load only canonical trusted evidence; legacy CSV is never a fallback."""
    from src.stage1_store import load_trusted_evidence_rows

    return pd.DataFrame(load_trusted_evidence_rows())


@st.cache_data(show_spinner="加载变量关系...")
def load_variable_relations() -> pd.DataFrame:
    path = _csv_path("variable_relation_dataset.csv")
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    return pd.DataFrame()


@st.cache_data(show_spinner="加载研究空白...")
def load_research_gap_dataset() -> pd.DataFrame:
    path = _csv_path("research_gap_dataset.csv")
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    return pd.DataFrame()


@st.cache_data(show_spinner="加载假设...")
def load_hypothesis_dataset() -> pd.DataFrame:
    path = _csv_path("hypothesis_dataset.csv")
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    return pd.DataFrame()


@st.cache_data(show_spinner="加载方程库...")
def load_equation_library() -> pd.DataFrame:
    path = _csv_path("equation_parameter_dataset.csv")
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    return pd.DataFrame()


@st.cache_data(show_spinner="加载疲劳方程库...")
def load_fatigue_equation_library() -> pd.DataFrame:
    path = _csv_path("fatigue_equation_library.csv")
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    return pd.DataFrame()


@st.cache_data(show_spinner="加载变量机制...")
def load_variable_mechanism() -> pd.DataFrame:
    path = _csv_path("variable_mechanism.csv")
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    return pd.DataFrame()


@st.cache_data(show_spinner="加载冲突检测...")
def load_conflict_claims() -> pd.DataFrame:
    path = _csv_path("conflict_claims.csv")
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    return pd.DataFrame()


@st.cache_data(show_spinner="加载基准评测...")
def load_benchmark_results() -> pd.DataFrame:
    path = _csv_path("benchmark_results.csv")
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    return pd.DataFrame()


@st.cache_data(show_spinner="加载钛合金疲劳基准...")
def load_titanium_fatigue_bench() -> pd.DataFrame:
    path = _csv_path("titanium_fatigue_bench.csv")
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    return pd.DataFrame()


@st.cache_data(show_spinner="加载基线对比...")
def load_baseline_comparison() -> pd.DataFrame:
    path = _csv_path("baseline_comparison.csv")
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    return pd.DataFrame()


@st.cache_data(show_spinner="加载消融结果...")
def load_ablation_results() -> pd.DataFrame:
    path = _csv_path("ablation_results.csv")
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    return pd.DataFrame()


@st.cache_data(show_spinner="加载追溯验证...")
def load_retrospective_validation_results() -> pd.DataFrame:
    path = _csv_path("retrospective_validation_results.csv")
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    return pd.DataFrame()


@st.cache_data(show_spinner="加载 Paris 验证...")
def load_paris_law_validation() -> pd.DataFrame:
    path = _csv_path("paris_law_validation_dataset.csv")
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    return pd.DataFrame()


@st.cache_data(show_spinner="加载失败案例...")
def load_failure_cases() -> pd.DataFrame:
    path = _csv_path("failure_cases.csv")
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    return pd.DataFrame()


@st.cache_data(show_spinner="加载追溯验证对...")
def load_retrospective_validation_pairs() -> pd.DataFrame:
    path = _csv_path("retrospective_validation_pairs.csv")
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    return pd.DataFrame()


# ── RAG 索引缓存 ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="加载 RAG 索引...")
def load_rag_index() -> Optional[Any]:
    """加载 Stage-1 唯一 RAG 真源。返回 None 表示未构建。"""
    try:
        from src.stage1_store import load_rag_index as load_canonical_rag_index

        index = load_canonical_rag_index()
        if not index:
            return None
        return {"ready": True, "engine": "keyword", "index": index}
    except Exception:
        return None


@st.cache_resource(show_spinner="加载嵌入模型...")
def load_embedding_model() -> Optional[Any]:
    """加载嵌入模型（占位，后续可接入实际模型）。"""
    return None


@st.cache_resource
def load_vector_store() -> Optional[Any]:
    """加载向量存储。"""
    return None


@st.cache_resource
def load_llm_client() -> Optional[Any]:
    """加载 DeepSeek 客户端，仅初始化一次。"""
    from src.api_keys import get_deepseek_settings

    settings = get_deepseek_settings()
    if not settings.configured:
        return None
    try:
        from src.deepseek_client import DeepSeekClient

        return DeepSeekClient(settings)
    except Exception:
        return None


# ── 缓存失效触发 ────────────────────────────────────────────────────────

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def build_pdf_duplicate_inventory(base_dir: Path = BASE_DIR) -> Dict[str, Any]:
    """Inventory PDFs without deleting or moving any user file."""
    from src.stage1_store import (
        load_paper_manifest,
        normalize_doi,
        normalize_title,
        sha256_file,
    )

    roots = [
        base_dir / "paper" / "pdfs",
        base_dir / "papers",
        base_dir / "early_papers",
        base_dir / "followup_papers",
    ]
    manifest = load_paper_manifest(base_dir)
    records_by_hash: Dict[str, List[Dict[str, Any]]] = {}
    for row in manifest:
        file_hash = str(row.get("file_hash_sha256") or "")
        if file_hash:
            records_by_hash.setdefault(file_hash, []).append(row)

    files_by_hash: Dict[str, List[Path]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.pdf"):
            try:
                files_by_hash.setdefault(sha256_file(path), []).append(
                    path.resolve()
                )
            except OSError:
                continue

    groups: Dict[str, Dict[str, Any]] = {}
    hash_to_group: Dict[str, str] = {}
    for file_hash, paths in files_by_hash.items():
        rows = records_by_hash.get(file_hash) or []
        row = rows[0] if rows else {}
        doi = normalize_doi(row.get("doi") or "")
        title = normalize_title(
            row.get("normalized_title") or row.get("title") or ""
        )
        canonical_paper_id = str(
            row.get("duplicate_of") or row.get("paper_id") or ""
        )
        group_key = (
            f"doi:{doi}"
            if doi
            else f"title:{title}"
            if title
            else f"paper:{canonical_paper_id}"
            if canonical_paper_id
            else f"sha256:{file_hash}"
        )
        group = groups.setdefault(
            group_key,
            {
                "canonical_paper_id": canonical_paper_id
                or f"SHA_{file_hash[:20].upper()}",
                "hashes": [],
                "paths": [],
                "paper_ids": [],
            },
        )
        group["hashes"].append(file_hash)
        group["paths"].extend(str(path) for path in paths)
        group["paper_ids"].extend(
            str(value.get("paper_id") or "")
            for value in rows
            if value.get("paper_id")
        )
        hash_to_group[file_hash] = group_key

    duplicate_rows: List[Dict[str, Any]] = []
    for group in groups.values():
        paths = list(dict.fromkeys(group["paths"]))
        if len(paths) < 2:
            continue
        canonical_path = paths[0]
        for linked_path in paths[1:]:
            duplicate_rows.append(
                {
                    "duplicate_of": canonical_path,
                    "linked_path": linked_path,
                    "canonical_paper_id": group["canonical_paper_id"],
                }
            )
    return {
        "local_pdf_file_count": sum(len(paths) for paths in files_by_hash.values()),
        "unique_pdf_sha256_count": len(files_by_hash),
        "unique_literature_count": len(groups),
        "duplicate_pdf_count": sum(
            max(0, len(paths) - 1) for paths in files_by_hash.values()
        ),
        "duplicate_relationships": duplicate_rows,
        "groups": groups,
        "hash_to_group": hash_to_group,
    }


def get_canonical_literature_counts(base_dir: Path = BASE_DIR) -> Dict[str, Any]:
    from src.corpus_statistics import read_corpus_statistics_snapshot

    snapshot = read_corpus_statistics_snapshot(base_dir)
    counts = dict(snapshot.get("counts") or {})
    return {
        "local_pdf_file_count": counts["pdf_file_count"],
        "unique_pdf_sha256_count": counts["unique_pdf_sha256_count"],
        # Backward-compatible field now means active canonical primary records.
        # It excludes related versions, archived/deleted records and aliases.
        "unique_literature_count": counts["current_logical_literature_count"],
        "active_canonical_primary_record_count": counts[
            "active_canonical_primary_record_count"
        ],
        "acquired_logical_literature_count": counts["acquired_logical_literature_count"],
        "duplicate_pdf_count": counts["duplicate_pdf_file_count"],
        "pdf_asset_count": counts["pdf_asset_count"],
        "canonical_paper_record_count": counts["canonical_paper_record_count"],
        "related_version_count": counts["related_version_count"],
        "archived_count": counts["archived_count"],
        "alias_old_id_count": counts["alias_old_id_count"],
        "historical_pre_cleanup_acquired_primary_count": counts[
            "historical_pre_cleanup_acquired_primary_count"
        ],
        "deep_read_complete_count": counts["deep_read_complete_count"],
        "indexed_count": counts["rag_paper_count"],
        "formal_indexed_count": counts["formal_indexed_count"],
        "complete_not_indexed_count": counts["complete_not_indexed_count"],
        "pending_processing_count": counts["pending_processing_count"],
        "processing_failed_count": counts["processing_failed_count"],
        "needs_human_review_count": counts["needs_human_review_count"],
        "pdf_not_acquired_count": counts["pdf_not_acquired_count"],
        "related_version_count": counts["related_version_count"],
        "out_of_scope_count": counts["out_of_scope_count"],
        "deleted_count": counts["deleted_count"],
        "duplicate_relationships": list(
            (snapshot.get("validation") or {}).get("duplicate_pdf_relationships") or []
        ),
    }


def _stats_signature(base_dir: Path = BASE_DIR) -> str:
    # Constant-time cache key: expensive crawls are reserved for explicit
    # refresh commands that rewrite this durable snapshot.
    paths = [base_dir / "data" / "system" / "corpus_statistics.json"]
    payload = []
    for path in paths:
        try:
            stat = path.stat()
            payload.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            continue
    return hashlib.sha256("\n".join(sorted(payload)).encode("utf-8")).hexdigest()


@st.cache_data
def _get_system_stats_cached(signature: str) -> Dict[str, Any]:
    """Read canonical literature counts; signature invalidates cross-process writes."""
    canonical = get_canonical_literature_counts(BASE_DIR)
    return {
        # Backward-compatible name now points to canonical unique literature.
        "n_papers": canonical["unique_literature_count"],
        **canonical,
        "legacy_csv_row_count": 0,
        "n_candidates": 0,
        "ev_count": 0,
        "vm_count": 0,
        "hypothesis_count": 0,
        "gap_count": 0,
        "benchmark_count": 0,
        "equation_count": 0,
        "primary_count": 0,
    }


def _file_version(*paths: Path) -> str:
    payload = []
    for path in paths:
        try:
            stat = path.stat(); payload.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            payload.append(f"{path}:missing")
    return hashlib.sha256("|".join(payload).encode()).hexdigest()


def literature_formula_version(base_dir: Path = BASE_DIR) -> str:
    snapshot = base_dir / "data/system/corpus_statistics.json"
    manifest = base_dir / "data/paper_manifest.jsonl"
    return _file_version(snapshot, manifest)


@st.cache_data(show_spinner="加载公式索引…")
def load_literature_formulas_cached(base_dir_text: str, dataset_version: str):
    from src.literature_formula_library import load_literature_formulas

    return load_literature_formulas(Path(base_dir_text))


def get_system_stats_cached() -> Dict[str, Any]:
    return _get_system_stats_cached(_stats_signature(BASE_DIR))


def clear_all_cache():
    """数据变更后主动清理缓存（上传/删除/重建后调用）。"""
    st.cache_data.clear()
    st.cache_resource.clear()


def mark_data_changed():
    """标记数据已变更（在 upload/delete/rebuild 后调用）。"""
    clear_all_cache()

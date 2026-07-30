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
    inventory = build_pdf_duplicate_inventory(base_dir)
    groups = inventory["groups"]
    deep_read_ids = set()
    deep_root = base_dir / "data" / "deep_read"
    if deep_root.exists():
        for status_path in deep_root.glob("*/extraction_status.json"):
            status = _read_json(status_path)
            if status.get("deep_read_complete") and status.get("paper_id"):
                deep_read_ids.add(str(status["paper_id"]))
    rag_manifest = _read_json(base_dir / "data" / "rag" / "manifest.json")
    indexed_ids = {
        str(value) for value in rag_manifest.get("paper_ids") or [] if value
    }
    # Each status layer already uses canonical paper_id and must not be
    # down-counted merely because older manifest rows lack a file hash.
    completed_count = len(deep_read_ids)
    indexed_count = len(indexed_ids)
    fully_processed_count = len(deep_read_ids & indexed_ids)
    pending_count = max(
        0, int(inventory["unique_literature_count"]) - fully_processed_count
    )
    return {
        **{
            key: value
            for key, value in inventory.items()
            if key not in {"groups", "hash_to_group"}
        },
        "deep_read_complete_count": completed_count,
        "indexed_count": indexed_count,
        "pending_or_failed_count": pending_count,
    }


def _stats_signature(base_dir: Path = BASE_DIR) -> str:
    paths = [
        base_dir / "data" / "paper_manifest.jsonl",
        base_dir / "data" / "rag" / "manifest.json",
        base_dir / "data" / "evidence" / "trusted_evidence.csv",
    ]
    for root in (
        base_dir / "paper" / "pdfs",
        base_dir / "papers",
        base_dir / "early_papers",
        base_dir / "followup_papers",
        base_dir / "data" / "deep_read",
    ):
        if root.exists():
            paths.extend(root.rglob("*.pdf"))
            paths.extend(root.glob("*/extraction_status.json"))
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
    lit = load_literature_database()
    candidates = load_candidate_papers()
    evidence = load_evidence_snippets()
    vr = load_variable_relations()
    hyp = load_hypothesis_dataset()
    gaps = load_research_gap_dataset()
    bm = load_benchmark_results()
    eq = load_equation_library()

    canonical = get_canonical_literature_counts(BASE_DIR)
    return {
        # Backward-compatible name now points to canonical unique literature.
        "n_papers": canonical["unique_literature_count"],
        **canonical,
        "legacy_csv_row_count": len(lit),
        "n_candidates": len(candidates),
        "ev_count": len(evidence),
        "vm_count": len(vr),
        "hypothesis_count": len(hyp),
        "gap_count": len(gaps),
        "benchmark_count": len(bm),
        "equation_count": len(eq),
        "primary_count": sum(1 for _, r in lit.iterrows() if str(r.get("paper_type_primary", "")).strip()),
    }


def get_system_stats_cached() -> Dict[str, Any]:
    return _get_system_stats_cached(_stats_signature(BASE_DIR))


def clear_all_cache():
    """数据变更后主动清理缓存（上传/删除/重建后调用）。"""
    st.cache_data.clear()
    st.cache_resource.clear()


def mark_data_changed():
    """标记数据已变更（在 upload/delete/rebuild 后调用）。"""
    clear_all_cache()

"""
data_cache.py — 统一的数据加载缓存层

使用 @st.cache_data 缓存所有 CSV 和 RAG 加载，
避免每次页面渲染重复读取。
"""

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

@st.cache_data
def get_system_stats_cached() -> Dict[str, Any]:
    """读取系统统计信息（缓存版）。"""
    lit = load_literature_database()
    candidates = load_candidate_papers()
    evidence = load_evidence_snippets()
    vr = load_variable_relations()
    hyp = load_hypothesis_dataset()
    gaps = load_research_gap_dataset()
    bm = load_benchmark_results()
    eq = load_equation_library()

    return {
        "n_papers": len(lit),
        "n_candidates": len(candidates),
        "ev_count": len(evidence),
        "vm_count": len(vr),
        "hypothesis_count": len(hyp),
        "gap_count": len(gaps),
        "benchmark_count": len(bm),
        "equation_count": len(eq),
        "primary_count": sum(1 for _, r in lit.iterrows() if str(r.get("paper_type_primary", "")).strip()),
    }


def clear_all_cache():
    """数据变更后主动清理缓存（上传/删除/重建后调用）。"""
    st.cache_data.clear()
    st.cache_resource.clear()


def mark_data_changed():
    """标记数据已变更（在 upload/delete/rebuild 后调用）。"""
    clear_all_cache()

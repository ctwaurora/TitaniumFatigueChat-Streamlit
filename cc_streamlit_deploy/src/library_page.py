"""
library_page.py — 文献库管理页面（性能优化版）

5 个区域：
1. 文献库概览（缓存统计）
2. 文献筛选与搜索
3. 轻量文献列表（分页）
4. 所选文献详情（按需加载）
5. 批量操作（按钮触发）
"""

import csv
import json
import time
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

import pandas as pd
import streamlit as st
from src.stage1_store import TRUSTED_EVIDENCE_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# ── 分页工具 ────────────────────────────────────────────────────────────

def paginate_dataframe(df: pd.DataFrame, page_size: int = 20, page_key: str = "lib_page") -> pd.DataFrame:
    """返回当前页的 DataFrame 切片，并渲染分页控件。"""
    total = len(df)
    total_pages = max(1, (total + page_size - 1) // page_size)

    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    if st.session_state[page_key] > total_pages:
        st.session_state[page_key] = total_pages

    start = (st.session_state[page_key] - 1) * page_size
    end = min(start + page_size, total)

    # 分页控件
    pgc = st.columns([1, 3, 1, 2])
    with pgc[0]:
        if st.button("◀ 上一页", disabled=(st.session_state[page_key] <= 1),
                     key=f"{page_key}_prev", use_container_width=True):
            st.session_state[page_key] = max(1, st.session_state[page_key] - 1)
            st.rerun()
    with pgc[1]:
        st.markdown(f"<div style='text-align:center;'>第 <b>{st.session_state[page_key]}</b>/{total_pages} 页</div>",
                    unsafe_allow_html=True)
    with pgc[2]:
        if st.button("下一页 ▶", disabled=(st.session_state[page_key] >= total_pages),
                     key=f"{page_key}_next", use_container_width=True):
            st.session_state[page_key] = min(total_pages, st.session_state[page_key] + 1)
            st.rerun()
    with pgc[3]:
        st.caption(f"显示 {start+1}-{end} / 共 {total} 条")

    return df.iloc[start:end]


def make_ui_key(prefix: str, text: str = "", row: int = 0) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", str(text))[:20]
    h = hashlib.md5(str(text).encode("utf-8")).hexdigest()[:6]
    return f"{prefix}_{row}_{h}"


# ── 中文映射 ────────────────────────────────────────────────────────────

PAPER_TYPE_CN = {
    "review": "综述文献",
    "pore_fatigue_life": "孔隙/缺陷-疲劳寿命",
    "micro_ct_defects": "micro-CT 缺陷表征",
    "surface_roughness": "表面粗糙度/表面状态",
    "hip_heat_treatment": "HIP/热处理",
    "fcgr_paris_law": "FCGR/Paris 裂纹扩展",
    "defect_tolerance_models": "缺陷容限模型",
    "ai_materials_fatigue": "AI/材料疲劳",
    "experimental_fatigue": "疲劳实验",
    "candidate": "候选未分类",
    "other": "其他/待确认",
}

STATUS_CN = {
    "verified": "已核验",
    "metadata_uncertain": "待确认",
    "duplicate": "重复",
    "rag_ready": "已入 RAG",
    "rag_pending": "未入 RAG",
    "downloaded": "已下载",
    "uploaded": "已上传",
}

RAG_STATUS_CN = {
    "indexed": "已入 RAG",
    "pending": "未入 RAG",
    "failed": "RAG 失败",
    "skipped": "跳过",
}


def paper_type_cn(code: str) -> str:
    return PAPER_TYPE_CN.get(code, code)


# ── 区域 1: 文献库概览（纯缓存统计） ────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_library_summary() -> Dict[str, Any]:
    """轻量统计：只读 CSV，不扫描磁盘。"""
    lit_path = DATA_DIR / "literature_database.csv"
    cand_path = DATA_DIR / "candidate_papers.csv"

    n_lit = 0
    n_cand = 0
    n_verified = 0
    n_unverified = 0
    n_rag_ready = 0
    n_rag_pending = 0

    if lit_path.exists():
        lit = pd.read_csv(lit_path, encoding="utf-8-sig", on_bad_lines="skip")
        n_lit = len(lit)
        if "verification_status" in lit.columns:
            n_verified += (lit["verification_status"] == "verified").sum()
        if "rag_status" in lit.columns:
            n_rag_ready += (lit["rag_status"] == "indexed").sum()
            n_rag_pending += (lit["rag_status"] == "pending").sum()

    if cand_path.exists():
        cand = pd.read_csv(cand_path, encoding="utf-8-sig", on_bad_lines="skip")
        n_cand = len(cand)
        if "status" in cand.columns:
            n_unverified = (cand["status"] != "verified").sum()
        if "rag_status" in cand.columns:
            n_rag_ready += (cand["rag_status"] == "indexed").sum()
            n_rag_pending += (cand["rag_status"] == "pending").sum()

    return {
        "n_lit": n_lit, "n_cand": n_cand,
        "n_verified": n_verified, "n_unverified": n_unverified,
        "n_rag_ready": n_rag_ready, "n_rag_pending": n_rag_pending,
    }


@st.cache_data(show_spinner=False)
def load_paper_types() -> List[str]:
    """获取所有文献分类（用于筛选）。"""
    types = set()
    for p in [DATA_DIR / "literature_database.csv", DATA_DIR / "candidate_papers.csv"]:
        if p.exists():
            df = pd.read_csv(p, encoding="utf-8-sig", on_bad_lines="skip")
            if "paper_type_primary" in df.columns:
                types.update(df["paper_type_primary"].dropna().unique())
    return sorted(types)


@st.cache_data(show_spinner=False)
def load_library_table(source: str = "all") -> pd.DataFrame:
    """加载文献列表（轻量表格数据，不含 evidence/chunks）。"""
    rows = []
    id_counter = 0

    for src_key, path, prefix in [
        ("candidate", DATA_DIR / "candidate_papers.csv", "CAND"),
        ("formal", DATA_DIR / "literature_database.csv", "P"),
    ]:
        if source not in ("all", src_key):
            continue
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
        for _, r in df.iterrows():
            id_counter += 1
            paper_id = str(r.get("paper_id", f"{prefix}_{id_counter:04d}"))
            title = str(r.get("title", "") or "")
            rows.append({
                "_id": id_counter,
                "paper_id": paper_id,
                "title": title[:100] if title else "(无题名)",
                "year": str(r.get("year", "") or ""),
                "type_code": str(r.get("paper_type_primary", "") or ""),
                "type_cn": paper_type_cn(str(r.get("paper_type_primary", "") or "")),
                "source": src_key,
                "verification": STATUS_CN.get(str(r.get("verification_status", "") or ""), "待确认"),
                "rag_status": RAG_STATUS_CN.get(str(r.get("rag_status", "") or ""), "未入 RAG"),
                "doi": str(r.get("doi", "") or "")[:60],
                "authors": str(r.get("authors", "") or "")[:60],
                "tags": str(r.get("tags", "") or ""),
                "storage_folder": str(r.get("storage_folder", "") or ""),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ── 区域 2: 按需加载详情 ────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_evidence_for_paper(paper_id: str) -> pd.DataFrame:
    """按需加载某篇文献的 evidence（只在点击详情后调用）。"""
    path = TRUSTED_EVIDENCE_PATH
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    if "paper_id" not in df.columns:
        return pd.DataFrame()
    return df[df["paper_id"] == paper_id]


@st.cache_data(show_spinner=False)
def load_chunks_for_paper(paper_id: str) -> List[Dict[str, Any]]:
    """按需加载某篇文献的 RAG chunks。"""
    path = DATA_DIR / "rag" / "chunks" / f"{paper_id}.json"
    if not path.exists():
        return []
    try:
        chunks = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return chunks if isinstance(chunks, list) else []


@st.cache_data(show_spinner=False)
def load_relations_for_paper(paper_id: str) -> pd.DataFrame:
    """按需加载某篇文献的变量关系。"""
    path = DATA_DIR / "variable_relation_dataset.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    if "supporting_evidence_ids" not in df.columns:
        return pd.DataFrame()
    # 检查哪些关系包含该文献的 evidence_id
    matched = df[df["supporting_evidence_ids"].str.contains(paper_id, na=False)]
    return matched


# ── 区域 4: 单篇文献详情渲染 ────────────────────────────────────────────

def render_paper_detail(paper_id: str, source: str):
    """渲染单篇文献的完整详情（按需加载）。"""
    # 从源 CSV 找到该文献
    csv_path = DATA_DIR / "literature_database.csv" if source == "formal" else DATA_DIR / "candidate_papers.csv"
    if not csv_path.exists():
        st.warning("文献数据文件不存在")
        return

    df = pd.read_csv(csv_path, encoding="utf-8-sig", on_bad_lines="skip")
    row = df[df["paper_id"] == paper_id]
    if row.empty:
        # 尝试用索引匹配
        try:
            idx = int(paper_id.split("_")[-1]) - 1
            if 0 <= idx < len(df):
                row = df.iloc[[idx]]
        except (ValueError, IndexError):
            pass
    if row.empty:
        st.warning(f"未找到文献: {paper_id}")
        return

    r = row.iloc[0]
    title = str(r.get("title", "") or "")

    st.markdown(f"### 📄 {title[:80]}")
    with st.expander("📋 基本信息", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"**题名**: {title}")
            st.markdown(f"**作者**: {str(r.get('authors', '') or '')}")
            st.markdown(f"**年份**: {str(r.get('year', '') or '')}")
            st.markdown(f"**期刊**: {str(r.get('doi', '') or '')}")
        with cols[1]:
            st.markdown(f"**DOI**: {str(r.get('doi', '') or '')}")
            st.markdown(f"**文献类型**: {paper_type_cn(str(r.get('paper_type_primary', '') or ''))}")
            st.markdown(f"**来源**: {source}")
            st.markdown(f"**核验状态**: {STATUS_CN.get(str(r.get('verification_status', '') or ''), '待确认')}")

    # 研究主题
    with st.expander("🔬 研究主题", expanded=False):
        topics = {
            "孔隙/缺陷": str(r.get("pore_size", "") or "") or str(r.get("defect_type", "") or ""),
            "表面粗糙度": str(r.get("surface_roughness_Ra", "") or "") or str(r.get("surface_state", "") or ""),
            "HIP/热处理": str(r.get("heat_treatment", "") or ""),
            "micro-CT": str(r.get("characterization_method", "") or ""),
            "FCGR/Paris": str(r.get("Paris_C", "") or "") or str(r.get("Paris_m", "") or ""),
            "S-N fatigue": str(r.get("Nf", "") or ""),
            "crack initiation": str(r.get("crack_initiation_site", "") or ""),
        }
        for topic, val in topics.items():
            st.markdown(f"- **{topic}**: {'✅ 涉及' if val else '❌ 未涉及'}")

    # 可提取变量
    with st.expander("📊 可提取变量", expanded=False):
        var_fields = [
            ("pore_size / area", "pore_size"), ("distance_to_surface", "distance_to_surface"),
            ("surface_state", "surface_state"), ("Ra/Rz", "surface_roughness_Ra"),
            ("Nf", "Nf"), ("crack_initiation_site", "crack_initiation_site"),
            ("DeltaK", "Delta_Kth"), ("da/dN", "da_dN"),
            ("Paris_C/m", "Paris_C"), ("fatigue_limit", "fatigue_limit"),
        ]
        cols = st.columns(3)
        for i, (name, field) in enumerate(var_fields):
            val = str(r.get(field, "") or "")
            with cols[i % 3]:
                st.markdown(f"- **{name}**: {'✅ ' + val if val else '❌'}")

    # 实验条件
    with st.expander("🧪 实验条件", expanded=False):
        cond_fields = [
            "material", "manufacturing_process", "heat_treatment", "surface_state",
            "stress_ratio_R", "fatigue_type", "characterization_method",
        ]
        for f in cond_fields:
            val = str(r.get(f, "") or "")
            st.markdown(f"- **{f}**: {val if val else '（未提取）'}")

    # 证据状态（按需加载）
    ev_df = load_evidence_for_paper(paper_id)
    with st.expander(f"📎 证据状态 (共 {len(ev_df)} 条)", expanded=False):
        if not ev_df.empty:
            n_direct = (ev_df["evidence_type"] == "direct_experimental_evidence").sum() if "evidence_type" in ev_df.columns else 0
            n_indirect = (ev_df["evidence_type"] == "indirect_mechanistic_evidence").sum() if "evidence_type" in ev_df.columns else 0
            n_conflict = (ev_df["evidence_type"] == "conflict_evidence").sum() if "evidence_type" in ev_df.columns else 0
            st.markdown(f"- 直接证据: {n_direct}")
            st.markdown(f"- 间接证据: {n_indirect}")
            st.markdown(f"- 冲突证据: {n_conflict}")

            # 缺失条件
            if "missing_condition_fields" in ev_df.columns:
                missing = ev_df["missing_condition_fields"].dropna().unique()
                if len(missing) > 0:
                    st.markdown(f"- 缺失条件字段: {missing[0][:80] if missing[0] else '无'}")
        else:
            st.info("该文献暂无结构化证据片段")

    # 系统用途
    with st.expander("🎯 系统用途", expanded=False):
        usage_items = [
            ("背景综述", True),
            ("变量关系分析", bool(r.get("pore_size", "")) or bool(r.get("surface_roughness_Ra", ""))),
            ("研究空白发现", len(ev_df) > 0),
            ("假设生成", len(ev_df) > 0),
            ("实验方案设计", bool(r.get("heat_treatment", "")) or bool(r.get("surface_state", ""))),
            ("公式/模型验证", bool(r.get("Paris_C", "")) or bool(r.get("Paris_m", "")) or bool(r.get("Nf", ""))),
        ]
        for name, supported in usage_items:
            st.markdown(f"- {'✅' if supported else '❌'} {name}")


# ── 区域 5: 批量操作 ────────────────────────────────────────────────────

def render_batch_operations(selected_ids: List[str], source: str):
    """批量操作面板（按钮触发）。"""
    if not selected_ids:
        st.info("请先在列表中选择文献")
        return

    st.markdown(f"**已选 {len(selected_ids)} 篇文献**")

    bcols = st.columns(4)
    with bcols[0]:
        if st.button("🗑️ 删除所选", key="batch_delete_btn", use_container_width=True):
            st.session_state["batch_delete_ids"] = selected_ids
            st.session_state["show_delete_confirm"] = True

    with bcols[1]:
        if st.button("🔍 分析研究空白", key="batch_gap_btn", use_container_width=True):
            st.session_state["batch_gap_ids"] = selected_ids
            st.session_state["run_batch_gap"] = True

    with bcols[2]:
        if st.button("🧪 生成候选假设", key="batch_hyp_btn", use_container_width=True):
            st.session_state["batch_hyp_ids"] = selected_ids
            st.session_state["run_batch_hyp"] = True

    with bcols[3]:
        if st.button("🔄 重建 RAG", key="batch_rag_btn", use_container_width=True):
            st.session_state["batch_rag_ids"] = selected_ids
            st.session_state["run_batch_rag"] = True

    # ── 删除确认 ──
    if st.session_state.get("show_delete_confirm") and st.session_state.get("batch_delete_ids") == selected_ids:
        st.warning(f"⚠️ 将删除 {len(selected_ids)} 篇文献。请输入 DELETE 确认。")
        confirm_text = st.text_input("确认删除", key="delete_confirm_input")
        with st.expander("删除选项", expanded=True):
            dc = st.columns(2)
            with dc[0]:
                del_csv = st.checkbox("删除 CSV 记录", value=True, key="del_csv")
                del_chunks = st.checkbox("删除 RAG chunks", value=True, key="del_chunks")
            with dc[1]:
                del_evidence = st.checkbox("删除 Evidence", value=True, key="del_ev")
                del_pdf = st.checkbox("删除本地 PDF", value=False, key="del_pdf")

        if confirm_text == "DELETE":
            if st.button("✅ 确认删除", key="confirm_del_exec"):
                with st.spinner(f"正在删除 {len(selected_ids)} 篇文献..."):
                    from src.pdf_upload_handler import delete_paper_record
                    for pid in selected_ids:
                        delete_paper_record(
                            paper_id=pid,
                            delete_pdf=del_pdf,
                            delete_chunks=del_chunks,
                            delete_evidence=del_evidence,
                            delete_record=del_csv,
                        )
                    st.success(f"已删除 {len(selected_ids)} 篇文献")
                    st.session_state["show_delete_confirm"] = False
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()

    # ── 批量研究空白 ──
    if st.session_state.get("run_batch_gap") and st.session_state.get("batch_gap_ids") == selected_ids:
        with st.spinner("正在分析研究空白..."):
            try:
                from src.research_gap_discovery import discover_research_gaps, save_gaps
                # 构造综述文本
                review_parts = []
                csv_path = DATA_DIR / "literature_database.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path, encoding="utf-8-sig", on_bad_lines="skip")
                    for _, r in df.iterrows():
                        if str(r.get("paper_id", "")) in selected_ids:
                            review_parts.append(str(r.get("main_conclusion", "") or ""))

                review_text = "\n\n".join(review_parts) if review_parts else ""
                gaps = discover_research_gaps(review_text=review_text)
                save_gaps(gaps)
                st.success(f"研究空白分析完成！发现 {len(gaps)} 个候选空白。")
                st.session_state["run_batch_gap"] = False
            except Exception as e:
                st.error(f"分析失败: {e}")
                st.session_state["run_batch_gap"] = False

    # ── 批量假设生成 ──
    if st.session_state.get("run_batch_hyp") and st.session_state.get("batch_hyp_ids") == selected_ids:
        with st.spinner("正在生成候选假设..."):
            try:
                from src.hypothesis_dataset import generate_hypotheses
                hyps = generate_hypotheses(paper_ids=selected_ids)
                st.success(f"假设生成完成！共 {len(hyps)} 个候选假设。")
                st.session_state["run_batch_hyp"] = False
            except Exception as e:
                st.error(f"假设生成失败: {e}")
                st.session_state["run_batch_hyp"] = False

    # ── 批量 RAG 重建 ──
    if st.session_state.get("run_batch_rag") and st.session_state.get("batch_rag_ids") == selected_ids:
        with st.spinner("正在重建 RAG..."):
            try:
                from src.pdf_upload_handler import rebuild_paper_rag
                for pid in selected_ids:
                    rebuild_paper_rag(pid)
                st.success(f"已为 {len(selected_ids)} 篇文献重建 RAG。")
                st.session_state["run_batch_rag"] = False
            except Exception as e:
                st.error(f"RAG 重建失败: {e}")
                st.session_state["run_batch_rag"] = False


# ── 主页面渲染 ──────────────────────────────────────────────────────────

def render_library_page():
    """文献库管理页面主入口。"""
    DEMO_MODE = os.environ.get("DEMO_MODE", "False").lower() in ("true", "1", "yes")
    PAGE_SIZE = 10 if DEMO_MODE else 20

    st.markdown("<h1 style='text-align: center;'>📚 文献库管理</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#888;font-size:0.9rem;'>"
        "文献库管理用于维护 L-PBF Ti-6Al-4V 疲劳研究语料，包括正式文献、候选文献、"
        "文献分类、RAG 状态和结构化证据状态。该模块为后续智能检索、研究空白发现、"
        "假设生成和实验方案辅助提供数据基础。</p>",
        unsafe_allow_html=True
    )
    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # 区域 1: 文献库概览（纯缓存统计）
    # ══════════════════════════════════════════════════════════════════════
    summary = load_library_summary()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.metric("正式文献", summary["n_lit"])
    with c2: st.metric("候选文献", summary["n_cand"])
    with c3: st.metric("已核验", summary["n_verified"])
    with c4: st.metric("待确认", summary["n_unverified"])
    with c5: st.metric("已入 RAG", summary["n_rag_ready"])
    with c6: st.metric("未入 RAG", summary["n_rag_pending"])

    st.caption("统计仅从 CSV 读取，不扫描磁盘文件")
    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # 区域 2: 文献筛选与搜索
    # ══════════════════════════════════════════════════════════════════════
    with st.expander("🔍 筛选与搜索", expanded=True):
        fcols = st.columns([1, 1, 2])
        with fcols[0]:
            source_filter = st.selectbox("文献来源", ["全部", "正式库", "候选库"], key="lib_source")
        with fcols[1]:
            types = ["全部"] + [paper_type_cn(t) for t in load_paper_types()]
            type_filter = st.selectbox("文献分类", types, key="lib_type")
        with fcols[2]:
            keyword = st.text_input("关键词（题名/作者/DOI/标签）", placeholder="输入关键词...", key="lib_keyword")

    # 加载列表数据
    source_map = {"全部": "all", "正式库": "formal", "候选库": "candidate"}
    df = load_library_table(source=source_map.get(source_filter, "all"))

    # 应用筛选
    if type_filter != "全部":
        df = df[df["type_cn"] == type_filter]
    if keyword.strip():
        kw = keyword.lower()
        df = df[df.apply(lambda r: any(kw in str(r.get(c, "") or "").lower()
                                         for c in ["title", "authors", "doi", "tags", "paper_id"]), axis=1)]

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # 区域 3: 轻量文献列表（分页 + data_editor）
    # ══════════════════════════════════════════════════════════════════════
    st.subheader(f"📋 文献列表 ({len(df)} 篇)")

    if df.empty:
        st.info("当前筛选条件下无文献。")
        return

    # 构建显示表格
    display_df = df[["paper_id", "title", "year", "type_cn", "verification",
                     "rag_status", "doi", "source"]].copy()
    display_df.insert(0, "选择", False)
    display_df.columns = ["选择", "编号", "题名", "年份", "分类", "状态", "RAG 状态", "DOI", "来源"]
    display_df = display_df.reset_index(drop=True)

    # 分页
    page_df = paginate_dataframe(display_df, page_size=PAGE_SIZE, page_key="lib_table")

    # data_editor
    edited_df = st.data_editor(
        page_df,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=["编号", "题名", "年份", "分类", "状态", "RAG 状态", "DOI", "来源"],
        key="lib_data_editor",
        column_config={
            "选择": st.column_config.CheckboxColumn("选择", default=False),
            "编号": st.column_config.TextColumn("编号", width="small"),
            "题名": st.column_config.TextColumn("题名", width="large"),
            "分类": st.column_config.TextColumn("分类", width="medium"),
        }
    )

    # 获取选中文献和查看详情
    selected = edited_df[edited_df["选择"] == True]
    selected_paper_ids = selected["编号"].tolist() if not selected.empty else []

    # 查看详情按钮
    vcols = st.columns([1, 3])
    with vcols[0]:
        detail_paper_id = st.selectbox(
            "选择文献查看详情",
            options=df["paper_id"].tolist(),
            format_func=lambda pid: next((r["title"][:50] for _, r in df[df["paper_id"] == pid].iterrows()), pid),
            key="lib_detail_selector",
        )
    with vcols[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        view_detail = st.button("📄 查看详情", key="view_detail_btn")

    if view_detail and detail_paper_id:
        detail_source = "formal" if detail_paper_id in df[df["source"] == "formal"]["paper_id"].values else "candidate"
        with st.container(border=True):
            render_paper_detail(detail_paper_id, detail_source)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # 区域 5: 批量操作（按钮触发）
    # ══════════════════════════════════════════════════════════════════════
    st.subheader(f"⚙️ 批量操作 ({len(selected_paper_ids)} 篇选中)")
    render_batch_operations(selected_paper_ids, source_filter)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # 折叠区域：上传 PDF / RAG 管理
    # ══════════════════════════════════════════════════════════════════════
    if not DEMO_MODE:
        with st.expander("📤 上传新文献 PDF", expanded=False):
            st.markdown("上传 PDF 文件到系统。上传后仅处理新增文件，不触发全库重建。")
            uploaded_file = st.file_uploader("选择 PDF 文件", type=["pdf"], key="lib_pdf_upload")
            if uploaded_file is not None:
                from src.pdf_upload_handler import process_uploaded_pdf
                with st.spinner("正在处理上传的 PDF..."):
                    result = process_uploaded_pdf(
                        bytes(uploaded_file.getbuffer()),
                        uploaded_file.name,
                    )
                    st.cache_data.clear()
                    if result.get("pdf_valid"):
                        st.success(f"上传完成: {result.get('paper_id', '?')}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(
                            "PDF 无效或已损坏："
                            f"{result.get('error_message') or result.get('error', '无法解析')}"
                        )

        with st.expander("🔄 RAG 索引管理", expanded=False):
            st.markdown("管理 RAG 索引状态。仅点击按钮时执行操作。")
            rcols = st.columns(3)
            with rcols[0]:
                if st.button("📊 查看 RAG 状态", key="check_rag_status"):
                    from src.stage1_store import load_rag_index
                    index = load_rag_index()
                    ready = {
                        key: value
                        for key, value in index.items()
                        if isinstance(value, dict)
                        and value.get("status", "READY") != "BROKEN_PATH"
                    }
                    n_chunks = sum(
                        int(value.get("n_chunks", 0) or 0)
                        for value in ready.values()
                    )
                    st.info(f"RAG: {len(ready)} 篇文献，{n_chunks} 条 chunks")
            with rcols[1]:
                if st.button("🔄 重建全部 RAG", key="rebuild_all_rag"):
                    st.warning("全库 RAG 重建可能较慢，确认执行？")
                    if st.button("✅ 确认重建", key="confirm_rebuild_rag"):
                        with st.spinner("正在重建全部 RAG..."):
                            from skills.rag_skill import index_pdf_chunks, _save_chunks, _load_chunks
                            # 清理旧 chunks
                            _save_chunks([])
                            from src.stage1_store import CANONICAL_PDF_DIR
                            pdf_dir = CANONICAL_PDF_DIR
                            if pdf_dir.exists():
                                for pdf_path in pdf_dir.glob("*.pdf"):
                                    file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
                                    index_pdf_chunks(str(pdf_path), file_hash)
                            st.success("RAG 重建完成！")
                            st.cache_data.clear()
        with st.expander("⚠️ 危险操作区", expanded=False):
            st.warning("以下操作可能导致数据不可恢复。请谨慎操作。")
            if st.button("🗑️ 清空所有缓存数据", key="clear_all_data"):
                if st.checkbox("确认清空所有缓存数据", key="confirm_clear_data"):
                    st.cache_data.clear()
                    st.success("缓存已清空。")
    else:
        st.caption("ℹ️ 演示模式: PDF 上传和 RAG 管理已折叠。完整功能请本地运行。")

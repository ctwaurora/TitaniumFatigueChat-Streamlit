"""Streamlit literature-library management page.

All scientific actions are backed by the canonical Stage-1/2/3 state.  Legacy
CSV files are used only as metadata compatibility inputs, never as the source
of truth for counts or RAG status.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import streamlit as st

from src.literature_library import (
    HUMAN_CONFIRMED,
    HUMAN_REVISION_REQUIRED,
    VALID_METADATA,
    add_doi_or_url,
    backfill_invalid_candidates,
    canonical_pdf_records,
    delete_invalid_candidate,
    eligible_paper_ids,
    ingest_uploaded_pdf,
    invalid_candidate_records,
    library_statistics,
    load_candidate_records,
    quarantine_record,
    repair_candidate_metadata,
    rebuild_unified_rag,
    set_evidence_review_status,
    trusted_evidence_rows,
    valid_candidate_records,
)
from src.storage_adapter import CLOUD_TEMPORARY, detect_storage_backend


PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


def paper_type_cn(code: str) -> str:
    return PAPER_TYPE_CN.get(str(code or ""), str(code or "") or "待分类")


def paginate_dataframe(
    df: pd.DataFrame,
    page_size: int = 20,
    page_key: str = "lib_page",
) -> pd.DataFrame:
    total = len(df)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(max(1, int(st.session_state.get(page_key, 1))), total_pages)
    st.session_state[page_key] = page
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    left, center, right, count = st.columns([1, 3, 1, 2])
    with left:
        if st.button(
            "◀ 上一页",
            disabled=page <= 1,
            key=f"{page_key}_prev",
            width="stretch",
        ):
            st.session_state[page_key] = page - 1
            st.rerun()
    with center:
        st.markdown(
            f"<div style='text-align:center;'>第 <b>{page}</b>/{total_pages} 页</div>",
            unsafe_allow_html=True,
        )
    with right:
        if st.button(
            "下一页 ▶",
            disabled=page >= total_pages,
            key=f"{page_key}_next",
            width="stretch",
        ):
            st.session_state[page_key] = page + 1
            st.rerun()
    with count:
        st.caption(f"显示 {start + 1 if total else 0}-{end} / 共 {total} 条")
    return df.iloc[start:end]


def _clear_library_cache() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()


def _display_title(row: Dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    year = str(row.get("year") or "").strip()
    doi = str(row.get("doi") or "").strip()
    suffix = doi.rsplit("/", 1)[-1] if doi else "无 DOI"
    return f"{title}（{year or '年份未知'}）— {suffix}"


def _render_storage_status(base_dir: Path, mode: str, warning: str) -> None:
    left, right = st.columns([1, 3])
    with left:
        st.metric("存储后端", mode)
    with right:
        st.caption(f"当前工作目录：{base_dir}")
        version_path = PROJECT_ROOT / "DEPLOY_VERSION.json"
        try:
            version = json.loads(version_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            version = {}
        if version.get("source_commit"):
            st.caption(
                f"部署版本：`{str(version['source_commit'])[:8]}` · "
                f"应用版本：{version.get('application_version') or 'unknown'}"
            )
        if mode == CLOUD_TEMPORARY:
            st.warning(warning)
            st.warning("当前云端为临时存储模式，应用重启后新增PDF和索引可能丢失。")
            st.info(
                "云端临时模式不运行独立后台 worker。短任务会在当前网页请求内同步执行并显示进度；"
                "较大 PDF、慢速 OA 下载或大规模重建可能超过 Community Cloud 的请求时限或内存限制，"
                "超时后不会在后台继续，也不会保留为等待领取的 PENDING 任务。"
            )
        else:
            st.success("当前存储后端具备持久化语义。")


def _render_pdf_upload(base_dir: Path) -> None:
    st.markdown("#### 上传本地 PDF 并深读")
    files = st.file_uploader(
        "选择一篇或多篇 PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="lib_pdf_upload",
        help="系统验证真实 PDF 结构并按 SHA-256 查重，HTML 伪 PDF 会被拒绝。",
    )
    if not files:
        return
    file_rows = [
        {"文件名": item.name, "大小": f"{item.size / 1024:.1f} KB"}
        for item in files
    ]
    st.dataframe(pd.DataFrame(file_rows), hide_index=True, width="stretch")
    st.caption(
        "若 PDF 内嵌元数据不完整，可在单篇上传时手动补全；系统不会用文件名猜测题名。"
    )
    meta_cols = st.columns(2)
    with meta_cols[0]:
        manual_title = st.text_input(
            "真实题名（可选）",
            key="upload_manual_title",
            disabled=len(files) != 1,
        )
        manual_authors = st.text_input(
            "作者（可选）",
            key="upload_manual_authors",
            disabled=len(files) != 1,
        )
    with meta_cols[1]:
        manual_year = st.text_input(
            "年份（可选）",
            key="upload_manual_year",
            disabled=len(files) != 1,
        )
        manual_doi = st.text_input(
            "DOI（自动入 RAG 时必需）",
            key="upload_manual_doi",
            disabled=len(files) != 1,
        )
    if not st.button(
        "开始上传并执行完整摄取",
        type="primary",
        key="process_library_uploads",
    ):
        return
    progress = st.progress(0, text="准备处理 PDF…")
    results = []
    for index, uploaded in enumerate(files, start=1):
        progress.progress(
            (index - 1) / len(files),
            text=f"正在验证、查重和逐页深读：{uploaded.name}",
        )
        payload = bytes(uploaded.getbuffer())
        metadata_override = None
        if len(files) == 1:
            metadata_override = {
                "title": manual_title.strip(),
                "authors": manual_authors.strip(),
                "publication_date": manual_year.strip(),
                "doi": manual_doi.strip(),
            }
        result = ingest_uploaded_pdf(
            payload,
            uploaded.name,
            base_dir=base_dir,
            metadata_override=metadata_override,
        )
        results.append(result)
        if not result.get("pdf_valid"):
            st.error(
                f"{uploaded.name}：{result.get('error_message') or result.get('error') or 'PDF 无效'}"
            )
        elif result.get("is_duplicate"):
            st.info(
                f"{uploaded.name}：SHA/DOI/题名命中已有主记录 "
                f"{result.get('paper_id')}，未重复计数。"
            )
        elif result.get("deep_read_complete"):
            if result.get("rag_status") == "INDEXED_STAGE3_UNIFIED":
                st.success(
                    f"{uploaded.name}：{result.get('paper_id')}，"
                    f"{result.get('real_page_count')} 页，"
                    f"{result.get('evidence_count')} 条可信证据，已写入统一 RAG。"
                )
            else:
                st.warning(
                    f"{uploaded.name} 已完成深读，但自动门禁未通过，未写入 RAG："
                    f"{'; '.join(result.get('quality_gate_reasons') or [])}"
                )
        else:
            st.warning(
                f"{uploaded.name} 已保存，但深读状态为 "
                f"{result.get('deep_read_status') or 'PARTIAL'}。"
            )
    progress.progress(1.0, text="上传处理完成")
    st.session_state["last_upload_results"] = results
    _clear_library_cache()


def _render_doi_add(base_dir: Path) -> None:
    st.markdown("#### 2. 通过 DOI / URL 添加")
    value = st.text_input(
        "DOI 或包含 DOI 的 URL",
        placeholder="10.xxxx/xxxxx 或 https://doi.org/10.xxxx/xxxxx",
        key="library_doi_input",
    )
    acquire = st.checkbox(
        "若存在合法 OA PDF，则自动获取并执行逐页深读",
        value=True,
        key="library_acquire_oa",
    )
    if st.button("核验元数据并添加", key="library_add_doi"):
        if not value.strip():
            st.warning("请输入 DOI 或 URL。")
            return
        with st.spinner("正在查询 OpenAlex / Crossref，并核验 OA 来源…"):
            result = add_doi_or_url(
                value,
                base_dir=base_dir,
                acquire_oa_pdf=acquire,
            )
        if result.get("validation_status") != VALID_METADATA:
            st.error(result.get("error") or "未获得有效真实元数据。")
        else:
            st.success(
                f"已核验：{result.get('title')}；元数据源："
                f"{result.get('metadata_source')}；状态：{result.get('status')}。"
            )
            oa = result.get("oa_ingest") or {}
            if oa:
                st.json(oa, expanded=False)
            _clear_library_cache()


def _render_oa_topup(base_dir: Path) -> None:
    st.markdown("### 自动发现OA文献")
    st.caption(
        "自动检索与 OA 全文补充会在当前网页请求内同步完成合法 OA 下载、逐页深读、证据门禁和统一 RAG；"
        "关闭页面后不会转入后台继续运行。"
    )
    query = st.text_input(
        "检索主题",
        value="L-PBF Ti-6Al-4V fatigue pore defect crack initiation",
        key="library_auto_oa_query",
    )
    controls = st.columns(4)
    with controls[0]:
        max_new = st.number_input(
            "最大新增数量",
            min_value=1,
            max_value=3,
            value=1,
            step=1,
            key="library_auto_oa_max",
        )
    with controls[1]:
        topic_filter = st.selectbox(
            "文献主题筛选",
            [
                "疲劳与裂纹萌生",
                "孔隙与缺陷",
                "表面状态",
                "热处理与HIP",
                "裂纹扩展",
                "不限",
            ],
            key="library_auto_oa_topic",
        )
    with controls[2]:
        year_from = st.number_input(
            "起始年份",
            min_value=1900,
            max_value=datetime.now().year,
            value=2000,
            step=1,
            key="library_auto_oa_year_from",
        )
    with controls[3]:
        year_to = st.number_input(
            "结束年份",
            min_value=1900,
            max_value=datetime.now().year,
            value=datetime.now().year,
            step=1,
            key="library_auto_oa_year_to",
        )
    core_only = st.checkbox(
        "仅L-PBF/SLM Ti-6Al-4V疲劳核心文献",
        value=True,
        key="library_auto_oa_core",
    )
    if st.button(
        "开始自动发现并处理",
        type="primary",
        key="library_auto_oa_start",
    ):
        from src.auto_oa_pipeline import PHASES, run_auto_oa_discovery

        progress = st.progress(0.0, text="SEARCHING：正在查询 OA 元数据源…")
        phase_rows: List[Dict[str, Any]] = []

        def on_progress(phase: str, payload: Dict[str, Any]) -> None:
            phase_rows.append(payload)
            position = PHASES.index(phase) + 1 if phase in PHASES else 1
            progress.progress(
                min(position / len(PHASES), 1.0),
                text=f"{phase}：{payload.get('title') or payload.get('paper_id') or query}",
            )

        result = run_auto_oa_discovery(
            query,
            max_new=int(max_new),
            topic_filter=topic_filter,
            year_from=int(year_from),
            year_to=int(year_to),
            core_only=core_only,
            base_dir=base_dir,
            progress_callback=on_progress,
        )
        st.session_state["last_auto_oa_result"] = result
        if result.get("completed_count"):
            st.success(
                f"同步处理完成：{result['completed_count']} 篇通过自动门禁并进入统一 RAG。"
            )
        else:
            st.warning(
                "本次没有文献通过完整门禁；未创建 PENDING 任务，也不会在后台继续。"
            )
        if result.get("results"):
            display_rows = []
            for row in result["results"]:
                display_rows.append(
                    {
                        "真实题名": row.get("title"),
                        "作者": row.get("authors"),
                        "年份": row.get("year"),
                        "DOI": row.get("doi"),
                        "OA来源": row.get("oa_source"),
                        "下载状态": row.get("download_status"),
                        "HTTP": row.get("http_status"),
                        "真实页数": row.get("real_page_count"),
                        "深读状态": row.get("deep_read_status"),
                        "EvidenceRecord": row.get("evidence_count"),
                        "自动质量门禁": row.get("quality_gate"),
                        "RAG状态": row.get("rag_status"),
                        "失败原因": row.get("failure_reason"),
                    }
                )
            st.dataframe(pd.DataFrame(display_rows), hide_index=True, width="stretch")
        with st.expander("查看数据源与阶段日志", expanded=False):
            st.json(
                {
                    "source_results": result.get("source_results"),
                    "errors": result.get("errors"),
                    "phases": phase_rows,
                }
            )
        _clear_library_cache()


def render_ingestion_entries(base_dir: Path) -> None:
    st.subheader("📥 添加文献")
    _render_oa_topup(base_dir)
    st.divider()
    upload_tab, doi_tab = st.tabs(["上传本地 PDF", "DOI / URL 添加"])
    with upload_tab:
        _render_pdf_upload(base_dir)
    with doi_tab:
        _render_doi_add(base_dir)


def _render_statistics(base_dir: Path) -> None:
    stats = library_statistics(base_dir)
    labels = [
        ("唯一文献", "unique_literature"),
        ("候选元数据", "candidate_metadata"),
        ("PDF 已获取", "pdf_acquired"),
        ("深读完成", "deep_read_complete"),
        ("证据待确认", "evidence_pending"),
        ("已确认", "evidence_confirmed"),
        ("已入统一 RAG", "rag_indexed"),
        ("待处理/失败", "pending_or_failed"),
        ("异常元数据", "invalid_metadata"),
    ]
    first = st.columns(5)
    second = st.columns(4)
    for column, (label, key) in zip([*first, *second], labels):
        with column:
            st.metric(label, stats[key])
    st.caption(
        "统计真源：canonical PDF SHA/DOI/题名分组、Stage-2 extraction_status、"
        "可信 EvidenceRecord 与 Stage-3 unified RAG manifest。"
    )


def _normal_library_rows(base_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for row in canonical_pdf_records(base_dir):
        if row.get("validation_status") != VALID_METADATA:
            continue
        rows.append(
            {
                **row,
                "source": "formal",
                "authors": str(row.get("authors") or ""),
                "type_cn": paper_type_cn(row.get("type_code", "")),
                "verification": (
                    "部署快照（只读）"
                    if row.get("snapshot_read_only")
                    else "已核验"
                ),
            }
        )
    for row in valid_candidate_records(base_dir):
        rows.append(
            {
                **row,
                "deep_read_complete": False,
                "evidence_status": "无全文证据",
                "rag_status": "NOT_INDEXED",
                "type_cn": paper_type_cn(row.get("type_code", "")),
                "verification": "候选元数据",
                "selectable": False,
            }
        )
    return rows


def _render_evidence_table(evidence: Sequence[Dict[str, Any]]) -> None:
    if not evidence:
        st.info("该文献暂无可信 EvidenceRecord。")
        return
    display = pd.DataFrame(evidence)
    columns = [
        column
        for column in (
            "evidence_id",
            "page_number",
            "directness",
            "claim",
            "original_text",
            "review_status",
        )
        if column in display.columns
    ]
    st.dataframe(display[columns], hide_index=True, width="stretch")


def render_paper_detail(record: Dict[str, Any], base_dir: Path) -> None:
    paper_id = str(record["paper_id"])
    if record.get("snapshot_read_only"):
        st.info(
            "这是部署时生成的 canonical 状态快照。当前云端临时工作区没有该文献的 "
            "PDF/EvidenceRecord 运行文件；请重新上传 PDF 或配置外部持久化存储后再执行科研操作。"
        )
    evidence = [
        row
        for row in trusted_evidence_rows(base_dir)
        if str(row.get("paper_id") or "") == paper_id
    ]
    directness = pd.Series(
        [str(row.get("directness") or "") for row in evidence], dtype="object"
    ).value_counts()
    deep_status_path = (
        base_dir / "data" / "deep_read" / paper_id / "extraction_status.json"
    )
    try:
        deep_status = json.loads(deep_status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        deep_status = {}

    st.markdown(f"### 📄 {record['title']}")
    left, right = st.columns(2)
    with left:
        st.markdown(f"**作者**：{record.get('authors') or '未报告'}")
        st.markdown(f"**年份**：{record.get('year') or '未报告'}")
        st.markdown(f"**DOI**：{record.get('doi') or '未报告'}")
        st.markdown(f"**PDF 来源**：{record.get('source_url') or '本地上传/迁移'}")
        st.markdown(f"**真实页数**：{record.get('real_page_count') or 0}")
        st.markdown(f"**canonical_paper_id**：`{paper_id}`")
    with right:
        st.markdown(
            f"**深读状态**：{'COMPLETED' if record.get('deep_read_complete') else record.get('deep_read_status')}"
        )
        st.markdown(f"**EvidenceRecord**：{len(evidence)}")
        st.markdown(
            "**DIRECT / INDIRECT / MENTION_ONLY**："
            f"{directness.get('DIRECT', 0)} / {directness.get('INDIRECT', 0)} / "
            f"{directness.get('MENTION_ONLY', 0)}"
        )
        st.markdown(
            f"**公式数量**：{deep_status.get('formula_evidence_count', 0)}"
        )
        st.markdown(
            f"**待人工看图**：{deep_status.get('figure_review_required_count', 0)}"
        )
        st.markdown(f"**证据确认状态**：{record.get('evidence_status')}")
        st.markdown(f"**RAG 状态**：{record.get('rag_status')}")
        st.markdown(
            f"**失败/隔离原因**：{record.get('quarantine_reason') or deep_status.get('error') or '无'}"
        )

    with st.expander("查看可信证据", expanded=False):
        _render_evidence_table(evidence)

    action = st.columns(5)
    with action[0]:
        if st.button("核验元数据", key=f"verify_{paper_id}"):
            if not record.get("doi"):
                st.warning("没有 DOI，无法调用真实元数据源核验。")
            else:
                with st.spinner("核验中…"):
                    result = add_doi_or_url(
                        record["doi"],
                        base_dir=base_dir,
                        acquire_oa_pdf=False,
                    )
                st.json(result, expanded=False)
    with action[1]:
        if st.button(
            "开始/重新深读",
            key=f"deep_{paper_id}",
            disabled=(
                not record.get("pdf_valid")
                or bool(record.get("snapshot_read_only"))
            ),
        ):
            paths = record.get("linked_versions") or []
            existing = next((Path(path) for path in paths if Path(path).exists()), None)
            if existing is None:
                st.error("canonical PDF 路径不存在。")
            else:
                from src.deep_read_pipeline import deep_read_pdf

                with st.spinner("正在逐页深读、审计漏提并生成 EvidenceRecord…"):
                    result = deep_read_pdf(
                        existing,
                        paper_id=paper_id,
                        title=record["title"],
                        base_dir=base_dir,
                        force=True,
                    )
                st.json(result, expanded=False)
                _clear_library_cache()
    with action[2]:
        if st.button(
            "确认全部证据",
            key=f"confirm_{paper_id}",
            disabled=not record.get("deep_read_complete") or not evidence,
        ):
            result = set_evidence_review_status(
                paper_id,
                HUMAN_CONFIRMED,
                base_dir=base_dir,
            )
            st.success(f"已确认 {result['updated_count']} 条证据。")
            _clear_library_cache()
            st.rerun()
    with action[3]:
        if st.button(
            "退回修改",
            key=f"reject_{paper_id}",
            disabled=not evidence,
        ):
            result = set_evidence_review_status(
                paper_id,
                HUMAN_REVISION_REQUIRED,
                base_dir=base_dir,
            )
            st.warning(f"已退回 {result['updated_count']} 条证据。")
            _clear_library_cache()
            st.rerun()
    with action[4]:
        if st.button(
            "写入/重建 RAG",
            key=f"rag_{paper_id}",
            disabled=record.get("evidence_status") != HUMAN_CONFIRMED,
        ):
            with st.spinner("正在调用统一 Stage-3 索引…"):
                result = rebuild_unified_rag([paper_id], base_dir=base_dir)
            if result["status"] == "COMPLETED":
                st.success("统一 RAG 索引完成。")
            else:
                st.error(f"RAG 门禁未通过：{result['rejected']}")
            _clear_library_cache()
            st.rerun()

    if st.button("隔离该记录", key=f"quarantine_{paper_id}"):
        result = quarantine_record(
            paper_id,
            "MANUAL_LIBRARY_QUARANTINE",
            base_dir=base_dir,
        )
        st.warning(f"记录状态：{result['status']}")
        _clear_library_cache()
        st.rerun()


def render_batch_operations(
    selected_ids: List[str],
    base_dir: Path,
) -> None:
    gate = eligible_paper_ids(selected_ids, base_dir=base_dir)
    eligible = gate["eligible"]
    if not selected_ids:
        st.info("请先选择具有有效全文和可信证据的 canonical 文献。")
        return
    if gate["rejected"]:
        st.warning(
            "以下记录已自动排除：" + json.dumps(gate["rejected"], ensure_ascii=False)
        )
    st.markdown(f"**已选 {len(selected_ids)} 条；科研操作可用 {len(eligible)} 篇。**")
    gap_col, hyp_col, rag_col = st.columns(3)
    with gap_col:
        run_gap = st.button(
            "🔍 分析研究空白",
            key="batch_gap_btn",
            disabled=not eligible,
            width="stretch",
        )
    with hyp_col:
        run_hyp = st.button(
            "🧪 生成候选假设",
            key="batch_hyp_btn",
            disabled=not eligible,
            width="stretch",
        )
    with rag_col:
        confirmed = eligible_paper_ids(
            eligible,
            base_dir=base_dir,
            require_confirmation=True,
        )["eligible"]
        run_rag = st.button(
            "🔄 写入/重建统一 RAG",
            key="batch_rag_btn",
            disabled=not confirmed,
            width="stretch",
        )

    if run_gap:
        from src.research_gap_service import analyze_research_gaps

        with st.spinner("正在基于可信正文证据和真实页码分析…"):
            result = analyze_research_gaps(eligible, base_dir=base_dir)
        st.session_state["library_gap_result"] = result
    if run_hyp:
        from src.hypothesis_service import generate_hypotheses

        with st.spinner("正在通过正式证据门禁生成候选假设…"):
            result = generate_hypotheses(eligible, base_dir=base_dir)
        st.session_state["library_hypothesis_result"] = result
    if run_rag:
        with st.spinner("正在调用统一 Stage-3 索引…"):
            result = rebuild_unified_rag(confirmed, base_dir=base_dir)
        st.session_state["library_rag_result"] = result
        _clear_library_cache()

    for key, label in (
        ("library_gap_result", "研究空白结果"),
        ("library_hypothesis_result", "候选假设结果"),
        ("library_rag_result", "RAG 结果"),
    ):
        result = st.session_state.get(key)
        if result:
            with st.expander(label, expanded=True):
                if result.get("status") == "INSUFFICIENT_EVIDENCE":
                    st.error("INSUFFICIENT_EVIDENCE")
                st.json(result, expanded=True)


def render_invalid_metadata(base_dir: Path) -> None:
    invalid = invalid_candidate_records(base_dir)
    with st.expander(f"⚠️ 异常元数据记录 ({len(invalid)})", expanded=False):
        st.caption(
            "这些记录不会进入正常列表、详情选择器、研究空白、假设生成或 RAG。"
        )
        if not invalid:
            st.success("当前没有异常元数据。")
            return
        display = pd.DataFrame(invalid)
        columns = [
            "candidate_id",
            "doi",
            "source_url",
            "metadata_source",
            "duplicate_status",
            "validation_status",
            "quarantine_reason",
        ]
        st.dataframe(
            display[[column for column in columns if column in display.columns]],
            hide_index=True,
            width="stretch",
        )
        if st.button(
            "批量按 DOI/URL 回填真实元数据",
            key="backfill_all_invalid_metadata",
        ):
            with st.spinner("正在查询 OpenAlex / Crossref；不会猜测缺失题名…"):
                result = backfill_invalid_candidates(base_dir=base_dir)
            st.success(
                f"回填完成：更新 {len(result['updated'])} 条，"
                f"仍失败 {len(result['failed'])} 条，"
                f"无标识符 {len(result['skipped_without_identifier'])} 条。"
            )
            st.json(result, expanded=False)
            _clear_library_cache()
        record_id = st.selectbox(
            "选择异常记录",
            options=[row["candidate_id"] for row in invalid],
            key="invalid_record_selector",
        )
        current = next(row for row in invalid if row["candidate_id"] == record_id)
        doi_or_url = st.text_input(
            "DOI / URL（可修改后重新获取）",
            value=current.get("doi") or current.get("source_url") or "",
            key=f"invalid_lookup_{record_id}",
        )
        repair, remove = st.columns(2)
        with repair:
            if st.button("从真实元数据源重新获取", key=f"repair_{record_id}"):
                result = repair_candidate_metadata(
                    record_id,
                    doi_or_url,
                    base_dir=base_dir,
                )
                if result.get("validation_status") == VALID_METADATA:
                    st.success(
                        "已使用真实元数据更新候选记录；若 DOI 与正式主记录重复，"
                        "它会自动合并为 duplicate_of，不会重复计数。"
                    )
                else:
                    st.error(result.get("error") or "补全失败，记录继续隔离。")
                st.json(result, expanded=False)
                _clear_library_cache()
        with remove:
            if st.button("删除无效记录", key=f"remove_{record_id}"):
                if delete_invalid_candidate(record_id, base_dir=base_dir):
                    st.success("无效记录已删除。")
                    _clear_library_cache()
                    st.rerun()
                else:
                    st.warning("记录不存在或已删除。")


def render_task_status(base_dir: Path) -> None:
    with st.expander("🧭 文献任务状态", expanded=False):
        from src.literature_tasks import list_literature_tasks

        tasks = list_literature_tasks(base_dir)
        if not tasks:
            st.info("暂无 Stage-3 文献任务。")
            return
        rows = [
            {
                "task_id": row.get("task_id"),
                "status": row.get("status"),
                "query": row.get("query"),
                "downloaded": row.get("downloaded_count"),
                "deep_read": row.get("deep_read_count"),
                "indexed": row.get("indexed_count"),
                "error": row.get("last_error") or row.get("error"),
            }
            for row in tasks
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_library_page() -> None:
    """Render the same complete library workflow locally and on cloud."""
    backend = detect_storage_backend(PROJECT_ROOT)
    base_dir = backend.prepare()

    st.markdown(
        "<h1 style='text-align:center;'>📚 文献库管理</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "上传、真实元数据核验、逐页深读、EvidenceRecord 审核、统一 RAG、"
        "研究空白和候选假设均使用同一套 canonical 工作流。"
    )
    _render_storage_status(base_dir, backend.mode, backend.warning)
    st.divider()

    # Upload and DOI/OA entry points are intentionally at the top and are
    # never hidden merely because the app is running on Community Cloud.
    render_ingestion_entries(base_dir)
    st.divider()
    _render_statistics(base_dir)
    render_task_status(base_dir)
    render_invalid_metadata(base_dir)
    st.divider()

    rows = _normal_library_rows(base_dir)
    frame = pd.DataFrame(rows)
    st.subheader("📋 正常文献与有效候选")
    if frame.empty:
        st.info("当前没有可显示的有效文献；可使用页面顶部入口添加。")
        return

    filter_cols = st.columns([1, 1, 2])
    with filter_cols[0]:
        source_filter = st.selectbox(
            "来源",
            ["全部", "canonical PDF", "候选元数据"],
            key="lib_source",
        )
    with filter_cols[1]:
        types = ["全部", *sorted(frame["type_cn"].dropna().unique())]
        type_filter = st.selectbox("分类", types, key="lib_type")
    with filter_cols[2]:
        keyword = st.text_input(
            "题名 / 作者 / DOI",
            placeholder="输入关键词…",
            key="lib_keyword",
        )
    if source_filter == "canonical PDF":
        frame = frame[frame["source"] == "formal"]
    elif source_filter == "候选元数据":
        frame = frame[frame["source"] == "candidate"]
    if type_filter != "全部":
        frame = frame[frame["type_cn"] == type_filter]
    if keyword.strip():
        needle = keyword.strip().lower()
        frame = frame[
            frame.apply(
                lambda row: any(
                    needle in str(row.get(key) or "").lower()
                    for key in ("title", "authors", "doi")
                ),
                axis=1,
            )
        ]

    display = pd.DataFrame(
        {
            "选择": False,
            "题名": frame["title"],
            "作者": frame["authors"].fillna(""),
            "年份": frame["year"].fillna(""),
            "分类": frame["type_cn"],
            "PDF 状态": frame["pdf_status"],
            "深读状态": frame["deep_read_complete"].map(
                {True: "COMPLETED", False: "PENDING"}
            ),
            "证据状态": frame["evidence_status"],
            "RAG 状态": frame["rag_status"],
            "DOI": frame["doi"].fillna(""),
            "来源": frame["source"].map(
                {"formal": "canonical PDF", "candidate": "候选元数据"}
            ),
            "_paper_id": frame["paper_id"],
            "_selectable": frame["selectable"],
        }
    ).reset_index(drop=True)
    page = paginate_dataframe(display, page_size=20, page_key="lib_table")
    editor = st.data_editor(
        page.drop(columns=["_paper_id", "_selectable"]),
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        disabled=[
            "题名",
            "作者",
            "年份",
            "分类",
            "PDF 状态",
            "深读状态",
            "证据状态",
            "RAG 状态",
            "DOI",
            "来源",
        ],
        key="lib_data_editor",
        column_config={
            "选择": st.column_config.CheckboxColumn("选择", default=False),
            "题名": st.column_config.TextColumn("题名", width="large"),
            "作者": st.column_config.TextColumn("作者", width="medium"),
        },
    )
    selected_indices = editor.index[editor["选择"] == True].tolist()
    selected_ids = [
        str(page.loc[index, "_paper_id"])
        for index in selected_indices
        if bool(page.loc[index, "_selectable"])
    ]
    excluded_selection = [
        str(page.loc[index, "_paper_id"])
        for index in selected_indices
        if not bool(page.loc[index, "_selectable"])
    ]
    if excluded_selection:
        st.warning(
            "以下元数据记录没有有效全文/深读证据，已自动排除："
            + "、".join(excluded_selection)
        )

    canonical = [
        row
        for row in rows
        if row.get("source") == "formal"
        and row.get("validation_status") == VALID_METADATA
    ]
    if canonical:
        options = [row["paper_id"] for row in canonical]
        by_id = {row["paper_id"]: row for row in canonical}
        detail_id = st.selectbox(
            "选择文献查看详情",
            options=options,
            format_func=lambda value: _display_title(by_id[value]),
            key="lib_detail_selector",
        )
        if st.button("📄 查看详情", key="view_detail_btn"):
            st.session_state["library_detail_id"] = detail_id
        active_detail = st.session_state.get("library_detail_id")
        if active_detail in by_id:
            with st.container(border=True):
                render_paper_detail(by_id[active_detail], base_dir)

    st.divider()
    st.subheader(f"⚙️ 批量科研操作（{len(selected_ids)} 篇有效文献）")
    render_batch_operations(selected_ids, base_dir)

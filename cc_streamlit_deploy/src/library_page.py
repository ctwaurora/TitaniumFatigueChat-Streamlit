"""Streamlit literature-library management page.

All scientific actions are backed by the canonical Stage-1/2/3 state.  Legacy
CSV files are used only as metadata compatibility inputs, never as the source
of truth for counts or RAG status.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
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
    archive_canonical_records,
    backfill_invalid_candidates,
    canonical_library_records,
    canonical_pdf_records,
    delete_invalid_candidate,
    eligible_paper_ids,
    ingest_uploaded_pdf,
    invalid_candidate_records,
    library_statistics,
    load_candidate_records,
    quarantine_record,
    permanently_delete_canonical_records,
    repair_candidate_metadata,
    rebuild_unified_rag,
    set_evidence_review_status,
    trusted_evidence_rows,
    valid_candidate_records,
)
from src.storage_adapter import CLOUD_TEMPORARY, detect_storage_backend
from src.library_management import reconcile_persistent_selection


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


@st.fragment
def _render_full_read_status(base_dir: Path) -> None:
    from src.full_library_deep_read import queue_summary

    summary = queue_summary(base_dir=base_dir)
    metrics = st.columns(6)
    labels = ("唯一文献", "已完成", "处理中", "待处理", "失败", "待人工审核")
    values = (
        summary["logical_document_count"], summary["completed"], summary["running"],
        summary["pending"], summary["failed"], summary["needs_human_review"],
    )
    for column, label, value in zip(metrics, labels, values):
        column.metric(label, value)
    page_cols = st.columns(5)
    for column, label, value in zip(page_cols, (
        "总页数", "已读页数", "EvidenceRecord", "已入RAG", "当前控制",
    ), (
        summary["total_pages"], summary["completed_pages"], summary["evidence_count"],
        summary["indexed"], summary["control"],
    )):
        column.metric(label, value)
    rows = summary.get("tasks") or []
    if rows:
        st.dataframe(pd.DataFrame([{
            "题名": row.get("canonical_title") or "待元数据核验",
            "PDF": row.get("original_filename"), "页数": row.get("real_page_count"),
            "已读页数": row.get("processed_pages"), "精读状态": row.get("status"),
            "终态分类": row.get("terminal_state") or "待终态核验",
            "EvidenceRecord": (row.get("gate") or {}).get("evidence_count", 0),
            "质量门禁": (row.get("gate") or {}).get("passed", False),
            "RAG状态": (row.get("index_result") or {}).get("status", "NOT_INDEXED"),
            "失败原因": row.get("last_error") or "",
        } for row in rows]), hide_index=True, width="stretch")


def _render_full_library_deep_read(base_dir: Path, mode: str) -> None:
    """Local durable batch controls; cloud never creates unusable tasks."""
    with st.expander("全库精读状态与控制", expanded=False):
        if mode == CLOUD_TEMPORARY:
            st.info("该功能仅在本地运行：Streamlit Cloud 无法读取用户电脑 C 盘中的 paper/pdfs。")
            return
        from src.full_library_deep_read import (
            build_full_library_queue, inventory_pdfs, set_queue_control,
        )

        pdf_dir = base_dir / "paper" / "pdfs"
        if st.button("刷新全量PDF清单", key="refresh_full_read_inventory"):
            with st.spinner("正在校验PDF、真实页数并执行分层查重……"):
                preview = inventory_pdfs(pdf_dir, base_dir=base_dir)
            st.session_state["full_read_inventory_preview"] = preview
        preview = st.session_state.get("full_read_inventory_preview")
        if preview:
            cols = st.columns(5)
            for column, label, value in zip(cols, (
                "PDF文件总数", "唯一逻辑文献", "完全重复", "PDF总页数", "预计DeepSeek调用",
            ), (
                preview["pdf_file_count"], preview["logical_document_count"],
                preview["exact_duplicate_count"], preview["total_pages"], 0,
            )):
                column.metric(label, value)
            st.caption(
                "当前精读器使用本地确定性逐页解析；未调用DeepSeek时，实际API调用为0。"
                f" 本地目录：{pdf_dir}"
            )
        _render_full_read_status(base_dir)
        concurrency = st.number_input("并发数", min_value=1, max_value=2, value=1, step=1)
        use_deepseek = st.checkbox(
            "启用 DeepSeek 语义增强",
            value=True,
            help="只显示是否启用；不会显示或记录 API Key。",
        )
        confirm = st.checkbox("我确认启动全部未完成PDF的逐篇逐页精读", key="confirm_full_read_all")
        start, pause, resume, stop, retry = st.columns(5)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        if start.button("开始全部精读", type="primary", disabled=not confirm, width="stretch"):
            build_full_library_queue(pdf_dir, base_dir=base_dir, resume=True)
            set_queue_control("RUN", base_dir=base_dir)
            command = [
                sys.executable, str(base_dir / "app.py"), "deep-read-all",
                "--pdf-dir", str(pdf_dir), "--resume", "--only-incomplete",
                "--include-review", "--retry-failed",
                "--concurrency", str(int(concurrency)),
            ]
            if use_deepseek:
                command.append("--use-deepseek")
            process = subprocess.Popen(command, cwd=base_dir, creationflags=flags)
            st.success(f"本地持久化任务已启动（PID {process.pid}）；关闭页面后仍可用继续按钮恢复。")
        if pause.button("暂停", width="stretch"):
            set_queue_control("PAUSE", base_dir=base_dir)
        if resume.button("继续", width="stretch"):
            set_queue_control("RUN", base_dir=base_dir)
            subprocess.Popen([
                sys.executable, str(base_dir / "app.py"), "deep-read-all",
                "--pdf-dir", str(pdf_dir), "--resume", "--only-incomplete",
                "--include-review", "--retry-failed",
                "--concurrency", str(int(concurrency)),
                *(["--use-deepseek"] if use_deepseek else []),
            ], cwd=base_dir, creationflags=flags)
        if stop.button("当前文献后停止", width="stretch"):
            set_queue_control("STOP_AFTER_CURRENT", base_dir=base_dir)
        if retry.button("仅重试失败", width="stretch"):
            subprocess.Popen([
                sys.executable, str(base_dir / "app.py"), "deep-read-all",
                "--pdf-dir", str(pdf_dir), "--resume", "--retry-failed",
                "--include-review", "--only-incomplete", "--concurrency", "1",
                *(["--use-deepseek"] if use_deepseek else []),
            ], cwd=base_dir, creationflags=flags)
        st.caption("状态从持久化队列读取；已COMPLETED文献默认跳过，默认并发1，最大2。")


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


def _deployment_version() -> Dict[str, Any]:
    try:
        value = json.loads(
            (PROJECT_ROOT / "DEPLOY_VERSION.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _storage_label(mode: str) -> str:
    return {
        CLOUD_TEMPORARY: "云端临时存储",
        "LOCAL_PERSISTENT": "本地持久化存储",
        "EXTERNAL_PERSISTENT": "外部持久化存储",
    }.get(mode, "存储状态未知")


def _render_storage_status(base_dir: Path, mode: str, warning: str) -> None:
    """Render only compact user-facing persistence guidance above the fold."""
    del base_dir, warning
    if mode == CLOUD_TEMPORARY:
        st.warning("当前为云端临时存储模式，应用重启后新增PDF和索引可能丢失。")
        st.caption("云端任务在当前页面同步执行，关闭网页或超时后不会在后台继续。")


def _render_system_status(base_dir: Path, mode: str) -> None:
    """Keep deployment and filesystem details out of the default library view."""
    version = _deployment_version()
    try:
        from src.unified_rag import rag_paths

        rag_exists = rag_paths(base_dir)["manifest"].exists()
    except (ImportError, KeyError, OSError):
        rag_exists = False
    with st.expander("系统状态与部署信息", expanded=False):
        st.markdown(f"**存储模式**：{_storage_label(mode)}")
        st.markdown(f"**工作目录**：`{base_dir}`")
        st.markdown(
            f"**部署版本**：`{version.get('source_commit') or '本地开发版本'}`"
        )
        st.markdown(
            f"**应用版本**：{version.get('application_version') or '未标记'}"
        )
        st.markdown(f"**统一 RAG**：{'已存在' if rag_exists else '尚未建立'}")
        st.markdown(
            f"**云端模式**：{'是' if mode == CLOUD_TEMPORARY else '否'}"
        )


def _render_pdf_upload(base_dir: Path) -> None:
    st.markdown("### 上传本地PDF")
    st.caption("支持单篇或多篇；上传后自动查重、精读、质量门禁和入RAG。")
    st.caption(
        "上传 → 元数据核验 → PDF校验 → 查重 → 逐页精读 → "
        "证据审计 → 质量门禁 → 写入RAG → 完成"
    )
    files = st.file_uploader(
        "选择一篇或多篇 PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="lib_pdf_upload",
        help="系统验证真实 PDF 结构并按 SHA-256 查重，HTML 伪 PDF 会被拒绝。",
    )
    manual_title = manual_authors = manual_year = manual_doi = ""
    if files:
        file_rows = [
            {"文件名": item.name, "大小": f"{item.size / 1024:.1f} KB"}
            for item in files
        ]
        st.dataframe(pd.DataFrame(file_rows), hide_index=True, width="stretch")
        if len(files) == 1:
            with st.expander("单篇元数据补充（可选）", expanded=False):
                st.caption("系统不会用文件名猜测题名。")
                manual_title = st.text_input(
                    "真实题名",
                    key="upload_manual_title",
                )
                manual_authors = st.text_input(
                    "作者",
                    key="upload_manual_authors",
                )
                manual_year = st.text_input(
                    "年份",
                    key="upload_manual_year",
                )
                manual_doi = st.text_input(
                    "DOI（自动入 RAG 时必需）",
                    key="upload_manual_doi",
                )
    if not st.button(
        "处理上传文献",
        type="primary",
        key="process_library_uploads",
        disabled=not files,
        width="stretch",
    ):
        return
    progress = st.progress(0, text="上传：准备接收 PDF…")
    results = []
    for index, uploaded in enumerate(files, start=1):
        progress.progress(
            (index - 1) / len(files),
            text=f"PDF校验 → 查重 → 逐页精读：{uploaded.name}",
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
    progress.progress(1.0, text="完成：上传文献已处理")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "真实题名": row.get("title") or row.get("filename") or "",
                    "作者": row.get("authors") or "",
                    "年份": row.get("year") or row.get("publication_date") or "",
                    "DOI": row.get("doi") or "",
                    "真实下载PDF": bool(row.get("pdf_valid")),
                    "PDF大小（字节）": row.get("file_size") or 0,
                    "SHA-256": row.get("file_hash_sha256") or "",
                    "真实页数": row.get("real_page_count") or 0,
                    "已精读页数": (
                        row.get("processed_page_count")
                        or row.get("page_record_count")
                        or 0
                    ),
                    "EvidenceRecord": row.get("evidence_count") or 0,
                    "DIRECT": row.get("direct_evidence_count") or 0,
                    "INDIRECT": row.get("indirect_evidence_count") or 0,
                    "MENTION_ONLY": row.get("mention_only_count") or 0,
                    "质量门禁": row.get("quality_gate") or "FAILED",
                    "库状态": row.get("library_status") or "QUARANTINED",
                    "RAG状态": row.get("rag_status") or "NOT_INDEXED",
                }
                for row in results
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.session_state["last_upload_results"] = results
    _clear_library_cache()


def _render_doi_add(base_dir: Path) -> None:
    st.markdown("#### 通过 DOI / URL 添加")
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
    st.markdown("### 自动发现并处理OA文献")
    st.caption(
        "SEARCHING → METADATA_VALIDATED → DOWNLOADING → PDF_VALIDATED → "
        "DEDUPLICATING → DEEP_READING → QUALITY_GATING → REINDEXING → COMPLETED"
    )
    query = st.text_input(
        "检索主题",
        value="L-PBF Ti-6Al-4V fatigue",
        key="library_auto_oa_query",
    )
    controls = st.columns(2)
    with controls[0]:
        max_new = st.number_input(
            "最大新增数量（1—3篇）",
            min_value=1,
            max_value=3,
            value=1,
            step=1,
            key="library_auto_oa_max",
        )
    with controls[1]:
        topic_filter = st.selectbox(
            "文献类型",
            [
                "全部钛合金疲劳",
                "Ti-6Al-4V / TC4疲劳",
                "增材制造钛合金疲劳",
                "LCF",
                "HCF",
                "VHCF",
                "裂纹起裂",
                "短裂纹",
                "疲劳裂纹扩展",
                "表面粗糙度",
                "孔隙与未熔合",
                "微观组织与织构",
                "残余应力",
                "热处理与HIP",
                "成形方向",
                "环境疲劳",
                "疲劳模型与公式",
            ],
            key="library_auto_oa_topic",
        )
    core_only = st.toggle(
        "仅限CORE：Ti-6Al-4V / TC4疲劳",
        value=False,
        key="library_auto_oa_core",
    )
    with st.expander("更多筛选（可选）", expanded=False):
        year_columns = st.columns(2)
        with year_columns[0]:
            year_from = st.number_input(
                "起始年份",
                min_value=1900,
                max_value=datetime.now().year,
                value=2000,
                step=1,
                key="library_auto_oa_year_from",
            )
        with year_columns[1]:
            year_to = st.number_input(
                "结束年份",
                min_value=1900,
                max_value=datetime.now().year,
                value=datetime.now().year,
                step=1,
                key="library_auto_oa_year_to",
            )
    if st.button(
        "开始自动发现并处理",
        type="primary",
        key="library_auto_oa_start",
        width="stretch",
    ):
        from src.auto_oa_pipeline import PHASES, run_auto_oa_discovery

        progress_labels = {
            "SEARCHING": "检索",
            "METADATA_VALIDATED": "元数据核验",
            "DOWNLOADING": "PDF校验",
            "PDF_VALIDATED": "PDF校验",
            "DEDUPLICATING": "查重",
            "DEEP_READING": "逐页精读",
            "QUALITY_GATING": "质量门禁",
            "REINDEXING": "写入RAG",
            "COMPLETED": "完成",
        }
        progress = st.progress(0.0, text="检索：正在查询 OA 元数据源…")
        phase_rows: List[Dict[str, Any]] = []

        def on_progress(phase: str, payload: Dict[str, Any]) -> None:
            phase_rows.append(payload)
            position = PHASES.index(phase) + 1 if phase in PHASES else 1
            progress.progress(
                min(position / len(PHASES), 1.0),
                text=(
                    f"{progress_labels.get(phase, phase)}："
                    f"{payload.get('title') or payload.get('paper_id') or query}"
                ),
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
        st.caption(
            "搜索候选 {search} · OA候选 {oa} · 下载尝试 {attempts} · "
            "真实下载成功 {success}".format(
                search=result.get("search_candidate_count", 0),
                oa=result.get("oa_candidate_count", 0),
                attempts=result.get("download_attempt_count", 0),
                success=result.get("download_success_count", 0),
            )
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
                        "元数据来源": row.get("metadata_source"),
                        "OA来源": row.get("oa_source"),
                        "领域分级": row.get("domain_scope"),
                        "真实下载PDF": row.get("downloaded_pdf"),
                        "下载状态": row.get("download_status"),
                        "HTTP": row.get("http_status"),
                        "PDF大小（字节）": row.get("file_size"),
                        "SHA-256": row.get("file_hash_sha256"),
                        "保存位置": row.get("saved_location"),
                        "真实页数": row.get("real_page_count"),
                        "已精读页数": row.get("processed_page_count"),
                        "深读状态": row.get("deep_read_status"),
                        "EvidenceRecord": row.get("evidence_count"),
                        "DIRECT": row.get("direct_evidence_count"),
                        "INDIRECT": row.get("indirect_evidence_count"),
                        "MENTION_ONLY": row.get("mention_only_count"),
                        "自动质量门禁": row.get("quality_gate"),
                        "库状态": row.get("library_status"),
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
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            _render_oa_topup(base_dir)
    with right:
        with st.container(border=True):
            _render_pdf_upload(base_dir)
    _render_local_pdf_scan(base_dir)


def _render_local_pdf_scan(base_dir: Path) -> None:
    from src.local_pdf_import import (
        local_pdf_directory_summary,
        scan_and_import_local_pdfs,
    )

    with st.expander("扫描本地PDF并批量导入文献库", expanded=False):
        directory_rows = local_pdf_directory_summary(base_dir)
        st.dataframe(
            pd.DataFrame(directory_rows),
            hide_index=True,
            width="stretch",
        )
        batch_mode = st.radio(
            "处理范围",
            ["小批量验收（最多3篇）", "全部目录"],
            horizontal=True,
            key="local_pdf_scan_mode",
        )
        total = sum(int(row["pdf_count"]) for row in directory_rows)
        run_scan = st.button(
            "扫描本地PDF",
            key="scan_local_pdf_button",
            disabled=total == 0,
        )
        if not run_scan:
            return
        progress = st.progress(0.0, text="正在递归扫描PDF…")
        events: List[Dict[str, Any]] = []

        def on_progress(phase: str, payload: Dict[str, Any]) -> None:
            events.append(payload)
            if phase == "SCANNING":
                index = int(payload.get("index") or 0)
                count = max(1, int(payload.get("total") or 1))
                progress.progress(
                    min(index / count, 0.85),
                    text=f"扫描、查重和导入：{payload.get('path') or ''}",
                )
            elif phase == "DEEP_READING":
                progress.progress(0.9, text="正在逐页精读并生成证据…")
            elif phase == "QUALITY_GATING":
                progress.progress(0.95, text="正在质量门禁并写入RAG…")

        full_run = batch_mode == "全部目录"
        result = scan_and_import_local_pdfs(
            base_dir=base_dir,
            max_files=None if full_run else 3,
            progress_callback=on_progress,
            report_name=(
                "pdf_storage_unification_report.json"
                if full_run
                else "pdf_import_smoke_3_report.json"
            ),
        )
        progress.progress(1.0, text="本地PDF扫描处理完成")
        metrics = st.columns(8)
        values = (
            ("扫描PDF总数", result.get("original_pdf_count", 0)),
            ("唯一PDF数", result.get("unique_pdf_count", 0)),
            ("重复数", result.get("duplicate_pdf_count", 0)),
            ("新增候选", result.get("new_candidate_count", 0)),
            ("新增正式", result.get("new_formal_count", 0)),
            ("深读成功", result.get("deep_read_success_count", 0)),
            ("已入RAG", result.get("indexed_count", 0)),
            ("失败", result.get("failure_count", 0)),
        )
        for column, (label, value) in zip(metrics, values):
            column.metric(label, value)
        if result.get("validation", {}).get("passed"):
            st.success("PDF、canonical、深读、EvidenceRecord和RAG关联验证通过。")
        else:
            st.error("关联验证未通过；请查看迁移报告，旧文件没有被删除。")
        display = pd.DataFrame(result.get("processing_results") or [])
        if not display.empty:
            columns = [
                name
                for name in (
                    "title",
                    "doi",
                    "domain_scope",
                    "real_page_count",
                    "status",
                    "deep_read_complete",
                    "evidence_count",
                    "rag_status",
                    "failure_reason",
                )
                if name in display.columns
            ]
            st.dataframe(
                display[columns], hide_index=True, width="stretch"
            )
        with st.expander("查看扫描日志", expanded=False):
            st.json(
                {
                    "report": Path(
                        str(result.get("report_path") or "")
                    ).name,
                    "path_updates": result.get("path_updates"),
                    "events": events,
                }
            )
        _clear_library_cache()


def _render_statistics(base_dir: Path) -> None:
    stats = library_statistics(base_dir)
    labels = [
        ("唯一文献", "unique_literature"),
        ("PDF 已获取", "pdf_acquired"),
        ("深读完成", "deep_read_complete"),
        ("已入统一 RAG", "rag_indexed"),
        ("待处理/失败", "pending_or_failed"),
    ]
    columns = st.columns(5)
    for column, (label, key) in zip(columns, labels):
        with column:
            st.metric(label, stats[key])


def _normal_library_rows(base_dir: Path) -> List[Dict[str, Any]]:
    return [
        {
            **row,
            "authors": str(row.get("authors") or ""),
            "type_cn": paper_type_cn(row.get("type_code", "")),
            "verification": (
                "正式库"
                if row.get("library_status") == "FORMAL"
                else "异常记录"
                if row.get("library_status") == "QUARANTINED"
                else "候选库"
            ),
        }
        for row in canonical_library_records(base_dir)
    ]


def _library_dataset_version(base_dir: Path) -> str:
    values = []
    for path in (base_dir / "data/paper_manifest.jsonl", base_dir / "data/stage3_5/candidate_metadata.jsonl"):
        try:
            stat = path.stat(); values.append(f"{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            values.append("missing")
    return hashlib.sha256("|".join(values).encode()).hexdigest()


@st.cache_data(show_spinner="加载文献摘要…")
def _normal_library_rows_cached(base_dir_text: str, dataset_version: str) -> List[Dict[str, Any]]:
    return _normal_library_rows(Path(base_dir_text))


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
    source_url = str(record.get("source_url") or "")
    display_source = (
        source_url
        if source_url.startswith(("https://", "http://"))
        else "本地上传/系统临时存储"
    )
    pdf_candidates = [
        record.get("canonical_pdf_path"),
        *(record.get("linked_versions") or []),
    ]
    pdf_path = next(
        (
            Path(value)
            for value in pdf_candidates
            if value and Path(value).is_file()
        ),
        None,
    )

    st.markdown(f"### 📄 {record['title']}")
    left, right = st.columns(2)
    with left:
        st.markdown(f"**作者**：{record.get('authors') or '未报告'}")
        st.markdown(f"**年份**：{record.get('year') or '未报告'}")
        st.markdown(f"**DOI**：{record.get('doi') or '未报告'}")
        st.markdown(f"**PDF 来源**：{display_source}")
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

    if pdf_path is not None:
        try:
            pdf_bytes = pdf_path.read_bytes()
        except OSError:
            pdf_bytes = b""
        if pdf_bytes.startswith(b"%PDF"):
            safe_title = re.sub(
                r'[\\/:*?"<>|]+',
                "_",
                str(record.get("title") or paper_id),
            ).strip(" ._")
            st.download_button(
                "下载PDF到本机",
                data=pdf_bytes,
                file_name=f"{safe_title or paper_id}.pdf",
                mime="application/pdf",
                key=f"download_pdf_{paper_id}",
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
            disabled=record.get("evidence_status")
            not in {"AUTO_VALIDATED", HUMAN_CONFIRMED},
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
    from src.library_management import (
        add_to_formal, archive as archive_records, audit_selected_evidence,
        export_records, rebuild_current_formal_rag, remove_from_formal, set_scope,
    )
    operation_reason = st.text_input(
        "操作原因（写入审计日志）", value="USER_LIBRARY_MANAGEMENT",
        key="batch_operation_reason",
    )
    manage1 = st.columns(4)
    rematch_clicked = manage1[0].button("重新匹配PDF", key="batch_rematch_pdf", width="stretch")
    retry_read_clicked = manage1[1].button("重试深读", key="batch_retry_read", width="stretch")
    reaudit_clicked = manage1[2].button("重新审计证据", key="batch_reaudit", width="stretch")
    add_formal_clicked = manage1[3].button("加入正式库", key="batch_add_formal", width="stretch")
    manage2 = st.columns(4)
    remove_formal_clicked = manage2[0].button("移出正式库", key="batch_remove_formal", width="stretch")
    add_rag_clicked = manage2[1].button("加入RAG", key="batch_add_rag", width="stretch")
    remove_rag_clicked = manage2[2].button("移出RAG", key="batch_remove_rag", width="stretch")
    context_clicked = manage2[3].button("标记CONTEXT", key="batch_context", width="stretch")
    manage3 = st.columns(3)
    out_scope_clicked = manage3[0].button("标记OUT_OF_SCOPE", key="batch_out_scope", width="stretch")
    archive_clicked = manage3[1].button("归档（推荐）", key="batch_archive_safe", width="stretch")
    manage3[2].download_button(
        "导出选中文献", data=export_records(selected_ids, base_dir=base_dir),
        file_name="selected_titanium_fatigue_papers.json", mime="application/json",
        key="batch_export_selected", width="stretch",
    )

    if rematch_clicked:
        from src.pdf_rematcher import audit_pdf_not_acquired
        with st.spinner("按资产ID、SHA、DOI、题名、作者年份与PDF首页重新匹配…"):
            results = [row for row in audit_pdf_not_acquired(base_dir) if row["canonical_paper_id"] in set(selected_ids)]
        st.session_state["library_rematch_result"] = results
    if retry_read_clicked:
        from src.deep_read_pipeline import deep_read_pdf
        by_id = {row["paper_id"]: row for row in canonical_library_records(base_dir)}
        results = []
        for paper_id in selected_ids:
            row = by_id.get(paper_id) or {}
            path = Path(str(row.get("canonical_pdf_path") or ""))
            if not path.is_file():
                results.append({"paper_id": paper_id, "status": "SKIPPED", "reason": "PDF_MISSING"}); continue
            results.append(deep_read_pdf(path, paper_id=paper_id, title=row.get("title") or "", base_dir=base_dir, force=False))
        st.session_state["library_retry_read_result"] = results
    if reaudit_clicked:
        st.session_state["library_reaudit_result"] = audit_selected_evidence(selected_ids, base_dir=base_dir)
    if add_formal_clicked or add_rag_clicked:
        result = add_to_formal(selected_ids, reason=operation_reason, base_dir=base_dir)
        if add_rag_clicked and result.get("updated"):
            result["rag"] = rebuild_current_formal_rag(base_dir)
        st.session_state["library_membership_result"] = result; _clear_library_cache(); st.rerun()
    if remove_formal_clicked or remove_rag_clicked:
        result = remove_from_formal(selected_ids, reason=operation_reason, base_dir=base_dir)
        result["rag"] = rebuild_current_formal_rag(base_dir)
        st.session_state["library_membership_result"] = result; _clear_library_cache(); st.rerun()
    if context_clicked:
        st.session_state["library_scope_result"] = set_scope(selected_ids, "CONTEXT", reason=operation_reason, base_dir=base_dir); _clear_library_cache(); st.rerun()
    if out_scope_clicked:
        result = set_scope(selected_ids, "OUT_OF_SCOPE", reason=operation_reason, base_dir=base_dir)
        result["rag"] = rebuild_current_formal_rag(base_dir)
        st.session_state["library_scope_result"] = result; _clear_library_cache(); st.rerun()
    if archive_clicked:
        result = archive_records(selected_ids, reason=operation_reason, base_dir=base_dir)
        result["rag"] = rebuild_current_formal_rag(base_dir)
        st.session_state["library_archive_result"] = result; _clear_library_cache(); st.rerun()

    gap_col, hyp_col, experiment_col = st.columns(3)
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
    with experiment_col:
        run_experiment = st.button(
            "🧫 生成实验设计",
            key="batch_experiment_btn",
            disabled=not eligible,
            width="stretch",
        )

    rag_col, archive_col, delete_col = st.columns(3)
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
    with archive_col:
        run_full = st.button(
            "生成完整科研推理任务",
            key="batch_full_reasoning_btn",
            disabled=not eligible,
            width="stretch",
        )
    with delete_col:
        confirm_delete = st.checkbox(
            "第一步：确认彻底删除",
            key="batch_delete_confirm",
            disabled=not eligible,
        )
        delete_phrase = st.text_input(
            f"第二步：输入 DELETE {len(selected_ids)} RECORDS",
            key="batch_delete_phrase", disabled=not eligible,
        )
        run_delete = st.button(
            "彻底删除",
            key="batch_delete_btn",
            disabled=not eligible or not confirm_delete or delete_phrase.strip() != f"DELETE {len(selected_ids)} RECORDS",
            width="stretch",
        )

    if run_gap or run_hyp or run_experiment or run_full:
        from src.research_reasoning import run_selected_research_workflow
        with st.spinner("正在生成证据矩阵、反证检索、空白、假设、实验、公式与机制图…"):
            workflow = run_selected_research_workflow(
                eligible,
                question="L-PBF Ti-6Al-4V fatigue mechanism transition under selected conditions",
                base_dir=base_dir,
            )
        if run_gap: st.session_state["library_gap_result"] = {"status": workflow["status"], "gaps": workflow["research_gaps"], "reverse_evidence_retrieval": workflow["reverse_evidence_retrieval"]}
        if run_hyp: st.session_state["library_hypothesis_result"] = {"status": workflow["status"], "hypotheses": workflow["hypotheses"]}
        if run_experiment: st.session_state["library_experiment_result"] = {"status": workflow["status"], "experiment_designs": workflow["experiment_designs"]}
        if run_full: st.session_state["library_full_reasoning_result"] = workflow
    if run_rag:
        with st.spinner("正在调用统一 Stage-3 索引…"):
            result = rebuild_unified_rag(confirmed, base_dir=base_dir)
        st.session_state["library_rag_result"] = result
        _clear_library_cache()
    if run_delete:
        result = permanently_delete_canonical_records(
            eligible, base_dir=base_dir
        )
        st.session_state["library_delete_result"] = result
        _clear_library_cache()
        st.rerun()

    for key, label in (
        ("library_rematch_result", "PDF重新匹配结果"),
        ("library_retry_read_result", "深读重试结果"),
        ("library_reaudit_result", "证据重审结果"),
        ("library_membership_result", "库/RAG状态结果"),
        ("library_scope_result", "范围标记结果"),
        ("library_gap_result", "研究空白结果"),
        ("library_hypothesis_result", "候选假设结果"),
        ("library_experiment_result", "实验设计结果"),
        ("library_full_reasoning_result", "完整科研推理结果"),
        ("library_rag_result", "RAG 结果"),
        ("library_archive_result", "归档结果"),
        ("library_delete_result", "彻底删除结果"),
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


def _render_secondary_sections(base_dir: Path, mode: str) -> None:
    st.divider()
    with st.expander("更多添加方式：DOI / URL", expanded=False):
        _render_doi_add(base_dir)
    render_task_status(base_dir)
    _render_system_status(base_dir, mode)


def render_library_page() -> None:
    """Render the same complete library workflow locally and on cloud."""
    backend = detect_storage_backend(PROJECT_ROOT)
    base_dir = backend.prepare()

    st.markdown(
        "<h1 style='text-align:center;'>📚 文献库管理</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "集中寻找、上传和管理可追溯的疲劳文献证据。"
    )
    _render_storage_status(base_dir, backend.mode, backend.warning)

    # The two primary workflows stay visible together above statistics.
    render_ingestion_entries(base_dir)
    _render_full_library_deep_read(base_dir, backend.mode)
    st.divider()
    st.subheader("文献统计")
    _render_statistics(base_dir)
    st.divider()

    rows = _normal_library_rows_cached(str(base_dir), _library_dataset_version(base_dir))
    frame = pd.DataFrame(rows)
    st.subheader("文献列表")
    view_filter = st.radio(
        "文献视图",
        ["全部", "正式库", "完整但未索引", "待人工审核", "候选", "隔离", "OUT_OF_SCOPE", "归档"],
        horizontal=True,
        key="library_view_filter",
    )
    if frame.empty:
        st.info("当前没有可显示的有效文献；可使用页面顶部入口添加。")
        _render_secondary_sections(base_dir, backend.mode)
        return

    if view_filter == "正式库":
        frame = frame[frame["library_status"] == "FORMAL"]
    elif view_filter == "完整但未索引":
        frame = frame[(frame["deep_read_complete"] == True) & (frame["library_status"] != "FORMAL") & (frame["domain_scope"] != "OUT_OF_SCOPE")]
    elif view_filter == "待人工审核":
        frame = frame[frame["evidence_status"].isin(["NEEDS_HUMAN_REVIEW", "PENDING_CONFIRMATION", "HUMAN_REVISION_REQUIRED"])]
    elif view_filter == "候选":
        frame = frame[frame["library_status"].isin(["CANDIDATE", "PDF_NOT_ACQUIRED"])]
    elif view_filter == "隔离":
        frame = frame[frame["library_status"] == "QUARANTINED"]
    elif view_filter == "OUT_OF_SCOPE":
        frame = frame[frame["domain_scope"] == "OUT_OF_SCOPE"]
    elif view_filter == "归档":
        frame = frame[frame["library_status"] == "ARCHIVED"]

    filter_cols = st.columns([1, 2])
    with filter_cols[0]:
        types = ["全部", *sorted(frame["type_cn"].dropna().unique())]
        type_filter = st.selectbox("分类", types, key="lib_type")
    with filter_cols[1]:
        keyword = st.text_input(
            "题名 / 作者 / DOI",
            placeholder="输入关键词…",
            key="lib_keyword",
        )
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

    persisted_before = set(st.session_state.get("library_selected_ids", []))
    display = pd.DataFrame(
        {
            "选择": frame["paper_id"].isin(persisted_before),
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
            "库状态": frame["library_status"],
            "来源": frame["source"].map(
                {
                    "formal": "正式库",
                    "candidate": "候选库",
                    "quarantined": "异常记录",
                    "archived": "已归档",
                }
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
            "库状态",
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
    page_selected_ids = [
        str(page.loc[index, "_paper_id"])
        for index in selected_indices
    ]
    page_ids = [str(value) for value in page["_paper_id"].tolist()]
    selected_ids = reconcile_persistent_selection(
        st.session_state.get("library_selected_ids", []), page_ids, page_selected_ids,
    )
    st.session_state["library_selected_ids"] = selected_ids
    selection_cols = st.columns([3, 1])
    selection_cols[0].caption(f"跨页持久选择：{len(selected_ids)} 篇")
    if selection_cols[1].button("清空选择", key="clear_library_selection", disabled=not selected_ids):
        st.session_state["library_selected_ids"] = []
        st.rerun()

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
    _render_secondary_sections(base_dir, backend.mode)

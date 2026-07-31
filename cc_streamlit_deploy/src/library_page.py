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
        value="L-PBF Ti-6Al-4V fatigue pore defect crack initiation",
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
                "疲劳与裂纹萌生",
                "孔隙与缺陷",
                "表面状态",
                "热处理与HIP",
                "裂纹扩展",
                "不限",
            ],
            key="library_auto_oa_topic",
        )
    core_only = st.toggle(
        "仅限L-PBF/SLM Ti-6Al-4V疲劳",
        value=True,
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
        run_archive = st.button(
            "归档",
            key="batch_archive_btn",
            disabled=not eligible,
            width="stretch",
        )
    with delete_col:
        confirm_delete = st.checkbox(
            "确认彻底删除",
            key="batch_delete_confirm",
            disabled=not eligible,
        )
        run_delete = st.button(
            "彻底删除",
            key="batch_delete_btn",
            disabled=not eligible or not confirm_delete,
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
    if run_experiment:
        from src.hypothesis_service import generate_hypotheses

        with st.spinner("正在基于可追溯正文证据生成可证伪实验设计…"):
            hypotheses = generate_hypotheses(
                eligible, base_dir=base_dir, persist=False
            )
            designs = [
                {
                    "hypothesis_id": hypothesis.get("hypothesis_id"),
                    "objective": hypothesis.get("hypothesis"),
                    "independent_variables": hypothesis.get(
                        "independent_variables", []
                    ),
                    "dependent_variables": hypothesis.get(
                        "dependent_variables", []
                    ),
                    "control_variables": hypothesis.get(
                        "control_variables", []
                    ),
                    "moderating_variables": hypothesis.get(
                        "moderating_variables", []
                    ),
                    "support_criteria": hypothesis.get(
                        "support_criteria", []
                    ),
                    "falsification_criteria": hypothesis.get(
                        "falsification_criteria", []
                    ),
                    "supporting_evidence": hypothesis.get(
                        "supporting_evidence", []
                    ),
                    "status": "SYSTEM_PROPOSAL_REQUIRES_HUMAN_REVIEW",
                }
                for hypothesis in hypotheses.get("hypotheses") or []
            ]
            result = {
                "status": (
                    "GENERATED"
                    if designs
                    else hypotheses.get("status", "INSUFFICIENT_EVIDENCE")
                ),
                "experiment_designs": designs,
                "rejected": hypotheses.get("rejected", {}),
            }
        st.session_state["library_experiment_result"] = result
    if run_rag:
        with st.spinner("正在调用统一 Stage-3 索引…"):
            result = rebuild_unified_rag(confirmed, base_dir=base_dir)
        st.session_state["library_rag_result"] = result
        _clear_library_cache()
    if run_archive:
        result = archive_canonical_records(eligible, base_dir=base_dir)
        st.session_state["library_archive_result"] = result
        _clear_library_cache()
        st.rerun()
    if run_delete:
        result = permanently_delete_canonical_records(
            eligible, base_dir=base_dir
        )
        st.session_state["library_delete_result"] = result
        _clear_library_cache()
        st.rerun()

    for key, label in (
        ("library_gap_result", "研究空白结果"),
        ("library_hypothesis_result", "候选假设结果"),
        ("library_experiment_result", "实验设计结果"),
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
    st.divider()
    st.subheader("文献统计")
    _render_statistics(base_dir)
    st.divider()

    rows = _normal_library_rows(base_dir)
    frame = pd.DataFrame(rows)
    st.subheader("文献列表")
    view_filter = st.radio(
        "文献视图",
        ["全部文献", "正式库", "候选库", "异常记录"],
        horizontal=True,
        key="library_view_filter",
    )
    if frame.empty:
        st.info("当前没有可显示的有效文献；可使用页面顶部入口添加。")
        _render_secondary_sections(base_dir, backend.mode)
        return

    if view_filter == "正式库":
        frame = frame[frame["library_status"] == "FORMAL"]
    elif view_filter == "候选库":
        frame = frame[
            ~frame["library_status"].isin(
                ["FORMAL", "QUARANTINED", "ARCHIVED"]
            )
        ]
    elif view_filter == "异常记录":
        frame = frame[frame["library_status"] == "QUARANTINED"]

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
    _render_secondary_sections(base_dir, backend.mode)

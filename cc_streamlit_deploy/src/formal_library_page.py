"""Formal-library-only Streamlit page."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from urllib.parse import quote

from src.library_management import delete_formal_papers, deletion_impact
from src.metadata_gate import read_jsonl
from src.pdf_watcher import ensure_background_watcher, public_status, set_control
from src.cloud_evidence_bundle import (
    FAIL_CLOSED_MESSAGE,
    cloud_bundle_required,
    cloud_bundle_status,
    cloud_records,
    load_cloud_bundle,
)


TIER_LABELS = {
    "TIER1_CORE_DIRECT": "核心直接证据",
    "TIER2_NEAR_DOMAIN": "近领域迁移证据",
    "TIER3_FOUNDATIONAL_MECHANICS": "基础机制证据",
}


def _active_dataset_index(base_dir: Path) -> dict[str, dict[str, Any]]:
    """Read existing frozen tier metadata without changing the dataset."""
    try:
        from src.dataset_versioning import load_active_dataset_manifest

        payload = load_active_dataset_manifest(base_dir)
    except (OSError, RuntimeError, ValueError):
        return {}
    return {
        str(row.get("document_id") or row.get("paper_id") or ""): dict(row)
        for row in (payload.get("papers") or payload.get("documents") or [])
        if row.get("document_id") or row.get("paper_id")
    }


def _tier_label(row: dict[str, Any]) -> str:
    tier = str(row.get("evidence_tier") or "").strip()
    return TIER_LABELS.get(tier, "证据层级未记录")


def _doi_value(row: dict[str, Any]) -> str:
    value = str(row.get("doi") or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.casefold().startswith(prefix):
            value = value[len(prefix):].strip()
            break
    return value if value.casefold().startswith("10.") else ""


def _doi_markdown(row: dict[str, Any]) -> str:
    doi = _doi_value(row)
    if not doi:
        return "DOI未记录"
    return f"[doi:{doi}](https://doi.org/{quote(doi, safe='/.:;()[]')})"


def _formal_rows(base_dir: Path) -> list[dict[str, Any]]:
    active = _active_dataset_index(base_dir)
    if cloud_bundle_required(base_dir):
        source_rows = cloud_records(load_cloud_bundle(base_dir).formal_literature)
        rows = []
        for source in source_rows:
            row = dict(source)
            frozen = active.get(str(row.get("paper_id") or ""), {})
            for key in (
                "evidence_tier", "tier_reason", "tier_role_limit",
                "evidence_record_count", "condition_evidence_record_count",
                "formula_record_count", "confirmed_formula_count", "journal",
            ):
                if not row.get(key) and frozen.get(key) not in (None, ""):
                    row[key] = frozen[key]
            rows.append(row)
        return rows
    # The legacy whitelist is retained for historical v1.0 reproduction only.
    # Current product views must follow the active dataset pointer.
    from src.dataset_versioning import (
        active_dataset_ids,
        active_dataset_manifest_path,
    )

    if not active_dataset_manifest_path(base_dir).is_file():
        return []
    active_ids = active_dataset_ids(base_dir)
    rows = []
    for source in read_jsonl(base_dir / "data/paper_manifest.jsonl"):
        paper_id = str(source.get("paper_id") or "")
        if paper_id not in active_ids or source.get("rag_status") != "INDEXED_STAGE3_UNIFIED":
            continue
        row = dict(source)
        frozen = active.get(paper_id, {})
        for key in (
            "evidence_tier", "tier_reason", "tier_role_limit",
            "evidence_record_count", "condition_evidence_record_count",
            "formula_record_count", "confirmed_formula_count", "journal",
        ):
            if not row.get(key) and frozen.get(key) not in (None, ""):
                row[key] = frozen[key]
        rows.append(row)
    return rows


def _counts(rows: list[dict[str, Any]], base_dir: Path) -> dict[str, int]:
    from src.dataset_versioning import (
        active_dataset_contract_path,
        get_active_dataset_manifest,
    )

    if not active_dataset_contract_path(base_dir).is_file():
        return {
            "formal": 0,
            "rag": 0,
            "evidence": 0,
            "conditions": 0,
            "formulas": 0,
            "pdfs": 0,
        }

    if cloud_bundle_required(base_dir):
        manifest = load_cloud_bundle(base_dir).manifest
        active = get_active_dataset_manifest(base_dir, mounted_manifest=manifest)
        return {
            "formal": int(active["paper_count"]),
            "rag": int(active["rag_count"]),
            "evidence": int(active["evidence_record_count"]),
            "conditions": int(active["condition_evidence_record_count"]),
            "formula_candidates": int(active["formula_candidate_count"]),
            "formula_confirmed": int(active["formula_confirmed_count"]),
            "pdfs": int(manifest["traceable_literature_count"]),
        }
    active = get_active_dataset_manifest(base_dir)
    return {
        "formal": int(active["paper_count"]),
        "rag": int(active["rag_count"]),
        "evidence": int(active["evidence_record_count"]),
        "conditions": int(active["condition_evidence_record_count"]),
        "formula_candidates": int(active["formula_candidate_count"]),
        "formula_confirmed": int(active["formula_confirmed_count"]),
        "pdfs": sum(Path(str(row.get("canonical_pdf_path") or "")).is_file() for row in rows),
    }


def _render_paper_detail(
    st: Any,
    *,
    row: dict[str, Any],
    paper_id: str,
    base_dir: Path,
    cloud_mode: bool,
) -> None:
    st.markdown("#### 文献详情")
    st.markdown(
        f"- **题名**：{row.get('title') or '题名未报告'}\n"
        f"- **作者**：{row.get('authors') or '未报告'}\n"
        f"- **年份**：{row.get('publication_date') or row.get('year') or '未报告'}\n"
        f"- **期刊**：{row.get('journal') or '未报告'}\n"
        f"- **DOI**：{_doi_markdown(row)}\n"
        f"- **Evidence Tier**：{_tier_label(row)}\n"
        f"- **主要主题**：{row.get('topics') or row.get('primary_topic') or '未报告'}\n"
        f"- **文献类型**：{row.get('document_type') or '未报告'}\n"
        f"- **正式状态**：正式收录"
    )
    if row.get("oa_url"):
        st.markdown(f"- **开放链接**：[打开来源]({row['oa_url']})")
    with st.expander("技术详情", expanded=False):
        st.markdown(
            f"- 文献ID：{paper_id}\n"
            f"- 内部RAG状态：{row.get('rag_status') or '未报告'}\n"
            f"- Tier原始值：{row.get('evidence_tier') or '未记录'}\n"
            f"- Tier依据：{row.get('tier_reason') or '未记录'}"
        )
    if not cloud_mode:
        return
    bundle = load_cloud_bundle(base_dir)
    evidence = bundle.evidence_records[
        bundle.evidence_records["paper_id"].astype(str) == paper_id
    ].head(8)
    conditions = bundle.condition_evidence_records[
        bundle.condition_evidence_records["paper_id"].astype(str) == paper_id
    ].head(8)
    formulas = bundle.formula_records[
        bundle.formula_records["paper_id"].astype(str) == paper_id
    ].head(5)
    st.markdown("##### 证据摘要")
    for item in cloud_records(evidence):
        st.markdown(
            f"- {item.get('claim') or item.get('short_excerpt') or '证据摘要未报告'} "
            f"**[{item.get('evidence_id') or 'Evidence ID未记录'} | "
            f"p.{item.get('page_number') or '未记录'}]**"
        )
    with st.expander("实验条件与公式", expanded=False):
        st.markdown("**实验条件**")
        for item in cloud_records(conditions):
            values = [
                f"{key}：{value}"
                for key, value in item.items()
                if key not in {
                    "condition_evidence_id", "evidence_id", "paper_id",
                    "claim", "page_number", "section", "directness",
                    "evidence_role", "independent_variables",
                    "dependent_variables",
                }
                and str(value or "").strip()
                and str(value).strip() not in {"NOT_REPORTED", "{}", "[]"}
            ]
            st.markdown(
                f"- **[{item.get('evidence_id') or 'Evidence ID未记录'} | "
                f"p.{item.get('page_number') or '未记录'}]** "
                + ("；".join(values[:8]) if values else "实验条件未完整报告。")
            )
        st.markdown("**相关公式**")
        for item in cloud_records(formulas):
            st.markdown(
                f"- {item.get('equation') or '公式未报告'} "
                f"（Formula ID：{item.get('formula_id') or '未记录'}；"
                f"页码：{item.get('page_number') or '未记录'}）"
            )


def render_formal_library_page(st: Any, *, base_dir: Path) -> None:
    st.caption("TitaniumFatigueChat")
    st.title("正式文献库")
    bundle_status = cloud_bundle_status(base_dir)
    cloud_mode = bool(bundle_status["required"])
    if cloud_mode and not bundle_status["ready"]:
        st.error(FAIL_CLOSED_MESSAGE)
        return
    rows = _formal_rows(base_dir)
    counts = _counts(rows, base_dir)
    metric_columns = st.columns(4)
    for column, (label, value) in zip(
        metric_columns,
        (
            ("正式文献数", counts["formal"]), ("已进入RAG数", counts["rag"]),
            ("EvidenceRecord数", counts["evidence"]),
            ("ConditionEvidenceRecord数", counts["conditions"]),
        ),
    ):
        column.metric(label, value)
    with st.expander("库数据详情", expanded=False):
        extra_columns = st.columns(3)
        extra_columns[0].metric("公式候选", counts["formula_candidates"])
        extra_columns[1].metric("严格确认", counts["formula_confirmed"])
        extra_columns[2].metric(
            "可追溯文献" if cloud_mode else "本地有效PDF数",
            counts["pdfs"],
        )

    if not cloud_mode:
        with st.expander("PDF自动入库", expanded=False):
            status = public_status(base_dir)
            status_columns = st.columns(4)
            status_columns[0].metric("自动监听", status["自动监听"])
            status_columns[1].metric("当前文件", status["当前文件"])
            status_columns[2].metric("页面进度", status["页面进度"])
            status_columns[3].metric("等待队列", status["等待队列"])
            st.caption(
                f"当前阶段：{status['当前阶段']}；最近成功：{status['最近成功']}；"
                f"最近删除：{status['最近删除']}"
            )
            start, pause = st.columns(2)
            if start.button("启动自动监听", use_container_width=True):
                set_control("RUN", base_dir=base_dir)
                ensure_background_watcher(base_dir)
                st.rerun()
            if pause.button("暂停自动监听", use_container_width=True):
                set_control("PAUSE", base_dir=base_dir)
                st.rerun()

    query = st.text_input("检索正式文献", placeholder="输入题名、作者、年份或 DOI")
    if query.strip():
        needle = query.casefold().strip()
        rows = [
            row for row in rows
            if needle in " ".join(
                str(row.get(key) or "")
                for key in ("title", "authors", "publication_date", "doi")
            ).casefold()
        ]
    page_size = 20
    total_pages = max(1, math.ceil(len(rows) / page_size))
    page = min(int(st.session_state.get("formal_library_page", 1)), total_pages)
    st.session_state["formal_library_page"] = page
    start_index = (page - 1) * page_size
    page_rows = rows[start_index : start_index + page_size]
    selected = set(st.session_state.get("formal_library_selected_ids", []))
    for row in page_rows:
        paper_id = str(row.get("paper_id") or "")
        with st.container(border=True):
            title_col, detail_col = st.columns([8, 1.35], vertical_alignment="center")
            title_col.markdown(f"**{row.get('title') or '题名未报告'}**")
            is_open = st.session_state.get("formal_library_detail_id") == paper_id
            if detail_col.button(
                "收起详情" if is_open else "查看详情",
                key=f"formal_detail_{paper_id}",
                use_container_width=True,
            ):
                st.session_state["formal_library_detail_id"] = None if is_open else paper_id
                st.rerun()
            st.caption(
                f"{row.get('authors') or '作者未报告'} · "
                f"{row.get('publication_date') or row.get('year') or '年份未报告'} · "
                f"{row.get('journal') or '期刊未报告'}"
            )
            st.markdown(
                f"<span class='tfc-tier-pill'>{_tier_label(row)}</span>&nbsp;&nbsp;"
                f"{_doi_markdown(row)} · 正式收录 · "
                f"Evidence：{row.get('evidence_record_count', 0)} · "
                f"ConditionEvidence：{row.get('condition_evidence_record_count', 0)} · "
                f"Formula：{row.get('formula_record_count', 0)}",
                unsafe_allow_html=True,
            )
            if not cloud_mode:
                checked = st.checkbox(
                    "选择用于文献库管理",
                    value=paper_id in selected,
                    key=f"formal_select_{paper_id}",
                )
                if checked:
                    selected.add(paper_id)
                else:
                    selected.discard(paper_id)
            if is_open:
                _render_paper_detail(
                    st,
                    row=row,
                    paper_id=paper_id,
                    base_dir=base_dir,
                    cloud_mode=cloud_mode,
                )
    st.session_state["formal_library_selected_ids"] = sorted(selected)

    previous, page_label, following = st.columns([1, 2, 1])
    if previous.button("上一页", disabled=page <= 1, use_container_width=True):
        st.session_state["formal_library_page"] = page - 1
        st.rerun()
    page_label.markdown(f"<p style='text-align:center'>第 {page} / {total_pages} 页</p>", unsafe_allow_html=True)
    if following.button("下一页", disabled=page >= total_pages, use_container_width=True):
        st.session_state["formal_library_page"] = page + 1
        st.rerun()

    request_delete = False
    if not cloud_mode:
        delete_col, clear_col = st.columns(2)
        request_delete = delete_col.button(
            "删除选中文献",
            disabled=not selected,
            use_container_width=True,
        )
        if clear_col.button("清空选择", disabled=not selected, use_container_width=True):
            st.session_state["formal_library_selected_ids"] = []
            for row in rows:
                st.session_state.pop(f"formal_select_{row.get('paper_id')}", None)
            st.rerun()
    if request_delete:
        st.session_state["formal_library_delete_pending"] = True
    if st.session_state.get("formal_library_delete_pending") and selected:
        impact = deletion_impact(sorted(selected), base_dir=base_dir)
        st.warning(
            f"将删除 {impact['paper_count']} 篇文献、{impact['pdf_count']} 个 PDF、"
            f"{impact['evidence_count']} 条 EvidenceRecord 和 {impact['formula_count']} 条公式，"
            "并从正式 RAG 中移除。"
        )
        for title in impact["titles"]:
            st.markdown(f"- {title}")
        confirmation = st.text_input("输入 DELETE 确认", key="formal_delete_confirmation")
        if st.button(
            "确认删除选中文献", type="primary", use_container_width=True,
            disabled=confirmation != "DELETE",
        ):
            with st.spinner("正在删除关联数据并重建索引"):
                delete_formal_papers(
                    sorted(selected), confirmation=confirmation, base_dir=base_dir
                )
            st.session_state["formal_library_selected_ids"] = []
            st.session_state["formal_library_delete_pending"] = False
            st.success("删除完成，正式文献库和 RAG 已更新。")
            st.rerun()

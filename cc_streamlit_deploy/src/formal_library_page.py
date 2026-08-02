"""Formal-library-only Streamlit page."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from src.library_management import delete_formal_papers, deletion_impact
from src.metadata_gate import read_jsonl
from src.pdf_watcher import ensure_background_watcher, public_status, set_control


def _formal_rows(base_dir: Path) -> list[dict[str, Any]]:
    whitelist_path = base_dir / "data/system/formal_rag_whitelist.json"
    if not whitelist_path.is_file():
        return []
    try:
        whitelist_payload = json.loads(whitelist_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    whitelist = set(whitelist_payload.get("paper_ids") or [])
    return [
        row for row in read_jsonl(base_dir / "data/paper_manifest.jsonl")
        if str(row.get("paper_id") or "") in whitelist
        and row.get("rag_status") == "INDEXED_STAGE3_UNIFIED"
    ]


def _counts(rows: list[dict[str, Any]], base_dir: Path) -> dict[str, int]:
    ids = {str(row.get("paper_id") or "") for row in rows}
    evidence_path = base_dir / "data/evidence/trusted_evidence.csv"
    evidence: list[dict[str, Any]] = []
    if evidence_path.is_file():
        with evidence_path.open("r", encoding="utf-8-sig", newline="") as stream:
            evidence = [
                row for row in csv.DictReader(stream)
                if str(row.get("canonical_paper_id") or row.get("paper_id") or "") in ids
            ]
    evidence_ids = {str(row.get("evidence_id") or "") for row in evidence}
    conditions = [
        row for row in read_jsonl(
            base_dir / "data/evidence/condition_evidence_records.jsonl"
        )
        if str(row.get("evidence_id") or "") in evidence_ids
    ]
    formulas = 0
    for paper_id in ids:
        for filename in ("equations.jsonl", "formula_records.jsonl"):
            path = base_dir / "data/deep_read" / paper_id / filename
            if path.exists():
                formulas += len(read_jsonl(path))
                break
    return {
        "formal": len(rows), "rag": len(rows), "evidence": len(evidence),
        "conditions": len(conditions), "formulas": formulas,
        "pdfs": sum(Path(str(row.get("canonical_pdf_path") or "")).is_file() for row in rows),
    }


def render_formal_library_page(st: Any, *, base_dir: Path) -> None:
    st.caption("TitaniumFatigueChat")
    st.title("正式文献库")
    rows = _formal_rows(base_dir)
    counts = _counts(rows, base_dir)
    metric_columns = st.columns(6)
    for column, (label, value) in zip(
        metric_columns,
        (
            ("正式文献数", counts["formal"]), ("已进入RAG数", counts["rag"]),
            ("EvidenceRecord数", counts["evidence"]),
            ("ConditionEvidenceRecord数", counts["conditions"]),
            ("公式数", counts["formulas"]), ("本地有效PDF数", counts["pdfs"]),
        ),
    ):
        column.metric(label, value)

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
        checked = st.checkbox(
            str(row.get("title") or "题名未报告"),
            value=paper_id in selected,
            key=f"formal_select_{paper_id}",
        )
        if checked:
            selected.add(paper_id)
        else:
            selected.discard(paper_id)
        st.caption(
            f"{row.get('authors') or '作者未报告'} · "
            f"{row.get('publication_date') or '年份未报告'} · "
            f"DOI：{row.get('doi') or '无'}"
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

    detail_col, delete_col, clear_col = st.columns(3)
    show_detail = detail_col.button("查看文献详情", disabled=not selected, use_container_width=True)
    request_delete = delete_col.button("删除选中文献", disabled=not selected, use_container_width=True)
    if clear_col.button("清空选择", disabled=not selected, use_container_width=True):
        st.session_state["formal_library_selected_ids"] = []
        for row in rows:
            st.session_state.pop(f"formal_select_{row.get('paper_id')}", None)
        st.rerun()

    by_id = {str(row.get("paper_id") or ""): row for row in rows}
    if show_detail:
        for paper_id in sorted(selected):
            row = by_id.get(paper_id)
            if not row:
                continue
            st.subheader(str(row.get("title") or "题名未报告"))
            st.markdown(
                f"- 作者：{row.get('authors') or '未报告'}\n"
                f"- 年份：{row.get('publication_date') or '未报告'}\n"
                f"- DOI：{row.get('doi') or '无'}\n"
                f"- 文献类型：{row.get('document_type') or '未报告'}\n"
                f"- 研究范围：{'钛合金疲劳核心' if row.get('domain_scope') == 'CORE' else '必要背景'}"
            )
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

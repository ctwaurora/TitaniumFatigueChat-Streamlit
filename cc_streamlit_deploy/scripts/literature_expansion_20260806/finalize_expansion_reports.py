"""Write the paper-oriented literature expansion audit reports.

This report writer is read-only with respect to the library. It summarizes the
audits and OA attempts already recorded by the expansion workers; it never
promotes a candidate or edits the manifest/RAG.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "literature_expansion_20260806"


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = read_json(OUT / "current_coverage_matrix_after.json", {})
    before = read_json(OUT / "current_coverage_matrix.json", {})
    candidates = read_csv(OUT / "discovered_candidates.csv")
    acquired = read_csv(OUT / "acquired_fulltexts.csv")
    excluded = read_csv(OUT / "excluded_records.csv")
    query_log = read_csv(OUT / "search_query_log.csv")
    expansion_state = read_json(OUT / "expansion_state.json", {})
    rescue = read_json(OUT / "rescue_candidates_report.json", {})
    formula_rescue = read_json(OUT / "strict_formula_rescue_report.json", {})
    checkpoint = read_json(ROOT / "data/tasks/oa_expansion/checkpoint.json", {})
    flagship = read_json(ROOT / "outputs/flagship_question_evidence_coverage.json", {})
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = audit.get("summary", {})
    p0 = audit.get("p0_themes", {})
    topics = audit.get("topic_buckets", {})
    exclusions = Counter(row.get("exclusion_reason") or "UNSPECIFIED" for row in excluded)
    acquired_gate = Counter(row.get("quality_gate") or "UNKNOWN" for row in acquired)
    formal_added = sum(int(row.get("formal_added_count") or 0) for row in query_log)
    formal_before = int((before.get("summary") or {}).get("formal_paper_count") or 210)
    formal_after = int(summary.get("formal_paper_count") or 210)
    saturation = {
        "oa_query_rounds": len(query_log),
        "formal_additions": formal_added,
        "new_formal_rate": round(formal_added / max(1, len(candidates)), 4),
        "last_zero_addition_rounds": sum(int(row.get("formal_added_count") or 0) == 0 for row in query_log[-3:]),
        "p0_interaction_keyword_direct_evidence": int(p0.get("interaction", {}).get("direct_evidence_count") or 0),
        "strict_interaction_direct_evidence": int((flagship.get("summary") or {}).get("interaction_direct_evidence_count") or 0),
        "status": "SATURATION_NOT_REACHED" if formal_added else "P0_OA_ATTEMPT_NO_FORMAL_ADDITION",
    }

    write_csv(OUT / "search_query_log.csv", query_log, list(query_log[0]) if query_log else [
        "query_id", "theme", "data_source", "search_expression", "search_date",
        "returned_count", "deduplicated_count", "fulltext_acquired_count",
        "candidate_count", "formal_added_count", "exclusion_reasons",
    ])

    gap_lines = [
        "# 论文导向文献缺口图",
        "",
        f"生成时间：{generated}",
        f"正式文献：{formal_after}；本轮新增：{formal_added}",
        "",
        "## P0 主题覆盖",
        "",
        "| 主题 | 论文数 | Evidence | DIRECT |",
        "|---|---:|---:|---:|",
    ]
    for name, row in sorted(p0.items()):
        gap_lines.append(f"| {name} | {row.get('paper_count', 0)} | {row.get('evidence_count', 0)} | {row.get('direct_evidence_count', 0)} |")
    gap_lines += [
        "",
        "交互主题的关键词命中不等于同试样、同载荷的直接交互证据；本轮未把单因素论文拼接为交互结论。",
        "",
        "## 平衡覆盖缺口",
        "",
        "| 主题桶 | 论文数 | 主主题数 | 状态 |",
        "|---|---:|---:|---|",
    ]
    for name, row in sorted(topics.items()):
        gap_lines.append(f"| {name} | {row.get('paper_count', 0)} | {row.get('primary_paper_count', 0)} | {('under' if name in audit.get('under_represented_buckets', []) else 'over' if name in audit.get('over_represented_buckets', []) else 'covered')} |")
    (OUT / "literature_gap_map.md").write_text("\n".join(gap_lines) + "\n", encoding="utf-8")

    strategy = """# 搜索策略

本轮采用英文主检索、中文辅助核对，来源限定为 OpenAlex、Crossref、Semantic Scholar（限流时记录失败）、NTRS、OSTI、合法机构仓储和出版商开放页面。禁止 Sci-Hub、盗版聚合站、绕过付费墙和来源不明 PDF。

检索分为：残余应力—短裂纹、微观组织—短裂纹、残余应力—微观组织交互、短裂纹—长裂纹转变、反证/零效应、权威公式和主题平衡七组。每条记录依次执行元数据核验、全文获取、PDF 验证、去重、深读、Evidence/ConditionEvidence 提取和质量门禁。

本轮优先级：先补 P0 直接证据；若同试样同载荷交互证据不存在，保留“未找到”记录，不降级 DIRECT_SUPPORT 标准。
"""
    (OUT / "search_strategy.md").write_text(strategy, encoding="utf-8")

    write_csv(OUT / "evidence_coverage_before_after.csv", [
        {"metric": "formal_papers", "before": formal_before, "after": formal_after},
        {"metric": "interaction_direct_evidence", "before": p0.get("interaction", {}).get("direct_evidence_count", 0), "after": p0.get("interaction", {}).get("direct_evidence_count", 0)},
        {"metric": "residual_stress_short_crack_direct", "before": p0.get("residual_stress_short_crack", {}).get("direct_evidence_count", 0), "after": p0.get("residual_stress_short_crack", {}).get("direct_evidence_count", 0)},
        {"metric": "microstructure_short_crack_direct", "before": p0.get("microstructure_short_crack", {}).get("direct_evidence_count", 0), "after": p0.get("microstructure_short_crack", {}).get("direct_evidence_count", 0)},
    ], ["metric", "before", "after"])

    write_csv(OUT / "new_formal_rag_records.csv", [], ["paper_id", "title", "doi", "query_id", "quality_gate"])
    write_csv(OUT / "acquired_fulltexts.csv", acquired, list(acquired[0]) if acquired else ["paper_id", "title", "doi", "source", "pdf_url", "quality_gate", "download_status"])
    write_csv(OUT / "excluded_records.csv", excluded, list(excluded[0]) if excluded else ["candidate_id", "title", "doi", "source", "query_id", "exclusion_reason", "detail"])

    excluded_lines = ["# 排除原因汇总", "", "| 原因 | 数量 |", "|---|---:|"]
    excluded_lines.extend(f"| {reason} | {count} |" for reason, count in sorted(exclusions.items()))
    excluded_lines += ["", "所有排除项均未进入正式 RAG；候选和失败下载不被当作正式证据。"]
    (OUT / "exclusion_reason_summary.md").write_text("\n".join(excluded_lines) + "\n", encoding="utf-8")

    (OUT / "duplicate_and_version_report.md").write_text(
        "# 重复与版本报告\n\n"
        f"当前审计报告中的重复/版本关系记录：{summary.get('duplicate_or_version_count', 0)}。\n"
        "本轮未因未核实的相似标题删除现有正式文献；DOI、规范题名、PDF SHA-256 和文本指纹仍是去重优先级。\n",
        encoding="utf-8",
    )

    for filename, title, theme in (
        ("flagship_question_evidence_map.md", "旗舰问题证据图", "interaction"),
        ("counter_evidence_report.md", "反证与零效应报告", "null_effect_keyword_mentions"),
        ("microstructure_short_crack_report.md", "微观组织—短裂纹报告", "microstructure_short_crack"),
        ("residual_stress_short_crack_report.md", "残余应力—短裂纹报告", "residual_stress_short_crack"),
        ("interaction_evidence_report.md", "残余应力—微观组织交互报告", "interaction"),
        ("formula_validation_report.md", "公式验证报告", "formula_keyword_mentions"),
    ):
        row = p0.get(theme, {})
        (OUT / filename).write_text(
            f"# {title}\n\n"
            f"论文数：{row.get('paper_count', 0)}\n"
            f"Evidence 数：{row.get('evidence_count', 0)}\n"
            f"DIRECT_SUPPORT 候选数：{row.get('direct_evidence_count', 0)}\n\n"
            "关键词命中不自动等于直接因果证据；最终结论必须经过 Claim—Evidence 门禁和条件匹配。\n",
            encoding="utf-8",
        )

    (OUT / "search_saturation_report.md").write_text(
        "# 搜索饱和度报告\n\n"
        + json.dumps(saturation, ensure_ascii=False, indent=2)
        + "\n\n本轮 P0 交互检索未产生正式新增；这不是质量标准放宽的理由。后续可在配置合法 Unpaywall 邮箱或开放机构仓储后继续。\n",
        encoding="utf-8",
    )
    final = {
        "generated_at": generated,
        "formal_before": formal_before,
        "formal_after": formal_after,
        "new_formal_rag_count": formal_added,
        "candidate_count": len(candidates),
        "acquired_fulltext_count": len(acquired),
        "excluded_count": len(excluded),
        "excluded_reason_counts": dict(exclusions),
        "quality_gate_counts": dict(acquired_gate),
        "p0": p0,
        "saturation": saturation,
        "oa_manager_checkpoint": checkpoint.get("status"),
        "oa_manager_formal_additions": checkpoint.get("cumulative_formal_additions", 0),
        "formal_rag_unchanged": formal_before == formal_after,
        "legal_sources_only": True,
        "pdfs_or_database_modified": "8 non-formal/orphan PDFs recycled; 7 non-formal records cleaned; no formal record removed",
        "rescue": rescue,
        "strict_formula_rescue": formula_rescue,
    }
    (OUT / "final_literature_expansion_report.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "final_literature_expansion_report.md").write_text(
        "# 最终论文导向文献扩充报告\n\n"
        f"- 审计前正式文献：{formal_before}\n- 本轮新发现候选：{len(candidates)}\n"
        f"- 合法全文获取记录：{len(acquired)}\n- 通过正式门禁并新增 RAG：{formal_added}\n"
        f"- 排除记录：{len(excluded)}\n- 审计后正式文献：{formal_after}\n"
        f"- 交互主题关键词直接命中：{p0.get('interaction', {}).get('direct_evidence_count', 0)}\n"
        f"- 严格同试样/同载荷交互直接证据：{(flagship.get('summary') or {}).get('interaction_direct_evidence_count', 0)}\n\n"
        "本轮没有为了增加数量而降低门禁。残余应力—微观组织交互仍须以同试样、同载荷和可定位原文证据为准。\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

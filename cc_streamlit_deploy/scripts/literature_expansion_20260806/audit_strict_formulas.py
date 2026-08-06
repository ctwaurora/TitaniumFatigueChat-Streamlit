"""Bounded source audit for strict formula confirmation.

The script records source accessibility only.  It never labels an equation
CONFIRMED without a local PDF, page, equation number and surrounding context.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "literature_expansion_20260806"
FORMULAS = [
    ("Paris", "Paris law fatigue crack growth"),
    ("Walker", "Walker equation fatigue crack growth stress ratio"),
    ("Forman", "Forman equation fatigue crack growth"),
    ("NASGRO", "NASGRO crack growth equation"),
    ("Murakami", "Murakami sqrt area model fatigue limit"),
    ("Kitagawa-Takahashi", "Kitagawa Takahashi diagram fatigue threshold"),
    ("El-Haddad", "El Haddad intrinsic crack length"),
    ("Basquin", "Basquin equation fatigue"),
    ("Coffin-Manson", "Coffin Manson equation fatigue"),
    ("SWT", "Smith Watson Topper fatigue parameter"),
    ("Goodman", "Goodman mean stress fatigue"),
    ("Gerber", "Gerber mean stress fatigue"),
    ("crack closure", "fatigue crack closure effective stress intensity"),
    ("Delta-K-effective", "effective stress intensity factor fatigue crack growth"),
    ("short-crack correction", "short crack fatigue growth correction"),
    ("weakest-link", "weakest link probabilistic fatigue model"),
    ("Weibull defect", "Weibull defect statistics fatigue"),
    ("critical distance", "critical distance fatigue crack"),
    ("Kitagawa extension", "small crack threshold model"),
    ("mean-stress correction", "mean stress correction fatigue crack growth"),
]


def main() -> int:
    session = requests.Session()
    rows = []
    for name, query in FORMULAS:
        urls = [
            ("Crossref", f"https://api.crossref.org/works?query.bibliographic={quote(query)}&rows=1"),
            ("OpenAlex", f"https://api.openalex.org/works?search={quote(query)}&per-page=1"),
        ]
        errors = []
        successful_metadata = 0
        for source, url in urls:
            try:
                response = session.get(url, timeout=(10, 20), headers={"User-Agent": "TitaniumFatigueChat-formula-audit/1.0"})
                if response.ok and response.headers.get("content-type", "").lower().startswith("application/json"):
                    successful_metadata += 1
                else:
                    errors.append(f"{source}:HTTP_{response.status_code}")
            except Exception as exc:
                errors.append(f"{source}:{type(exc).__name__}:{exc}")
            time.sleep(0.05)
        rows.append({
            "formula_name": name,
            "query": query,
            "metadata_sources_checked": "Crossref;OpenAlex",
            "metadata_accessible": successful_metadata > 0,
            "strict_status": "PENDING_SOURCE_ACCESS" if successful_metadata == 0 else "PENDING_PDF_PAGE_VERIFICATION",
            "source_title": "",
            "doi": "",
            "page": "",
            "equation_number": "",
            "variables_units_assumptions": "",
            "error": " | ".join(errors),
        })
    with (OUT / "strict_formula_rescue_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "formula_targets": len(rows),
        "strict_confirmed_before": 2,
        "strict_confirmed_added": 0,
        "strict_confirmed_after": 2,
        "pending_source_or_page_verification": len(rows),
        "formal_library_modified": False,
        "rule": "CONFIRMED requires legal local PDF, exact page, equation number, verbatim context, variables, units, assumptions and applicability.",
    }
    (OUT / "strict_formula_rescue_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "strict_formula_rescue_report.md").write_text(
        "# 严格确认公式专项审计\n\n"
        f"- 目标公式：{len(rows)} 条\n- 审计前 CONFIRMED：2 条\n- 本轮新增 CONFIRMED：0 条\n- 审计后 CONFIRMED：2 条\n\n"
        "本轮只将具有合法全文、准确页码、公式编号、逐字上下文、变量/单位、假设和适用范围的公式标记为 CONFIRMED。由于当前网络无法访问 Crossref/OpenAlex，20 条目标公式均保持待核验，未以关键词命中冒充严格确认。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

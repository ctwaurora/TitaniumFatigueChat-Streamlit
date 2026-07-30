"""Generate baseline and ablation reports from existing CSV data."""
import csv
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

def generate():
    # --- Baseline ---
    bpath = DATA_DIR / "baseline_comparison.csv"
    if bpath.exists():
        with open(bpath, encoding="utf-8-sig") as f:
            cr = csv.DictReader(f); rows = list(cr); fn = cr.fieldnames
        tcol = "task" if "task" in fn else "test_case"
        scols = [c for c in fn if c not in ("task", "test_case", "system_version")]
        aggr = {}
        for r in rows:
            sv = r.get("system_version","")
            if sv not in aggr: aggr[sv] = {c:0 for c in scols}; aggr[sv]["_n"]=0
            aggr[sv]["_n"] += 1
            for c in scols:
                try: aggr[sv][c] += float(r.get(c,0) or 0)
                except: pass

        with open(OUTPUTS_DIR / "baseline_comparison_report.md", "w") as f:
            f.write("# Baseline Comparison Report\n\n")
            f.write(f"Systems: {len(aggr)}, Tasks: {len(set(r.get(tcol,'') for r in rows))}\n\n")
            f.write("| System | " + " | ".join(scols) + " |\n")
            f.write("|--------|" + "|".join(":---:" for _ in scols) + "|\n")
            for sv in sorted(aggr, key=lambda s: aggr[s][scols[-1]]/max(aggr[s]["_n"],1), reverse=True):
                vals = [f"{aggr[sv][c]/aggr[sv]['_n']:.1f}" for c in scols]
                f.write(f"| {sv} | {' | '.join(vals)} |\n")
        print(f"[OK] baseline_comparison_report.md")

    # --- Ablation ---
    apath = DATA_DIR / "ablation_results.csv"
    if apath.exists():
        with open(apath, encoding="utf-8-sig") as f:
            cr = csv.DictReader(f); rows = list(cr); fn = cr.fieldnames
        scols = [c for c in fn if c not in ("ablation_version",)]
        with open(OUTPUTS_DIR / "ablation_study_report.md", "w") as f:
            f.write("# Ablation Study Report\n\n")
            f.write(f"Configurations: {len(rows)}\n\n")
            f.write("| Version | " + " | ".join(scols) + " |\n")
            f.write("|--------|" + "|".join(":---:" for _ in scols) + "|\n")
            for r in rows:
                vals = [r.get(c,"") for c in scols]
                f.write(f"| {r['ablation_version']} | {' | '.join(vals)} |\n")
        print(f"[OK] ablation_study_report.md")

if __name__ == "__main__":
    generate()

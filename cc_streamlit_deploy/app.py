#!/usr/bin/env python3
"""
TitaniumFatigueChat — 面向钛合金疲劳研究的证据约束型 AI Scientist

使用:
    python app.py ingest       # 构建文献库
    python app.py discover     # 发现研究空白
    python app.py validate     # 验证假设/生成推荐卡片
    python app.py demo         # 完整流程 + 导出比赛包

配置:
    DEEPSEEK_API_KEY 或 .streamlit/secrets.toml — DeepSeek API Key
    config/task_profile.yaml — 榜题配置
    # (teacher constraints integrated into validator logic)
"""

import sys
import os
import json
from pathlib import Path

# Windows encoding fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

try:
    import typer
    from rich.console import Console
    from rich.markup import escape
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    print("缺少依赖：请运行 pip install typer rich")
    print("全部依赖：pip install -r requirements.txt")
    sys.exit(1)

from src import __version__, __app_name__
from src.ingestion import run_ingest, write_evidence_map
from src.discovery import run_discover
from src.validator import run_validate
from src.exporter import run_export
from src.debug_outputs import write_debug_scope_report, write_debug_extraction_quality
from src.scorecard import run_scorecard
from src.deepseek_usage import run_deepseek_usage_report
from src.evidence_gate import run_evidence_gate
from src.reproducibility import run_reproducibility

app = typer.Typer(
    name=__app_name__,
    help="面向钛合金疲劳研究的证据约束型 AI Scientist",
    add_completion=False,
)
console = Console()


def _print_banner():
    """打印项目 Banner。"""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║              {__app_name__} v{__version__}                  ║
║     Evidence-Constrained AI Scientist for                  ║
║          Titanium Alloy Fatigue Research                   ║
╚══════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="bold cyan")
    console.print(Panel(
        "[yellow]本项目是钛合金疲劳科研助手，L-PBF Ti-6Al-4V 孔隙问题是当前主案例，\n"
        "通过文献证据抽取、RAG 文献分析、研究空白验证和假设生成，\n"
        "构建可追溯、可验证的领域知识库与假设生成系统。[/yellow]",
        title="定位",
        border_style="blue",
    ))


def _check_deepseek_key(required: bool = True) -> bool:
    """检查 DeepSeek API Key 是否存在。

    只读取环境变量或 Streamlit secrets。
    如果没有配置 API key，不会崩溃，只会提示部分功能受限。
    """
    from src.api_keys import get_deepseek_api_key

    key = get_deepseek_api_key()
    if not key:
        msg = "未配置 DEEPSEEK_API_KEY。部分生成能力不可用，但文献库浏览和已有结果仍可使用。"
        if required:
            console.print(f"[red]错误: {msg}[/red]")
        else:
            console.print(f"[yellow]提示: {msg}[/yellow]")
        return False
    return True


def _count_literature() -> int:
    """简单统计文献数量。"""
    try:
        from skills.library_skill import get_all_papers
        return len(get_all_papers())
    except Exception:
        return 0


# ── 命令 1: ingest ───────────────────────────────────────────────────────


@app.command()
def ingest(
    scan_only: bool = typer.Option(
        False,
        "--scan-only",
        help="仅递归扫描并验证 PDF 基础信息，不写文献卡或证据。",
    ),
    stage2_config: Path = typer.Option(
        None,
        "--stage2-config",
        help="仅逐页精读本地 JSON 配置中列出的 PDF，不扫描整个语料库。",
    ),
    force_deep_read: bool = typer.Option(
        False,
        "--force-deep-read",
        help="忽略相同文件哈希的阶段2幂等缓存。",
    ),
):
    """构建钛合金疲劳文献库

    从 papers/, early_papers/, followup_papers/ 读取 PDF，
    抽取文献卡片，生成 literature_database.csv。
    """
    _print_banner()
    if stage2_config is not None:
        from src.deep_read_pipeline import deep_read_pdf

        try:
            payload = json.loads(stage2_config.read_text(encoding="utf-8"))
            configured = payload.get("papers", payload) if isinstance(payload, dict) else payload
            if not isinstance(configured, list) or not configured:
                raise ValueError("配置必须包含非空 papers 列表")
        except Exception as exc:
            console.print(f"[red]阶段2配置无效: {escape(str(exc))}[/red]")
            raise typer.Exit(1)
        results = []
        for item in configured:
            raw_path = item.get("path", "") if isinstance(item, dict) else str(item)
            path = Path(raw_path)
            if not path.is_absolute():
                path = (stage2_config.parent / path).resolve()
            result = deep_read_pdf(path, force=force_deep_read)
            results.append(result)
            console.print(
                f"  {escape(path.name)}: {result.get('status')} | "
                f"pages={result.get('processed_page_count', 0)} | "
                f"evidence={result.get('evidence_count', 0)}"
            )
        failed = [row for row in results if row.get("status") == "FAILED"]
        if failed:
            raise typer.Exit(1)
        console.print(f"[green]阶段2逐页精读完成: {len(results)} 篇[/green]")
        return
    if scan_only:
        from src.ingestion import _scan_paper_dirs
        from src.stage1_store import validate_pdf_path

        pdfs = _scan_paper_dirs()
        invalid = [path for path in pdfs if not validate_pdf_path(path)["pdf_valid"]]
        console.print(f"\n[bold]Stage-1 ingest scan-only[/bold]")
        console.print(f"  递归发现 PDF: {len(pdfs)}")
        console.print(f"  有效 PDF: {len(pdfs) - len(invalid)}")
        console.print(f"  损坏/不可解析 PDF: {len(invalid)}")
        if invalid:
            raise typer.Exit(1)
        return
    if not _check_deepseek_key():
        raise typer.Exit(1)

    console.print("\n[bold]Step 1/4: 构建文献库 (ingest)[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("处理 PDF 并抽取文献卡片...", total=None)

        stats = run_ingest()

        progress.update(task, completed=True)

    # 生成 evidence map
    ev_map_path = write_evidence_map(stats)
    console.print(f"  Evidence Map: [green]{escape(str(ev_map_path))}[/green]")

    console.print(f"\n[green]✓ ingest 完成[/green]")
    console.print(f"  处理 {stats.get('total_pdfs', 0)} 个 PDF")
    console.print(f"  成功 {stats.get('processed', 0)}，跳过 {stats.get('skipped_duplicates', 0)}，失败 {stats.get('failed', 0)}")
    console.print(f"  非钛合金方向 {stats.get('out_of_scope', 0)}")
    console.print(f"  当前文献库共 {_count_literature()} 篇")


# ── 命令 2: discover ─────────────────────────────────────────────────────


@app.command()
def discover():
    """发现研究空白

    构建变量—性能—机制—证据表、覆盖矩阵、
    检测候选研究空白、筛除伪空白。
    """
    _print_banner()

    lit_count = _count_literature()
    if lit_count == 0:
        console.print("[yellow]警告: 文献库为空。请先运行 python app.py ingest[/yellow]")

    console.print("\n[bold]Step 2/4: 发现研究空白 (discover)[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("构建变量-机制表和覆盖矩阵...", total=None)
        stats = run_discover()
        progress.update(task, completed=True)

    console.print(f"\n[green]✓ discover 完成[/green]")
    console.print(f"  变量—机制关系记录: {stats.get('variable_records', 0)}")
    console.print(f"  候选研究空白: {stats.get('candidate_gaps', 0)}")
    console.print(f"  保留空白: {stats.get('real_gaps', 0)}")
    console.print(f"  伪空白（已拒绝）: {stats.get('pseudo_gaps', 0)}")
    console.print("  输出：data/variable_mechanism.csv, data/coverage_matrix.csv")
    console.print("  报告：outputs/02_gap_diagnosis.md")

    # 生成调试输出
    debug_scope = write_debug_scope_report()
    debug_extraction = write_debug_extraction_quality()
    console.print(f"  调试：{debug_scope}", markup=False)
    console.print(f"  调试：{debug_extraction}", markup=False)

    if lit_count < 5:
        console.print("\n[yellow][!] 文献库规模较小，覆盖分析仅供参考。[/yellow]")


# ── 命令 3: validate ─────────────────────────────────────────────────────


@app.command()
def validate():
    """验证研究假设并生成推荐卡片

    质量门禁、可行性判断、历史回溯、基线对比。
    """
    _print_banner()

    lit_count = _count_literature()
    if lit_count == 0:
        console.print("[yellow]警告: 文献库为空。建议先运行 python app.py ingest[/yellow]")

    console.print("\n[bold]Step 3/4: 验证假设 (validate)[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("运行质量门禁和基线对比...", total=None)
        stats = run_validate()
        progress.update(task, completed=True)

    console.print(f"\n[green]✓ validate 完成[/green]")
    qg = stats.get("quality_gate_results", {})
    console.print(f"  质量门禁输入: {qg.get('total', 0)} 项")
    console.print(f"  通过: {qg.get('passed', 0)} 项")
    console.print(f"  拒绝: {qg.get('rejected', 0)} 项")

    has_ev = stats.get("has_evidence_recommendations", False)
    if has_ev:
        console.print(f"  科学假设数: {stats.get('recommendation_count', 0)} 个")
        console.print("  辅助输出:")
        console.print("    - outputs/03_hypothesis_summary.md")
        console.print("    - outputs/03_recommendation_cards_pretty.md")
        console.print("    - outputs/03_recommendation_cards_pretty.html")
    else:
        console.print("  [yellow]正式推荐方向: 无（质量门禁未通过，暂不生成证据支持型推荐）[/yellow]")
        console.print("  [yellow]格式示例: outputs/example_recommendation_templates.md[/yellow]")


# ── 命令 4: demo ─────────────────────────────────────────────────────────


@app.command()
def demo():
    """一键运行完整流程

    环境变量或 Streamlit secrets 中有 DeepSeek Key 时运行 ingest → discover → validate → audit → export。
    没有 DeepSeek Key 时，如已有 data/literature_database.csv，则使用 cached evidence mode
    重新生成审计、证据门禁、可复现声明和 competition_package。
    """
    _print_banner()
    has_deepseek_key = _check_deepseek_key(required=False)
    cached_mode = not has_deepseek_key
    if cached_mode and not (BASE_DIR / "data" / "literature_database.csv").exists():
        console.print("[red]错误: 未配置 DEEPSEEK_API_KEY，且 data/literature_database.csv 不存在，无法运行 demo。[/red]")
        raise typer.Exit(1)

    console.print("\n[bold][RUN] 运行完整 Demo 流程[/bold]\n")
    if cached_mode:
        console.print("Cached evidence mode: 跳过需要实时 DeepSeek 的 ingest/discover/validate，使用现有 data/ 与 outputs/ 重新生成审计、证据门禁和提交包。", markup=False)

    if not cached_mode:
        # Phase 1: Ingest
        console.print("\n[bold cyan]Phase 1/4: Ingest — 构建文献库[/bold cyan]")
        console.print("═" * 40, markup=False)
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("处理 PDF 中...", total=None)
            ingest_stats = run_ingest()
            progress.update(task, completed=True)
        console.print(f"  -> 文献库: {_count_literature()} 篇\n")

        # Phase 2: Discover
        console.print("[bold cyan]Phase 2/4: Discover — 发现研究空白[/bold cyan]")
        console.print("═" * 40, markup=False)
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("构建覆盖矩阵...", total=None)
            discover_stats = run_discover()
            progress.update(task, completed=True)
        console.print(f"  -> 候选空白: {discover_stats.get('candidate_gaps', 0)} 个\n")
        write_debug_scope_report()
        write_debug_extraction_quality()

        # Phase 3: Validate
        console.print("[bold cyan]Phase 3/4: Validate — 验证假设[/bold cyan]")
        console.print("═" * 40, markup=False)
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("运行质量门禁...", total=None)
            validate_stats = run_validate()
            progress.update(task, completed=True)
        if validate_stats.get("has_evidence_recommendations"):
            console.print(f"  -> 科学假设: {validate_stats.get('recommendation_count', 0)} 个")
        else:
            console.print("  -> 科学假设: 无（证据不足，暂不生成）", markup=False)
        console.print("")
    else:
        console.print("\n[bold cyan]Phase 1–3: Cached Evidence — 使用现有文献库与结构化结果[/bold cyan]")
        console.print("═" * 40, markup=False)
        console.print(f"  -> 缓存文献库: {_count_literature()} 篇", markup=False)
        write_debug_scope_report()
        write_debug_extraction_quality()

    # Phase 3.5: Quality modules
    console.print("[bold cyan]Phase 3.5/4: Quality — 质量验证模块[/bold cyan]")
    console.print("═" * 40, markup=False)
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("运行证据片段追溯...", total=None)
        from src.evidence_trace import run_evidence_trace
        trace_stats = run_evidence_trace()
        progress.update(task, completed=True)
    console.print(f"  -> 09_evidence_trace_report.md | 证据片段: {trace_stats.get('total_snippets', 0)} 条\n")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("运行 Evidence Quality Gate...", total=None)
        eg_stats = run_evidence_gate()
        progress.update(task, completed=True)
    console.print(f"  -> 12_evidence_quality_gate.md | 通过: {eg_stats.get('passed_snippets', 0)} | 未通过: {eg_stats.get('failed_snippets', 0)} | 证据等级: {eg_stats.get('evidence_level', '?')}\n")

    # If cached mode, regenerate the scientific plan after evidence level is known.
    if cached_mode:
        try:
            from src.validator import _write_scientific_hypothesis_plan, _run_baseline_comparison
            _write_scientific_hypothesis_plan([])
            _run_baseline_comparison()
        except Exception as e:
            console.print(f"  -> cached mode plan/baseline refresh failed: {e}", markup=False)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("运行历史回溯验证...", total=None)
        from src.retrospective_validation import run_retrospective_validation
        retro_stats = run_retrospective_validation()
        progress.update(task, completed=True)
    console.print(f"  -> 07_retrospective_validation.md | 早期文献: {retro_stats.get('early_count', 0)} 篇 | 后续文献: {retro_stats.get('followup_count', 0)} 篇\n")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("运行消融实验...", total=None)
        from src.ablation_study import run_ablation_study
        ablation_stats = run_ablation_study()
        progress.update(task, completed=True)
    console.print(f"  -> 08_ablation_study.md | 完整系统总分: {ablation_stats.get('full_system_score', '?')}\n")

    # Phase 3.75: Award readiness, DeepSeek usage, Reproducibility
    console.print("[bold cyan]Phase 3.75/4: Scorecard & Audit — 完成度评分与审计[/bold cyan]")
    console.print("═" * 40, markup=False)
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("生成 DeepSeek Usage Report...", total=None)
        qu_stats = run_deepseek_usage_report()
        progress.update(task, completed=True)
    console.print(f"  -> 11_deepseek_usage_report.md | 调用记录: {qu_stats.get('total_calls', 0)} 条 (data/deepseek_call_log.csv)\n")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("生成 Reproducibility Manifest...", total=None)
        rp_stats = run_reproducibility()
        progress.update(task, completed=True)
    console.print("  -> 13_reproducibility_manifest.md | 运行命令: competition_package/run_command.txt\n")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("生成 Award Readiness Scorecard...", total=None)
        sc_stats = run_scorecard()
        progress.update(task, completed=True)
    console.print(f"  -> 10_award_readiness_scorecard.md | Overall: {sc_stats.get('overall', '?')} | Pass: {sc_stats.get('pass_count', 0)}/6\n")

    # Phase 4: Export
    console.print("[bold cyan]Phase 4/4: Export — 导出比赛包[/bold cyan]")
    console.print("═" * 40, markup=False)
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("导出比赛包...", total=None)
        export_stats = run_export()
        progress.update(task, completed=True)

    console.print("\n" + "═" * 50, markup=False)
    console.print("[bold green][DONE] Demo 运行完成![/bold green]")
    console.print("═" * 50, markup=False)

    table = Table(title="核心输出")
    table.add_column("文件", style="cyan")
    table.add_column("说明", style="white")
    table.add_row("outputs/01_evidence_map.md", "文献证据地图")
    table.add_row("outputs/02_gap_diagnosis.md", "研究空白诊断")
    table.add_row("outputs/03_hypothesis_summary.md", "科学假设摘要")
    table.add_row("outputs/04_baseline_comparison.md", "基线对比报告")
    table.add_row("outputs/05_scientific_hypothesis_plan.md", "科学假设与研究计划")
    table.add_row("outputs/06_competition_readiness.md", "题目完成度自检")
    table.add_row("outputs/07_retrospective_validation.md", "历史回溯验证")
    table.add_row("outputs/08_ablation_study.md", "消融实验")
    table.add_row("outputs/09_evidence_trace_report.md", "证据片段追溯")
    table.add_row("outputs/10_award_readiness_scorecard.md", "奖项完成度评分卡")
    table.add_row("outputs/11_deepseek_usage_report.md", "DeepSeek 使用报告")
    table.add_row("outputs/12_evidence_quality_gate.md", "证据质量门禁")
    table.add_row("outputs/13_reproducibility_manifest.md", "可复现性声明")
    table.add_row("outputs/final_demo_report.md", "系统运行报告")
    console.print(table)
    console.print("数据文件: data/evidence_snippets.csv, data/ablation_results.csv, data/retrospective_validation_pairs.csv, data/minimum_validation_dataset_schema.csv", markup=False)
    console.print(f"\n[PKG] 比赛包: [green]{escape(export_stats.get('competition_package', ''))}[/green]")
    for f in export_stats.get("files", []):
        try:
            console.print(f"    ├ {Path(f).relative_to(BASE_DIR)}")
        except Exception:
            console.print(f"    ├ {f}", markup=False)
    console.print("\n[bold]运行方式:[/bold] python app.py demo")
    console.print("[bold]分步运行:[/bold] python app.py ingest / discover / validate")


# ── 命令 5: collect-papers ────────────────────────────────────────────────


@app.command(name="collect-papers")
def collect_papers():
    """自动下载开放获取（Open Access）文献

    使用 OpenAlex API 按标题搜索钛合金疲劳目标文献，
    下载公开 PDF 保存到 early_papers/ papers/ followup_papers/。
    不下载付费墙文献，不绕过版权限制。
    """
    _print_banner()

    console.print("\n[bold]收集开放获取文献 (collect-papers)[/bold]\n")
    console.print("搜索 OpenAlex → 定位 OA PDF → 下载 → 生成报告\n", markup=False)

    try:
        from src.paper_collector import run_collect_papers
    except ImportError as e:
        console.print(f"[red]模块导入失败: {escape(str(e))}[/red]")
        console.print("[yellow]请确认 src/paper_collector.py 存在且无语法错误[/yellow]")
        raise typer.Exit(1)

    console.print("[yellow]⚠️  本工具仅下载开放获取（Open Access）PDF。[/yellow]")
    console.print("[yellow]  不绕过付费墙，不从非法镜像下载。[/yellow]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("搜索并下载文献...", total=None)
        stats = run_collect_papers()
        progress.update(task, completed=True)

    console.print(f"\n[green]✓ collect-papers 完成[/green]")
    console.print(f"  目标文献: {stats.get('total_target', 0)} 篇")
    console.print(f"  [green]成功下载: {stats.get('downloaded', 0)} 篇[/green]")
    console.print(f"  [yellow]已存在跳过: {stats.get('skipped', 0)} 篇[/yellow]")
    console.print(f"  [red]未找到 OA PDF: {stats.get('missing', 0)} 篇[/red]")
    console.print(f"  [red]下载失败: {stats.get('failed', 0)} 篇[/red]")
    console.print(f"\n  报告: [cyan]{escape(stats.get('report_path', ''))}[/cyan]")
    console.print(f"  已下载清单: [cyan]{escape(stats.get('downloaded_csv', ''))}[/cyan]")
    console.print(f"  缺失清单: [cyan]{escape(stats.get('missing_csv', ''))}[/cyan]")

    if stats.get("downloaded", 0) > 0:
        console.print(f"\n[bold]下一步:[/bold] python app.py ingest   # 将下载的 PDF 导入文献库")
    if stats.get("missing", 0) > 0:
        console.print(f"\n[bold]缺失文献:[/bold] 请查看 {stats.get('missing_csv', '')}")
        console.print("  可通过 institutional access 或联系作者手动获取。")


# ── 命令 6: auto ────────────────────────────────────────────────────────────


@app.command()
def auto():
    """自动采集开放获取文献并运行完整 Pipeline

    1. 按 10 个关键词从 OpenAlex 搜索钛合金疲劳 OA 文献
    2. 下载开放获取 PDF（不绕过付费墙）
    3. 自动去重（DOI / arXiv ID / 标题）
    4. 保存到 papers/
    5. 自动运行 ingest → discover → validate → demo
    """
    _print_banner()

    console.print("\n[bold][AUTO] 自动开放文献采集 + 完整 Pipeline[/bold]\n")

    # ── Phase A: Auto-collect ──
    console.print("[bold cyan]Phase A: 自动采集开放文献[/bold cyan]")
    console.print("[dim]" + "═" * 40 + "[/dim]")

    try:
        from src.auto_collector import run_auto_collect
    except ImportError as e:
        console.print(f"[red]模块导入失败: {escape(str(e))}[/red]")
        console.print("[yellow]请确认 src/auto_collector.py 存在且无语法错误[/yellow]")
        raise typer.Exit(1)

    console.print("[yellow]⚠️  本工具仅下载开放获取（Open Access）PDF。[/yellow]")
    console.print("[yellow]  不绕过付费墙，不从非法镜像下载。[/yellow]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("搜索并下载 OA 文献...", total=None)
        collect_stats = run_auto_collect()
        progress.update(task, completed=True)

    console.print(f"\n[green]✓ 采集完成[/green]")
    console.print(f"  搜索关键词: {collect_stats['searched']} 个")
    console.print(f"  发现唯一文献: {collect_stats['total_found']} 篇")
    console.print(f"  [green]成功下载: {collect_stats['downloaded']} 篇[/green]")
    console.print(f"  [yellow]已存在跳过: {collect_stats['skipped_exists']} 篇[/yellow]")
    console.print(f"  [red]未找到 OA PDF: {collect_stats['missing_oa']} 篇[/red]")
    console.print(f"  [red]下载失败: {collect_stats['failed']} 篇[/red]")
    console.print(f"\n  已下载清单: [cyan]{escape(collect_stats.get('downloaded_csv', ''))}[/cyan]")
    console.print(f"  缺失清单: [cyan]{escape(collect_stats.get('missing_csv', ''))}[/cyan]")
    console.print(f"  Pipeline 报告: [cyan]{escape(collect_stats.get('report_path', ''))}[/cyan]")

    if collect_stats.get("failed", 0) > 0:
        console.print("\n[yellow]部分下载失败。继续运行 Pipeline（已有文献仍可处理）...[/yellow]")
    if collect_stats.get("downloaded", 0) == 0:
        console.print("\n[yellow]未下载新文献。检查缺失列表手动获取。[/yellow]")

    # ── Phase B: Pipeline (ingest → discover → validate → demo) ──
    console.print("\n[bold cyan]Phase B: 运行完整 Pipeline[/bold cyan]")
    console.print("[dim]" + "═" * 40 + "[/dim]")

    if not _check_deepseek_key():
        console.print("[red]缺少 API Key，无法运行 Pipeline。[/red]")
        console.print("[yellow]请设置 DEEPSEEK_API_KEY 或 Streamlit secrets，然后手动运行:[/yellow]")
        console.print("  python app.py ingest")
        console.print("  python app.py discover")
        console.print("  python app.py validate")
        console.print("  python app.py demo")
        raise typer.Exit(1)

    # B1: Ingest
    console.print("\n[bold]Step 1/4: Ingest — 构建文献库[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("处理 PDF 中...", total=None)
        ingest_stats = run_ingest()
        progress.update(task, completed=True)
    console.print(f"  -> 文献库: {_count_literature()} 篇")

    # B2: Discover
    console.print("\n[bold]Step 2/4: Discover — 发现研究空白[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("构建覆盖矩阵...", total=None)
        discover_stats = run_discover()
        progress.update(task, completed=True)
    console.print(f"  -> 候选空白: {discover_stats.get('candidate_gaps', 0)} 个")
    write_debug_scope_report()
    write_debug_extraction_quality()

    # B3: Validate
    console.print("\n[bold]Step 3/4: Validate — 验证假设[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("运行质量门禁...", total=None)
        validate_stats = run_validate()
        progress.update(task, completed=True)
    if validate_stats.get("has_evidence_recommendations"):
        console.print(f"  -> 正式推荐方向: {validate_stats.get('recommendation_count', 0)} 张卡片")
    else:
        console.print("  -> [yellow]正式推荐方向: 无（证据不足）[/yellow]")

    # B3.5: Quality modules
    console.print("\n[bold]Step 3.5/4: Quality — 质量验证模块[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("运行质量验证模块...", total=None)
        from src.retrospective_validation import run_retrospective_validation
        from src.ablation_study import run_ablation_study
        from src.evidence_trace import run_evidence_trace
        run_retrospective_validation()
        run_ablation_study()
        run_evidence_trace()
        progress.update(task, completed=True)
    console.print("  -> 07_retrospective_validation.md, 08_ablation_study.md, 09_evidence_trace_report.md")

    # B4: Export
    console.print("\n[bold]Step 4/4: Export — 导出比赛包[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("导出中...", total=None)
        export_stats = run_export()
        progress.update(task, completed=True)

    # Summary
    console.print("\n" + "═" * 50)
    console.print("[bold green][DONE] Auto Pipeline 运行完成![/bold green]")
    console.print("═" * 50)

    table = Table(title="核心输出")
    table.add_column("文件", style="cyan")
    table.add_column("说明", style="white")
    table.add_row("data/auto_downloaded_papers.csv", "自动下载文献清单")
    table.add_row("data/auto_missing_papers.csv", "缺失 OA PDF 清单")
    table.add_row("outputs/auto_pipeline_report.md", "自动采集 Pipeline 报告")
    table.add_row("outputs/01_evidence_map.md", "文献证据地图")
    table.add_row("outputs/02_gap_diagnosis.md", "研究空白诊断")
    table.add_row("outputs/03_hypothesis_summary.md", "科学假设摘要")
    table.add_row("outputs/03_recommendation_cards.md", "推荐卡片（完整版）")
    table.add_row("outputs/04_baseline_comparison.md", "基线对比报告")
    table.add_row("outputs/05_scientific_hypothesis_plan.md", "科学假设与研究计划")
    table.add_row("outputs/06_competition_readiness.md", "题目完成度自检")
    table.add_row("outputs/07_retrospective_validation.md", "历史回溯验证")
    table.add_row("outputs/08_ablation_study.md", "消融实验")
    table.add_row("outputs/09_evidence_trace_report.md", "证据片段追溯")
    table.add_row("outputs/10_award_readiness_scorecard.md", "奖项完成度评分卡")
    table.add_row("outputs/11_deepseek_usage_report.md", "DeepSeek 使用报告")
    table.add_row("outputs/12_evidence_quality_gate.md", "证据质量门禁")
    table.add_row("outputs/13_reproducibility_manifest.md", "可复现性声明")
    table.add_row("outputs/final_demo_report.md", "系统运行报告")
    console.print(table)

    console.print(f"\n[PKG] 比赛包: [green]{escape(export_stats.get('competition_package', ''))}[/green]")
    console.print("\n[bold]本地已有文献运行:[/bold] python app.py demo")
    console.print("[bold]自动采集并运行:[/bold] python app.py auto")


@app.command("rag-build")
def rag_build(
    paper_ids: str = typer.Option(
        "", "--paper-ids", help="Comma-separated Stage-2-complete paper IDs."
    ),
):
    """Build the Stage-3 unified RAG from trusted Stage-2 artifacts."""
    from src.unified_rag import build_unified_rag

    selected = [value.strip() for value in paper_ids.split(",") if value.strip()]
    if not selected:
        raise typer.BadParameter("--paper-ids is required")
    console.print_json(data=build_unified_rag(selected, base_dir=BASE_DIR))


@app.command("rag-query")
def rag_query(
    question: str = typer.Argument(...),
    top_k: int = typer.Option(10, "--top-k"),
):
    """Run hybrid retrieval without changing the Streamlit UI."""
    from src.unified_rag import answer_research_question

    console.print_json(
        data=answer_research_question(question, top_k=top_k, base_dir=BASE_DIR)
    )


@app.command("literature-task")
def literature_task(
    question: str = typer.Argument(...),
    batch_size: int = typer.Option(3, "--batch-size", min=3, max=5),
):
    """Persist an OA top-up task for the independent worker."""
    from src.literature_tasks import create_literature_task

    console.print_json(
        data=create_literature_task(
            question, batch_size=batch_size, base_dir=BASE_DIR
        )
    )


@app.command("deep-read-all")
def deep_read_all(
    pdf_dir: Path = typer.Option(Path("paper/pdfs"), "--pdf-dir"),
    resume: bool = typer.Option(False, "--resume"),
    retry_failed: bool = typer.Option(False, "--retry-failed"),
    only_unread: bool = typer.Option(False, "--only-unread"),
    include_review: bool = typer.Option(
        False,
        "--include-review",
        help="Re-open NEEDS_HUMAN_REVIEW tasks for one repair/deep-read attempt.",
    ),
    only_incomplete: bool = typer.Option(
        False,
        "--only-incomplete",
        help=(
            "Process only papers without a final A-D classification or without "
            "the requested DeepSeek semantic enhancement."
        ),
    ),
    limit: int = typer.Option(None, "--limit", min=1),
    concurrency: int = typer.Option(1, "--concurrency", min=1, max=2),
    dry_run: bool = typer.Option(False, "--dry-run"),
    use_deepseek: bool = typer.Option(
        False,
        "--use-deepseek",
        help=(
            "Enable real DeepSeek semantic enrichment for structure, evidence "
            "relations, conditions, mechanisms, and formula context."
        ),
    ),
    stop_after_pages: int = typer.Option(None, "--stop-after-pages", min=1, hidden=True),
):
    """Deep-read every unique valid local PDF with durable page checkpoints."""
    from collections import Counter

    from src.api_keys import get_deepseek_settings
    from src.full_library_deep_read import build_full_library_queue, run_full_library_queue

    settings = get_deepseek_settings(project_root=BASE_DIR)
    built = build_full_library_queue(
        pdf_dir, base_dir=BASE_DIR, resume=resume, dry_run=dry_run
    )
    inventory = built["inventory"]
    states = Counter(str(task.get("status") or "PENDING") for task in built.get("tasks", []))
    completed = states["COMPLETED"]
    pending_unique = sum(
        str(task.get("status") or "PENDING") == "PENDING"
        or (
            retry_failed
            and str(task.get("status") or "") in {"FAILED_RETRYABLE", "PAUSED"}
        )
        or (
            include_review
            and str(task.get("status") or "") == "NEEDS_HUMAN_REVIEW"
        )
        or (
            not only_unread
            and use_deepseek
            and str(task.get("status") or "") == "COMPLETED"
            and not bool(task.get("deepseek_enhancement_applied"))
        )
        for task in built.get("tasks", [])
    )
    console.print_json(data={
        "deepseek_key": "FOUND" if settings.configured else "MISSING",
        "deepseek_config_source": settings.source,
        "deepseek_enhancement_enabled": use_deepseek,
        "pending_unique_papers": pending_unique,
        "completed_papers": completed,
        "duplicate_files_skipped": inventory["exact_duplicate_count"],
    })
    if use_deepseek and not settings.configured:
        console.print("[red]--use-deepseek requires a detected DEEPSEEK_API_KEY.[/red]")
        raise typer.Exit(code=2)

    if dry_run:
        console.print_json(data={
            "dry_run": True,
            "pdf_file_count": inventory["pdf_file_count"],
            "logical_document_count": inventory["logical_document_count"],
            "exact_duplicate_count": inventory["exact_duplicate_count"],
            "different_version_count": inventory["different_version_count"],
            "total_pages": inventory["total_pages"],
        })
        return
    result = run_full_library_queue(
        pdf_dir, base_dir=BASE_DIR, resume=resume,
        retry_failed=retry_failed, only_unread=only_unread,
        include_review=include_review, only_incomplete=only_incomplete, limit=limit,
        concurrency=concurrency, stop_after_pages=stop_after_pages,
        use_deepseek=use_deepseek,
    )
    report = result["report"]
    console.print_json(data={
        "processed_tasks": result["processed_tasks"],
        "scanned_pdf_count": report["scanned_pdf_count"],
        "unique_logical_document_count": report["unique_logical_document_count"],
        "completed_paper_count": report["completed_paper_count"],
        "page_coverage_ratio": report["page_coverage_ratio"],
        "evidence_record_count": report["evidence_record_count"],
        "formal_paper_count": report["formal_paper_count"],
        "indexed_paper_count": report["indexed_paper_count"],
        "failed_paper_count": report["failed_paper_count"],
        "needs_human_review_count": report["needs_human_review_count"],
        "deepseek_enhancement_enabled": result["deepseek_enabled"],
        "deepseek_api_call_count": result["deepseek_usage"]["api_call_count"],
        "deepseek_success_count": result["deepseek_usage"]["success_count"],
        "deepseek_failure_count": result["deepseek_usage"]["failure_count"],
        "deepseek_retry_count": result["deepseek_usage"]["retry_count"],
        "deepseek_prompt_tokens": result["deepseek_usage"]["prompt_tokens"],
        "deepseek_completion_tokens": result["deepseek_usage"]["completion_tokens"],
        "deepseek_total_tokens": result["deepseek_usage"]["total_tokens"],
        "terminal_state_counts": result.get("terminal_counts") or {},
        "report": str((BASE_DIR / "outputs" / "full_library_deep_read_report.json").resolve()),
    })


@app.command("config-check")
def config_check():
    """Diagnose DeepSeek configuration without printing any credential text."""
    from src.api_keys import get_deepseek_settings
    from src.deepseek_client import DeepSeekClient

    settings = get_deepseek_settings(project_root=BASE_DIR)
    ready = False
    if settings.configured:
        try:
            DeepSeekClient(settings)
            ready = True
        except Exception:
            ready = False
    console.print(
        f"DEEPSEEK_API_KEY: {'FOUND' if settings.configured else 'MISSING'}"
    )
    console.print(f"source: {settings.source}")
    console.print(f"deepseek_client_ready: {str(ready).lower()}")


# ── Health check ─────────────────────────────────────────────────────────


@app.command()
def check():
    """检查环境和依赖。"""
    _print_banner()

    console.print("\n[bold]环境检查[/bold]\n")

    # Python
    console.print(f"  Python: {sys.version}")

    # DeepSeek API key
    from src.api_keys import get_deepseek_settings

    settings = get_deepseek_settings(project_root=BASE_DIR)
    detected = "已检测" if settings.configured else "未检测"
    console.print(f"  DeepSeek配置: {detected}")
    console.print(f"  配置来源: {settings.source}")

    # Dependencies
    deps = {"typer": "typer", "rich": "rich", "fitz": "PyMuPDF", "pandas": "pandas", "requests": "requests"}
    all_ok = True
    for mod_name, pip_name in deps.items():
        try:
            __import__(mod_name)
            console.print(f"  [OK] {pip_name}: [green]已安装[/green]")
        except ImportError:
            console.print(f"  [FAIL] {pip_name}: [red]未安装[/red] (pip install {pip_name})")
            all_ok = False

    # Paper dirs
    for d in ["paper/pdfs", "papers", "early_papers", "followup_papers"]:
        d_path = BASE_DIR / d
        pdf_count = len(list(d_path.rglob("*.pdf"))) if d_path.exists() else 0
        status = f"{pdf_count} 个 PDF" if pdf_count > 0 else "空"
        console.print(f"  [DIR] {d}/: {status}")

    # Config
    if (BASE_DIR / "config" / "task_profile.yaml").exists():
        console.print("  [OK] config/task_profile.yaml: [green]存在[/green]")
    if (BASE_DIR / "docs" / "teacher_constraints.md").exists():
        console.print("  [OK] docs/teacher_constraints.md: [green]存在[/green]")

    if all_ok:
        console.print("\n[green][OK] 环境检查通过！可以运行 python app.py demo[/green]")
    else:
        console.print("\n[yellow][WARN] 部分依赖缺失，请运行 pip install -r requirements.txt[/yellow]")


if __name__ == "__main__":
    app()

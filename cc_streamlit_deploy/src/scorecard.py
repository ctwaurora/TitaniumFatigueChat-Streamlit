"""Award readiness scorecard with non-contradictory status logic."""
from __future__ import annotations
from pathlib import Path
from src.stage1_store import TRUSTED_EVIDENCE_PATH
from typing import Dict, Any, List
import csv, re
BASE_DIR=Path(__file__).resolve().parent.parent
OUTPUTS_DIR=BASE_DIR/'outputs'; DATA_DIR=BASE_DIR/'data'; COMP_DIR=BASE_DIR/'competition_package'
from skills.library_skill import get_all_papers

def _exists(p:Path)->bool: return p.exists() and p.stat().st_size>0

def _contains(path:Path, kws:List[str])->bool:
    if not _exists(path): return False
    t=path.read_text(encoding='utf-8', errors='ignore').lower()
    return all(k.lower() in t for k in kws)

def _csv_count(path:Path)->int:
    if not _exists(path): return 0
    try:
        with path.open('r',encoding='utf-8-sig') as f: return max(sum(1 for _ in f)-1,0)
    except Exception: return 0

def _qwen_success_count(path:Path)->int:
    if not _exists(path): return 0
    try:
        with path.open('r',encoding='utf-8-sig') as f:
            rows=list(csv.DictReader(f))
        return sum(1 for r in rows if str(r.get('success_or_fail','')).lower().strip()=='success')
    except Exception:
        return 0

def _status(checks:Dict[str,bool], required:List[str]=None)->str:
    vals=list(checks.values())
    if vals and all(vals): return 'pass'
    if required and any(not checks.get(r,False) for r in required):
        return 'partial' if any(vals) else 'fail'
    return 'partial' if any(vals) else 'fail'

def run_scorecard()->Dict[str,Any]:
    n=len(get_all_papers())
    files={
        'hypothesis_plan':OUTPUTS_DIR/'05_scientific_hypothesis_plan.md',
        'hypothesis_summary':OUTPUTS_DIR/'03_hypothesis_summary.md',
        'gap':OUTPUTS_DIR/'02_gap_diagnosis.md',
        'baseline':OUTPUTS_DIR/'04_baseline_comparison.md',
        'retro':OUTPUTS_DIR/'07_retrospective_validation.md',
        'ablation':OUTPUTS_DIR/'08_ablation_study.md',
        'trace':OUTPUTS_DIR/'09_evidence_trace_report.md',
    'snippets':TRUSTED_EVIDENCE_PATH,
        'ablation_csv':DATA_DIR/'ablation_results.csv',
        'retro_csv':DATA_DIR/'retrospective_validation_pairs.csv',
        'min_schema':DATA_DIR/'minimum_validation_dataset_schema.csv',
        'litdb':DATA_DIR/'literature_database.csv',
        'run_command':COMP_DIR/'run_command.txt',
    }
    dims=[]
    c1={
        'predictive_hypothesis':_contains(files['hypothesis_plan'], ['控制表面粗糙度','应力比 r','较低的疲劳寿命']),
        'controlled_variables':_contains(files['hypothesis_summary'], ['controlled_variables']) or _contains(files['hypothesis_plan'], ['控制表面粗糙度']),
        'expected_trend':_contains(files['hypothesis_plan'], ['较高的早期 da/dn','较低的疲劳寿命']) or _contains(files['hypothesis_plan'], ['更高的早期 da/dn']),
        'falsification_logic':_contains(files['hypothesis_plan'], ['若该趋势','不成立','主导']),
    }
    dims.append(_dim('Scientific Hypothesis Quality',c1,'03_hypothesis_summary.md, 05_scientific_hypothesis_plan.md','需要继续补强原始数据以验证预测趋势','补充可提取的 FCGR / Paris 参数数据，验证预测趋势。'))
    c2={
        'paper_id':_csv_count(files['snippets'])>0,
        'evidence_id':_csv_count(files['snippets'])>0,
        'evidence_snippet':_contains(files['trace'], ['top evidence snippets']) and _csv_count(files['snippets'])>0,
        'references_traceable':_contains(files['hypothesis_plan'], ['paper_id','evidence_ids']) or _contains(files['trace'], ['evidence id']),
        # Strict: evidence grounding is not full Pass unless the quality gate itself reaches candidate level.
        'evidence_supported_level_reached': _contains(OUTPUTS_DIR/'12_evidence_quality_gate.md', ['evidence_supported_candidate']),
    }
    dims.append(_dim('Evidence Grounding',c2,'evidence_snippets.csv, 09_evidence_trace_report.md','仍为文本级证据，未解析图表曲线','继续清洗 evidence snippets，提升核心证据通过率，并避免背景句、截断句进入核心证据。'))
    c3={
        'missing_evidence':_contains(files['gap'], ['缺失']) or _contains(files['hypothesis_plan'], ['仍缺失']),
        'well_partial_missing':_contains(files['hypothesis_plan'], ['已充分研究','部分研究但未闭合','仍缺失']),
        'pseudo_gap_avoidance':_contains(files['hypothesis_plan'], ['不是伪空白']) or _contains(files['gap'], ['伪空白']),
    }
    dims.append(_dim('Missing Evidence Diagnosis',c3,'02_gap_diagnosis.md, 05_scientific_hypothesis_plan.md','缺失证据仍需更细粒度分层','将 missing evidence 细分为数据缺失、实验缺失、机制缺失和模型缺失。'))
    c4={
        'minimum_cost_validation':_contains(files['hypothesis_plan'], ['minimum-cost validation']) or _contains(files['hypothesis_plan'], ['最低成本']),
        'full_validation':_contains(files['hypothesis_plan'], ['full validation']) or _contains(files['hypothesis_plan'], ['完整验证']),
        'target_dataset':_contains(files['hypothesis_plan'], ['target dataset']),
        'expected_results':_contains(files['hypothesis_plan'], ['expected results']),
        'falsification_conditions':_contains(files['hypothesis_plan'], ['falsification conditions']) or _contains(files['hypothesis_plan'], ['推翻']),
        # Planning is not equivalent to executed validation. Keep this dimension Partial until real data/experiment/simulation is executed.
        'validation_executed': _contains(files['hypothesis_plan'], ['executed validation data attached']) or _contains(files['hypothesis_plan'], ['实测数据已验证']),
    }
    dims.append(_dim('Validation Design',c4,'05_scientific_hypothesis_plan.md','验证路径仍是设计层面，尚未执行真实实验/仿真','执行 minimum validation dataset 的文献数据复现，至少完成孔隙特征—Nf / da/dN / Paris 参数的结构化表。'))
    qwen_calls = _csv_count(DATA_DIR / 'qwen_call_log.csv')
    qwen_success = _qwen_success_count(DATA_DIR / 'qwen_call_log.csv')
    c5={
        'baseline':_exists(files['baseline']),
        'ablation':_exists(files['ablation']),
        'retrospective':_exists(files['retro']),
        'vs_direct_qwen':_contains(files['baseline'], ['direct qwen']) or _contains(files['baseline'], ['直接 qwen']),
        # Strict: attempted-but-failed calls do not prove live Qwen success.
        'qwen_success_recorded': qwen_success > 0,
        # Current ablation is still approximate unless explicitly marked as actual rerun.
        'actual_ablation_rerun': _contains(files['ablation'], ['actual rerun confirmed']) or _contains(files['ablation'], ['真实重跑']),
    }
    dims.append(_dim('System Effectiveness',c5,'04_baseline_comparison.md, 08_ablation_study.md, 07_retrospective_validation.md','当前 baseline 和 approximate ablation 初步表明完整系统优于直接 Qwen，但 actual_ablation_rerun 仍为 ❌，因此尚不能视为强验证；后续需实际重跑各版本以增强结论可信度。','实际重跑 direct Qwen、summary Qwen、full system 三种版本，并保存真实调用日志和评分结果。'))
    c6={
        'run_command':_exists(files['run_command']),
        'literature_database':_exists(files['litdb']),
        'evidence_snippets':_exists(files['snippets']),
        'ablation_results':_exists(files['ablation_csv']),
        'retrospective_pairs':_exists(files['retro_csv']),
        'minimum_schema':_exists(files['min_schema']),
    }
    dims.append(_dim('Reproducibility',c6,'competition_package/run_command.txt, data/*.csv','Qwen API key 需用户本地配置；网络调用可能有差异','固化环境检查、运行日志和 data_preview，确保他人可以按 run_command.txt 复现。'))
    pass_count=sum(1 for d in dims if d['status']=='pass'); partial_count=sum(1 for d in dims if d['status']=='partial')
    # Strict ceiling rules:
    # - fewer than 30 papers: cannot exceed small-case validation;
    # - empty qwen_call_log: cannot exceed small-case validation because current run did not prove live Qwen calls;
    if pass_count == 6 and n >= 30 and qwen_success > 0:
        overall = 'evidence-supported case'
    elif pass_count >= 5 and n >= 30 and qwen_success > 0:
        overall = 'approaching evidence-supported'
    elif pass_count >= 3:
        overall = 'small-case validation'
    else:
        overall = 'prototype validation'
    _write(dims,overall,n)
    return {'overall':overall,'pass_count':pass_count,'partial_count':partial_count,'fail_count':6-pass_count-partial_count}

def _dim(name,checks,evidence,limitation,next_improvement=''):
    st=_status(checks)
    return {'name':name,'status':st,'checks':checks,'evidence_file':evidence,'current_limitation':limitation,'next_improvement':next_improvement}

def _write(dims,overall,n):
    icon={'pass':'✅ Pass','partial':'⚠️ Partial','fail':'❌ Fail'}
    lines=['# Award Readiness Scorecard（奖项完成度评分卡）','', '> 本评分卡按 AI Scientist 榜题要求自检，不自吹满分；若证据不足必须降级。', f'> **文献库规模**: {n} 篇', '', f'## Overall Judgment: **{overall}**', '']
    for i,d in enumerate(dims,1):
        lines += [f"### {i}. {d['name']} — {icon[d['status']]}", '', f"**Evidence file**: {d['evidence_file']}", '', '| Check | Status |', '|---|---|']
        for k,v in d['checks'].items(): lines.append(f"| {k} | {'✅' if v else '❌'} |")
        lines += ['', f"**Current limitation**: {d['current_limitation']}", f"**Next improvement**: {d['next_improvement']}", '']
    lines += ['## Dimension Summary','', '| Dimension | Status |','|---|---|']
    for d in dims: lines.append(f"| {d['name']} | {icon[d['status']]} |")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR/'10_award_readiness_scorecard.md').write_text('\n'.join(lines),encoding='utf-8')

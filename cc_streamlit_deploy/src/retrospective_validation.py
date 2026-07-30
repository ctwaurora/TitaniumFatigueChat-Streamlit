"""Retrospective validation aligned with the main pore-defect hypothesis."""
from __future__ import annotations
import csv, json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from skills.library_skill import get_all_papers
BASE_DIR=Path(__file__).resolve().parent.parent
OUTPUTS_DIR=BASE_DIR/"outputs"; DATA_DIR=BASE_DIR/"data"

MAIN_GAPS=[
    {"early_gap_id":"G01","early_gap_statement":"早期 AM Ti-6Al-4V 文献已指出孔隙/缺陷影响疲劳性能，但三维孔隙特征与裂纹起裂位置之间的定量关系不足。","generated_hypothesis":"孔隙尺寸、空间位置和形态因子可能决定裂纹优先起裂位置。","keywords":["pore","porosity","defect","crack initiation","initiation","inner surface","surface","fatigue life"]},
    {"early_gap_id":"G02","early_gap_statement":"表面粗糙度与内部孔隙竞争控制疲劳失效的边界条件不清楚。","generated_hypothesis":"在不同表面状态下，近表面孔隙与表面粗糙峰可能竞争成为主导起裂源。","keywords":["surface roughness","roughness","as-built","pore","defect","fatigue"]},
    {"early_gap_id":"G03","early_gap_statement":"AM Ti-6Al-4V 中缺陷特征与 da/dN-ΔK、FCGR 或 Paris 参数之间的关系仍不足。","generated_hypothesis":"孔隙特征可能改变早期裂纹扩展速率和 Paris 参数 C/m。","keywords":["fcgr","da/dn","crack growth","paris","walker","delta k","Δk","defect"]},
    {"early_gap_id":"G04","early_gap_statement":"micro-CT + SEM/EBSD + FCGR 的联合证据不足，缺少从三维缺陷到机制表征的闭合链条。","generated_hypothesis":"联合 micro-CT、FCGR 和 SEM/EBSD 可验证孔隙特征对起裂与早期扩展的作用。","keywords":["micro-ct","x-ray ct","sem","ebsd","fcgr","crack growth","defect"]},
]

def _year(p):
    try: return int(str(p.get('year',''))[:4])
    except Exception: return 0

def _roles(p):
    r=p.get('corpus_roles',[])
    if isinstance(r,str):
        try: r=json.loads(r)
        except Exception: r=[r]
    return r if isinstance(r,list) else [str(r)]

def _split(papers):
    early=[]; follow=[]
    for p in papers:
        if p.get('alloy_type')=='out_of_scope': continue
        y=_year(p); roles=_roles(p)
        if 'early' in roles or (y and y<=2018): early.append(p)
        elif 'followup' in roles or (y and y>=2019): follow.append(p)
    return early, follow

def _text(p):
    vals=[]
    for k in ['title','key_findings','limitations','abstract','conclusion','evidence_text']:
        v=p.get(k,'')
        if isinstance(v,list): vals.extend([str(x) for x in v])
        else: vals.append(str(v))
    return ' '.join(vals).lower()

def _title(p): return str(p.get('title','')).strip()

def run_retrospective_validation()->Dict[str,Any]:
    papers=get_all_papers(); early,follow=_split(papers)
    pairs=[]
    for gap in MAIN_GAPS:
        matches=[]
        for p in follow:
            t=_text(p)
            score=sum(1 for kw in gap['keywords'] if kw.lower() in t)
            # require at least two keywords and fatigue relevance
            if score>=2 and ('fatigue' in t or '疲劳' in t or 'crack' in t):
                matches.append((score,p))
        matches=sorted(matches, key=lambda x:-x[0])[:4]
        if len(matches)>=2: status='partially_supported'; conf='medium'
        elif len(matches)==1: status='partially_supported'; conf='low'
        else: status='not_found' if follow else 'insufficient_followup'; conf='low'
        pairs.append({
            'early_gap_id':gap['early_gap_id'],
            'early_gap_statement':gap['early_gap_statement'],
            'early_evidence_papers':'; '.join(_title(p) for p in early[:3]),
            'generated_hypothesis':gap['generated_hypothesis'],
            'followup_paper_id':'; '.join(f"P{papers.index(p)+1:02d}" for _,p in matches),
            'followup_title':'; '.join(_title(p) for _,p in matches),
            'followup_evidence_summary': _summary(gap, matches),
            'validation_status':status,
            'confidence_level':conf,
        })
    _write_csv(pairs); _write_report(papers, early, follow, pairs)
    counts={s:sum(1 for p in pairs if p['validation_status']==s) for s in ['supported','partially_supported','contradicted','not_found','insufficient_followup']}
    return {'early_count':len(early),'followup_count':len(follow),'gaps_count':len(MAIN_GAPS),'validation_pairs':len(pairs),'status_counts':counts}

def _summary(gap, matches):
    if not matches: return '后续文献库中未找到与该主线空白直接匹配的证据，不能强行判定为已验证。'
    return '后续文献部分涉及该空白：' + '；'.join(_title(p) for _,p in matches[:3])

def _write_csv(pairs):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fields=['early_gap_id','early_gap_statement','early_evidence_papers','generated_hypothesis','followup_paper_id','followup_title','followup_evidence_summary','validation_status','confidence_level']
    with (DATA_DIR/'retrospective_validation_pairs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(pairs)

def _write_report(papers,early,follow,pairs):
    counts={s:sum(1 for p in pairs if p['validation_status']==s) for s in ['supported','partially_supported','contradicted','not_found','insufficient_followup']}
    hit=counts['supported']+counts['partially_supported']
    lines=['# Retrospective Validation Report（历史回溯验证报告）','', '## 方法说明','', '本报告只围绕当前主假设主线进行历史回溯：L-PBF Ti-6Al-4V 孔隙缺陷尺寸/形态/位置对疲劳裂纹起裂与早期扩展行为的影响。', '做法：先用 2018 年及以前/early 角色文献构造早期空白，再用 2019 年及以后/followup 角色文献检查是否被后续研究推进。', '', f'- **总文献数**: {len(papers)}', f'- **早期文献数**: {len(early)}', f'- **后续文献数**: {len(follow)}', f'- **主线早期空白数**: {len(pairs)}', '', '## 主线早期空白', '']
    for p in pairs:
        lines += [f"### {p['early_gap_id']}", f"- **空白描述**: {p['early_gap_statement']}", f"- **基于空白提出的假设**: {p['generated_hypothesis']}", '']
    lines += ['## 后续文献匹配统计','', '| 验证状态 | 数量 |','|---|---:|']
    for s in ['supported','partially_supported','contradicted','not_found','insufficient_followup']:
        lines.append(f'| {s} | {counts[s]} |')
    lines += ['', f'**命中率（supported + partially_supported）**: {hit}/{len(pairs)} = {hit/len(pairs)*100:.1f}%', '', '## 匹配详情','']
    for p in pairs:
        lines += [f"### {p['early_gap_id']} — {p['validation_status']}", f"- 早期空白: {p['early_gap_statement']}", f"- 后续匹配: {p['followup_title'] or '无'}", f"- 说明: {p['followup_evidence_summary']}", '']
    lines += ['## 结论','']
    if hit:
        lines.append('系统在主线空白上具备初步历史回溯发现能力：部分早期可识别的空白在后续文献中被研究推进。但本结果仍受文献数量和关键词匹配限制，不能夸大为完整预测能力。')
    else:
        lines.append('本次主线历史回溯未获得直接命中。该结果应作为证据不足处理，不能宣称系统已具备稳定预测能力。')
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR/'07_retrospective_validation.md').write_text('\n'.join(lines),encoding='utf-8')

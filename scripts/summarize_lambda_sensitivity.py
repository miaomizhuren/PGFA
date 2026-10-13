#!/usr/bin/env python3
"""Build a provenance-preserving lambda sensitivity summary from real metrics."""
from __future__ import annotations
import csv, sqlite3
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'experiments/summaries/lambda_sensitivity'; ROUNDS=tuple(range(100,201,10))
CASES=(('CE-only',0,0,'full run','configs/mainexp/mainexp_02_ce_only_uap_baseline.yaml'),('6/3',6,3,'quick-search','configs/search/search2_q02_e1_lambda_6_3.yaml'),('9/5',9,5,'full run','configs/mainexp/mainexp_04_proto_uap_layer4_proposed.yaml'),('12/7',12,7,'quick-search','configs/search/search2_q03_e1_lambda_12_7.yaml'))
FIELDS=['setting','lambda_delta','lambda_train','source_type','config_path',*[f'CleanACC@{r}' for r in ROUNDS],*[f'ASR@{r}' for r in ROUNDS],'AUC@100-150','AUC@100-200','Half-life','PSNR','SSIM','L_inf','note']
def metric_data(path):
    with sqlite3.connect(path) as c:
        rows=c.execute('SELECT round_index, clean_accuracy, attack_success_rate, psnr, ssim, linf FROM round_metrics ORDER BY round_index').fetchall()
    return {r[0]:{'clean':r[1],'asr':r[2],'psnr':r[3],'ssim':r[4],'linf':r[5]} for r in rows}
def f(value,digits=2): return '' if value in (None,'') else f'{float(value):.{digits}f}'
def auc(data,end):
    pts=[(r,data[r]['asr']) for r in sorted(data) if 100<=r<=end and data[r]['asr'] is not None]
    if len(pts)<2 or pts[-1][0]<end: return ''
    return f(sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(pts,pts[1:]))/(pts[-1][0]-pts[0][0]))
def main():
    OUT.mkdir(parents=True,exist_ok=True); rows=[]; maps=[]
    for setting,ld,lt,source_type,relative in CASES:
        config_path=ROOT/relative; cfg=yaml.safe_load(config_path.read_text()); db=ROOT/cfg['experiment']['output_dir']/cfg['tracking']['database_name']; log=ROOT/'experiments/logs'/('mainexp' if source_type=='full run' else 'search2')/f"{cfg['experiment']['name']}.log"
        try: data=metric_data(db); last=max(data); error=''
        except (OSError,sqlite3.Error) as exc: data={}; last=0; error=f'missing/unreadable metrics DB: {exc}'
        note=error or ('complete through Round 200' if last>=200 else f'quick-search complete through Round {last}; no values after Round {last}')
        row={'setting':setting,'lambda_delta':ld,'lambda_train':lt,'source_type':source_type,'config_path':relative,'note':note}
        row.update({f'CleanACC@{r}':f(data.get(r,{}).get('clean')) for r in ROUNDS}); row.update({f'ASR@{r}':f(data.get(r,{}).get('asr')) for r in ROUNDS})
        row.update({'AUC@100-150':f(auc(data,150)),'AUC@100-200':f(auc(data,200)),'Half-life':next((r for r in sorted(data) if r>=100 and data[r]['asr'] is not None and data[r]['asr']<50),'>150' if last>=150 and last<200 else ('>200' if last>=200 else '')),'PSNR':f(next((data[r]['psnr'] for r in sorted(data,reverse=True) if data[r]['psnr'] is not None),None)),'SSIM':f(next((data[r]['ssim'] for r in sorted(data,reverse=True) if data[r]['ssim'] is not None),None),4),'L_inf':f(next((data[r]['linf'] for r in sorted(data,reverse=True) if data[r]['linf'] is not None),None),4)})
        rows.append(row); maps.append((setting,relative,db.relative_to(ROOT),log.relative_to(ROOT),source_type,last,note))
    with (OUT/'lambda_sensitivity_summary.csv').open('w',newline='',encoding='utf-8') as h: w=csv.DictWriter(h,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    maps_text=['# Lambda sensitivity source map','', '| Setting | Config | SQLite DB | Log | Source | Rounds | Coverage / note |','|---|---|---|---|---|---:|---|']
    maps_text += [f'| {a} | `{b}` | `{c}` | `{d}` | {e} | {g} | {h} |' for a,b,c,d,e,g,h in maps]
    maps_text += ['', 'Medium (6/3) and Very strong (12/7) have valid local_epochs=1 quick-search results through Round 150, but lack Round 160–200 coverage. No supplemental configuration is required for the requested 100–150 trend analysis; a 200-round follow-up remains advisable before making long-horizon comparisons.']
    (OUT/'lambda_sensitivity_source_map.md').write_text('\n'.join(maps_text)+'\n',encoding='utf-8')
    by={r['setting']:r for r in rows}
    md=['# 原型损失权重敏感性真实结果','', '## 表 A：数据来源表','', '| setting | λ_delta | λ_train | source_type | config_path | rounds | note |','|---|---:|---:|---|---|---:|---|']
    md += [f"| {r['setting']} | {r['lambda_delta']} | {r['lambda_train']} | {r['source_type']} | `{r['config_path']}` | {200 if r['source_type']=='full run' else 150} | {r['note']} |" for r in rows]
    md += ['', '## 表 B：ASR 曲线表','', '| setting | '+' | '.join(f'ASR@{r}' for r in ROUNDS)+' |','|---|'+'---:|'*len(ROUNDS)]
    md += [f"| {r['setting']} | "+' | '.join(r[f'ASR@{x}'] or '—' for x in ROUNDS)+' |' for r in rows]
    md += ['', '## 表 C：综合指标表','', '| setting | CleanACC@150 | ASR@150 | AUC@100-150 | AUC@100-200 | Half-life | PSNR | SSIM | L_inf |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    md += [f"| {r['setting']} | {r['CleanACC@150']} | {r['ASR@150']} | {r['AUC@100-150']} | {r['AUC@100-200'] or '—'} | {r['Half-life']} | {r['PSNR']} | {r['SSIM']} | {r['L_inf']} |" for r in rows]
    md += ['', 'Medium 和 Very strong 目前来自 search2 quick-search，因此主要用于敏感性趋势分析；Strong 和 CE-only 来自 mainexp 200-round full run。', '', '## 可直接替换论文 6.2 的文本', '', '### 6.2 原型损失权重敏感性分析', '', f"原型损失权重决定了特征空间约束相对于目标交叉熵损失的强度。本文比较了 CE-only、Medium(λ_delta=6, λ_train=3)、Strong(λ_delta=9, λ_train=5) 和 Very strong(λ_delta=12, λ_train=7) 四种设置。其中 CE-only 不使用原型约束，Strong 对应本文主方法默认配置。CE-only 与 Strong 使用 mainexp 的 200-round full run；Medium 与 Very strong 使用设置一致的 search2 150-round quick-search，故后两者仅用于停攻后早期持久性趋势分析。", '', f"在 Round 150，CE-only、Medium、Strong 与 Very strong 的 ASR 分别为 {by['CE-only']['ASR@150']}%、{by['6/3']['ASR@150']}%、{by['9/5']['ASR@150']}% 和 {by['12/7']['ASR@150']}%。在现有结果中，Strong 的 ASR@150 最高；其 200-round AUC 为 {by['9/5']['AUC@100-200']}，Clean ACC@150 为 {by['9/5']['CleanACC@150']}%。Medium 与 Very strong 的 100–150 AUC 分别为 {by['6/3']['AUC@100-150']} 和 {by['12/7']['AUC@100-150']}，均低于 Strong 的 {by['9/5']['AUC@100-150']}。Very strong 在 ASR@150 上低于 Strong，说明进一步增大权重并不必然带来更强的早期持久性。由于 6/3 和 12/7 尚无 200-round full run，上述比较应视为 quick-search 趋势，后续可补 full run 验证长期结论。"]
    (OUT/'lambda_sensitivity_summary.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
if __name__=='__main__': main()

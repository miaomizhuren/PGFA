#!/usr/bin/env python3
"""Create a resilient, method-ordered main-experiment metrics summary."""
from __future__ import annotations
import csv, re, sqlite3
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/summaries/mainexp"
SPECS = [("Fixed Patch", "mainexp_01_fixed_patch_baseline.yaml"), ("CE-only UAP", "mainexp_02_ce_only_uap_baseline.yaml"), ("Proto-UAP Layer3", "mainexp_03_proto_uap_layer3_baseline.yaml"), ("Proto-UAP Layer4 Proposed", "mainexp_04_proto_uap_layer4_proposed.yaml")]
ACC = (100,120,130,150,180,200); ASR = (100,110,120,130,140,150,160,170,180,190,200)

def metrics(db):
    if not db.exists(): return {}, f"missing db: {db.relative_to(ROOT)}"
    try:
        with sqlite3.connect(db) as con:
            cols={r[1] for r in con.execute('PRAGMA table_info(round_metrics)')}
            required={'round_index','clean_accuracy','attack_success_rate'}
            if not required <= cols: return {}, f"unsupported round_metrics columns: {sorted(cols)}"
            select=['round_index','clean_accuracy','attack_success_rate']+[x for x in ('psnr','ssim','linf') if x in cols]
            rows=con.execute(f"SELECT {', '.join(select)} FROM round_metrics ORDER BY round_index").fetchall()
        return {r[0]:dict(zip(['round','clean','asr',*select[3:]],r)) for r in rows}, ''
    except sqlite3.Error as e: return {}, f"sqlite error: {e}"

def log_metrics(path):
    if not path.exists(): return {}, ''
    pattern=re.compile(r"Round\s+(\d+).*?Clean ACC:\s+([\d.]+)%.*?ASR:\s+([\d.]+|N/A)(?:.*?PSNR:\s+([\d.]+)\s+dB)?(?:.*?SSIM:\s+([\d.]+))?(?:.*?L_inf:\s+([\d.]+))?")
    data={}
    for m in pattern.finditer(path.read_text(encoding='utf-8',errors='replace')):
        r,clean,asr,psnr,ssim,linf=m.groups(); data[int(r)]={'round':int(r),'clean':float(clean),'asr':None if asr=='N/A' else float(asr),'psnr':float(psnr) if psnr else None,'ssim':float(ssim) if ssim else None,'linf':float(linf) if linf else None}
    return data, f"log fallback: {path.relative_to(ROOT)}" if data else ''

def point(data, r, key): return data.get(r,{}).get(key, 'N/A')
def latest(data, key): return next((data[r].get(key) for r in sorted(data,reverse=True) if data[r].get(key) is not None),'N/A')
def auc(data):
    pts=[(r,data[r]['asr']) for r in sorted(data) if 100<=r<=200 and data[r].get('asr') is not None]
    return 'N/A' if len(pts)<2 else sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(pts,pts[1:]))/(pts[-1][0]-pts[0][0])

def main():
    OUT.mkdir(parents=True,exist_ok=True); rows=[]
    for method, filename in SPECS:
        path=ROOT/'configs/mainexp'/filename; cfg=yaml.safe_load(path.read_text()); db=ROOT/cfg['experiment']['output_dir']/cfg['tracking']['database_name']; data,note=metrics(db)
        if not data:
            log_data, log_note = log_metrics(ROOT/'experiments/logs/mainexp'/f"{cfg['experiment']['name']}.log")
            data, note = log_data, note if not log_data else log_note
        row={'method':method,'config_path':str(path.relative_to(ROOT)),'status':'complete' if data and max(data)>=200 else ('missing' if not data else 'incomplete'),'note':note or (f"missing rounds through 200; last={max(data)}" if data and max(data)<200 else '')}
        row.update({f'CleanACC@{r}':point(data,r,'clean') for r in ACC}); row.update({f'ASR@{r}':point(data,r,'asr') for r in ASR})
        row.update({'PSNR':latest(data,'psnr'),'SSIM':latest(data,'ssim'),'L_inf':latest(data,'linf'),'post_attack_auc_100_200':auc(data),'half_life_round':next((r for r in sorted(data) if r>=100 and data[r].get('asr') is not None and data[r]['asr']<50),'N/A')}); rows.append(row)
    fields=list(rows[0]);
    with (OUT/'mainexp_summary.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    table=['# Main experiment summary','', '| Method | Status | Clean@150 | ASR@120 | ASR@150 | AUC@100–200 | PSNR | SSIM | L∞ |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    table += [f"| {r['method']} | {r['status']} | {r['CleanACC@150']} | {r['ASR@120']} | {r['ASR@150']} | {r['post_attack_auc_100_200']} | {r['PSNR']} | {r['SSIM']} | {r['L_inf']} |" for r in rows]
    table += ['', '## Paper analysis template — requires confirmation from real results', '', 'The proposed Proto-UAP Layer4 should be compared with the three baselines using post-attack ASR retention, AUC, clean accuracy, and trigger stealth. Confirm all claims against the completed CSV before use in the thesis.', '', '## Missing/incomplete diagnostics', *[f"- {r['method']}: {r['note'] or 'none'}" for r in rows]]
    (OUT/'mainexp_summary.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
if __name__=='__main__': main()

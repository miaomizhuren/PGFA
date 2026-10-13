#!/usr/bin/env python3
"""Plot real lambda-sensitivity summaries without hard-coded measurements."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'experiments/summaries/lambda_sensitivity'
def save(fig,stem):
    for ext in ('png','pdf','svg'): fig.savefig(OUT/f'{stem}.{ext}',dpi=180)
    plt.close(fig)
def main():
    rows=list(csv.DictReader((OUT/'lambda_sensitivity_summary.csv').open(encoding='utf-8'))); settings=[r['setting'] for r in rows]; rounds=range(100,201,10)
    with (OUT/'plot_data_lambda_asr150.csv').open('w',newline='',encoding='utf-8') as h: w=csv.DictWriter(h,fieldnames=['setting','ASR@150']); w.writeheader(); w.writerows({'setting':r['setting'],'ASR@150':r['ASR@150']} for r in rows)
    wide=[{'round':r,**{row['setting']:row[f'ASR@{r}'] for row in rows}} for r in rounds]
    with (OUT/'plot_data_lambda_asr_curve.csv').open('w',newline='',encoding='utf-8') as h: w=csv.DictWriter(h,fieldnames=['round',*settings]); w.writeheader(); w.writerows(wide)
    fig,ax=plt.subplots(figsize=(6,4)); ax.bar(settings,[float(r['ASR@150']) for r in rows]); ax.set(xlabel='λ_delta / λ_train',ylabel='ASR@150 (%)'); ax.grid(axis='y',alpha=.25); fig.tight_layout(); save(fig,'lambda_asr150_bar')
    for end,stem in ((150,'lambda_asr_curve_100_150'),(200,'lambda_asr_curve_100_200')):
        selected=list(range(100,end+1,10))
        if not all(all(r[f'ASR@{x}']!='' for x in selected) for r in rows): continue
        fig,ax=plt.subplots(figsize=(7,4.5))
        for row in rows: ax.plot(selected,[float(row[f'ASR@{x}']) for x in selected],marker='o',label=row['setting'])
        ax.axvline(100,color='black',linestyle='--',linewidth=1,label='Attack stops'); ax.set(xlabel='Round',ylabel='ASR (%)',xlim=(100,end)); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); save(fig,stem)
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Plot feature-layer ablation figures from the generated real-results CSV."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'experiments/summaries/feature_ablation'; CSV=OUT/'feature_layer_ablation_summary.csv'; ROUNDS=[100,110,120,130,140,150,160,170,180,190,200]
def save(fig, stem):
    for ext in ('png','pdf','svg'): fig.savefig(OUT/f'{stem}.{ext}',dpi=180)
    plt.close(fig)
def main():
    rows=list(csv.DictReader(CSV.open(encoding='utf-8'))); OUT.mkdir(parents=True,exist_ok=True)
    by_layer={r['feature_layer']:r for r in rows}
    line_data=[{'round':r,'Proto-UAP Layer3':by_layer['layer3'][f'ASR@{r}'],'Proto-UAP Layer4':by_layer['layer4'][f'ASR@{r}']} for r in ROUNDS]
    with (OUT/'plot_data_feature_layer_asr.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=['round','Proto-UAP Layer3','Proto-UAP Layer4']); w.writeheader(); w.writerows(line_data)
    bars=[{'feature_layer':r['feature_layer'],'ASR@150':r['ASR@150'],'AUC@100-200':r['AUC@100-200'],'Half-life':r['Half-life'],'CleanACC@150':r['CleanACC@150'],'PSNR':r['PSNR'],'SSIM':r['SSIM'],'L_inf':r['L_inf']} for r in rows]
    with (OUT/'plot_data_feature_layer_bars.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(bars[0])); w.writeheader(); w.writerows(bars)
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    for row,label in zip(rows,['Proto-UAP Layer3','Proto-UAP Layer4']): ax.plot(ROUNDS,[float(row[f'ASR@{r}']) for r in ROUNDS],marker='o',label=label)
    ax.axvline(100,color='black',linestyle='--',linewidth=1,label='Attack stops'); ax.set(xlabel='Round',ylabel='ASR (%)',xlim=(100,200)); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); save(fig,'feature_layer_asr_curve')
    for field,stem,ylabel in [('ASR@150','feature_layer_asr150_bar','ASR@150 (%)'),('AUC@100-200','feature_layer_auc_bar','AUC@100–200 (%)'),('Half-life','feature_layer_halflife_bar','Half-life round')]:
        fig,ax=plt.subplots(figsize=(5,4)); ax.bar([r['feature_layer'] for r in rows],[float(r[field]) for r in rows],color=['#5B8FF9','#61DDAA']); ax.set(xlabel='Feature layer',ylabel=ylabel); ax.grid(axis='y',alpha=.25); fig.tight_layout(); save(fig,stem)
if __name__=='__main__': main()

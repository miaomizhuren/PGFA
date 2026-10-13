#!/usr/bin/env python3
"""Plot main-experiment ASR and clean-accuracy trajectories from SQLite metrics."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'experiments/summaries/mainexp'
SPECS=[('Fixed Patch','mainexp_01_fixed_patch_baseline.yaml'),('CE-only UAP','mainexp_02_ce_only_uap_baseline.yaml'),('Proto-UAP Layer3','mainexp_03_proto_uap_layer3_baseline.yaml'),('Proto-UAP Layer4 Proposed','mainexp_04_proto_uap_layer4_proposed.yaml')]
def load(filename):
    c=yaml.safe_load((ROOT/'configs/mainexp'/filename).read_text()); db=ROOT/c['experiment']['output_dir']/c['tracking']['database_name']
    if not db.exists(): return [],[],[]
    with sqlite3.connect(db) as con: rows=con.execute('SELECT round_index, clean_accuracy, attack_success_rate FROM round_metrics ORDER BY round_index').fetchall()
    return [r[0] for r in rows],[r[1] for r in rows],[r[2] for r in rows]
def save(metric, ylabel, stem):
    fig,ax=plt.subplots(figsize=(8,4.5)); found=False
    for method,file in SPECS:
        rounds,clean,asr=load(file); values=asr if metric=='asr' else clean
        pairs=[(r,v) for r,v in zip(rounds,values) if r>=100 and v is not None]
        if pairs: ax.plot(*zip(*pairs),label=method,linewidth=2); found=True
    ax.axvline(100,color='black',linestyle='--',linewidth=1,label='Attack stops'); ax.set(xlabel='Round',ylabel=ylabel,xlim=(100,200)); ax.grid(alpha=.25)
    if found: ax.legend(fontsize=8)
    else: ax.text(.5,.5,'No completed mainexp metrics found',ha='center',transform=ax.transAxes)
    fig.tight_layout()
    for ext in ('png','pdf','svg'): fig.savefig(OUT/f'{stem}.{ext}',dpi=180)
    plt.close(fig)
def main(): OUT.mkdir(parents=True,exist_ok=True); save('asr','ASR (%)','mainexp_asr_curve'); save('clean','Clean ACC (%)','mainexp_clean_acc_curve')
if __name__=='__main__': main()

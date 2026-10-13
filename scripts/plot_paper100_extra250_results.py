#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'experiments/summaries/paper100_extra250'; FIG=ROOT/'experiments/figures/paper100_extra250'; R=[110,150,180,200,240,250]
def save(fig,name):
 FIG.mkdir(parents=True,exist_ok=True)
 for ext in ('png','pdf'):fig.savefig(FIG/f'{name}.{ext}',bbox_inches='tight',dpi=180)
 plt.close(fig)
def curve(data,title,name):
 fig,ax=plt.subplots(figsize=(7,4))
 for _,x in data.iterrows():
  y=[pd.to_numeric(x.get(f'ASR@{r}'),errors='coerce') for r in R];ax.plot(R,y,marker='o',label=x['experiment_id'])
 ax.set(xlabel='Round',ylabel='ASR (%)',title=title);ax.grid(alpha=.3);ax.legend(fontsize=7);save(fig,name)
def main():
 path=OUT/'paper100_extra250_summary_all.csv'
 if not path.exists():print(f'Missing summary CSV: {path}',file=sys.stderr);return 1
 d=pd.read_csv(path)
 if d.empty:print('Summary CSV is empty.',file=sys.stderr);return 1
 for group,name in [('baselines','extra250_baseline_asr_curve'),('targets','extra250_target_label_asr'),('noniid','extra250_noniid_asr_curve')]:curve(d[d.group==group],group,name)
 m=OUT/'paper100_extra250_seed_mean_std.csv'
 if m.exists():
  s=pd.read_csv(m);s=s[s.metric=='asr'];fig,ax=plt.subplots(figsize=(7,4));ax.errorbar(s['round'],s['mean'],yerr=s['std'],marker='o');ax.set(xlabel='Round',ylabel='ASR mean ± std (%)',title='Proposed seeds (n may vary)');ax.grid(alpha=.3);save(fig,'extra250_seed_mean_std_asr')
 vals=pd.to_numeric(d['CleanACC@250'].fillna(d['CleanACC@final']),errors='coerce');fig,ax=plt.subplots(figsize=(9,4));ax.bar(range(len(d)),vals);ax.set_xticks(range(len(d)),d.experiment_id,rotation=60,ha='right',fontsize=6);ax.set_ylabel('Clean ACC@250/final (%)');save(fig,'extra250_cleanacc_bar')
 return 0
if __name__=='__main__':raise SystemExit(main())

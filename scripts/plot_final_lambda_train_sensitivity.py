from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];SAVE=ROOT/'outputs/saved_results/final_lambda_train_sensitivity_60_150_r350_20260912';FIG=SAVE/'figures'
def out(fig,n):
 FIG.mkdir(parents=True,exist_ok=True)
 for e in ('pdf','png','svg'):fig.savefig(FIG/f'{n}.{e}',bbox_inches='tight',dpi=300)
 plt.close(fig)
def main():
 c=pd.read_csv(SAVE/'FINAL_curve_data_lambda_train_asr.csv');t=pd.read_csv(SAVE/'FINAL_TABLE_lambda_train_sensitivity.csv');fig,ax=plt.subplots(figsize=(6.8,4))
 for lam in [3,5,7,9]:
  x=c[c.lambda_train==lam].sort_values('relative_post_round');ax.plot(x.relative_post_round,x.asr,label=f'$\\lambda_{{train}}={lam}$',lw=2)
 ax.set(title='Sensitivity to prototype supervision strength',xlabel='Rounds After Attack Termination',ylabel='ASR (%)',xlim=(0,200),ylim=(0,105),xticks=[0,60,120,180,200]);ax.grid(alpha=.25);ax.legend();out(fig,'lambda_train_asr_persistence')
 ax=t.plot.bar(x='lambda_train',y='PA-AUC_0:200',legend=False,figsize=(5.5,3.8),color='#0072b2');ax.set(xlabel='lambda_train',ylabel='PA-AUC_0:200');ax.grid(axis='y',alpha=.25);out(ax.figure,'lambda_train_pa_auc_bar')
 fig,ax=plt.subplots(figsize=(5.5,4));ax.scatter(t['CA@350'],t['PA-AUC_0:200'],color='#0072b2')
 for _,r in t.dropna(subset=['CA@350','PA-AUC_0:200']).iterrows():ax.annotate(str(int(r.lambda_train)),(r['CA@350'],r['PA-AUC_0:200']))
 ax.set(xlabel='CA@350',ylabel='PA-AUC_0:200');ax.grid(alpha=.25);out(fig,'lambda_train_ca_asr_tradeoff')
if __name__=='__main__':main()

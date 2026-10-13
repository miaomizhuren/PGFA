"""Create publication-ready duration-sensitivity figures after export."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; SAVE=ROOT/'outputs/saved_results/final_attack_duration_sensitivity_r350_20260912'; FIG=SAVE/'figures'
def save(fig,name):
 FIG.mkdir(parents=True,exist_ok=True)
 for ext in ('pdf','png','svg'):fig.savefig(FIG/f'{name}.{ext}',bbox_inches='tight',dpi=300)
 plt.close(fig)
def main():
 curve=pd.read_csv(SAVE/'FINAL_curve_data_attack_duration_asr.csv'); table=pd.read_csv(SAVE/'FINAL_TABLE_attack_duration_sensitivity.csv')
 fig,axs=plt.subplots(1,3,figsize=(13,3.6),sharey=True)
 for ax,w in zip(axs,['60-110','60-150','60-180']):
  for m,c in [('CE-UAP','#d55e00'),('PBU','#0072b2')]:
   x=curve[(curve.attack_window==w)&(curve.method==m)].sort_values('relative_post_round'); ax.plot(x.relative_post_round,x.asr,label=m,color=c,lw=2)
  ax.set(title=f'Attack {w}',xlim=(0,170),ylim=(0,105),xticks=[0,60,120,170],xlabel='Rounds After Attack Termination'); ax.grid(alpha=.25)
 axs[0].set_ylabel('ASR (%)'); axs[-1].legend(); fig.suptitle('ASR persistence under different attack durations'); save(fig,'attack_duration_asr_persistence')
 for metric,name,ylabel in [('PA-AUC_0:170','attack_duration_pa_auc_bar','PA-AUC_0:170')]:
  pivot=table.pivot(index='Attack_Window',columns='Method',values=metric).reindex(['60-110','60-150','60-180']); ax=pivot.plot.bar(color=['#d55e00','#0072b2'],figsize=(6,3.8)); ax.set(xlabel='Attack window',ylabel=ylabel); ax.grid(axis='y',alpha=.25); save(ax.figure,name)
 long=table.melt(id_vars=['Attack_Window','Method'],value_vars=['L50','L30'],var_name='Metric',value_name='Rounds'); long['Rounds']=long.Rounds.astype(str).str.replace('>170','170').astype(float); ax=long.pivot_table(index='Attack_Window',columns=['Metric','Method'],values='Rounds').reindex(['60-110','60-150','60-180']).plot.bar(figsize=(8,3.8)); ax.set(xlabel='Attack window',ylabel='Rounds after attack termination'); ax.grid(axis='y',alpha=.25); save(ax.figure,'attack_duration_l50_l30_bar')
if __name__=='__main__':main()

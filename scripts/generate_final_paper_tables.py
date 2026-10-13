"""Clean the audit master table and emit thesis tables without changing metrics."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; AUDIT=ROOT/'outputs/saved_results/final_paper_data_audit_20260913'; OUT=ROOT/'outputs/saved_results/final_paper_tables_20260913'
def tex(df,path,columns):
 def f(x): return '--' if pd.isna(x) else (f'{x:.2f}' if isinstance(x,float) else str(x))
 lines=[r'\begin{tabular}{'+'l'*len(columns)+r'}',r'\toprule',' & '.join(columns)+r' \\',r'\midrule']
 for _,r in df.iterrows(): lines.append(' & '.join(f(r[c]) for c in columns)+r' \\')
 path.write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular}'])+'\n')
def main():
 OUT.mkdir(parents=True,exist_ok=True); raw=pd.read_csv(AUDIT/'PAPER_MASTER_METRICS.csv'); clean=raw.copy(); clean.insert(clean.columns.get_loc('attack_start'), 'attack_window', pd.NA); clean['attack_end']=clean['attack_end'].astype(object); clean['attack_rounds']=clean['attack_rounds'].astype(object)
 ab=clean.experiment_group.eq('Ablation'); clean.loc[ab,'method']=['CE-only','Delta-only','Train-only','Full PBU']
 de=clean.experiment_group.eq('Defense'); clean.loc[de & clean.defense.isna(),'defense']='None/FedAvg'; clean.loc[de & clean.method.isna(),'method']='PBU'
 dur=clean.experiment_group.eq('Attack Duration')
 for i,row in clean[dur].iterrows():
  window=str(row['attack_end']); end=int(window.split('-')[1]); clean.loc[i,['attack_window','attack_end','attack_rounds']]=[window,end,end-60+1]
 clean.to_csv(AUDIT/'PAPER_MASTER_METRICS_CLEAN.csv',index=False,encoding='utf-8-sig')
 specs=[('TABLE_1_main_comparison','Main Comparison',['method']),('TABLE_2_defense','Defense',['defense']),('TABLE_3_ablation','Ablation',['method']),('TABLE_5_attack_duration','Attack Duration',['attack_window','method']),('TABLE_6_lambda_train','Prototype Strength',['lambda_train']),('TABLE_7_cross_dataset','Cross-Dataset',['dataset','method']),('TABLE_8_noniid','Non-IID',['partition','method'])]
 metrics=['CA_final','ASR_post0','ASR_post60','ASR_post120','ASR_post180','ASR_post200','L50','L30','PA_AUC']
 for name,group,ids in specs:
  x=clean[clean.experiment_group.eq(group)].copy(); cols=ids+metrics; x[cols].to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig');tex(x,OUT/f'{name}.tex',cols)
 mech=pd.read_csv(AUDIT/'PAPER_MECHANISM_METRICS.csv'); mech.to_csv(OUT/'TABLE_4_mechanism.csv',index=False,encoding='utf-8-sig');tex(mech,OUT/'TABLE_4_mechanism.tex',list(mech.columns))
 report=AUDIT/'PAPER_DATA_ONLY_AUDIT_REPORT.md'; report.write_text(report.read_text()+'\n- 原始 `PAPER_MASTER_METRICS.csv` 已保留；`PAPER_MASTER_METRICS_CLEAN.csv` 仅修复字段标签和 attack_window，用于论文表格生成，未修改任何实验数值。\n')
if __name__=='__main__':main()

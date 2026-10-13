"""Draft exporter for completed CIFAR-10/GTSRB cross-dataset runs."""
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];SAVE=ROOT/'outputs/saved_results/final_gtsrb_cross_dataset_60_150_r350_20260912'
SOURCES=[('CIFAR-10','CE-UAP',ROOT/'outputs/final_ablation_60_150_r350/abl01_ce_only_s60_e150_r350'),('CIFAR-10','PBU',ROOT/'outputs/extended_attack_single_proto/ea01_s60_e150_r350'),('GTSRB','CE-UAP',ROOT/'outputs/final_gtsrb_cross_dataset_60_150_r350/gtsrb_ce_uap_s60_e150_r350'),('GTSRB','PBU',ROOT/'outputs/final_gtsrb_cross_dataset_60_150_r350/gtsrb_pbu_s60_e150_r350')]
def main():
 SAVE.mkdir(parents=True,exist_ok=True);rows=[];curves=[]
 for ds,m,d in SOURCES:
  ps=sorted(d.rglob('metrics*.db'));x=None
  if ps:
   with sqlite3.connect(ps[0]) as c:x=pd.read_sql_query('select * from round_metrics',c).rename(columns={'round_index':'round','clean_accuracy':'clean_acc','attack_success_rate':'asr'})
  def v(r,c):
   if x is None:return None
   z=x.loc[x['round']==r,c];return None if z.empty else z.iloc[-1]
  if x is None:
   l50=l30=area=None
  else:
   post=x.loc[(x['round']>=150)&(x['round']<=350)&x['asr'].notna(),['round','asr']].drop_duplicates('round',keep='last').sort_values('round')
   def below(threshold):
    hit=post.loc[post['asr']<threshold,'round']
    return '>200' if hit.empty else int(hit.iloc[0]-150)
   l50,l30=below(50),below(30)
   area=(float(np.trapezoid(post['asr'].to_numpy(float),post['round'].to_numpy(float))/200)
         if not post.empty and post.iloc[0]['round']==150 and post.iloc[-1]['round']==350 else None)
  rows.append({'Dataset':ds,'Method':m,'CA@350':v(350,'clean_acc'),'ASR-0':v(150,'asr'),'ASR-60':v(210,'asr'),'ASR-120':v(270,'asr'),'ASR-180':v(330,'asr'),'ASR-200':v(350,'asr'),'L50':l50,'L30':l30,'PA-AUC_0:200':area,'source_dir':str(d)})
  if x is not None:
   curve=x.copy();curve['dataset']=ds;curve['method']=m;curve['global_round']=curve['round'];curve['post_attack_round']=curve['round']-150;curve['source_dir']=str(d);curve['source_file']=str(ps[0])
   for col in ['clean_acc','asr','attack_active']:
    if col not in curve:curve[col]=None
   curves.append(curve[['dataset','method','global_round','post_attack_round','clean_acc','asr','attack_active','source_dir','source_file']])
 table=pd.DataFrame(rows);table.to_csv(SAVE/'FINAL_TABLE_cross_dataset_gtsrb.csv',index=False,encoding='utf-8-sig')
 pd.concat(curves,ignore_index=True).to_csv(SAVE/'FINAL_curve_data_cross_dataset_asr.csv',index=False,encoding='utf-8-sig')
 cols=['CA@350','ASR-0','ASR-60','ASR-120','ASR-180','ASR-200','L50','L30','PA-AUC_0:200']
 def fmt(value): return '--' if value is None or pd.isna(value) else (f'{value:.2f}' if isinstance(value,(float,np.floating)) else str(value))
 lines=[r'\begin{table}[t]',r'\centering',r'\caption{Cross-dataset persistence comparison on CIFAR-10 and GTSRB.}',r'\label{tab:cross_dataset_gtsrb}',r'\begin{tabular}{llccccccccc}',r'\toprule',r'Dataset & Method & CA@350 & ASR-0 & ASR-60 & ASR-120 & ASR-180 & ASR-200 & $L_{50}$ & $L_{30}$ & PA-AUC$_{0:200}$ \\',r'\midrule']
 for _,row in table.iterrows(): lines.append(' & '.join([row.Dataset,row.Method]+[fmt(row[c]) for c in cols])+r' \\')
 (SAVE/'FINAL_TABLE_cross_dataset_gtsrb_latex.tex').write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular}',r'\end{table}'])+'\n')
 (SAVE/'README_where_to_open.txt').write_text(f'Open first: {SAVE}/FINAL_TABLE_cross_dataset_gtsrb.csv\nLaTeX: {SAVE}/FINAL_TABLE_cross_dataset_gtsrb_latex.tex\nL50/L30 and PA-AUC_0:200 use existing per-round metrics; no training was rerun.\n')
if __name__=='__main__':main()

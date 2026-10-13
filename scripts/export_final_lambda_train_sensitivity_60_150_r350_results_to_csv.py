"""Export fixed-lambda-delta PBU malicious-training sensitivity results."""
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; SAVE=ROOT/'outputs/saved_results/final_lambda_train_sensitivity_60_150_r350_20260912'
SPECS=[(3,[ROOT/'outputs/final_lambda_train_sensitivity_60_150_r350/lambda_train_3_pbu_s60_e150_r350'],'new run'),(5,[ROOT/'outputs/final_lambda_train_sensitivity_60_150_r350/lambda_train_5_pbu_s60_e150_r350'],'new run'),(7,[ROOT/'outputs/extended_attack_single_proto/ea01_s60_e150_r350'],'reused default PBU'),(9,[ROOT/'outputs/final_lambda_train_sensitivity_60_150_r350/lambda_train_9_pbu_s60_e150_r350'],'new run')]
COLS=['round','relative_post_round','lambda_train','lambda_delta','method','clean_acc','asr','attack_active','selected_malicious_clients','num_selected_malicious','num_attack_samples','num_successful_attacks','psnr','ssim','linf','source_file']
def find(ds):
 for d in ds:
  x=sorted(d.rglob('metrics*.db')) if d.is_dir() else []
  if x:return x[0]
def load(p):
 if not p:return None
 with sqlite3.connect(p) as c:x=pd.read_sql_query('select * from round_metrics order by round_index',c)
 x=x.rename(columns={'round_index':'round','clean_accuracy':'clean_acc','attack_success_rate':'asr'});x['round']=pd.to_numeric(x['round'],errors='coerce');return x.dropna(subset=['round']).assign(round=lambda z:z['round'].astype(int))
def v(x,r,c):
 if x is None:return None
 a=x.loc[x['round']==r,c];return None if a.empty else a.iloc[-1]
def below(x,t):
 if x is None:return None
 a=x[(x['round']>=150)&(x['round']<=350)&x['asr'].notna()];h=a.loc[a['asr']<t,'round'];return '>200' if h.empty else int(h.iloc[0]-150)
def auc(x):
 if x is None:return None
 a=x[(x['round']>=150)&(x['round']<=350)&x['asr'].notna()][['round','asr']].drop_duplicates('round',keep='last').sort_values('round');return None if a.empty or a.iloc[0]['round']!=150 or a.iloc[-1]['round']!=350 else float(np.trapezoid(a['asr'],a['round'])/200)
def main():
 SAVE.mkdir(parents=True,exist_ok=True); rows=[]; curves=[]
 for lam,ds,note in SPECS:
  p=find(ds);x=load(p);rows.append({'lambda_train':lam,'lambda_delta':12,'Method':'PBU','CA@350':v(x,350,'clean_acc'),'ASR-0':v(x,150,'asr'),'ASR-60':v(x,210,'asr'),'ASR-120':v(x,270,'asr'),'ASR-180':v(x,330,'asr'),'ASR-200':v(x,350,'asr'),'L50':below(x,50),'L30':below(x,30),'PA-AUC_0:200':auc(x),'source_dir':str(p.parent) if p else None,'notes':note})
  if x is None:full=pd.DataFrame(columns=COLS)
  else:
   full=x.copy();full['relative_post_round']=full['round']-150;full['lambda_train']=lam;full['lambda_delta']=12;full['method']='PBU';full['source_file']=str(p)
   for c in COLS:
    if c not in full:full[c]=None
   full=full[COLS];curves.append(full[['round','relative_post_round','lambda_train','lambda_delta','method','clean_acc','asr','attack_active','source_file']])
  full.to_csv(SAVE/f'FINAL_all_round_metrics_lambda_train_{lam}.csv',index=False,encoding='utf-8-sig');full[full['round'].isin([59,60,150,210,270,330,350])].to_csv(SAVE/f'FINAL_key_rounds_lambda_train_{lam}.csv',index=False,encoding='utf-8-sig')
 table=pd.DataFrame(rows);table.to_csv(SAVE/'FINAL_TABLE_lambda_train_sensitivity.csv',index=False,encoding='utf-8-sig');(pd.concat(curves,ignore_index=True) if curves else pd.DataFrame()).to_csv(SAVE/'FINAL_curve_data_lambda_train_asr.csv',index=False,encoding='utf-8-sig')
 def f(z):return '--' if pd.isna(z) else (f'{z:.2f}' if isinstance(z,(float,np.floating)) else str(z))
 cs=['CA@350','ASR-0','ASR-60','ASR-120','ASR-180','ASR-200','L50','L30','PA-AUC_0:200'];lines=[r'\begin{table}[t]',r'\centering',r'\caption{Sensitivity to prototype supervision strength in malicious training.}',r'\label{tab:lambda_train_sensitivity}',r'\begin{tabular}{lccccccccc}',r'\toprule',r'$\lambda_{\mathrm{train}}$ & CA@350 & ASR-0 & ASR-60 & ASR-120 & ASR-180 & ASR-200 & $L_{50}$ & $L_{30}$ & PA-AUC$_{0:200}$ \\',r'\midrule']
 for _,r in table.iterrows():lines.append(str(r.lambda_train)+' & '+' & '.join(f(r[c]) for c in cs)+r' \\')
 (SAVE/'FINAL_TABLE_lambda_train_sensitivity_latex.tex').write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular}',r'\end{table}'])+'\n');(SAVE/'README_where_to_open.txt').write_text(f'Main table: {SAVE}/FINAL_TABLE_lambda_train_sensitivity.csv\nLaTeX table: {SAVE}/FINAL_TABLE_lambda_train_sensitivity_latex.tex\nCurve data: {SAVE}/FINAL_curve_data_lambda_train_asr.csv\nlambda_train=7 is reused from ea01; lambda_delta is fixed at 12. Only lambda_train changes; no lambda_delta sensitivity is included to avoid coupled-weight interpretation.\n')
if __name__=='__main__':main()

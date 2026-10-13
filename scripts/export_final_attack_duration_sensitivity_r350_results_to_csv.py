"""Export the controlled IID attack-duration sensitivity comparison."""
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAVE = ROOT / 'outputs/saved_results/final_attack_duration_sensitivity_r350_20260912'
SPECS = [
 ('60-110',51,110,'CE-UAP',[ROOT/'outputs/final_attack_duration_sensitivity_r350/dur_s60_e110_ce_uap_r350']),
 ('60-110',51,110,'PBU',[ROOT/'outputs/final_attack_duration_sensitivity_r350/dur_s60_e110_pbu_r350']),
 ('60-150',91,150,'CE-UAP',[ROOT/'outputs/final_ablation_60_150_r350/abl01_ce_only_s60_e150_r350', ROOT/'outputs/final_main_60_150_r350/ce_uap_s60_e150_r350']),
 ('60-150',91,150,'PBU',[ROOT/'outputs/extended_attack_single_proto/ea01_s60_e150_r350', ROOT/'outputs/mechanism_feature_margin_60_150_r350/mech_pbu_s60_e150_r350_saveckpt']),
 ('60-180',121,180,'CE-UAP',[ROOT/'outputs/final_attack_duration_sensitivity_r350/dur_s60_e180_ce_uap_r350']),
 ('60-180',121,180,'PBU',[ROOT/'outputs/window_search_350_single_proto/ws350_04_s60_e180_r350']),
]
COLS=['round','relative_post_round','attack_window','attack_rounds','method','clean_acc','asr','attack_active','selected_malicious_clients','num_selected_malicious','num_attack_samples','num_successful_attacks','psnr','ssim','linf','source_file']
def db(candidates):
 for d in candidates:
  xs=sorted(d.rglob('metrics*.db')) if d.is_dir() else []
  if xs: return xs[0]
 return None
def load(p):
 if not p:return None
 with sqlite3.connect(p) as c: x=pd.read_sql_query('select * from round_metrics order by round_index',c)
 x=x.rename(columns={'round_index':'round','clean_accuracy':'clean_acc','attack_success_rate':'asr'}); x['round']=pd.to_numeric(x['round'],errors='coerce'); return x.dropna(subset=['round']).assign(round=lambda z:z['round'].astype(int))
def val(x,r,c):
 if x is None or c not in x:return None
 y=x.loc[x['round']==r,c]; return None if y.empty else y.iloc[-1]
def below(x,end,t):
 if x is None:return None
 y=x[(x['round']>=end)&(x['round']<=end+170)&x['asr'].notna()]; h=y.loc[y['asr']<t,'round']; return '>170' if h.empty else int(h.iloc[0]-end)
def auc(x,end):
 if x is None:return None
 y=x[(x['round']>=end)&(x['round']<=end+170)&x['asr'].notna()][['round','asr']].drop_duplicates('round',keep='last').sort_values('round')
 return None if y.empty or y.iloc[0]['round']!=end or y.iloc[-1]['round']!=end+170 else float(np.trapezoid(y['asr'],y['round'])/170)
def main():
 SAVE.mkdir(parents=True,exist_ok=True); frames={}; src={}; rows=[]; curves=[]
 for w,n,e,m,dirs in SPECS:
  k=(w,m); src[k]=db(dirs); frames[k]=load(src[k]); x=frames[k]; row={'Attack_Window':w,'Attack_Rounds':n,'Method':m,'CA@350':val(x,350,'clean_acc'),'ASR-0':val(x,e,'asr'),'ASR-60':val(x,e+60,'asr'),'ASR-120':val(x,e+120,'asr'),'ASR-170':val(x,e+170,'asr'),'L50':below(x,e,50),'L30':below(x,e,30),'PA-AUC_0:170':auc(x,e),'source_dir':str(src[k].parent) if src[k] else None}; rows.append(row)
  tag=f's60_e{e}_{m}'
  if x is None: full=pd.DataFrame(columns=COLS)
  else:
   full=x.copy(); full['relative_post_round']=full['round']-e; full['attack_window']=w; full['attack_rounds']=n; full['method']=m; full['source_file']=str(src[k]);
   for c in COLS:
    if c not in full: full[c]=None
   full=full[COLS]; curves.append(full[['round','relative_post_round','attack_window','attack_rounds','method','clean_acc','asr','attack_active','source_file']])
  full.to_csv(SAVE/f'FINAL_all_round_metrics_{tag}.csv',index=False,encoding='utf-8-sig'); full[full['round'].isin([59,60,e,e+60,e+120,e+170,350])].to_csv(SAVE/f'FINAL_key_rounds_{tag}.csv',index=False,encoding='utf-8-sig')
 table=pd.DataFrame(rows)
 for w in table.Attack_Window.unique():
  ce=table[(table.Attack_Window==w)&(table.Method=='CE-UAP')].iloc[0]; i=table.index[(table.Attack_Window==w)&(table.Method=='PBU')][0]
  if pd.notna(table.loc[i,'PA-AUC_0:170']) and pd.notna(ce['PA-AUC_0:170']):table.loc[i,'PBU_minus_CE_PA_AUC']=table.loc[i,'PA-AUC_0:170']-ce['PA-AUC_0:170']
  if pd.notna(table.loc[i,'ASR-170']) and pd.notna(ce['ASR-170']):table.loc[i,'PBU_minus_CE_ASR_170']=table.loc[i,'ASR-170']-ce['ASR-170']
 table.to_csv(SAVE/'FINAL_TABLE_attack_duration_sensitivity.csv',index=False,encoding='utf-8-sig')
 curve=pd.concat(curves,ignore_index=True) if curves else pd.DataFrame(columns=['round','relative_post_round','attack_window','attack_rounds','method','clean_acc','asr','attack_active','source_file']); curve.to_csv(SAVE/'FINAL_curve_data_attack_duration_asr.csv',index=False,encoding='utf-8-sig')
 def fmt(v):return '--' if pd.isna(v) else (f'{v:.2f}' if isinstance(v,(float,np.floating)) else str(v))
 lines=[r'\begin{table}[t]',r'\centering',r'\caption{Sensitivity of post-attack persistence to attack duration.}',r'\label{tab:attack_duration_sensitivity}',r'\begin{tabular}{llccccccccc}',r'\toprule',r'Window & Method & CA@350 & ASR-0 & ASR-60 & ASR-120 & ASR-170 & $L_{50}$ & $L_{30}$ & PA-AUC$_{0:170}$ \\',r'\midrule']
 for _,r in table.iterrows():lines.append(' & '.join([r.Attack_Window,r.Method]+[fmt(r[c]) for c in ['CA@350','ASR-0','ASR-60','ASR-120','ASR-170','L50','L30','PA-AUC_0:170']])+r' \\')
 (SAVE/'FINAL_TABLE_attack_duration_sensitivity_latex.tex').write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular}',r'\end{table}'])+'\n')
 (SAVE/'README_where_to_open.txt').write_text('Main table: '+str(SAVE/'FINAL_TABLE_attack_duration_sensitivity.csv')+'\nLaTex table: '+str(SAVE/'FINAL_TABLE_attack_duration_sensitivity_latex.tex')+'\nCurve data: '+str(SAVE/'FINAL_curve_data_attack_duration_asr.csv')+'\n\nReused: 60-150 CE-UAP/PBU and 60-180 PBU. New runs: 60-110 CE-UAP/PBU and 60-180 CE-UAP.\nPA-AUC uses 0:170, not 0:200, because attack 60-180 with total_rounds=350 has only 170 post-attack rounds.\n')
if __name__=='__main__':main()

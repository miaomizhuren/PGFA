"""Read-only audit of final thesis experiment artifacts."""
from pathlib import Path
import sqlite3
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/saved_results/final_paper_data_audit_20260913'
GROUPS={
'Main Comparison':('final_main_60_150_r350_20260909/FINAL_TABLE_main_persistence.csv',4,'FINAL_curve_data_main_asr.csv'),
'Defense':('final_defense_60_150_r350_20260910/FINAL_TABLE_defense_persistence.csv',5,'FINAL_curve_data_defense_asr.csv'),
'Ablation':('final_ablation_60_150_r350_20260911/FINAL_TABLE_ablation_persistence.csv',4,'FINAL_curve_data_ablation_asr.csv'),
'Non-IID':('final_noniid_60_150_r350_20260912/FINAL_TABLE_noniid_persistence.csv',6,'FINAL_curve_data_noniid_asr.csv'),
'Attack Duration':('final_attack_duration_sensitivity_r350_20260912/FINAL_TABLE_attack_duration_sensitivity.csv',6,'FINAL_curve_data_attack_duration_asr.csv'),
'Prototype Strength':('final_lambda_train_sensitivity_60_150_r350_20260912/FINAL_TABLE_lambda_train_sensitivity.csv',4,'FINAL_curve_data_lambda_train_asr.csv'),
'Cross-Dataset':('final_gtsrb_cross_dataset_60_150_r350_20260912/FINAL_TABLE_cross_dataset_gtsrb.csv',4,'FINAL_curve_data_cross_dataset_asr.csv')}
SAVED=ROOT/'outputs/saved_results'
def add_master(rows,group,table,path):
 for _,r in table.iterrows():
  get=lambda *ks:next((r[k] for k in ks if k in r and pd.notna(r[k])),None)
  rows.append({'experiment_group':group,'dataset':get('Dataset') or ('CIFAR-10' if group!='Cross-Dataset' else None),'method':get('Method'),'defense':get('Defense'),'partition':get('Partition'),'dirichlet_alpha':None,'attack_start':60,'attack_end':get('Attack_Window') or 150,'attack_rounds':get('Attack_Rounds'),'total_rounds':350,'seed':2026,'lambda_delta':get('lambda_delta'),'lambda_train':get('lambda_train'),'poison_ratio':.60,'CA_final':get('CA@350'),'ASR_post0':get('ASR-0'),'ASR_post60':get('ASR-60'),'ASR_post120':get('ASR-120'),'ASR_post170':get('ASR-170'),'ASR_post180':get('ASR-180'),'ASR_post200':get('ASR-200'),'L50':get('L50'),'L30':get('L30'),'PA_AUC':get('PA-AUC_0:200','PA-AUC_0:170'),'PA_AUC_range':'0:170' if group=='Attack Duration' else '0:200','PSNR':None,'SSIM':None,'Linf':None,'source_dir':get('source_dir'),'source_table':str(path),'notes':get('notes')})
def main():
 OUT.mkdir(parents=True,exist_ok=True); status=[]; missing=[]; master=[]; curves=[]; stealth=[]
 for group,(rel,need,curve_name) in GROUPS.items():
  path=SAVED/rel; table=pd.read_csv(path) if path.exists() else pd.DataFrame(); found=len(table); required=['CA@350','ASR-0','ASR-60','ASR-120','L50','L30']+(['ASR-170'] if group=='Attack Duration' else ['ASR-180','ASR-200'])+(['PA-AUC_0:170'] if group=='Attack Duration' else ['PA-AUC_0:200']); complete=int(sum(all(c in r and pd.notna(r[c]) for c in required) for _,r in table.iterrows())) if found else 0; curve_path=path.parent/curve_name if curve_name else None
  figure_dir=path.parent/'figures'
  if complete<need: state='DATA_PARTIAL'
  elif not curve_path or not curve_path.exists(): state='NEEDS_EXPORT_ONLY'
  else: state='DATA_COMPLETE'
  status.append({'experiment_group':group,'required_rows':need,'found_rows':found,'complete_rows':complete,'missing_rows':need-complete,'status':state,'main_table_path':str(path),'curve_data_path':str(curve_path) if curve_path else None,'figure_paths':str(figure_dir),'notes':'figures_not_checked'})
  if not curve_path or not curve_path.exists(): missing.append({'experiment_group':group,'item':'round curves','missing_type':'missing_curve_data','expected_field_or_file':'curve data CSV','current_status':'missing','recommended_action':'export from DB','suggested_script_or_path':str(path.parent)})
  if complete<need: missing.append({'experiment_group':group,'item':'table rows/metrics','missing_type':'missing_metric','expected_field_or_file':','.join(required),'current_status':f'{complete}/{need} complete','recommended_action':'audit source/export','suggested_script_or_path':str(path)})
  if found:add_master(master,group,table,path)
  if curve_path and curve_path.exists():
   x=pd.read_csv(curve_path); x['experiment_group']=group;curves.append(x)
   if all(col in x for col in ['psnr','ssim','linf']):
    method_col='method' if 'method' in x else None
    for method,part in x.groupby(method_col) if method_col else [('unknown',x)]:
     z=part.dropna(subset=['psnr','ssim','linf'])
     if not z.empty:
      last=z.iloc[-1]; stealth.append({'experiment_group':group,'dataset':last.get('dataset','CIFAR-10'),'method':method,'defense':None,'partition':last.get('partition'), 'attack_start':60,'attack_end':150,'PSNR':last['psnr'],'SSIM':last['ssim'],'Linf':last['linf'],'round_measured':last.get('round'), 'source_dir':last.get('source_file'),'notes':'from final available round metrics'})
 pd.DataFrame(status).to_csv(OUT/'PAPER_EXPERIMENT_STATUS_AUDIT.csv',index=False);pd.DataFrame(missing).to_csv(OUT/'PAPER_MISSING_ITEMS.csv',index=False);pd.DataFrame(master).to_csv(OUT/'PAPER_MASTER_METRICS.csv',index=False);pd.concat(curves,ignore_index=True).to_csv(OUT/'PAPER_ROUND_CURVES.csv',index=False)
 mech=SAVED/'mechanism_feature_margin_20260911/FEATURE_MARGIN_CE_vs_PBU.csv'; pd.read_csv(mech).to_csv(OUT/'PAPER_MECHANISM_METRICS.csv',index=False) if mech.exists() else pd.DataFrame().to_csv(OUT/'PAPER_MECHANISM_METRICS.csv',index=False)
 pd.DataFrame(stealth,columns=['experiment_group','dataset','method','defense','partition','attack_start','attack_end','PSNR','SSIM','Linf','round_measured','source_dir','notes']).to_csv(OUT/'PAPER_STEALTH_METRICS.csv',index=False)
 ready=[r['experiment_group'] for r in status if r['status']=='DATA_COMPLETE']; not_ready=[f"{r['experiment_group']}: {r['status']}" for r in status if r['status']!='DATA_COMPLETE']; (OUT/'README_WHAT_IS_READY.md').write_text('# 论文实验数据-only审计\n\n本次只整理数据，不处理图。数据完整：'+', '.join(ready)+'。\n\n数据待处理：'+(', '.join(not_ready) if not_ready else '无')+'。\n\n图未处理不影响本次状态；优先打开 PAPER_MASTER_METRICS.csv；DBA enhanced 配置不建议作为主表。\n')
if __name__=='__main__':main()

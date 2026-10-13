#!/usr/bin/env python3
"""Summarize isolated paper100_extra250 SQLite results and explicit reuses."""
import csv, sqlite3
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]; CFG=ROOT/'configs/paper100_extra250'; OUT=ROOT/'experiments/summaries/paper100_extra250'; R=(110,150,180,200,240,250)
F=['experiment_id','group','method','status','config_path','source_result','num_clients','clients_per_round','participation_rate','local_epochs','partition','dirichlet_alpha','min_samples_per_client','seed','target_label','attack_start','attack_end','total_rounds','final_round','feature_layer','poison_ratio','lambda_delta','lambda_train','epsilon','steps','alpha','client_lr',*[f'CleanACC@{r}' for r in R],*[f'ASR@{r}' for r in R],'CleanACC@final','ASR@final','PSNR','SSIM','L_inf','AUC@110-150','AUC@110-200','AUC@110-250','Half-life','note']
def fmt(v,d=2): return '' if v is None else f'{float(v):.{d}f}'
def read_db(path):
 if not path.is_file(): return {},None
 try:
  with sqlite3.connect(path) as c: rows=c.execute('select round_index,clean_accuracy,attack_success_rate,psnr,ssim,linf from round_metrics order by round_index').fetchall()
  return {r:(a,s,p,q,l) for r,a,s,p,q,l in rows},path
 except sqlite3.Error:return {},None
def auc(d,end):
 p=[(r,d[r][1]) for r in sorted(d) if 110<=r<=end and d[r][1] is not None]
 return None if len(p)<2 or p[0][0]!=110 or p[-1][0]!=end else sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(p,p[1:]))/(end-110)
def row_from(config_path,source=None,label=None):
 c=yaml.safe_load(config_path.read_text()); e=c['experiment']; a=c['attack']; ad=a['trigger'].get('adversarial',{}); p=ad.get('prototype_contrastive',{}); group=config_path.parent.name
 row={k:'' for k in F}; row.update(experiment_id=e['name'],group=group,method=a.get('method','patch'),config_path=str(config_path.relative_to(ROOT)),num_clients=c['partition']['num_clients'],clients_per_round=c['federated']['clients_per_round'],participation_rate='10%',local_epochs=c['federated']['local_epochs'],partition=c['partition'].get('type','iid'),dirichlet_alpha=c['partition'].get('alpha',''),min_samples_per_client=c['partition'].get('min_samples_per_client',''),seed=e['seed'],target_label=a['target_label'],attack_start=a['attack_start_round'],attack_end=a['attack_end_round'],total_rounds=c['federated']['num_rounds'],feature_layer=p.get('feature_layer','patch'),poison_ratio=fmt(a['poison_ratio']),lambda_delta=fmt(p.get('lambda_delta')),lambda_train=fmt(p.get('lambda_train')),epsilon=fmt(ad.get('epsilon')),steps=ad.get('steps',''),alpha=fmt(ad.get('alpha'),3),client_lr=fmt(c['training']['learning_rate'],4))
 dbpath=ROOT/e['output_dir']/c['tracking']['database_name'] if source is None else ROOT/source
 d,used=read_db(dbpath); last=max(d,default=0); row['source_result']=str(source or '')
 row['status']='complete' if last>=250 and 250 in d and d[250][0] is not None and d[250][1] is not None else 'not_completed'
 row['final_round']=last or ''
 for r in R:
  if r in d: row[f'CleanACC@{r}'],row[f'ASR@{r}']=fmt(d[r][0]),fmt(d[r][1])
 if d:
  v=d[last]; row.update(**{'CleanACC@final':fmt(v[0]),'ASR@final':fmt(v[1]),'PSNR':fmt(v[2]),'SSIM':fmt(v[3],4),'L_inf':fmt(v[4],4),'AUC@110-150':fmt(auc(d,150)),'AUC@110-200':fmt(auc(d,200)),'AUC@110-250':fmt(auc(d,250)),'Half-life':next((str(r) for r in sorted(d) if r>110 and d.get(110,(None,None))[1] is not None and d[r][1] is not None and d[r][1]<=d[110][1]/2),'>250' if last>=250 else '')})
 row['note']=('reused source; ' if source else '')+(f'DB: {used.relative_to(ROOT)}' if used else 'missing metrics DB')
 if source and 250 not in d: row['note']+='; missing Round250 in reused source'
 if label: row['method']=label
 return row
def main():
 OUT.mkdir(parents=True,exist_ok=True); rows=[row_from(p) for p in sorted(CFG.rglob('*.yaml'))]
 base=ROOT/'outputs/paper100_mechanism250'; ce=base/'mech250_01_ce_only_uap/metrics_mech250_01_ce_only_uap.db'; prop=base/'mech250_03_proto_uap_layer4/metrics_mech250_03_proto_uap_layer4.db'
 templates={r['group']:r for r in rows if r['group'] in ('baselines','seeds','targets','noniid')}
 for group,name,src,label in [('baselines','extra250_reuse_ce_uap',ce,'ce_uap'),('baselines','extra250_reuse_proposed',prop,'proto_uap'),('seeds','extra250_reuse_default_seed',prop,'proto_uap'),('targets','extra250_reuse_target_0',prop,'proto_uap'),('noniid','extra250_reuse_iid',prop,'proto_uap')]:
  t=next(r for r in rows if r['group']==group); cp=ROOT/t['config_path']; r=row_from(cp,src,label); r.update(experiment_id=name,source_result=str(src.relative_to(ROOT)),group=group); rows.append(r)
  if group=='seeds': r['seed']=2026
  if group=='targets': r['target_label']=0
  if group=='noniid': r.update(partition='iid',dirichlet_alpha='',min_samples_per_client='')
 with (OUT/'paper100_extra250_summary_all.csv').open('w',newline='') as h: w=csv.DictWriter(h,F);w.writeheader();w.writerows(rows)
 for group in ('baselines','seeds','targets','noniid'):
  subset=[r for r in rows if r['group']==group]
  with (OUT/f'paper100_extra250_{group}_summary.csv').open('w',newline='') as h:w=csv.DictWriter(h,F);w.writeheader();w.writerows(subset)
 seedrows=[r for r in rows if r['group']=='seeds']; mean=[]
 for r in R:
  for metric,prefix in [('clean_acc','CleanACC'),('asr','ASR')]:
   values=[float(x[f'{prefix}@{r}']) for x in seedrows if x[f'{prefix}@{r}']!='']; n=len(values); mean.append({'round':r,'metric':metric,'mean':fmt(sum(values)/n) if n else '','std':fmt((sum((v-sum(values)/n)**2 for v in values)/n)**.5) if n else '','n':n,'seeds_used':','.join(str(x['seed']) for x in seedrows if x[f'{prefix}@{r}']!='')})
 with (OUT/'paper100_extra250_seed_mean_std.csv').open('w',newline='') as h:w=csv.DictWriter(h,['round','metric','mean','std','n','seeds_used']);w.writeheader();w.writerows(mean)
 (OUT/'paper100_extra250_summary_all.md').write_text('# paper100_extra250 summary\n\nGenerated from real SQLite DBs and explicitly labelled reused sources. Missing Round250 values are left blank.\n',encoding='utf-8')
 print(OUT/'paper100_extra250_summary_all.csv')
if __name__=='__main__':main()

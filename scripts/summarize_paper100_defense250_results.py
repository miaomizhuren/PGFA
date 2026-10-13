import csv,sqlite3,yaml
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/summaries/paper100_defense250';R=(110,150,180,200,240,250)
F=['experiment_id','defense_method','defense_enabled','clip_norm','noise_std','trim_ratio','status','config_path','source_result','num_clients','clients_per_round','participation_rate','local_epochs','partition','seed','target_label','attack_start','attack_end','total_rounds','final_round','feature_layer','poison_ratio','lambda_delta','lambda_train','epsilon','steps','alpha','client_lr',*[f'CleanACC@{r}'for r in R],*[f'ASR@{r}'for r in R],'CleanACC@final','ASR@final','PSNR','SSIM','L_inf','AUC@110-150','AUC@110-200','AUC@110-250','Half-life','note']
def f(x,d=2):return ''if x is None else f'{float(x):.{d}f}'
def read(p):
 if not p.exists():return {}
 with sqlite3.connect(p)as c:z=c.execute('select round_index,clean_accuracy,attack_success_rate,psnr,ssim,linf from round_metrics').fetchall()
 return {r:(a,s,p,q,l)for r,a,s,p,q,l in z}
def au(d,e):
 x=[(r,d[r][1])for r in sorted(d)if 110<=r<=e and d[r][1]is not None];return None if len(x)<2 or x[0][0]!=110 or x[-1][0]!=e else sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(x,x[1:]))/(e-110)
def make(cp,src=None,name=None):
 c=yaml.safe_load(cp.read_text());a=c['attack'];ad=a['trigger']['adversarial'];p=ad['prototype_contrastive'];de=c.get('defense',{});r={k:''for k in F};r.update(experiment_id=name or c['experiment']['name'],defense_method=de.get('method','none'),defense_enabled=str(de.get('enabled',False)).lower(),clip_norm=de.get('clip_norm',''),noise_std=de.get('noise_std',''),trim_ratio=de.get('trim_ratio',''),config_path=str(cp.relative_to(ROOT)),source_result=str(src.relative_to(ROOT))if src else '',num_clients=100,clients_per_round=10,participation_rate='10%',local_epochs=3,partition='iid',seed=c['experiment']['seed'],target_label=0,attack_start=60,attack_end=110,total_rounds=250,feature_layer=p['feature_layer'],poison_ratio=f(a['poison_ratio']),lambda_delta=f(p['lambda_delta']),lambda_train=f(p['lambda_train']),epsilon=f(ad['epsilon']),steps=ad['steps'],alpha=f(ad['alpha'],3),client_lr='0.0100')
 d=read(src or ROOT/c['experiment']['output_dir']/c['tracking']['database_name']);last=max(d,default=0);r['final_round']=last or '';r['status']='complete'if last>=250 and 250 in d and d[250][0]is not None and d[250][1]is not None else'not_completed'
 for x in R:
  if x in d:r[f'CleanACC@{x}'],r[f'ASR@{x}']=f(d[x][0]),f(d[x][1])
 if d:
  v=d[last];r.update(**{'CleanACC@final':f(v[0]),'ASR@final':f(v[1]),'PSNR':f(v[2]),'SSIM':f(v[3],4),'L_inf':f(v[4],4),'AUC@110-150':f(au(d,150)),'AUC@110-200':f(au(d,200)),'AUC@110-250':f(au(d,250)),'Half-life':next((str(x)for x in sorted(d)if x>110 and d[x][1]is not None and d[x][1]<=d[110][1]/2),'>250'if last>=250 else'')})
 r['note']=('reused source; 'if src else'')+(f'DB: {(src or ROOT/c["experiment"]["output_dir"]/c["tracking"]["database_name"]).relative_to(ROOT)}'if d else'missing metrics DB');return r
def main():
 OUT.mkdir(parents=True,exist_ok=True);ps=sorted((ROOT/'configs/paper100_defense250').glob('*.yaml'));rows=[make(p)for p in ps];src=ROOT/'outputs/paper100_mechanism250/mech250_03_proto_uap_layer4/metrics_mech250_03_proto_uap_layer4.db';x=make(ps[0],src,'reuse_no_defense_proposed');x.update(defense_method='none',defense_enabled='false',clip_norm=0,noise_std=0,trim_ratio=0);rows=[x]+rows
 for n in ('paper100_defense250_summary_all.csv','paper100_defense250_table_for_paper.csv'):
  with(OUT/n).open('w',newline='')as h:w=csv.DictWriter(h,F);w.writeheader();w.writerows(rows)
 (OUT/'paper100_defense250_summary_all.md').write_text('# paper100 defense250 summary\n\nReal DB metrics only; the no-defense row is explicitly reused.\n');print(OUT/'paper100_defense250_summary_all.csv')
if __name__=='__main__':main()

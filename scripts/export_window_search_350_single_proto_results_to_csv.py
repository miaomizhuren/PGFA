"""Export four fixed-prototype window-search runs and an optional fixed baseline."""
from pathlib import Path
import re, sqlite3
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; SAVE=ROOT/'outputs/saved_results/window_search_350_single_proto_20260909'
SPECS={'ws350_01_s50_e150_r350':(50,150),'ws350_02_s50_e140_r350':(50,140),'ws350_03_s40_e140_r350':(40,140),'ws350_04_s60_e180_r350':(60,180)}
def norm(f):
 a={'round_index':'round','round_id':'round','clean_accuracy':'clean_acc','cleanacc':'clean_acc','attack_success_rate':'asr','attack_success':'asr'}; f=f.copy(); f.columns=[a.get(re.sub(r'[^a-z0-9]+','_',str(c).lower()).strip('_'),re.sub(r'[^a-z0-9]+','_',str(c).lower()).strip('_')) for c in f.columns]; return f
def prep(f):
 f=norm(f); f['round']=pd.to_numeric(f['round'],errors='coerce'); f=f.dropna(subset=['round']).copy(); f['round']=f['round'].astype(int)
 for c in ('clean_acc','asr','psnr','ssim','linf'):
  if c in f:f[c]=pd.to_numeric(f[c],errors='coerce')
 return f.sort_values('round').drop_duplicates('round',keep='last')
def load(name):
 base=ROOT/'outputs/window_search_350_single_proto'/name; found=[]
 for path in [*base.rglob('*.db'),*base.rglob('*.sqlite'),*base.rglob('*.sqlite3')]:
  with sqlite3.connect(path) as conn:
   for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    f=norm(pd.read_sql_query(f'SELECT * FROM "{table}"',conn))
    if {'round','clean_acc','asr'} <= set(f): found.append((path,table,f))
 for path in base.rglob('*.csv'):
  f=norm(pd.read_csv(path))
  if {'round','clean_acc','asr'} <= set(f): found.append((path,'',f))
 if not found: raise SystemExit(f'No metrics found for {name}; train it first.')
 path,table,f=found[0]; return prep(f),path
def at(f,r,c):
 x=f.loc[f['round']==r,c] if c in f else pd.Series(dtype=float); return x.iloc[-1] if not x.empty else float('nan')
def auc(f,start,end):
 p=f[(f['round']>=start)&(f['round']<=end)][['round','asr']].dropna().sort_values('round')
 return float('nan') if len(p)<2 else sum((a+b)*(x2-x1)/2 for x1,a,x2,b in zip(p['round'],p['asr'],p['round'].iloc[1:],p['asr'].iloc[1:]))/(end-start)
def row(name,f,start,end,source):
 final=f[f['round']==350].tail(1); d={'experiment':name,'attack_start':start,'attack_end':end,'total_rounds':350,'attack_length':end-start+1,'post_attack_length':350-end,'CleanACC@attack_start_minus1':at(f,start-1,'clean_acc'),'CleanACC@attack_end':at(f,end,'clean_acc'),'CleanACC@final':at(f,350,'clean_acc'),'ASR@attack_end':at(f,end,'asr')}
 for n in (1,10,20,40,70,90,120,130,150,170,200): d[f'ASR@post{n}']=at(f,end+n,'asr') if end+n<=350 else float('nan')
 d.update({'ASR@final':at(f,350,'asr'),'PA-AUC@post':auc(f,end+1,350),'PSNR':at(final,350,'psnr'),'SSIM':at(final,350,'ssim'),'Linf':at(final,350,'linf'),'source_file':str(source)}); return d
SAVE.mkdir(parents=True,exist_ok=True); rows=[]
for name,(start,end) in SPECS.items():
 f,path=load(name); keys=sorted(set([start-1,start,100,110,120,130,140,150,180,end,end+1,end+10,end+20,end+40,end+70,end+90,end+120,end+130,end+150,end+170,end+200,350])); keys=[r for r in keys if r<=350]
 f.to_csv(SAVE/f'FINAL_all_round_metrics_{name}.csv',index=False,encoding='utf-8-sig'); f[f['round'].isin(keys)].to_csv(SAVE/f'FINAL_key_rounds_{name}.csv',index=False,encoding='utf-8-sig'); rows.append(row(name,f,start,end,path))
pd.DataFrame(rows).to_csv(SAVE/'FINAL_compare_4groups.csv',index=False,encoding='utf-8-sig')
baseline=ROOT/'outputs/saved_results/extended_attack_single_proto_ea01_20260909/FINAL_key_rounds.csv'; compare=rows.copy()
if baseline.exists(): compare.insert(0,row('baseline_s60_e150_r350',prep(pd.read_csv(baseline)),60,150,baseline))
pd.DataFrame(compare).to_csv(SAVE/'COMPARE_with_s60_e150_baseline.csv',index=False,encoding='utf-8-sig')
(SAVE/'README_where_to_open.txt').write_text('用户最终打开：\n'+str(SAVE/'FINAL_compare_4groups.csv')+'\n'+str(SAVE/'COMPARE_with_s60_e150_baseline.csv')+'\n',encoding='utf-8')

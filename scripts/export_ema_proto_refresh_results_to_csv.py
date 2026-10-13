"""Export EMA results and compare them against the fixed ea01 baseline."""
from pathlib import Path
import re, sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'outputs/ema_proto_refresh/ema01_beta09_int15_s60_e150_r350'
SAVE = ROOT / 'outputs/saved_results/ema_proto_refresh_20260909'
KEY = [59,60,75,90,105,120,135,150,151,160,170,190,220,240,270,280,300,320,350]

def norm(frame):
    aliases={'round_index':'round','round_id':'round','clean_accuracy':'clean_acc','cleanacc':'clean_acc','attack_success_rate':'asr','attack_success':'asr'}
    frame=frame.copy(); frame.columns=[aliases.get(re.sub(r'[^a-z0-9]+','_',str(c).lower()).strip('_'),re.sub(r'[^a-z0-9]+','_',str(c).lower()).strip('_')) for c in frame.columns]
    return frame
def score(frame): return 10*('round' in frame)+10*('asr' in frame)+8*('clean_acc' in frame)
def prep(frame):
    frame=norm(frame); frame['round']=pd.to_numeric(frame['round'],errors='coerce'); frame=frame.dropna(subset=['round']).copy(); frame['round']=frame['round'].astype(int)
    for c in ('clean_acc','asr','psnr','ssim','linf'):
        if c in frame: frame[c]=pd.to_numeric(frame[c],errors='coerce')
    return frame.sort_values('round').drop_duplicates('round',keep='last')
def load_result():
    found=[]
    for path in [*RESULT.rglob('*.db'),*RESULT.rglob('*.sqlite'),*RESULT.rglob('*.sqlite3')]:
        with sqlite3.connect(path) as conn:
            for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                frame=norm(pd.read_sql_query(f'SELECT * FROM "{table}"',conn))
                if score(frame)>=20: found.append((score(frame),path,table,frame))
    for path in RESULT.rglob('*.csv'):
        frame=norm(pd.read_csv(path))
        if score(frame)>=20: found.append((score(frame),path,'',frame))
    if not found: raise SystemExit(f'No completed EMA metrics under {RESULT}. Train ema01 first.')
    _,path,table,frame=sorted(found,key=lambda x:x[0],reverse=True)[0]; return prep(frame),path,table
def at(frame,round_index,field):
    found=frame.loc[frame['round']==round_index,field] if field in frame else pd.Series(dtype=float)
    return found.iloc[-1] if not found.empty else float('nan')
def auc(frame,start,end):
    points=frame[(frame['round']>=start)&(frame['round']<=end)][['round','asr']].dropna().sort_values('round')
    if len(points)<2:return float('nan')
    return sum((a+b)*(x2-x1)/2 for x1,a,x2,b in zip(points['round'],points['asr'],points['round'].iloc[1:],points['asr'].iloc[1:]))/(end-start)
def row(name,frame,policy,interval,beta,source):
    final=frame[frame['round']==350].tail(1); end=150
    result={'experiment':name,'refresh_policy':policy,'refresh_interval':interval,'ema_beta':beta,'attack_start':60,'attack_end':150,'total_rounds':350,'CleanACC@final':at(frame,350,'clean_acc'),'ASR@attack_end':at(frame,150,'asr'),'ASR@post1':at(frame,151,'asr'),'ASR@post10':at(frame,160,'asr'),'ASR@post20':at(frame,170,'asr'),'ASR@post40':at(frame,190,'asr'),'ASR@post70':at(frame,220,'asr'),'ASR@post90':at(frame,240,'asr'),'ASR@post120':at(frame,270,'asr'),'ASR@post130':at(frame,280,'asr'),'ASR@post150':at(frame,300,'asr'),'ASR@post170':at(frame,320,'asr'),'ASR@post200':at(frame,350,'asr'),'ASR@final':at(frame,350,'asr'),'PA-AUC@post':auc(frame,151,350),'PSNR':at(final,350,'psnr'),'SSIM':at(final,350,'ssim'),'Linf':at(final,350,'linf'),'source_file':str(source)}
    return result
SAVE.mkdir(parents=True,exist_ok=True)
ema,source,table=load_result(); ema.to_csv(SAVE/'FINAL_all_round_metrics.csv',index=False,encoding='utf-8-sig'); ema[ema['round'].isin(KEY)].to_csv(SAVE/'FINAL_key_rounds.csv',index=False,encoding='utf-8-sig')
rows=[row('ema01_beta09_int15_s60_e150_r350',ema,'ema',15,0.9,source)]
fixed=ROOT/'outputs/saved_results/extended_attack_single_proto_ea01_20260909/FINAL_key_rounds.csv'
if fixed.exists(): rows.append(row('fixed_ea01_s60_e150_r350',prep(pd.read_csv(fixed)),'fixed',15,0.9,fixed))
compare=SAVE/'COMPARE_with_fixed_ea01.csv'; pd.DataFrame(rows).to_csv(compare,index=False,encoding='utf-8-sig')
(SAVE/'README_where_to_open.txt').write_text('用户最终需要打开：\n'+str(compare)+'\n'+str(SAVE/'FINAL_key_rounds.csv')+'\n',encoding='utf-8')
print(f'Exported {compare}')

#!/usr/bin/env python3
from pathlib import Path
import csv,sqlite3,yaml
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"experiments/summaries/paper100_lambda_long200"; R=range(100,201,10)
N=["p100ll_lambda_00_00_ce_only","p100ll_lambda_03_015_weak","p100ll_lambda_06_03_medium","p100ll_lambda_09_05_proposed","p100ll_lambda_12_07_strong","p100ll_lambda_15_09_very_strong"]
F=["experiment_id","method","status","config_path","num_clients","clients_per_round","participation_rate","local_epochs","attack_start","attack_end","total_rounds","feature_layer","poison_ratio","lambda_delta","lambda_train","epsilon","steps","alpha","client_lr",*[f"CleanACC@{r}" for r in R],*[f"ASR@{r}" for r in R],"PSNR","SSIM","L_inf","AUC@110-150","AUC@110-200","AUC@attack_end-200","Half-life","score","note"]
def f(x,d=2): return "" if x is None else f"{float(x):.{d}f}"
def auc(d,e):
 p=[(r,d[r][1]) for r in sorted(d) if 110<=r<=e and d[r][1] is not None]
 return None if len(p)<2 or p[0][0]!=110 or p[-1][0]!=e else sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(p,p[1:]))/(e-110)
def main():
 OUT.mkdir(parents=True,exist_ok=True); rows=[]
 for n in N:
  path=ROOT/"configs/paper100_lambda_long200"/f"{n}.yaml"; c=yaml.safe_load(path.read_text()); a=c["attack"]; ad=a["trigger"]["adversarial"]; p=ad["prototype_contrastive"]; row={k:"" for k in F}; row.update(experiment_id=n,method=("CE-only UAP" if p["lambda_delta"]==0 else f"Lambda {p['lambda_delta']:g}/{p['lambda_train']:g}"),status="not_completed",config_path=str(path.relative_to(ROOT)),num_clients="100",clients_per_round="10",participation_rate="10%",local_epochs="3",attack_start="60",attack_end="110",total_rounds="200",feature_layer="layer4",poison_ratio=f(a["poison_ratio"]),lambda_delta=f(p["lambda_delta"]),lambda_train=f(p["lambda_train"]),epsilon=f(ad["epsilon"]),steps=str(ad["steps"]),alpha=f(ad["alpha"],3),client_lr=f(c["training"]["learning_rate"],4))
  db=ROOT/c["experiment"]["output_dir"]/c["tracking"]["database_name"]
  if not db.exists(): row["note"]=f"missing metrics DB: {db.relative_to(ROOT)}"; rows.append(row); continue
  with sqlite3.connect(db) as con: z=con.execute("select round_index,clean_accuracy,attack_success_rate,psnr,ssim,linf from round_metrics").fetchall()
  d={r:(ca,asr,ps,ss,li) for r,ca,asr,ps,ss,li in z}; last=max(d,default=0); row["status"]="complete" if last>=200 else "not_completed"; row["note"]=f"complete through Round {last}" if last>=200 else f"last recorded round: {last}"
  for r in R:
   if r in d: row[f"CleanACC@{r}"]=f(d[r][0]);row[f"ASR@{r}"]=f(d[r][1])
  if d:
   v=d[max(d)];a150=auc(d,150);a200=auc(d,200);row.update(PSNR=f(v[2]),SSIM=f(v[3],4),L_inf=f(v[4],4),**{"AUC@110-150":f(a150),"AUC@110-200":f(a200),"AUC@attack_end-200":f(a200)})
   threshold=d.get(110,(None,None))[1]; row["Half-life"]=next((str(r) for r in range(120,201,10) if threshold is not None and r in d and d[r][1]<=.5*threshold),">200" if last>=200 else "")
   if None not in (d.get(200,(None,None))[1],a200,d.get(150,(None,None))[1],d.get(200,(None,None))[0],v[2]): row["score"]=f(.35*d[200][1]+.25*a200+.20*d[150][1]+.10*d[200][0]+.10*min(v[2],35))
  rows.append(row)
 with (OUT/"paper100_lambda_long200_summary.csv").open("w",newline="") as h: w=csv.DictWriter(h,F);w.writeheader();w.writerows(rows)
 ranked=sorted((x for x in rows if x["score"]),key=lambda x:float(x["score"]),reverse=True)
 lines=["# Paper100 lambda long200 summary",""]+[f"{i+1}. {x['method']}: score={x['score']}" for i,x in enumerate(ranked)]
 if not ranked: lines.append("No completed runs yet.")
 (OUT/"paper100_lambda_long200_summary.md").write_text("\n".join(lines)+"\n")
if __name__=="__main__": main()

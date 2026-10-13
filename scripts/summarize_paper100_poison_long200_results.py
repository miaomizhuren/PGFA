#!/usr/bin/env python3
from pathlib import Path
import csv,sqlite3,yaml
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"experiments/summaries/paper100_poison_long200";R=range(100,201,10);N=[f"p100pl_poison_{x:03d}" for x in (30,40,50,60,70,80)]
F=["experiment_id","method","status","config_path","source_result","num_clients","clients_per_round","participation_rate","local_epochs","attack_start","attack_end","total_rounds","feature_layer","poison_ratio","lambda_delta","lambda_train","epsilon","steps","alpha","client_lr",*[f"CleanACC@{r}" for r in R],*[f"ASR@{r}" for r in R],"PSNR","SSIM","L_inf","AUC@110-150","AUC@110-200","AUC@attack_end-200","Half-life","score","note"]
def f(x,d=2):return "" if x is None else f"{float(x):.{d}f}"
def auc(d,e):
 p=[(r,d[r][1])for r in sorted(d)if 110<=r<=e]
 return None if len(p)<2 or p[0][0]!=110 or p[-1][0]!=e else sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(p,p[1:]))/(e-110)
def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=[]
 for n in N:
  path=ROOT/"configs/paper100_poison_long200"/f"{n}.yaml";c=yaml.safe_load(path.read_text());db=ROOT/c["experiment"]["output_dir"]/c["tracking"]["database_name"];src=""
  if n.endswith("070"): db=ROOT/"outputs/paper100_lambda_long200/p100ll_lambda_12_07_strong/metrics_p100ll_lambda_12_07_strong.db";src="paper100_lambda_long200 Lambda 12/7"
  row={k:""for k in F};a=c["attack"];ad=a["trigger"]["adversarial"];p=ad["prototype_contrastive"];row.update(experiment_id=n,method=f"Poison {a['poison_ratio']:.2f}",status="not_completed",config_path=str(path.relative_to(ROOT)),source_result=src,num_clients="100",clients_per_round="10",participation_rate="10%",local_epochs="3",attack_start="60",attack_end="110",total_rounds="200",feature_layer="layer4",poison_ratio=f(a["poison_ratio"]),lambda_delta="12.00",lambda_train="7.00",epsilon="0.10",steps="100",alpha="0.005",client_lr="0.0100")
  if not db.exists():row["note"]=f"missing metrics DB: {db.relative_to(ROOT)}";rows.append(row);continue
  with sqlite3.connect(db)as con:z=con.execute("select round_index,clean_accuracy,attack_success_rate,psnr,ssim,linf from round_metrics").fetchall()
  d={r:(ca,asr,ps,ss,li)for r,ca,asr,ps,ss,li in z};last=max(d,default=0);row["status"]="complete"if last>=200 else"not_completed";row["note"]="reused equivalent long200 result"if src and last>=200 else f"complete through Round {last}"
  for r in R:
   if r in d:row[f"CleanACC@{r}"]=f(d[r][0]);row[f"ASR@{r}"]=f(d[r][1])
  v=d[max(d)];a150=auc(d,150);a200=auc(d,200);row.update(PSNR=f(v[2]),SSIM=f(v[3],4),L_inf=f(v[4],4),**{"AUC@110-150":f(a150),"AUC@110-200":f(a200),"AUC@attack_end-200":f(a200),"Half-life":next((str(r)for r in range(120,201,10)if d[r][1]<=.5*d[110][1]),">200")})
  row["score"]=f(.35*d[200][1]+.25*a200+.20*d[150][1]+.10*d[200][0]+.10*min(v[2],35));rows.append(row)
 with(OUT/"paper100_poison_long200_summary.csv").open("w",newline="")as h:w=csv.DictWriter(h,F);w.writeheader();w.writerows(rows)
 (OUT/"paper100_poison_long200_summary.md").write_text("# Paper100 poison long200 summary\n")
if __name__=="__main__":main()

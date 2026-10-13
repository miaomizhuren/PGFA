#!/usr/bin/env python3
from pathlib import Path
import csv,sqlite3,yaml
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"experiments/summaries/paper100_final310";R=(100,110,120,130,140,150,160,170,180,190,200,220,240,260,280,300,310)
F=["experiment_id","group","method","status","config_path","source_result","num_clients","clients_per_round","participation_rate","local_epochs","attack_start","attack_end","total_rounds","final_round","feature_layer","poison_ratio","lambda_delta","lambda_train","epsilon","steps","alpha","client_lr",*[f"CleanACC@{r}"for r in R],*[f"ASR@{r}"for r in R],"CleanACC@final","ASR@final","PSNR","SSIM","L_inf","AUC@110-150","AUC@110-200","AUC@110-310","AUC@110-final","AUC@attack_end-310","Half-life","note"]
def f(x,d=2):return ""if x is None else f"{float(x):.{d}f}"
def db(c):
 out=ROOT/c["experiment"]["output_dir"];n=c["experiment"]["name"];x=out/c["tracking"]["database_name"]
 if x.exists():return x
 z=[p for p in out.rglob("*")if p.suffix in(".db",".sqlite")]+[p for p in(ROOT/"outputs/paper100_final310").rglob("*")if p.suffix in(".db",".sqlite")and n in str(p)]
 return next(iter(sorted(z,key=lambda p:p.stat().st_mtime,reverse=True)),None)
def au(d,end):
 p=[(r,d[r][1])for r in sorted(d)if 110<=r<=end]
 return None if len(p)<2 or p[0][0]!=110 or p[-1][0]!=end else sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(p,p[1:]))/(end-110)
def main():
 OUT.mkdir(parents=True,exist_ok=True);epoch={r["experiment_id"]:r for r in csv.DictReader((ROOT/"experiments/summaries/paper100_epoch_final310/paper100_epoch_final310_summary.csv").open())};rows=[];reuse={"p100f_main_04_proto_uap_layer4_proposed":"p100ef_epoch_3","p100f_lambda_12_07_proposed":"p100ef_epoch_3","p100f_poison_060":"p100ef_epoch_3","p100f_ablation_delta_train":"p100ef_epoch_3","p100f_epoch_1":"p100ef_epoch_1","p100f_epoch_2":"p100ef_epoch_2","p100f_epoch_3":"p100ef_epoch_3","p100f_epoch_4":"p100ef_epoch_4","p100f_feat_layer4":"p100ef_epoch_3"}
 for path in sorted((ROOT/"configs/paper100_final310").rglob("*.yaml")):
  c=yaml.safe_load(path.read_text());n=c["experiment"]["name"];g=path.parent.name;a=c["attack"];ad=a["trigger"].get("adversarial",{});p=ad.get("prototype_contrastive",{});row={k:""for k in F};row.update(experiment_id=n,group=g,method=n,status="not_completed",config_path=str(path.relative_to(ROOT)),num_clients=str(c["partition"]["num_clients"]),clients_per_round=str(c["federated"]["clients_per_round"]),participation_rate="10%",local_epochs=str(c["federated"]["local_epochs"]),attack_start="60",attack_end="110",total_rounds=str(c["federated"]["num_rounds"]),feature_layer=p.get("feature_layer","patch"),poison_ratio=f(a["poison_ratio"]),lambda_delta=f(p.get("lambda_delta")),lambda_train=f(p.get("lambda_train")),epsilon=f(ad.get("epsilon")),steps=str(ad.get("steps","")),alpha=f(ad.get("alpha"),3),client_lr=f(c["training"]["learning_rate"],4))
  src=reuse.get(n)
  if src and src in epoch:
   row.update({k:v for k,v in epoch[src].items() if k in row});row.update(experiment_id=n,group=g,config_path=str(path.relative_to(ROOT)),source_result=f"paper100_epoch_final310 {src}",status=epoch[src]["status"],note="reused equivalent completed result");rows.append(row);continue
  x=db(c)
  if not x:row["note"]="awaiting final310 metrics DB";print(n+": DB not found");rows.append(row);continue
  print(n+": using DB "+str(x.relative_to(ROOT)))
  with sqlite3.connect(x)as con:z=con.execute("select round_index,clean_accuracy,attack_success_rate,psnr,ssim,linf from round_metrics").fetchall()
  d={r:(ca,asr,ps,ss,li)for r,ca,asr,ps,ss,li in z};last=max(d,default=0);row["final_round"]=str(last);row["status"]="complete"if last>=c["federated"]["num_rounds"]else"not_completed";row["note"]=f"DB: {x.relative_to(ROOT)}"
  for r in R:
   if r in d:row[f"CleanACC@{r}"]=f(d[r][0]);row[f"ASR@{r}"]=f(d[r][1])
  if d:
   v=d[last];row.update(**{"CleanACC@final":f(v[0]),"ASR@final":f(v[1]),"PSNR":f(v[2]),"SSIM":f(v[3],4),"L_inf":f(v[4],4),"AUC@110-150":f(au(d,150)),"AUC@110-200":f(au(d,200)),"AUC@110-310":f(au(d,310)),"AUC@110-final":f(au(d,last)),"AUC@attack_end-310":f(au(d,310)),"Half-life":next((str(r)for r in R if r>110 and d.get(r)and d[r][1]<=d[110][1]/2),f">{last}")})
  rows.append(row)
 by_id={r["experiment_id"]:r for r in rows}
 for target,source in {"p100f_feat_layer3":"p100f_main_03_proto_uap_layer3","p100f_lambda_00_00_ce_only":"p100f_main_02_ce_only_uap","p100f_ablation_ce_only":"p100f_main_02_ce_only_uap"}.items():
  if target in by_id and source in by_id and by_id[source]["status"]=="complete":
   target_row,source_row=by_id[target],by_id[source]
   for key in F:
    if key.startswith(("CleanACC@","ASR@","AUC@")) or key in ("PSNR","SSIM","L_inf","Half-life","final_round","CleanACC@final","ASR@final"):
     target_row[key]=source_row[key]
   target_row["status"]="complete";target_row["source_result"]=source;target_row["note"]="reused equivalent completed result"
 for r in rows:
  if r["status"]=="complete" and not r["final_round"]:
   r["final_round"]=r["total_rounds"];r["CleanACC@final"]=r.get(f"CleanACC@{r['final_round']}","");r["ASR@final"]=r.get(f"ASR@{r['final_round']}","");r["AUC@110-final"]=r.get("AUC@110-310","")
 with(OUT/"paper100_final310_summary_all.csv").open("w",newline="")as h:w=csv.DictWriter(h,F);w.writeheader();w.writerows(rows)
 for g in("main","feature_layer","lambda","poison","local_epochs","ablation"):
  q=[r for r in rows if r["group"]==g]
  with(OUT/f"paper100_final310_{g}_summary.csv").open("w",newline="")as h:w=csv.DictWriter(h,F);w.writeheader();w.writerows(q)
  (OUT/f"paper100_final310_{g}_summary.md").write_text(f"# {g}\n")
 (OUT/"paper100_final310_summary_all.md").write_text("# Paper100 final310 summary\n")
if __name__=="__main__":main()

#!/usr/bin/env python3
"""Summarize epoch-final310 runs with configuration-first DB discovery."""
from __future__ import annotations
import csv, sqlite3
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"experiments/summaries/paper100_epoch_final310"
ROUNDS=(100,110,120,130,140,150,160,170,180,190,200,220,240,260,280,300,310)
FIELDS=["experiment_id","method","status","config_path","num_clients","clients_per_round","participation_rate","local_epochs","attack_start","attack_end","total_rounds","feature_layer","poison_ratio","lambda_delta","lambda_train","epsilon","steps","alpha","client_lr",*[f"CleanACC@{r}" for r in ROUNDS],*[f"ASR@{r}" for r in ROUNDS],"PSNR","SSIM","L_inf","AUC@110-150","AUC@110-200","AUC@110-310","AUC@attack_end-310","Half-life","elapsed_seconds","score_no_time","score_with_time","note"]
def fmt(v,d=2): return "" if v is None else f"{float(v):.{d}f}"
def valid(path):
 try:
  with sqlite3.connect(path) as con: return "round_metrics" in {x[0] for x in con.execute("select name from sqlite_master where type='table'")}
 except sqlite3.Error: return False
def find_db(cfg):
 out=ROOT/cfg["experiment"]["output_dir"]; named=out/cfg["tracking"].get("database_name","metrics.db")
 name=cfg["experiment"]["name"]
 candidates=[named] if named.exists() else []
 candidates += [p for p in out.rglob("*") if p.is_file() and p.suffix.lower() in (".db",".sqlite")]
 root=ROOT/"outputs/paper100_epoch_final310"
 candidates += [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".db",".sqlite") and name in str(p)]
 candidates=sorted({p.resolve() for p in candidates if valid(p)},key=lambda p:p.stat().st_mtime,reverse=True)
 return (candidates[0] if candidates else None), named, out
def auc(data,end):
 pts=[(r,data[r]["asr"]) for r in sorted(data) if 110<=r<=end and data[r]["asr"] is not None]
 if len(pts)<2 or pts[0][0]!=110 or pts[-1][0]!=end:return None
 return sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(pts,pts[1:]))/(end-110)
def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=[]
 for e in range(1,6):
  name="p100ef_epoch_5_optional" if e==5 else f"p100ef_epoch_{e}"; path=ROOT/"configs/paper100_epoch_final310"/f"{name}.yaml"; cfg=yaml.safe_load(path.read_text());a=cfg["attack"];ad=a["trigger"]["adversarial"];pc=ad["prototype_contrastive"]
  row={f:"" for f in FIELDS};row.update(experiment_id=name,method=f"Epoch {e}"+(" optional" if e==5 else ""),status="not_completed",config_path=str(path.relative_to(ROOT)),num_clients=str(cfg["partition"]["num_clients"]),clients_per_round=str(cfg["federated"]["clients_per_round"]),participation_rate="10%",local_epochs=str(e),attack_start=str(a["attack_start_round"]),attack_end=str(a["attack_end_round"]),total_rounds=str(cfg["federated"]["num_rounds"]),feature_layer=pc["feature_layer"],poison_ratio=fmt(a["poison_ratio"]),lambda_delta=fmt(pc["lambda_delta"]),lambda_train=fmt(pc["lambda_train"]),epsilon=fmt(ad["epsilon"]),steps=str(ad["steps"]),alpha=fmt(ad["alpha"],3),client_lr=fmt(cfg["training"]["learning_rate"],4))
  db,named,out=find_db(cfg)
  if not db:
   row["note"]=("optional not run; " if e==5 else "")+f"missing metrics DB; searched {named.relative_to(ROOT)} and recursive .db/.sqlite under {out.relative_to(ROOT)} and outputs/paper100_epoch_final310";print(f"{name}: DB not found");rows.append(row);continue
  print(f"{name}: using DB {db.relative_to(ROOT)}")
  with sqlite3.connect(db) as con: rec=con.execute("select round_index,clean_accuracy,attack_success_rate,psnr,ssim,linf from round_metrics order by round_index").fetchall()
  data={r:{"clean":c,"asr":s,"psnr":p,"ssim":ss,"linf":li} for r,c,s,p,ss,li in rec};last=max(data,default=0);row["status"]="complete" if last>=310 else "not_completed";row["note"]=f"complete through Round {last}; DB: {db.relative_to(ROOT)}" if last>=310 else f"last recorded round: {last}; DB: {db.relative_to(ROOT)}"
  for r in ROUNDS: row[f"CleanACC@{r}"]=fmt(data.get(r,{}).get("clean"));row[f"ASR@{r}"]=fmt(data.get(r,{}).get("asr"))
  lastv=data.get(last,{})
  a150,a200,a310=auc(data,150),auc(data,200),auc(data,310);row.update(PSNR=fmt(lastv.get("psnr")),SSIM=fmt(lastv.get("ssim"),4),L_inf=fmt(lastv.get("linf"),4),**{"AUC@110-150":fmt(a150),"AUC@110-200":fmt(a200),"AUC@110-310":fmt(a310),"AUC@attack_end-310":fmt(a310)})
  base=data.get(110,{}).get("asr");row["Half-life"]=next((str(r) for r in ROUNDS if r>110 and base is not None and data.get(r,{}).get("asr") is not None and data[r]["asr"]<=base/2),">310" if row["status"]=="complete" else "")
  if None not in (data.get(310,{}).get("asr"),a310,data.get(200,{}).get("asr"),data.get(310,{}).get("clean"),lastv.get("psnr")): row["score_no_time"]=fmt(.30*data[310]["asr"]+.25*a310+.20*data[200]["asr"]+.10*data[310]["clean"]+.10*min(lastv["psnr"],35));row["score_with_time"]=row["score_no_time"]
  rows.append(row)
 with (OUT/"paper100_epoch_final310_summary.csv").open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,FIELDS);w.writeheader();w.writerows(rows)
 (OUT/"paper100_epoch_final310_summary.md").write_text("# Paper100 local epochs final310\n\n"+"\n".join(f"- {r['method']}: {r['status']}; {r['note']}" for r in rows)+"\n",encoding="utf-8")
if __name__=="__main__":main()

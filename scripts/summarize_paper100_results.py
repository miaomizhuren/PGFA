#!/usr/bin/env python3
"""Summarize paper100 configurations and reuse completed q04 when applicable."""
from pathlib import Path
import csv, sqlite3, yaml
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"experiments/summaries/paper100"; R=range(100,151,10)
SPECS=[("main","Fixed Patch","main/p100_main_01_fixed_patch"),("main","CE-only UAP","main/p100_main_02_ce_only_uap"),("main","Proto-UAP Layer3","main/p100_main_03_proto_uap_layer3"),("main","Proto-UAP Layer4 Proposed","main/p100_main_04_proto_uap_layer4_proposed"),("feature_layer","Layer3","feature_layer/p100_feat_layer3"),("feature_layer","Layer4","feature_layer/p100_feat_layer4"),("lambda","CE-only","lambda/p100_lambda_00_ce_only"),("lambda","Lambda 6/3","lambda/p100_lambda_06_03"),("lambda","Lambda 9/5","lambda/p100_lambda_09_05"),("lambda","Lambda 12/7","lambda/p100_lambda_12_07"),("poison","Poison 0.50","poison/p100_poison_050"),("poison","Poison 0.60","poison/p100_poison_060"),("poison","Poison 0.70","poison/p100_poison_070"),("poison","Poison 0.80","poison/p100_poison_080"),("local_epochs","Epoch 1","local_epochs/p100_epoch_1"),("local_epochs","Epoch 2","local_epochs/p100_epoch_2"),("local_epochs","Epoch 3","local_epochs/p100_epoch_3"),("ablation","CE-only","ablation/p100_ablation_ce_only"),("ablation","Delta only","ablation/p100_ablation_delta_only"),("ablation","Train only","ablation/p100_ablation_train_only"),("ablation","Delta + Train","ablation/p100_ablation_delta_train")]
F=["experiment_id","group","method","status","config_path","source_result","num_clients","clients_per_round","participation_rate","local_epochs","attack_start","attack_end","total_rounds","feature_layer","poison_ratio","lambda_delta","lambda_train","epsilon","steps","alpha","client_lr",*[f"CleanACC@{r}" for r in R],*[f"ASR@{r}" for r in R],"PSNR","SSIM","L_inf","AUC@110-150","AUC@attack_end-150","Half-life","score","note"]
def fmt(x,d=2): return "" if x is None else f"{float(x):.{d}f}"
def metrics(db):
 if not db.exists(): return {},0
 with sqlite3.connect(db) as c: z=c.execute("select round_index,clean_accuracy,attack_success_rate,psnr,ssim,linf from round_metrics").fetchall()
 return {r:(a,b,p,s,l) for r,a,b,p,s,l in z},max((x[0] for x in z),default=0)
def main():
 OUT.mkdir(parents=True,exist_ok=True); rows=[]
 for group,method,rel in SPECS:
  path=ROOT/"configs/paper100"/(rel+".yaml"); cfg=yaml.safe_load(path.read_text()); a=cfg["attack"]; adv=a["trigger"].get("adversarial",{}); pc=adv.get("prototype_contrastive",{})
  row={k:"" for k in F}; row.update(experiment_id=cfg["experiment"]["name"],group=group,method=method,status="not_completed",config_path=str(path.relative_to(ROOT)),num_clients=str(cfg["partition"]["num_clients"]),clients_per_round=str(cfg["federated"]["clients_per_round"]),participation_rate=f"{100*cfg['federated']['clients_per_round']/cfg['partition']['num_clients']:.0f}%",local_epochs=str(cfg["federated"]["local_epochs"]),attack_start=str(a["attack_start_round"]),attack_end=str(a["attack_end_round"]),total_rounds=str(cfg["federated"]["num_rounds"]),feature_layer=pc.get("feature_layer","patch"),poison_ratio=fmt(a["poison_ratio"]),lambda_delta=fmt(pc.get("lambda_delta")),lambda_train=fmt(pc.get("lambda_train")),epsilon=fmt(adv.get("epsilon")),steps=str(adv.get("steps","")),alpha=fmt(adv.get("alpha"),3),client_lr=fmt(cfg["training"]["learning_rate"],4))
  db=ROOT/cfg["experiment"]["output_dir"]/cfg["tracking"]["database_name"]; source=""
  if cfg["experiment"]["name"]=="p100_main_04_proto_uap_layer4_proposed" and (ROOT/"outputs/client_search_fast/csf_q04/metrics_csf_q04.db").exists(): db=ROOT/"outputs/client_search_fast/csf_q04/metrics_csf_q04.db"; source="client_search_fast q04"
  data,last=metrics(db); row["source_result"]=source
  row["status"]="complete" if last>=150 else "not_completed"; row["note"]=(f"reused completed q04 result" if source and last>=150 else f"complete through Round {last}" if last>=150 else f"missing/incomplete metrics DB: {db.relative_to(ROOT)}")
  for r in R:
   if r in data: row[f"CleanACC@{r}"]=fmt(data[r][0]); row[f"ASR@{r}"]=fmt(data[r][1])
  if data:
   z=data[max(data)]; row.update(PSNR=fmt(z[2]),SSIM=fmt(z[3],4),L_inf=fmt(z[4],4))
  rows.append(row)
 with (OUT/"paper100_summary_all.csv").open("w",newline="") as h: w=csv.DictWriter(h,F);w.writeheader();w.writerows(rows)
 for group in ("main","feature_layer","lambda","poison","local_epochs","ablation"):
  sub=[r for r in rows if r["group"]==group]
  with (OUT/f"paper100_{group}_summary.csv").open("w",newline="") as h: w=csv.DictWriter(h,F);w.writeheader();w.writerows(sub)
  (OUT/f"paper100_{group}_summary.md").write_text(f"# Paper100 {group}\n\n{len(sub)} configurations.\n")
 (OUT/"paper100_summary_all.md").write_text("# Paper100 summary\n\nGenerated non-destructively.\n")
if __name__=="__main__": main()

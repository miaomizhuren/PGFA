#!/usr/bin/env python3
"""Non-destructively summarize client-search-fast results."""
from __future__ import annotations
import csv
import sqlite3
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/summaries/client_search_fast"
ROUNDS = tuple(range(100, 151, 10))
FIELDS = ["qid","status","config_path","local_epochs","attack_start","attack_end","poison_ratio","lambda_delta","lambda_train","epsilon","steps","alpha","client_lr",*[f"CleanACC@{r}" for r in ROUNDS],*[f"ASR@{r}" for r in ROUNDS],"PSNR","SSIM","L_inf","AUC@110-150","AUC@attack_end-150","Half-life","score","note"]

def fmt(value, digits=2): return "" if value is None else f"{float(value):.{digits}f}"
def auc(data, start, end):
    pts=[(r,data[r]["asr"]) for r in sorted(data) if start<=r<=end and data[r]["asr"] is not None]
    if len(pts)<2 or pts[0][0]!=start or pts[-1][0]!=end: return None
    return sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(pts,pts[1:]))/(end-start)
def latest(data, key):
    return next((data[r][key] for r in sorted(data,reverse=True) if data[r][key] is not None),None)

def make_row(index):
    qid=f"q{index:02d}"; relative=f"configs/client_search_fast/csf_{qid}.yaml"
    row={field:"" for field in FIELDS}; row.update(qid=qid,status="not_completed",config_path=relative)
    path=ROOT/relative
    if not path.exists(): row["note"]=f"missing config: {relative}"; return row
    cfg=yaml.safe_load(path.read_text(encoding="utf-8")); attack=cfg["attack"]; fed=cfg["federated"]; adv=attack["trigger"]["adversarial"]; proto=adv["prototype_contrastive"]
    row.update(local_epochs=str(fed["local_epochs"]),attack_start=str(attack["attack_start_round"]),attack_end=str(attack["attack_end_round"]),poison_ratio=fmt(attack["poison_ratio"]),lambda_delta=fmt(proto["lambda_delta"]),lambda_train=fmt(proto["lambda_train"]),epsilon=fmt(adv["epsilon"]),steps=str(adv["steps"]),alpha=fmt(adv["alpha"],3),client_lr=fmt(cfg["training"]["learning_rate"],4))
    db=ROOT/cfg["experiment"]["output_dir"]/cfg["tracking"]["database_name"]
    if not db.exists(): row["note"]=f"not completed; missing metrics DB: {db.relative_to(ROOT)}"; return row
    try:
        with sqlite3.connect(db) as con: records=con.execute("SELECT round_index, clean_accuracy, attack_success_rate, psnr, ssim, linf FROM round_metrics ORDER BY round_index").fetchall()
    except sqlite3.Error as error: row["note"]=f"sqlite error: {error}"; return row
    data={r:{"clean":clean,"asr":asr,"psnr":psnr,"ssim":ssim,"linf":linf} for r,clean,asr,psnr,ssim,linf in records}; last=max(data,default=0)
    row["status"]="complete" if last>=fed["num_rounds"] else "not_completed"; row["note"]=f"complete through Round {last}" if row["status"]=="complete" else f"not completed; last recorded round: {last}"
    row.update({f"CleanACC@{r}":fmt(data.get(r,{}).get("clean")) for r in ROUNDS}); row.update({f"ASR@{r}":fmt(data.get(r,{}).get("asr")) for r in ROUNDS})
    a110=auc(data,110,150); post=auc(data,attack["attack_end_round"],150); psnr=latest(data,"psnr")
    row.update(PSNR=fmt(psnr),SSIM=fmt(latest(data,"ssim"),4),L_inf=fmt(latest(data,"linf"),4),**{"AUC@110-150":fmt(a110),"AUC@attack_end-150":fmt(post),"Half-life":next((str(r) for r in sorted(data) if r>=attack["attack_end_round"] and data[r]["asr"] is not None and data[r]["asr"]<50),">150" if row["status"]=="complete" else "")})
    acc=data.get(150,{}).get("clean"); asr=data.get(150,{}).get("asr")
    if None not in (acc,asr,post,psnr): row["score"]=fmt(.45*asr+.25*post+.20*acc+.10*min(psnr,35))
    return row

def main():
    OUT.mkdir(parents=True,exist_ok=True); rows=[make_row(i) for i in range(1,25)]
    with (OUT/"client_search_fast_summary.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    ranked=sorted((r for r in rows if r["score"]),key=lambda r:float(r["score"]),reverse=True)[:10]
    lines=["# Client-search-fast summary","","## Top 10 by score",""]
    if ranked:
        lines+=["| Rank | QID | Score | CleanACC@150 | ASR@150 | AUC@attack_end-150 |","|---:|---|---:|---:|---:|---:|"]
        lines += [f"| {i} | {r['qid']} | {r['score']} | {r['CleanACC@150']} | {r['ASR@150']} | {r['AUC@attack_end-150']} |" for i,r in enumerate(ranked,1)]
    else: lines.append("No completed configurations yet.")
    lines += ["",f"Completed: {sum(r['status']=='complete' for r in rows)}/24."]
    (OUT/"client_search_fast_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
if __name__=="__main__": main()

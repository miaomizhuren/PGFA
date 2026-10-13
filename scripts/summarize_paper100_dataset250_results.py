#!/usr/bin/env python3
"""Summarize real dataset250 SQLite metrics without inferring missing rounds."""
import csv
import sqlite3
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/paper100_dataset250"
OUT = ROOT / "experiments/summaries/paper100_dataset250"
ROUNDS = (110, 150, 180, 200, 240, 250)
FIELDS = ["experiment_id","dataset","method","status","config_path","num_classes","num_clients","clients_per_round","participation_rate","local_epochs","partition","seed","target_label","attack_start","attack_end","total_rounds","final_round","feature_layer","poison_ratio","lambda_delta","lambda_train","epsilon","steps","alpha","client_lr",*[f"CleanACC@{r}" for r in ROUNDS],*[f"ASR@{r}" for r in ROUNDS],"CleanACC@final","ASR@final","PSNR","SSIM","L_inf","AUC@110-150","AUC@110-200","AUC@110-250","Half-life","note"]

def fmt(value, digits=2):
    return "" if value is None else f"{float(value):.{digits}f}"

def read_metrics(db):
    if not db.is_file():
        return {}, None
    try:
        with sqlite3.connect(db) as con:
            rows = con.execute("SELECT round_index, clean_accuracy, attack_success_rate, psnr, ssim, linf FROM round_metrics ORDER BY round_index").fetchall()
        return {r: values for r, *values in rows}, db
    except sqlite3.Error:
        return {}, None

def auc(data, endpoint):
    points = [(r, data[r][1]) for r in sorted(data) if 110 <= r <= endpoint and data[r][1] is not None]
    if len(points) < 2 or points[0][0] != 110 or points[-1][0] != endpoint:
        return None
    return sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(points, points[1:]))/(endpoint-110)

def make_row(config_path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    e, d, a = config["experiment"], config["dataset"], config["attack"]
    ad = a["trigger"]["adversarial"]; proto = ad.get("prototype_contrastive", {})
    row = {key: "" for key in FIELDS}
    row.update(experiment_id=e["name"], dataset=d["name"], method=a.get("method",""),
        config_path=str(config_path.relative_to(ROOT)), num_classes=d["num_classes"],
        num_clients=config["partition"]["num_clients"], clients_per_round=config["federated"]["clients_per_round"],
        participation_rate=f"{100*config['federated']['clients_per_round']/config['partition']['num_clients']:.0f}%",
        local_epochs=config["federated"]["local_epochs"], partition=config["partition"].get("type","iid"),
        seed=e["seed"], target_label=a["target_label"], attack_start=a["attack_start_round"],
        attack_end=a["attack_end_round"], total_rounds=config["federated"]["num_rounds"],
        feature_layer=proto.get("feature_layer",""), poison_ratio=fmt(a["poison_ratio"]),
        lambda_delta=fmt(proto.get("lambda_delta")), lambda_train=fmt(proto.get("lambda_train")),
        epsilon=fmt(ad.get("epsilon")), steps=ad.get("steps",""), alpha=fmt(ad.get("alpha"),3),
        client_lr=fmt(config["training"]["learning_rate"],4))
    db = ROOT / e["output_dir"] / config.get("tracking",{}).get("database_name","metrics.db")
    metrics, used = read_metrics(db); final_round = max(metrics, default=0)
    row["final_round"] = final_round or ""
    for round_index in ROUNDS:
        if round_index in metrics:
            row[f"CleanACC@{round_index}"], row[f"ASR@{round_index}"] = fmt(metrics[round_index][0]), fmt(metrics[round_index][1])
    if metrics:
        latest = metrics[final_round]
        row.update(**{"CleanACC@final":fmt(latest[0]), "ASR@final":fmt(latest[1]), "PSNR":fmt(latest[2]), "SSIM":fmt(latest[3],4), "L_inf":fmt(latest[4],4), "AUC@110-150":fmt(auc(metrics,150)), "AUC@110-200":fmt(auc(metrics,200)), "AUC@110-250":fmt(auc(metrics,250)), "Half-life":next((str(r) for r in sorted(metrics) if r > 110 and metrics.get(110,(None,None))[1] is not None and metrics[r][1] is not None and metrics[r][1] <= metrics[110][1]/2), ">250" if final_round >= 250 else "")})
    row["status"] = "complete" if final_round >= 250 and 250 in metrics and metrics[250][0] is not None and metrics[250][1] is not None else "not_completed"
    row["note"] = f"DB: {used.relative_to(ROOT)}" if used else "missing metrics DB"
    return row

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [make_row(path) for path in sorted(CFG.glob("*.yaml"))]
    for name in ("paper100_dataset250_summary_all.csv", "paper100_dataset250_table_for_paper.csv"):
        with (OUT/name).open("w", newline="", encoding="utf-8") as handle:
            writer=csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    lines=["# paper100_dataset250 summary","","Only real SQLite values are reported; absent/incomplete rounds remain blank.","",f"| experiment_id | dataset | method | status | final_round | ASR@250 | CleanACC@250 |","|---|---|---|---|---:|---:|---:|"]
    lines += [f"| {r['experiment_id']} | {r['dataset']} | {r['method']} | {r['status']} | {r['final_round'] or '—'} | {r['ASR@250'] or '—'} | {r['CleanACC@250'] or '—'} |" for r in rows]
    (OUT/"paper100_dataset250_summary_all.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(OUT/"paper100_dataset250_summary_all.csv")
if __name__ == "__main__":
    main()


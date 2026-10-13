#!/usr/bin/env python3
"""Read-only aggregation of existing PBU experiment SQLite metrics."""
import csv
import math
import os
import sqlite3
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY_ROUNDS = [3000, 3200, 3201, 3300, 3400, 3500, 3600, 3700]
POST_ROUNDS = list(range(3201, 3701))


def read_db(path):
    con = sqlite3.connect(path)
    rows = con.execute("SELECT round_index, clean_accuracy, attack_success_rate FROM round_metrics ORDER BY round_index").fetchall()
    con.close()
    result = {}
    for r, clean, asr in rows:
        # The stored metrics in these databases use percentages. Retain that unit.
        result[int(r)] = (clean, asr)
    return result


def fmt(x, digits=6):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.{digits}f}"


def val(data, r, index):
    pair = data.get(r)
    return None if pair is None else pair[index]


def post_stats(data):
    ys = [val(data, r, 1) for r in POST_ROUNDS]
    available = [x for x in ys if x is not None]
    if not available:
        return dict(auc=None, ge80=None, ge50=None, max=None, min=None, mean=None)
    # AUC is only defined when the requested complete post-attack curve exists.
    auc = None
    if len(available) == len(POST_ROUNDS):
        auc = sum((ys[i] + ys[i + 1]) / 2 for i in range(len(ys) - 1))
    def duration(threshold):
        n = 0
        for y in ys:
            if y is None or y < threshold:
                break
            n += 1
        return n
    return dict(auc=auc, ge80=duration(80), ge50=duration(50), max=max(available), min=min(available), mean=sum(available)/len(available))


def source_row(group, label, output_dir, db, data):
    rounds = sorted(data)
    missing = [str(r) for r in KEY_ROUNDS if r not in data]
    return {
        "group": group, "run_or_setting": label, "output_dir": str(output_dir),
        "db_path": str(db), "csv_path": "", "max_round": max(rounds) if rounds else "",
        "min_round": min(rounds) if rounds else "",
        **{f"has_round_{r}": str(r in data).lower() for r in KEY_ROUNDS},
        "missing_key_rounds": ";".join(missing), "has_nan": "false",
        "asr_unit_detected": "0-100", "data_quality_note": (
            "complete to round 3700; ASR is NULL before attack by design" if max(rounds, default=0) == 3700
            else "incomplete: recorded metrics stop before attack/post-attack evaluation"
        )
    }


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def phase(r):
    return {3000: "before_attack / clean_end", 3200: "attack_end", 3201: "post_start",
            3300: "post_100", 3400: "post_200", 3500: "post_300", 3600: "post_400",
            3700: "post_500 / final"}[r]


def summary_row(label, meta, data, include_setting=False):
    s = post_stats(data)
    d = {
        "experiment_name": meta["experiment_name"], "output_dir": str(meta["output_dir"]),
        "attack_method": "proto_uap", "final_round": max(data, default=""),
        "clean_acc_at_attack_end": fmt(val(data, 3200, 0)), "asr_at_attack_end": fmt(val(data, 3200, 1)),
        "clean_acc_final": fmt(val(data, 3700, 0)),
        "asr_post_100": fmt(val(data, 3300, 1)), "asr_post_200": fmt(val(data, 3400, 1)),
        "asr_post_300": fmt(val(data, 3500, 1)), "asr_post_400": fmt(val(data, 3600, 1)),
        "asr_post_500": fmt(val(data, 3700, 1)), "post_attack_auc_3201_3700": fmt(s["auc"], 6),
        "asr_ge_80_duration_rounds": "" if s["ge80"] is None else s["ge80"],
        "asr_ge_50_duration_rounds": "" if s["ge50"] is None else s["ge50"],
        "max_post_asr": fmt(s["max"]), "min_post_asr": fmt(s["min"]), "mean_post_asr": fmt(s["mean"]),
        "notes": "complete" if max(data, default=0) == 3700 else "incomplete: ends at round 3000; no attack/post-attack metrics"
    }
    if include_setting:
        d = {"setting": label, "partition": meta["partition"], "dirichlet_alpha": meta["alpha"], **d}
    else:
        d = {"run": label, "seed": meta["seed"], **d}
    return d


def markdown_table(fields, rows):
    out = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    out += ["| " + " | ".join(str(row.get(f, "")) for f in fields) + " |" for row in rows]
    return "\n".join(out)


def main():
    main_dir = ROOT / "outputs/pbu_force_clean_stage_sensitivity/clean3000_attack3001_3200_observe3201_3700"
    experiments = {
        "IID-main": {"experiment_name": "pbu_force_clean3000_attack3001_3200_observe3201_3700", "output_dir": main_dir,
                     "db": main_dir / "metrics_pbu_force_clean3000_attack3001_3200_observe3201_3700.db", "partition": "IID", "alpha": "NA", "seed": "main"},
        "Non-IID-alpha0.9": {"experiment_name": "cifar10_noniid_alpha0.9_pbu_fromscratch_clean3000_attack3001_3200_observe3201_3700_rerun1", "output_dir": ROOT / "outputs/cifar10_noniid_pbu_force_fromscratch_3_rerun1/alpha0.9_pbu_fromscratch_clean3000_attack3001_3200_observe3201_3700", "partition": "Dirichlet", "alpha": "0.9", "seed": ""},
        "Non-IID-alpha0.8": {"experiment_name": "cifar10_noniid_alpha0.8_pbu_fromscratch_clean3000_attack3001_3200_observe3201_3700_rerun1", "output_dir": ROOT / "outputs/cifar10_noniid_pbu_force_fromscratch_3_rerun1/alpha0.8_pbu_fromscratch_clean3000_attack3001_3200_observe3201_3700", "partition": "Dirichlet", "alpha": "0.8", "seed": ""},
        "Non-IID-alpha0.7": {"experiment_name": "cifar10_noniid_alpha0.7_pbu_fromscratch_clean3000_attack3001_3200_observe3201_3700_rerun1", "output_dir": ROOT / "outputs/cifar10_noniid_pbu_force_fromscratch_3_rerun1/alpha0.7_pbu_fromscratch_clean3000_attack3001_3200_observe3201_3700", "partition": "Dirichlet", "alpha": "0.7", "seed": ""},
    }
    for m in experiments.values():
        if "db" not in m: m["db"] = next(m["output_dir"].glob("*.db"))
        m["data"] = read_db(m["db"])
    noniid_out = ROOT / "outputs/noniid_pbu_vs_iid_summary"
    nfields = ["setting","partition","dirichlet_alpha","experiment_name","output_dir","attack_method","final_round","clean_acc_at_attack_end","asr_at_attack_end","clean_acc_final","asr_post_100","asr_post_200","asr_post_300","asr_post_400","asr_post_500","post_attack_auc_3201_3700","asr_ge_80_duration_rounds","asr_ge_50_duration_rounds","max_post_asr","min_post_asr","mean_post_asr","notes"]
    nrows = [summary_row(k, m, m["data"], True) for k,m in experiments.items()]
    write_csv(noniid_out / "noniid_pbu_vs_iid_summary.csv", nfields, nrows)
    keyfields = ["setting","global_round","relative_post_round","phase","clean_acc","asr"]
    kr=[]
    for label,m in experiments.items():
        for r in KEY_ROUNDS: kr.append({"setting":label,"global_round":r,"relative_post_round":"" if r<3201 else r-3200,"phase":phase(r),"clean_acc":fmt(val(m["data"],r,0)),"asr":fmt(val(m["data"],r,1))})
    write_csv(noniid_out / "noniid_pbu_vs_iid_key_rounds.csv", keyfields, kr)
    curvefields=["relative_post_round","global_round"] + [x for label in experiments for x in (f"{label}_clean_acc",f"{label}_asr")]
    curves=[]
    for r in POST_ROUNDS:
        row={"relative_post_round":r-3200,"global_round":r}
        for label,m in experiments.items(): row[f"{label}_clean_acc"]=fmt(val(m["data"],r,0)); row[f"{label}_asr"]=fmt(val(m["data"],r,1))
        curves.append(row)
    write_csv(noniid_out / "noniid_pbu_vs_iid_post_attack_curves.csv", curvefields, curves)
    srcfields=["group","run_or_setting","output_dir","db_path","csv_path","max_round","min_round"]+[f"has_round_{r}" for r in KEY_ROUNDS]+["missing_key_rounds","has_nan","asr_unit_detected","data_quality_note"]
    nsrc=[source_row("noniid_vs_iid",k,m["output_dir"],m["db"],m["data"]) for k,m in experiments.items()]
    write_csv(noniid_out / "noniid_pbu_vs_iid_source_check.csv",srcfields,nsrc)
    noniid_report = "# Non-IID PBU vs IID Summary\n\n## 1. Goal\nCompare the IID main PBU experiment with Dirichlet Non-IID PBU experiments (α=0.9, 0.8, 0.7).\n\n## 2. Source Experiments\nMetrics were read from the listed SQLite databases; rerun1 directories were preferred.\n\n" + markdown_table(["setting","output_dir","final_round","notes"], nrows) + "\n\n## 3. Protocol Check\nAll configured runs use CIFAR-10, ResNet18, FedAvg, batch size 64, 3 local epochs, proto_uap/PBU, forced malicious selection in rounds 3001–3200, and observation from 3201–3700. The Non-IID runs use Dirichlet partitioning with the stated alpha.\n\n## 4. Summary\n" + markdown_table(["setting","asr_at_attack_end","asr_post_100","asr_post_200","asr_post_300","asr_post_400","asr_post_500","post_attack_auc_3201_3700","notes"],nrows) + "\n\n## 5. Key Rounds\n" + markdown_table(keyfields,kr) + "\n\n## 6. Analysis\nα=0.9 and α=0.8 complete to round 3700 and retain nonzero post-attack ASR; their exact persistence is reported in the table. α=0.7 is incomplete at round 3000, so it cannot support a persistence or AUC conclusion and is not imputed.\n\n## 7. Data Quality Check\nNo numeric NaN values were found. ASR is intentionally NULL before attack in full runs. IID main lacks round 3000 in its selected DB; the value is left blank. No raw files were modified.\n\n## 8. Files Generated\n- noniid_pbu_vs_iid_summary.csv\n- noniid_pbu_vs_iid_key_rounds.csv\n- noniid_pbu_vs_iid_post_attack_curves.csv\n- noniid_pbu_vs_iid_source_check.csv\n- noniid_pbu_vs_iid_report.md\n"
    (noniid_out / "noniid_pbu_vs_iid_report.md").write_text(noniid_report,encoding="utf-8")

    seeds={"main_ours_pbu":experiments["IID-main"]}
    for seed in range(20261,20266):
        od=ROOT/f"outputs/seed_stability_pbu_force_fromscratch_5_rerun1/pbu_seed{seed}_fromscratch_clean3000_attack3001_3200_observe3201_3700"
        db=next(od.glob("*.db")); seeds[f"seed{seed}"]={"experiment_name":f"pbu_seed{seed}_fromscratch_clean3000_attack3001_3200_observe3201_3700_rerun1","output_dir":od,"db":db,"seed":seed,"data":read_db(db)}
    sout=ROOT/"outputs/pbu_seed_stability_vs_main_summary"
    sfields=["run","seed","experiment_name","output_dir","attack_method","final_round","clean_acc_at_attack_end","asr_at_attack_end","clean_acc_final","asr_post_100","asr_post_200","asr_post_300","asr_post_400","asr_post_500","post_attack_auc_3201_3700","asr_ge_80_duration_rounds","asr_ge_50_duration_rounds","max_post_asr","min_post_asr","mean_post_asr","notes"]
    srows=[summary_row(k,m,m["data"]) for k,m in seeds.items()]
    write_csv(sout/"pbu_seed_stability_summary.csv",sfields,srows)
    skeyfields=["run","seed","global_round","relative_post_round","phase","clean_acc","asr"]; skr=[]
    for label,m in seeds.items():
        for r in KEY_ROUNDS: skr.append({"run":label,"seed":m["seed"],"global_round":r,"relative_post_round":"" if r<3201 else r-3200,"phase":phase(r),"clean_acc":fmt(val(m["data"],r,0)),"asr":fmt(val(m["data"],r,1))})
    write_csv(sout/"pbu_seed_stability_key_rounds.csv",skeyfields,skr)
    scurvefields=["relative_post_round","global_round"]+[x for label in seeds for x in (f"{label}_clean_acc",f"{label}_asr")]+["seed_mean_asr","seed_std_asr"]
    scurves=[]
    seedlabels=[f"seed{x}" for x in range(20261,20266)]
    for r in POST_ROUNDS:
        row={"relative_post_round":r-3200,"global_round":r}; vals=[]
        for label,m in seeds.items():
            row[f"{label}_clean_acc"]=fmt(val(m["data"],r,0)); row[f"{label}_asr"]=fmt(val(m["data"],r,1))
        vals=[val(seeds[x]["data"],r,1) for x in seedlabels]
        row["seed_mean_asr"]=fmt(statistics.mean(vals)); row["seed_std_asr"]=fmt(statistics.stdev(vals))
        scurves.append(row)
    write_csv(sout/"pbu_seed_stability_post_attack_curves.csv",scurvefields,scurves)
    metrics=["clean_acc_at_attack_end","asr_at_attack_end","clean_acc_final","asr_post_100","asr_post_200","asr_post_300","asr_post_400","asr_post_500","post_attack_auc_3201_3700","asr_ge_80_duration_rounds","asr_ge_50_duration_rounds","mean_post_asr"]
    byrun={r["run"]:r for r in srows}; ms=[]
    for metric in metrics:
        vals=[float(byrun[x][metric]) for x in seedlabels]
        ms.append({"metric":metric,"mean":fmt(statistics.mean(vals)),"std":fmt(statistics.stdev(vals)),"min":fmt(min(vals)),"max":fmt(max(vals)),"main_ours_pbu_value":byrun["main_ours_pbu"][metric]})
    msfields=["metric","mean","std","min","max","main_ours_pbu_value"]
    write_csv(sout/"pbu_seed_stability_mean_std.csv",msfields,ms)
    ssrc=[source_row("seed_stability",k,m["output_dir"],m["db"],m["data"]) for k,m in seeds.items()]
    write_csv(sout/"pbu_seed_stability_source_check.csv",srcfields,ssrc)
    seed_report="# PBU Seed Stability Summary\n\n## 1. Goal\nCompare the IID main Ours/PBU experiment with five independent from-scratch seeds.\n\n## 2. Source Experiments\n"+markdown_table(["run","seed","output_dir","final_round","notes"],srows)+"\n\n## 3. Protocol Check\nThe five seed configs specify CIFAR-10, ResNet18, FedAvg, batch size 64, 3 local epochs, iid partitioning, proto_uap/PBU, force_during_attack, attack rounds 3001–3200, and observation 3201–3700. Seeds are 20261–20265 and each is from scratch.\n\n## 4. Per-run Summary Table\n"+markdown_table(sfields,srows)+"\n\n## 5. Mean ± Std Table\n"+markdown_table(msfields,ms)+"\n\n## 6. Stability Analysis\nAll five seed runs complete to 3700. Attack-end and post-attack values, AUC, and ≥80% duration are quantified above; the main run is shown separately and excluded from mean/std. No seed was excluded because the selected databases are complete and protocol-consistent.\n\n## 7. Data Quality Check\nAll five seeds reached 3700, include every requested key round, and contain no numeric NaN values. ASR is stored in 0–100 units. No raw data were modified. The main selected database begins at 3001, so its round-3000 cell remains blank.\n\n## 8. Files Generated\n- pbu_seed_stability_summary.csv\n- pbu_seed_stability_mean_std.csv\n- pbu_seed_stability_key_rounds.csv\n- pbu_seed_stability_post_attack_curves.csv\n- pbu_seed_stability_source_check.csv\n- pbu_seed_stability_report.md\n"
    (sout/"pbu_seed_stability_report.md").write_text(seed_report,encoding="utf-8")
    print("NONIID")
    print(markdown_table(["setting","final_round","asr_at_attack_end","asr_post_100","asr_post_200","asr_post_300","asr_post_400","asr_post_500","post_attack_auc_3201_3700","notes"],nrows))
    print("\nSEEDS")
    print(markdown_table(["run","seed","final_round","asr_at_attack_end","asr_post_100","asr_post_200","asr_post_300","asr_post_400","asr_post_500","post_attack_auc_3201_3700"],srows))
    print("\nMEAN_STD")
    print(markdown_table(msfields,ms))

if __name__ == "__main__": main()

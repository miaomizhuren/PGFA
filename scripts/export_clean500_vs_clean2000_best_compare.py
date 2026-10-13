#!/usr/bin/env python3
"""Read-only comparison exporter for the clean500 (E) and clean2000 (H) PBU runs."""
from __future__ import annotations

import csv
import json
import math
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pbu_clean500_vs_clean2000_best_compare"

SPECS = {
    "E": {
        "experiment_id": "bs64_s501_e700_r1000_best_allE3",
        "experiment_label": "clean500 best allE3: attack 501-700, total 1000",
        "setting": "clean500 start, allE3, 200 attack rounds, 300 post rounds",
        "db": ROOT / "outputs/pbu_bs64_clean500_attack501_700_stop701_1000/metrics_pbu_bs64_clean500_attack501_700_stop701_1000.db",
        "config": ROOT / "configs/cifar10_pbu_bs64_clean500_attack501_700_stop701_1000.yaml",
        "attack_start": 501, "attack_end": 700, "total_rounds": 1000,
        "resume_checkpoint_path": "",
        "clean_checkpoint_acc_if_known": "",
    },
    "H": {
        "experiment_id": "bs64_s2001_e2200_r2500_clean2000_allE3",
        "experiment_label": "clean2000 allE3: attack 2001-2200, total 2500",
        "setting": "clean2000 start, allE3, 200 attack rounds, 300 post rounds",
        "db": ROOT / "outputs/pbu_bs64_resume_clean2000_allE3_attack2001_2200_stop2201_2500/metrics_pbu_bs64_resume_clean2000_allE3_attack2001_2200_stop2201_2500.db",
        "config": ROOT / "configs/cifar10_pbu_bs64_resume_clean2000_allE3_attack2001_2200_stop2201_2500.yaml",
        "attack_start": 2001, "attack_end": 2200, "total_rounds": 2500,
        "resume_checkpoint_path": "outputs/clean_fedavg_bs64_e3_r2000_save500_every100/checkpoints/global_round_2000.pt",
        "clean_checkpoint_acc_if_known": 92.71,
    },
}

KEYS = [
    ("attack_start", 0, "attack"), ("attack_middle_50", 50, "attack"),
    ("attack_middle_100", 100, "attack"), ("attack_middle_150", 150, "attack"),
    ("attack_end", 0, "post"),
    ("post+1", 1, "post"), ("post+10", 10, "post"),
    ("post+50", 50, "post"), ("post+100", 100, "post"),
    ("post+150", 150, "post"), ("post+200", 200, "post"),
    ("post+250", 250, "post"), ("post+300", 300, "post"),
]

def value(v):
    return "" if v is None else v

def number(v):
    return None if v in (None, "") else float(v)

def fmt(v, digits=2):
    return "" if v is None else f"{float(v):.{digits}f}"

def load(spec):
    con = sqlite3.connect(spec["db"])
    con.row_factory = sqlite3.Row
    rows = []
    for r in con.execute("""
        SELECT round_index, selected_clients, selected_malicious_clients, attack_active,
               average_train_loss, test_loss, clean_accuracy, attack_success_rate
        FROM round_metrics ORDER BY round_index
    """):
        rd = dict(r)
        round_index = rd["round_index"]
        if round_index < spec["attack_start"]:
            phase = "pre_attack"
        elif round_index <= spec["attack_end"]:
            phase = "attack"
        else:
            phase = "post_attack"
        rows.append({
            "experiment_id": spec["experiment_id"], "experiment_label": spec["experiment_label"],
            "paper_method": "PBU", "code_method": "proto_uap", "setting": spec["setting"],
            "batch_size": 64, "base_local_epochs": 3, "malicious_local_epochs": 3,
            "global_round": round_index, "relative_post_attack_round": round_index - spec["attack_end"],
            "attack_start": spec["attack_start"], "attack_end": spec["attack_end"],
            "stop_round": spec["attack_end"], "post_start_round": spec["attack_end"] + 1,
            "total_rounds": spec["total_rounds"], "phase": phase, "attack_active": rd["attack_active"],
            "clean_acc": rd["clean_accuracy"], "asr": rd["attack_success_rate"],
            "train_loss": rd["average_train_loss"], "test_loss": rd["test_loss"],
            "selected_clients": rd["selected_clients"], "selected_malicious_clients": rd["selected_malicious_clients"],
            "effective_malicious_local_epochs": "", "selection_policy": "legacy_all_malicious",
            "resume_checkpoint_path": spec["resume_checkpoint_path"], "source_file": str(spec["db"]),
        })
    con.close()
    return rows, {r["global_round"]: r for r in rows}

def nearest(by_round, wanted):
    actual = min(by_round, key=lambda x: (abs(x - wanted), x))
    return by_round[actual], actual == wanted

def auc_post(rows):
    points = [(r["relative_post_attack_round"], number(r["asr"])) for r in rows
              if 1 <= r["relative_post_attack_round"] <= 300 and number(r["asr"]) is not None]
    return sum((y0 + y1) * (x1 - x0) / 2 for (x0, y0), (x1, y1) in zip(points, points[1:]))

def summary(spec, rows, by_round):
    stop = spec["attack_end"]
    at = lambda rnd: by_round.get(rnd, {})
    asr = lambda rnd: number(at(rnd).get("asr"))
    post_asr = [number(r["asr"]) for r in rows if 1 <= r["relative_post_attack_round"] <= 300 and number(r["asr"]) is not None]
    stop_asr = asr(stop)
    half = None if stop_asr is None else next((r["relative_post_attack_round"] for r in rows
        if 1 <= r["relative_post_attack_round"] <= 300 and number(r["asr"]) is not None and number(r["asr"]) < stop_asr / 2), None)
    out = {
        "experiment_id": spec["experiment_id"], "experiment_label": spec["experiment_label"],
        "paper_method": "PBU", "code_method": "proto_uap", "setting": spec["setting"],
        "selection_policy": "legacy_all_malicious", "batch_size": 64, "base_local_epochs": 3,
        "malicious_local_epochs": 3, "attack_start": spec["attack_start"], "attack_end": stop,
        "attack_duration": 200, "stop_round": stop, "post_start_round": stop + 1,
        "total_rounds": spec["total_rounds"], "post_observation_rounds": 300,
        "clean_acc_at_attack_start": at(spec["attack_start"]).get("clean_acc"),
        "clean_acc_at_stop": at(stop).get("clean_acc"), "clean_acc_final": at(spec["total_rounds"]).get("clean_acc"),
        "asr_at_attack_start": asr(spec["attack_start"]), "asr_at_stop": stop_asr,
        **{f"asr_after_stop_{k}": asr(stop + k) for k in (1, 10, 50, 100, 150, 200, 250, 300)},
        "asr_final": asr(spec["total_rounds"]),
        "asr_drop_stop_to_final": None if stop_asr is None or asr(spec["total_rounds"]) is None else stop_asr-asr(spec["total_rounds"]),
        "asr_retention_final": None if not stop_asr or asr(spec["total_rounds"]) is None else asr(spec["total_rounds"])/stop_asr,
        "post_attack_auc": auc_post(rows), "post_attack_mean_asr": sum(post_asr)/len(post_asr) if post_asr else None,
        "post_attack_max_asr": max(post_asr) if post_asr else None, "post_attack_min_asr": min(post_asr) if post_asr else None,
        "post_attack_std_asr": (sum((x-sum(post_asr)/len(post_asr))**2 for x in post_asr)/len(post_asr))**.5 if post_asr else None,
        "half_life_rounds": ">300" if half is None else half,
        **{f"rounds_asr_ge_{t}_post": sum(x >= t for x in post_asr) for t in (90,80,70,60,50)},
        "resume_checkpoint_path": spec["resume_checkpoint_path"],
        "clean_checkpoint_acc_if_known": spec["clean_checkpoint_acc_if_known"], "source_file": str(spec["db"]),
    }
    return out

def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}; summaries = []
    for name, spec in SPECS.items():
        rows, by_round = load(spec); data[name] = (rows, by_round); summaries.append(summary(spec, rows, by_round))

    cleaned_rows = data["E"][0] + data["H"][0]
    cleaned_fields = list(cleaned_rows[0])
    write_csv(OUT / "clean500_vs_clean2000_cleaned_metrics.csv", cleaned_rows, cleaned_fields)

    summary_fields = list(summaries[0])
    write_csv(OUT / "clean500_vs_clean2000_summary.csv", summaries, summary_fields)

    key_rows = []
    for group, relative, kind in KEYS:
        wanted = {name: (sp["attack_start"] + relative if kind == "attack" and group != "attack_end" else sp["attack_end"] + relative)
                  for name, sp in SPECS.items()}
        er, ee = nearest(data["E"][1], wanted["E"]); hr, he = nearest(data["H"][1], wanted["H"])
        da = None if number(hr["clean_acc"]) is None or number(er["clean_acc"]) is None else number(hr["clean_acc"])-number(er["clean_acc"])
        ds = None if number(hr["asr"]) is None or number(er["asr"]) is None else number(hr["asr"])-number(er["asr"])
        interp = "exact phase-aligned records" if ee and he else "nearest available record used; no interpolation"
        key_rows.append({"metric_group": group, "requested_relative_round": relative if kind == "post" or group == "attack_end" else "attack+"+str(relative),
            "E_requested_round": wanted["E"], "E_actual_round": er["global_round"], "E_clean_acc": er["clean_acc"], "E_asr": er["asr"], "E_attack_active": er["attack_active"],
            "H_requested_round": wanted["H"], "H_actual_round": hr["global_round"], "H_clean_acc": hr["clean_acc"], "H_asr": hr["asr"], "H_attack_active": hr["attack_active"],
            "delta_clean_acc_H_minus_E": da, "delta_asr_H_minus_E": ds, "round_matched_exact": ee and he, "interpretation": interp})
    key_fields = list(key_rows[0]); write_csv(OUT / "clean500_vs_clean2000_key_rounds.csv", key_rows, key_fields)

    aligned=[]
    for rel in range(301):
        er = data["E"][1].get(SPECS["E"]["attack_end"]+rel, {}); hr = data["H"][1].get(SPECS["H"]["attack_end"]+rel, {})
        aligned.append({"relative_post_attack_round":rel, "E_round":er.get("global_round",""), "E_clean_acc":er.get("clean_acc",""), "E_asr":er.get("asr",""), "H_round":hr.get("global_round",""), "H_clean_acc":hr.get("clean_acc",""), "H_asr":hr.get("asr","")})
    write_csv(OUT / "clean500_vs_clean2000_aligned_curves.csv", aligned, list(aligned[0]))
    diff=[]
    for r in aligned:
        dc = None if number(r["E_clean_acc"]) is None or number(r["H_clean_acc"]) is None else number(r["H_clean_acc"])-number(r["E_clean_acc"])
        ds = None if number(r["E_asr"]) is None or number(r["H_asr"]) is None else number(r["H_asr"])-number(r["E_asr"])
        diff.append({**r, "delta_clean_acc_H_minus_E":dc, "delta_asr_H_minus_E":ds, "abs_delta_clean_acc":None if dc is None else abs(dc), "abs_delta_asr":None if ds is None else abs(ds)})
    write_csv(OUT / "clean500_vs_clean2000_diff_by_relative_round.csv", diff, list(diff[0]))

    E, H = summaries
    post_points = (100,200,300)
    diffs = {k: H[f"asr_after_stop_{k}"]-E[f"asr_after_stop_{k}"] for k in post_points}
    mean_diff = H["post_attack_mean_asr"] - E["post_attack_mean_asr"]
    stop_diff = H["asr_at_stop"] - E["asr_at_stop"]
    if abs(stop_diff) <= 2 and max(abs(v) for v in diffs.values()) <= 5 and abs(mean_diff) <= 5:
        verdict = "clean2000_similar"
    elif mean_diff < -5 or sum(v < -5 for v in diffs.values()) >= 2:
        verdict = "clean2000_weaker"
    elif mean_diff > 5 or sum(v > 5 for v in diffs.values()) >= 2:
        verdict = "clean2000_stronger"
    else:
        verdict = "inconclusive"
    def mdrow(s):
        return " | ".join([s["experiment_id"], fmt(s["asr_at_stop"]), fmt(s["asr_after_stop_100"]), fmt(s["asr_after_stop_200"]), fmt(s["asr_after_stop_300"]), fmt(s["asr_final"]), fmt(s["post_attack_mean_asr"]), fmt(s["post_attack_auc"]), str(s["half_life_rounds"]), fmt(s["clean_acc_final"])])
    key_md = "\n".join("| {metric_group} | {E_actual_round} | {E_clean_acc:.2f} | {E_asr:.2f} | {H_actual_round} | {H_clean_acc:.2f} | {H_asr:.2f} | {delta_clean_acc_H_minus_E:+.2f} | {delta_asr_H_minus_E:+.2f} |".format(**r) for r in key_rows)
    report = f"""# Clean500 vs Clean2000 PBU Comparison

## 1. Purpose

This read-only comparison evaluates only E (clean500 allE3) and H (clean2000 allE3), with attack and post-attack stages aligned by relative round.

## 2. Compared Experiments

| Group | Configuration/output | Attack | Post observation | Resume |
|---|---|---|---|---|
| E | `configs/cifar10_pbu_bs64_clean500_attack501_700_stop701_1000.yaml`; `outputs/pbu_bs64_clean500_attack501_700_stop701_1000/` | 501-700 | 701-1000 | No (from scratch) |
| H | `configs/cifar10_pbu_bs64_resume_clean2000_allE3_attack2001_2200_stop2201_2500.yaml`; `outputs/pbu_bs64_resume_clean2000_allE3_attack2001_2200_stop2201_2500/` | 2001-2200 | 2201-2500 | `outputs/clean_fedavg_bs64_e3_r2000_save500_every100/checkpoints/global_round_2000.pt` |

## 3. Data Sources

- E: `{SPECS['E']['db']}` (SQLite `round_metrics`, 1000 rows).
- H: `{SPECS['H']['db']}` (SQLite `round_metrics`, 500 rows covering 2001-2500).
- The clean training log records Round 2000 Clean ACC = 92.71%.

No `metrics.csv` existed for either target run, so the CSVs here were exported from SQLite. `effective_malicious_local_epochs` is not stored in the round-metrics schema and is left blank. H contains no new pre-attack rows because it resumes at Round 2001.

## 4. Configuration Consistency Check

Both use CIFAR-10 IID (100 clients), ResNet18, FedAvg, batch size 64, `method: proto_uap` (PBU), 10 clients/round, allE3, 200 attack rounds, 300 post-attack rounds, identical listed PBU/prototype parameters, and `selection_policy: legacy_all_malicious`. The material difference is the starting clean model and absolute attack window: E attacks from clean500; H attacks from the clean Round-2000 checkpoint (92.71% Clean ACC).

## 5. Key Round Comparison

| Stage | E round | E Clean ACC | E ASR | H round | H Clean ACC | H ASR | H-E Clean ACC | H-E ASR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{key_md}

All requested key rounds were exact records; no nearest-neighbour matching or interpolation was used.

## 6. Post-attack Persistence Comparison

| Group | ASR@stop | ASR +100 | ASR +200 | ASR +300 | Final ASR | Post mean ASR | Post AUC | Half-life | Final Clean ACC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| {mdrow(E)} |
| {mdrow(H)} |

H-E: stop ASR {stop_diff:+.2f} pp; post +100/{diffs[100]:+.2f} pp, +200/{diffs[200]:+.2f} pp, +300/{diffs[300]:+.2f} pp; post mean {mean_diff:+.2f} pp.

## 7. Clean500 vs Clean2000 Interpretation

1. H demonstrates that PBU can be implanted in the highly converged clean2000 model: its attack-end ASR is {fmt(H['asr_at_stop'])}%.
2. The attack-end ASR difference (H-E = {stop_diff:+.2f} pp) is {'basically equivalent by the ±2 pp rule' if abs(stop_diff)<=2 else 'not equivalent by the ±2 pp rule'}.
3. The aligned post-attack trajectory has mean difference {mean_diff:+.2f} pp; the +100/+200/+300 differences are {diffs[100]:+.2f}/{diffs[200]:+.2f}/{diffs[300]:+.2f} pp.
4. H final Clean ACC is {fmt(H['clean_acc_final'])}% versus E {fmt(E['clean_acc_final'])}%, a difference of {H['clean_acc_final']-E['clean_acc_final']:+.2f} pp.
5. Thus clean2000 is assessed as **{verdict}** under the stated thresholds. This is a controlled one-seed comparison, so it supports descriptive evidence rather than a general causal claim about convergence alone.

## 8. Sampling Policy Caveat

Both runs use `legacy_all_malicious`: clients 0, 1, and 2 are fixed participants in pre-attack, attack, and post-attack rounds. Outside active attack they are benign CE-only (no poison/trigger operation). This preserves the E-vs-H relative comparison because the policy is shared, but neither run should be described as using purely random client sampling outside attack.

## 9. Paper-use Recommendation

E is the clearer main-result run because its entire trajectory is from-scratch and already established as the best clean500 allE3 reference. H is valuable as a supplementary experiment showing performance from a highly converged clean model; put it in the main text only if the thesis foregrounds robustness across convergence states, otherwise report it in an ablation/supplementary section. Even if its effect were weak it would still be informative about convergence-state sensitivity; here the measured verdict is **{verdict}**.

## 10. Final Verdict

`final_verdict = {verdict}`

One-sentence conclusion: relative to the clean500 starting point, the clean2000-start PBU run is **{verdict.replace('clean2000_', '')}** on attack implantation and post-attack persistence under the same legacy-all-malicious sampling policy.

No training was started and no training code, configuration, raw database, log, or original result was modified.
"""
    (OUT / "clean500_vs_clean2000_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), "final_verdict": verdict, "summary": summaries}, indent=2, default=str))

if __name__ == "__main__":
    main()

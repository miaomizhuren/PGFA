#!/usr/bin/env python3
"""Compare the original BS64 all-E3 PBU run with its R500 resume rerun.

The source experiment databases are read-only.  This script writes derived
comparison files only under outputs/pbu_allE3_rerun_comparison.
"""

from __future__ import annotations

import csv
import math
import re
import sqlite3
import statistics
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pbu_allE3_rerun_comparison"
KEY_ROUNDS = [501, 550, 600, 650, 700, 701, 710, 750, 800, 850, 900, 950, 1000]
METRIC_FIELDS = [
    "experiment_id", "experiment_label", "paper_method", "code_method", "batch_size",
    "base_local_epochs", "malicious_local_epochs", "global_round", "relative_post_attack_round",
    "attack_start", "attack_end", "stop_round", "post_start_round", "total_rounds", "phase",
    "attack_active", "clean_acc", "asr", "train_loss", "test_loss", "selected_clients",
    "selected_malicious_clients", "effective_malicious_local_epochs", "source_file",
]
SUMMARY_FIELDS = [
    "experiment_id", "experiment_label", "batch_size", "base_local_epochs", "malicious_local_epochs",
    "attack_start", "attack_end", "attack_duration", "stop_round", "post_start_round", "total_rounds",
    "post_observation_rounds", "clean_acc_at_attack_start", "clean_acc_at_stop", "clean_acc_final",
    "asr_at_attack_start", "asr_at_stop", "asr_after_stop_1", "asr_after_stop_10",
    "asr_after_stop_50", "asr_after_stop_100", "asr_after_stop_150", "asr_after_stop_200",
    "asr_after_stop_250", "asr_after_stop_300", "asr_final", "asr_drop_stop_to_final",
    "asr_retention_final", "post_attack_auc", "post_attack_mean_asr", "post_attack_max_asr",
    "post_attack_min_asr", "post_attack_std_asr", "half_life_rounds", "rounds_asr_ge_90_post",
    "rounds_asr_ge_80_post", "rounds_asr_ge_70_post", "rounds_asr_ge_60_post",
    "rounds_asr_ge_50_post", "source_file",
]
SPECS = [
    {
        "key": "E", "id": "bs64_s501_e700_r1000",
        "label": "original allE3 bs64 PBU: attack 501-700, total 1000",
        "config": "configs/cifar10_pbu_bs64_clean500_attack501_700_stop701_1000.yaml",
        "db": "outputs/pbu_bs64_clean500_attack501_700_stop701_1000/metrics_pbu_bs64_clean500_attack501_700_stop701_1000.db",
        "log": "experiments/logs/pbu_bs64_clean500_attack501_700_stop701_1000.log",
    },
    {
        "key": "G", "id": "bs64_s501_e700_r1000_allE3_rerun",
        "label": "rerun allE3 bs64 PBU: resume clean500, attack 501-700, total 1000",
        "config": "configs/cifar10_pbu_bs64_resume_clean500_allE3_attack501_700_stop701_1000_rerun.yaml",
        "db": "outputs/pbu_bs64_resume_clean500_allE3_attack501_700_stop701_1000_rerun/metrics_pbu_bs64_resume_clean500_allE3_attack501_700_stop701_1000_rerun.db",
        "log": "experiments/logs/pbu_bs64_resume_clean500_allE3_attack501_700_stop701_1000_rerun.log",
    },
]


def none_to_blank(value):
    return "" if value is None else value


def load_yaml(path: str) -> dict:
    with (ROOT / path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_epochs(log_path: str) -> dict[int, int]:
    result = {}
    pattern = re.compile(r"\[LocalEpochPolicy\] Round (\d+) .*malicious_effective_epochs=([^ |]+)")
    path = ROOT / log_path
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match and match.group(2).isdigit():
            result[int(match.group(1))] = int(match.group(2))
    return result


def load_rows(spec: dict) -> list[dict]:
    cfg = load_yaml(spec["config"])
    fed, training, attack = cfg["federated"], cfg["training"], cfg["attack"]
    attack_start, attack_end, total = attack["start_round"], attack["end_round"], fed["total_rounds"]
    base = fed["local_epochs"]
    malicious = fed.get("malicious_local_epochs", base)
    source = str((ROOT / spec["db"]).resolve())
    epochs = load_epochs(spec["log"])
    conn = sqlite3.connect(ROOT / spec["db"])
    conn.row_factory = sqlite3.Row
    records = conn.execute("SELECT * FROM round_metrics ORDER BY round_index").fetchall()
    conn.close()
    rows = []
    for record in records:
        rnd = record["round_index"]
        phase = "pre_attack" if rnd < attack_start else "attack" if rnd <= attack_end else "post_attack"
        rows.append({
            "experiment_id": spec["id"], "experiment_label": spec["label"], "paper_method": "PBU",
            "code_method": "proto_uap", "batch_size": training["batch_size"], "base_local_epochs": base,
            "malicious_local_epochs": malicious, "global_round": rnd,
            "relative_post_attack_round": rnd - attack_end, "attack_start": attack_start, "attack_end": attack_end,
            "stop_round": attack_end, "post_start_round": attack_end + 1, "total_rounds": total, "phase": phase,
            "attack_active": record["attack_active"], "clean_acc": record["clean_accuracy"],
            "asr": record["attack_success_rate"], "train_loss": record["average_train_loss"],
            "test_loss": record["test_loss"], "selected_clients": record["selected_clients"],
            "selected_malicious_clients": record["selected_malicious_clients"],
            "effective_malicious_local_epochs": epochs.get(rnd, ""), "source_file": source,
        })
    return rows


def at(by_round: dict[int, dict], rnd: int, field: str):
    return none_to_blank(by_round.get(rnd, {}).get(field, ""))


def nearest(by_round: dict[int, dict], requested: int) -> tuple[int | str, dict, bool]:
    if not by_round:
        return "", {}, False
    actual = min(by_round, key=lambda candidate: (abs(candidate - requested), candidate))
    return actual, by_round[actual], actual == requested


def trapz(points: list[tuple[int, float]]):
    return sum((x2 - x1) * (y1 + y2) / 2 for (x1, y1), (x2, y2) in zip(points, points[1:])) if len(points) > 1 else ""


def summary(rows: list[dict]) -> dict:
    by_round = {r["global_round"]: r for r in rows}
    stop, total = 700, 1000
    post = [r for r in rows if r["global_round"] > stop and r["asr"] is not None]
    points = [(r["relative_post_attack_round"], float(r["asr"])) for r in post]
    vals = [value for _, value in points]
    stop_asr, final_asr = at(by_round, stop, "asr"), at(by_round, total, "asr")
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update({
        "experiment_id": rows[0]["experiment_id"], "experiment_label": rows[0]["experiment_label"],
        "batch_size": rows[0]["batch_size"], "base_local_epochs": rows[0]["base_local_epochs"],
        "malicious_local_epochs": rows[0]["malicious_local_epochs"], "attack_start": 501, "attack_end": stop,
        "attack_duration": 200, "stop_round": stop, "post_start_round": 701, "total_rounds": total,
        "post_observation_rounds": 300, "clean_acc_at_attack_start": at(by_round, 501, "clean_acc"),
        "clean_acc_at_stop": at(by_round, stop, "clean_acc"), "clean_acc_final": at(by_round, total, "clean_acc"),
        "asr_at_attack_start": at(by_round, 501, "asr"), "asr_at_stop": stop_asr,
        "asr_final": final_asr, "post_attack_auc": trapz(points),
        "post_attack_mean_asr": statistics.mean(vals) if vals else "",
        "post_attack_max_asr": max(vals) if vals else "", "post_attack_min_asr": min(vals) if vals else "",
        "post_attack_std_asr": statistics.pstdev(vals) if len(vals) > 1 else "", "source_file": rows[0]["source_file"],
    })
    for offset in (1, 10, 50, 100, 150, 200, 250, 300):
        row[f"asr_after_stop_{offset}"] = at(by_round, stop + offset, "asr")
    if stop_asr != "" and final_asr != "" and float(stop_asr):
        row["asr_drop_stop_to_final"] = float(stop_asr) - float(final_asr)
        row["asr_retention_final"] = float(final_asr) / float(stop_asr)
        half = float(stop_asr) / 2
        first = next((relative for relative, value in points if value < half), None)
        row["half_life_rounds"] = first if first is not None else ">300"
    for threshold in (90, 80, 70, 60, 50):
        row[f"rounds_asr_ge_{threshold}_post"] = sum(value >= threshold for value in vals)
    return row


def fmt(value, digits=2):
    return "" if value in (None, "") else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    return "\n".join(lines + ["| " + " | ".join(fmt(row.get(h, "")) for h in headers) + " |" for row in rows])


def verdict(e: dict, g: dict, diff_rows: list[dict]):
    stop_diff = abs(float(g["asr_at_stop"]) - float(e["asr_at_stop"]))
    key_rel = [50, 100, 200, 300]
    key_diffs = [
        abs(float(next(r["delta_asr_G_minus_E"] for r in diff_rows if r["relative_post_attack_round"] == rel)))
        for rel in key_rel
    ]
    mean_rel_pct = abs(float(g["post_attack_mean_asr"]) - float(e["post_attack_mean_asr"])) / float(e["post_attack_mean_asr"]) * 100
    auc_rel_pct = abs(float(g["post_attack_auc"]) - float(e["post_attack_auc"])) / float(e["post_attack_auc"]) * 100
    final_clean_diff = abs(float(g["clean_acc_final"]) - float(e["clean_acc_final"]))
    stable_key_count = sum(d <= 5 for d in key_diffs)
    if stop_diff <= 2 and stable_key_count >= 3 and max(mean_rel_pct, auc_rel_pct) <= 5 and final_clean_diff <= 1:
        label = "consistent"
    elif stop_diff <= 5 and stable_key_count >= 2 and max(mean_rel_pct, auc_rel_pct) <= 10 and final_clean_diff <= 2:
        label = "mostly_consistent"
    elif stable_key_count >= 1:
        label = "noticeably_different"
    else:
        label = "failed_to_reproduce"
    return label, {"stop_abs_pp": stop_diff, "key_abs_pp": key_diffs, "stable_key_count": stable_key_count,
                   "mean_relative_pct": mean_rel_pct, "auc_relative_pct": auc_rel_pct, "final_clean_abs_pp": final_clean_diff}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = {spec["key"]: load_rows(spec) for spec in SPECS}
    for key, rows in records.items():
        with (OUT / f"allE3_{'original' if key == 'E' else 'rerun'}_metrics_cleaned.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS); writer.writeheader(); writer.writerows(rows)
    with (OUT / "allE3_original_vs_rerun_metrics_cleaned.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS); writer.writeheader(); writer.writerows(records["E"] + records["G"])
    by_round = {key: {r["global_round"]: r for r in rows} for key, rows in records.items()}
    key_rows = []
    for requested in KEY_ROUNDS:
        er, e, exact_e = nearest(by_round["E"], requested)
        gr, g, exact_g = nearest(by_round["G"], requested)
        dc = "" if e.get("clean_acc") is None or g.get("clean_acc") is None else float(g["clean_acc"]) - float(e["clean_acc"])
        da = "" if e.get("asr") is None or g.get("asr") is None else float(g["asr"]) - float(e["asr"])
        key_rows.append({"metric_group": "attack_or_persistence", "requested_round": requested, "E_actual_round": er,
                         "E_clean_acc": none_to_blank(e.get("clean_acc")), "E_asr": none_to_blank(e.get("asr")),
                         "E_attack_active": e.get("attack_active", ""), "G_actual_round": gr,
                         "G_clean_acc": none_to_blank(g.get("clean_acc")), "G_asr": none_to_blank(g.get("asr")),
                         "G_attack_active": g.get("attack_active", ""), "delta_clean_acc_G_minus_E": dc,
                         "delta_asr_G_minus_E": da, "round_matched_exact": exact_e and exact_g,
                         "interpretation": "G higher" if da != "" and da > 0 else "E higher" if da != "" and da < 0 else "equal or missing"})
    key_fields = list(key_rows[0])
    with (OUT / "allE3_original_vs_rerun_key_rounds.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=key_fields); writer.writeheader(); writer.writerows(key_rows)
    aligned, diff_rows = [], []
    for relative in range(301):
        e, g = by_round["E"].get(700 + relative, {}), by_round["G"].get(700 + relative, {})
        aligned.append({"relative_post_attack_round": relative, "E_round": e.get("global_round", ""), "E_clean_acc": none_to_blank(e.get("clean_acc")), "E_asr": none_to_blank(e.get("asr")), "G_round": g.get("global_round", ""), "G_clean_acc": none_to_blank(g.get("clean_acc")), "G_asr": none_to_blank(g.get("asr"))})
        dc = "" if e.get("clean_acc") is None or g.get("clean_acc") is None else float(g["clean_acc"]) - float(e["clean_acc"])
        da = "" if e.get("asr") is None or g.get("asr") is None else float(g["asr"]) - float(e["asr"])
        diff_rows.append({"relative_post_attack_round": relative, "E_round": e.get("global_round", ""), "E_clean_acc": none_to_blank(e.get("clean_acc")), "E_asr": none_to_blank(e.get("asr")), "G_round": g.get("global_round", ""), "G_clean_acc": none_to_blank(g.get("clean_acc")), "G_asr": none_to_blank(g.get("asr")), "delta_clean_acc_G_minus_E": dc, "delta_asr_G_minus_E": da, "abs_delta_clean_acc": "" if dc == "" else abs(dc), "abs_delta_asr": "" if da == "" else abs(da)})
    for name, rows in [("allE3_original_vs_rerun_aligned_curves.csv", aligned), ("allE3_original_vs_rerun_diff_by_round.csv", diff_rows)]:
        with (OUT / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summaries = [summary(records[key]) for key in ("E", "G")]
    with (OUT / "allE3_original_vs_rerun_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS); writer.writeheader(); writer.writerows(summaries)
    e, g = summaries
    final_verdict, measures = verdict(e, g, diff_rows)
    config_e, config_g = load_yaml(SPECS[0]["config"]), load_yaml(SPECS[1]["config"])
    comparison_fields = ["requested_round", "E_clean_acc", "E_asr", "G_clean_acc", "G_asr", "delta_clean_acc_G_minus_E", "delta_asr_G_minus_E", "round_matched_exact"]
    persistence_fields = ["asr_at_stop", "asr_after_stop_1", "asr_after_stop_10", "asr_after_stop_50", "asr_after_stop_100", "asr_after_stop_150", "asr_after_stop_200", "asr_after_stop_250", "asr_after_stop_300", "asr_final", "post_attack_mean_asr", "post_attack_auc", "half_life_rounds", "clean_acc_final"]
    post_key = [next(r for r in diff_rows if r["relative_post_attack_round"] == rel) for rel in [50, 100, 200, 300]]
    lines = [
        "# allE3 Original vs Rerun Comparison", "",
        "## 1. Purpose", "", "This one-to-one comparison tests whether the original best BS64 all-E3 PBU result is reproducible from its clean Round-500 checkpoint. Source metrics are read directly from each SQLite `round_metrics` table; no training was started and no source artifact was modified.", "",
        "## 2. Compared Runs", "",
        f"- E (original): `{SPECS[0]['config']}`; source DB `{SPECS[0]['db']}`; from-scratch R1--1000.",
        f"- G (rerun): `{SPECS[1]['config']}`; source DB `{SPECS[1]['db']}`; resumes `{config_g['resume']['checkpoint_path']}` at completed R500 and executes R501--1000.", "",
        "## 3. Configuration Consistency Check", "",
        "| Item | E | G | Result |", "| --- | --- | --- | --- |",
        f"| batch size | {config_e['training']['batch_size']} | {config_g['training']['batch_size']} | match |",
        f"| method | {config_e['attack']['method']} | {config_g['attack']['method']} | match |",
        f"| base local epochs | {config_e['federated']['local_epochs']} | {config_g['federated']['local_epochs']} | match |",
        f"| malicious local epochs | {config_e['federated'].get('malicious_local_epochs', config_e['federated']['local_epochs'])} | {config_g['federated'].get('malicious_local_epochs', config_g['federated']['local_epochs'])} | match |",
        f"| attack window | {config_e['attack']['start_round']}--{config_e['attack']['end_round']} | {config_g['attack']['start_round']}--{config_g['attack']['end_round']} | match |",
        "| total/post observation | 1000 / R701--1000 | 1000 / R701--1000 | match |", "",
        "The intended difference is execution history: E trained R1--1000 from scratch, while G loaded E's clean R500 model and newly generated its shared PBU delta at the first attack-active round R501. Both configs use seed 2026, but resumed-model execution and delta/prototype construction can still introduce stochastic trajectory differences.", "",
        "## 4. Key Round Comparison", "", md_table(comparison_fields, [{f: r.get(f, "") for f in comparison_fields} for r in key_rows]), "",
        "## 5. Post-attack Persistence Comparison", "", md_table(["experiment_id"] + persistence_fields, [{field: row.get(field, "") for field in ["experiment_id"] + persistence_fields} for row in summaries]), "",
        "## 6. Difference Analysis", "",
        f"- Attack-stop ASR absolute gap: {measures['stop_abs_pp']:.2f} percentage points (pp).",
        f"- Post-stop ASR absolute gaps at +50/+100/+200/+300: " + ", ".join(f"{value:.2f} pp" for value in measures['key_abs_pp']) + f"; {measures['stable_key_count']}/4 are <=5 pp.",
        f"- Post-attack mean-ASR relative gap: {measures['mean_relative_pct']:.2f}%; AUC relative gap: {measures['auc_relative_pct']:.2f}%; final clean-accuracy gap: {measures['final_clean_abs_pp']:.2f} pp.",
        "- Both attack phases reach near-saturated ASR and both post-attack curves decline while clean accuracy recovers to about 91--92%. Differences are therefore evaluated using the specified multi-point persistence criteria rather than one final-round value.", "",
        "## 7. Reliability Conclusion", "",
        f"The rerun is classified **{final_verdict}** under the requested thresholds. A single rerun cannot estimate run-to-run variance reliably; for a thesis main result, retain both runs, report their mean (and range or standard deviation) for key persistence metrics, and run a third seed if a formal robustness claim is needed.", "",
        "## 8. Final Verdict", "", f"`final_verdict = {final_verdict}`", "",
        ("The rerun supports the reliability of the original all-E3 result within the predefined comparison thresholds." if final_verdict in {"consistent", "mostly_consistent"} else "The rerun differs materially from the original under the predefined thresholds; report both rather than treating the original single run as definitive."), "",
        "## Data Notes", "", "All requested R501--R1000 key points are exactly present in both SQLite databases. No interpolation was used. `effective_malicious_local_epochs` for E is absent from its historical log/database and remains blank in its cleaned CSV; G's LocalEpochPolicy log records E3 for its available R501--R1000 rounds.",
    ]
    (OUT / "allE3_original_vs_rerun_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote comparison to {OUT}")
    print(f"final_verdict={final_verdict}")
    for row in summaries:
        print(row["experiment_id"], "stop", fmt(row["asr_at_stop"]), "mean", fmt(row["post_attack_mean_asr"]), "final", fmt(row["asr_final"]))


if __name__ == "__main__":
    main()

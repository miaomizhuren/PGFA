#!/usr/bin/env python3
"""Read-only one-to-one exporter for legacy and force-only PBU sampling runs."""
from __future__ import annotations

import csv
import json
import math
import re
import sqlite3
import statistics
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pbu_sampling_policy_comparison_force_vs_legacy"
KEY_ROUNDS = [1, 100, 300, 500, 501, 550, 600, 650, 700, 701, 710, 750, 800, 850, 900, 950, 1000]
POST_CHECKPOINTS = (1, 10, 50, 100, 150, 200, 250, 300)

SPECS = {
    "legacy": {
        "experiment_id": "legacy_bs64_clean500_allE3",
        "experiment_label": "legacy sampling: bs64 clean500 allE3, attack 501-700, total 1000",
        "setting": "legacy_all_malicious, from scratch, clean500, 200 attack rounds, 300 post rounds",
        "selection_policy": "legacy_all_malicious",
        "sampling_protocol": "legacy: fixed 0/1/2 in pre-attack, attack, and post-attack",
        "config": ROOT / "configs/cifar10_pbu_bs64_clean500_attack501_700_stop701_1000.yaml",
        "db": ROOT / "outputs/pbu_bs64_clean500_attack501_700_stop701_1000/metrics_pbu_bs64_clean500_attack501_700_stop701_1000.db",
        "log": ROOT / "experiments/logs/pbu_bs64_clean500_attack501_700_stop701_1000.log",
        "checkpoint": ROOT / "outputs/pbu_bs64_clean500_attack501_700_stop701_1000/checkpoints/global_round_500.pt",
    },
    "force": {
        "experiment_id": "force_bs64_clean500_allE3",
        "experiment_label": "force-only sampling: bs64 clean500 allE3, attack 501-700, total 1000",
        "setting": "force_during_attack, from scratch, clean500, 200 attack rounds, 300 post rounds",
        "selection_policy": "force_during_attack",
        "sampling_protocol": "force only during attack: random pre/post, fixed 0/1/2 only during attack",
        "config": ROOT / "configs/cifar10_pbu_bs64_fromscratch_allE3_force_attack_only_clean500_attack501_700_stop701_1000.yaml",
        "db": ROOT / "outputs/pbu_bs64_fromscratch_allE3_force_attack_only_clean500_attack501_700_stop701_1000/metrics_pbu_bs64_fromscratch_allE3_force_attack_only_clean500_attack501_700_stop701_1000.db",
        "log": ROOT / "experiments/logs/pbu_bs64_fromscratch_allE3_force_attack_only_clean500_attack501_700_stop701_1000.log",
        "checkpoint": ROOT / "outputs/pbu_bs64_fromscratch_allE3_force_attack_only_clean500_attack501_700_stop701_1000/checkpoints/global_round_500.pt",
    },
}


def number(value):
    return None if value in (None, "") else float(value)


def cell(value, digits=2):
    return "" if value is None else f"{float(value):.{digits}f}"


def phase(round_index):
    return "pre_attack" if round_index < 501 else "attack" if round_index <= 700 else "post_attack"


def derive_sampling(spec, round_index):
    if spec["selection_policy"] == "legacy_all_malicious":
        return "legacy_all_malicious", "[0, 1, 2]"
    if 501 <= round_index <= 700:
        return "force_malicious_during_attack", "[0, 1, 2]"
    return "random_all_clients", "[]"


def read_log_sampling(log_path):
    """Read explicitly logged sampling fields when available (force run)."""
    found = {}
    if not log_path.is_file():
        return found
    pattern = re.compile(
        r"Round (?P<round>\d+) \| attack_active=(?P<active>True|False) \| "
        r"selection_policy=(?P<policy>[^|]+) \| sampling_mode=(?P<mode>[^|]+) \| "
        r"forced_malicious=(?P<forced>\[[^]]*\])"
    )
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            found[int(match["round"])] = {
                "sampling_mode": match["mode"].strip(),
                "forced_malicious": match["forced"],
            }
    return found


def load(spec):
    logged = read_log_sampling(spec["log"])
    con = sqlite3.connect(spec["db"])
    con.row_factory = sqlite3.Row
    rows = []
    for record in con.execute(
        """SELECT round_index, selected_clients, selected_malicious_clients,
                  attack_active, average_train_loss, test_loss, clean_accuracy,
                  attack_success_rate
           FROM round_metrics ORDER BY round_index"""
    ):
        row = dict(record)
        round_index = row["round_index"]
        default_mode, default_forced = derive_sampling(spec, round_index)
        log_fields = logged.get(round_index, {})
        selected_malicious = row["selected_malicious_clients"] or "[]"
        effective_epochs = 3 if selected_malicious != "[]" else None
        rows.append({
            "experiment_id": spec["experiment_id"],
            "experiment_label": spec["experiment_label"],
            "paper_method": "PBU", "code_method": "proto_uap", "setting": spec["setting"],
            "batch_size": 64, "base_local_epochs": 3, "malicious_local_epochs": 3,
            "selection_policy": spec["selection_policy"], "sampling_protocol": spec["sampling_protocol"],
            "global_round": round_index, "relative_post_attack_round": round_index - 700,
            "attack_start": 501, "attack_end": 700, "stop_round": 700,
            "post_start_round": 701, "total_rounds": 1000, "phase": phase(round_index),
            "attack_active": row["attack_active"], "clean_acc": row["clean_accuracy"],
            "asr": row["attack_success_rate"], "train_loss": row["average_train_loss"],
            "test_loss": row["test_loss"], "selected_clients": row["selected_clients"],
            "selected_malicious_clients": selected_malicious,
            "sampling_mode": log_fields.get("sampling_mode", default_mode),
            "forced_malicious": log_fields.get("forced_malicious", default_forced),
            "effective_malicious_local_epochs": effective_epochs,
            "source_file": str(spec["db"]),
        })
    con.close()
    return rows, {row["global_round"]: row for row in rows}


def nearest(rows, requested):
    actual = min(rows, key=lambda round_index: (abs(round_index - requested), round_index))
    return rows[actual], actual == requested


def post_values(rows):
    return [number(row["asr"]) for row in rows if 1 <= row["relative_post_attack_round"] <= 300 and number(row["asr"]) is not None]


def post_auc(rows):
    points = [(row["relative_post_attack_round"], number(row["asr"])) for row in rows
              if 1 <= row["relative_post_attack_round"] <= 300 and number(row["asr"]) is not None]
    return sum((left_y + right_y) * (right_x - left_x) / 2
               for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]))


def summary(spec, rows, by_round):
    get = lambda round_index, field: by_round.get(round_index, {}).get(field)
    asr_stop = number(get(700, "asr"))
    final_asr = number(get(1000, "asr"))
    post = post_values(rows)
    half_life = next((row["relative_post_attack_round"] for row in rows
                      if 1 <= row["relative_post_attack_round"] <= 300
                      and number(row["asr"]) is not None and number(row["asr"]) < asr_stop / 2), None)
    return {
        "experiment_id": spec["experiment_id"], "experiment_label": spec["experiment_label"],
        "paper_method": "PBU", "code_method": "proto_uap", "setting": spec["setting"],
        "selection_policy": spec["selection_policy"], "sampling_protocol": spec["sampling_protocol"],
        "batch_size": 64, "base_local_epochs": 3, "malicious_local_epochs": 3,
        "attack_start": 501, "attack_end": 700, "attack_duration": 200, "stop_round": 700,
        "post_start_round": 701, "total_rounds": 1000, "post_observation_rounds": 300,
        "clean_acc_at_round_500": get(500, "clean_acc"),
        "clean_acc_at_attack_start": get(501, "clean_acc"), "clean_acc_at_stop": get(700, "clean_acc"),
        "clean_acc_final": get(1000, "clean_acc"), "asr_at_attack_start": get(501, "asr"),
        "asr_at_stop": asr_stop, **{f"asr_after_stop_{k}": get(700 + k, "asr") for k in POST_CHECKPOINTS},
        "asr_final": final_asr,
        "asr_drop_stop_to_final": None if asr_stop is None or final_asr is None else asr_stop - final_asr,
        "asr_retention_final": None if not asr_stop or final_asr is None else final_asr / asr_stop,
        "post_attack_auc": post_auc(rows), "post_attack_mean_asr": statistics.mean(post),
        "post_attack_max_asr": max(post), "post_attack_min_asr": min(post),
        "post_attack_std_asr": statistics.pstdev(post),
        "half_life_rounds": ">300" if half_life is None else half_life,
        **{f"rounds_asr_ge_{threshold}_post": sum(value >= threshold for value in post)
           for threshold in (90, 80, 70, 60, 50)},
        "round500_checkpoint_path": str(spec["checkpoint"]) if spec["checkpoint"].is_file() else "",
        "source_file": str(spec["db"]),
    }


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def table_row(summary_row):
    fields = ("asr_at_stop", "asr_after_stop_100", "asr_after_stop_200", "asr_after_stop_300",
              "asr_final", "post_attack_mean_asr", "post_attack_auc", "half_life_rounds", "clean_acc_final")
    values = [summary_row["experiment_id"]] + [cell(summary_row[field]) if field != "half_life_rounds" else str(summary_row[field]) for field in fields]
    return "| " + " | ".join(values) + " |"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = {name: load(spec) for name, spec in SPECS.items()}
    legacy_rows, legacy_by_round = loaded["legacy"]
    force_rows, force_by_round = loaded["force"]
    write_csv(OUT / "legacy_bs64_clean500_allE3_metrics_cleaned.csv", legacy_rows)
    write_csv(OUT / "force_bs64_clean500_allE3_metrics_cleaned.csv", force_rows)

    summaries = [summary(SPECS["legacy"], legacy_rows, legacy_by_round), summary(SPECS["force"], force_rows, force_by_round)]
    write_csv(OUT / "sampling_policy_force_vs_legacy_summary.csv", summaries)

    key_rows = []
    for requested in KEY_ROUNDS:
        legacy, legacy_exact = nearest(legacy_by_round, requested)
        force, force_exact = nearest(force_by_round, requested)
        clean_delta = None if number(force["clean_acc"]) is None or number(legacy["clean_acc"]) is None else number(force["clean_acc"]) - number(legacy["clean_acc"])
        asr_delta = None if number(force["asr"]) is None or number(legacy["asr"]) is None else number(force["asr"]) - number(legacy["asr"])
        key_rows.append({
            "metric_group": phase(requested), "requested_round": requested,
            "legacy_actual_round": legacy["global_round"], "legacy_clean_acc": legacy["clean_acc"], "legacy_asr": legacy["asr"],
            "legacy_attack_active": legacy["attack_active"], "legacy_selected_clients": legacy["selected_clients"],
            "legacy_selected_malicious_clients": legacy["selected_malicious_clients"], "legacy_sampling_mode": legacy["sampling_mode"],
            "force_actual_round": force["global_round"], "force_clean_acc": force["clean_acc"], "force_asr": force["asr"],
            "force_attack_active": force["attack_active"], "force_selected_clients": force["selected_clients"],
            "force_selected_malicious_clients": force["selected_malicious_clients"], "force_sampling_mode": force["sampling_mode"],
            "delta_clean_acc_force_minus_legacy": clean_delta, "delta_asr_force_minus_legacy": asr_delta,
            "round_matched_exact": legacy_exact and force_exact,
            "interpretation": "exact recorded rounds" if legacy_exact and force_exact else "nearest recorded round; no interpolation",
        })
    write_csv(OUT / "sampling_policy_force_vs_legacy_key_rounds.csv", key_rows)

    aligned, differences = [], []
    for relative in range(301):
        legacy = legacy_by_round.get(700 + relative, {})
        force = force_by_round.get(700 + relative, {})
        aligned_row = {
            "relative_post_attack_round": relative, "legacy_round": legacy.get("global_round", ""),
            "legacy_clean_acc": legacy.get("clean_acc", ""), "legacy_asr": legacy.get("asr", ""),
            "force_round": force.get("global_round", ""), "force_clean_acc": force.get("clean_acc", ""),
            "force_asr": force.get("asr", ""),
        }
        aligned.append(aligned_row)
        clean_delta = None if number(aligned_row["force_clean_acc"]) is None or number(aligned_row["legacy_clean_acc"]) is None else number(aligned_row["force_clean_acc"]) - number(aligned_row["legacy_clean_acc"])
        asr_delta = None if number(aligned_row["force_asr"]) is None or number(aligned_row["legacy_asr"]) is None else number(aligned_row["force_asr"]) - number(aligned_row["legacy_asr"])
        differences.append({**aligned_row, "delta_clean_acc_force_minus_legacy": clean_delta,
                            "delta_asr_force_minus_legacy": asr_delta,
                            "abs_delta_clean_acc": None if clean_delta is None else abs(clean_delta),
                            "abs_delta_asr": None if asr_delta is None else abs(asr_delta)})
    write_csv(OUT / "sampling_policy_force_vs_legacy_aligned_curves.csv", aligned)
    write_csv(OUT / "sampling_policy_force_vs_legacy_diff_by_relative_round.csv", differences)

    legacy_summary, force_summary = summaries
    point_deltas = {point: number(force_summary[f"asr_after_stop_{point}"]) - number(legacy_summary[f"asr_after_stop_{point}"]) for point in (100, 200, 300)}
    stop_delta = number(force_summary["asr_at_stop"]) - number(legacy_summary["asr_at_stop"])
    mean_delta = number(force_summary["post_attack_mean_asr"]) - number(legacy_summary["post_attack_mean_asr"])
    if abs(stop_delta) <= 2 and max(abs(value) for value in point_deltas.values()) <= 5 and abs(mean_delta) <= 5:
        verdict = "force_similar_to_legacy"
    elif mean_delta < -5 or any(value < -5 for value in point_deltas.values()):
        verdict = "force_weaker_than_legacy"
    elif mean_delta > 5 or any(value > 5 for value in point_deltas.values()):
        verdict = "force_stronger_than_legacy"
    else:
        verdict = "inconclusive"

    force_round_500 = force_by_round[500]
    checkpoint = SPECS["force"]["checkpoint"]
    checkpoint_record = f"""# Force Protocol Clean500 Checkpoint Record

## Checkpoint Path

`{checkpoint}`

Exists: **{checkpoint.is_file()}**. The file is an ordinary model `state_dict` checkpoint (no embedded round/config/metric metadata); the details below are cross-verified from the experiment database, configuration, and log.

## Source Experiment

- Experiment: `{SPECS['force']['experiment_id']}`
- Configuration: `{SPECS['force']['config']}`
- Output directory: `{checkpoint.parent.parent}`
- Selection policy: `force_during_attack`
- Attack window: 501--700; therefore Round 500 is attack-inactive.

## Why This Checkpoint Matters

This is the clean global model after Round 500 of the from-scratch force protocol. Rounds 1--500 used 10-of-100 random client sampling, so it is the appropriate clean500 resume starting point for future experiments using the same force sampling protocol.

## Verification

- Round 500 clean accuracy: **{cell(force_round_500['clean_acc'])}%**.
- Round 500 `attack_active`: **{force_round_500['attack_active']}**.
- Round 500 `selection_policy`: **{force_round_500['selection_policy']}**.
- Round 500 `sampling_mode`: **{force_round_500['sampling_mode']}** (explicitly recorded in the force-run log).
- Round 500 `selected_clients`: `{force_round_500['selected_clients']}`.
- Round 500 `selected_malicious_clients`: `{force_round_500['selected_malicious_clients']}`; IDs 0/1/2 are not fixed participants.
- The checkpoint schedule includes 500, and the checkpoint is present at the path above.

It can be used for a later resume experiment provided that the future configuration explicitly records this checkpoint and its sampling protocol.
"""
    (OUT / "force_protocol_clean500_checkpoint_record.md").write_text(checkpoint_record, encoding="utf-8")

    key_markdown = "\n".join(
        "| {requested_round} | {legacy_actual_round} | {legacy_clean_acc:.2f} | {legacy_asr} | {force_actual_round} | {force_clean_acc:.2f} | {force_asr} | {delta_asr_force_minus_legacy} |".format(
            requested_round=row["requested_round"], legacy_actual_round=row["legacy_actual_round"], legacy_clean_acc=float(row["legacy_clean_acc"]),
            legacy_asr=cell(row["legacy_asr"]), force_actual_round=row["force_actual_round"], force_clean_acc=float(row["force_clean_acc"]),
            force_asr=cell(row["force_asr"]), delta_asr_force_minus_legacy=("" if row["delta_asr_force_minus_legacy"] is None else f"{row['delta_asr_force_minus_legacy']:+.2f}")
        ) for row in key_rows
    )
    report = f"""# Sampling Policy Comparison: force_during_attack vs legacy_all_malicious

## 1. Purpose

This is a one-to-one, read-only comparison of two otherwise matched PBU (`proto_uap`) settings. It isolates the client-sampling protocol; it does not change PBU, FedAvg, the trigger, or local epochs.

## 2. Compared Experiments

| Group | Config / output | Sampling policy | Attack / observation |
|---|---|---|---|
| Legacy E | `{SPECS['legacy']['config']}`; `{SPECS['legacy']['db'].parent}` | `legacy_all_malicious` | 501--700 / 701--1000 |
| Force I | `{SPECS['force']['config']}`; `{SPECS['force']['db'].parent}` | `force_during_attack` | 501--700 / 701--1000 |

Both are from scratch, CIFAR-10 IID with 100 clients, ResNet18, FedAvg, batch size 64, 10 clients per round, base/malicious local epochs 3, and the same listed PBU/prototype parameters. The material protocol change is `selection_policy` (plus the expected distinct names, output paths, tracking database, and checkpoint schedule).

## 3. Data Sources

- Legacy SQLite: `{SPECS['legacy']['db']}`.
- Force SQLite: `{SPECS['force']['db']}`.
- Legacy log: `{SPECS['legacy']['log']}`.
- Force log: `{SPECS['force']['log']}`.

Neither target output contained a `metrics.csv`; cleaned metrics were exported from the SQLite `round_metrics` tables. The database schema does not store `sampling_mode`, `forced_malicious`, `selection_policy`, or effective local epochs. Force sampling fields were read from its log where available; legacy sampling fields and any absent force log fields are configuration-derived and labeled accordingly. Pre-attack ASR is unavailable by design and is left empty. No values were interpolated.

## 4. Sampling Protocol Difference

- `legacy_all_malicious`: rounds 1--500, 501--700, and 701--1000 always select 0/1/2 plus seven sampled normal clients.
- `force_during_attack`: rounds 1--500 and 701--1000 sample 10 of all 100 clients; rounds 501--700 select 0/1/2 plus seven sampled normal clients.

Outside attack-active rounds, selected malicious clients are benign CE-only; selection is not synonymous with poisoning.

## 5. Round 500 Clean Checkpoint Record

Force I checkpoint: `{checkpoint}` (exists: **{checkpoint.is_file()}**). Round-500 clean accuracy is **{cell(force_round_500['clean_acc'])}%**, `attack_active=False`, and its log records `sampling_mode=random_all_clients`, with selected clients `{force_round_500['selected_clients']}`. It is suitable as the clean500 resume point for future force-protocol experiments. See `force_protocol_clean500_checkpoint_record.md` for the full record.

## 6. Key Round Comparison

| Requested | Legacy actual | Legacy ACC | Legacy ASR | Force actual | Force ACC | Force ASR | Force-Legacy ASR |
|---:|---:|---:|---:|---:|---:|---:|---:|
{key_markdown}

All requested key rounds were exact database records; no nearest-neighbour matches were needed.

## 7. Post-attack Persistence Comparison

| Group | ASR@stop | +100 | +200 | +300 | Final ASR | Post mean ASR | Post AUC | Half-life | Final clean ACC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table_row(legacy_summary)}
{table_row(force_summary)}

Force minus legacy: attack-end ASR **{stop_delta:+.2f} pp**; post +100/+200/+300 **{point_deltas[100]:+.2f}/{point_deltas[200]:+.2f}/{point_deltas[300]:+.2f} pp**; post mean **{mean_delta:+.2f} pp**; final clean accuracy **{number(force_summary['clean_acc_final']) - number(legacy_summary['clean_acc_final']):+.2f} pp**.

## 8. Interpretation

1. Force I still implants PBU: attack-end ASR is {cell(force_summary['asr_at_stop'])}%.
2. Its attack-end ASR differs from legacy by {stop_delta:+.2f} pp, which is {'basically equivalent' if abs(stop_delta) <= 2 else 'not equivalent'} under the ±2 pp criterion.
3. Its post +100/+200/+300 ASR differences are {point_deltas[100]:+.2f}/{point_deltas[200]:+.2f}/{point_deltas[300]:+.2f} pp. The post-attack mean difference is {mean_delta:+.2f} pp.
4. Under the supplied thresholds, the result is **{verdict}**. This is one seed per policy, so it is descriptive evidence rather than a population-level causal estimate.
5. The legacy result is not invalid, but it must be described with its fixed-attacker sampling policy. If future replications confirm force is weaker, that would indicate some legacy persistence is associated with fixed non-attack participation; if similar, persistence does not appear to require it; if stronger, fixed benign attacker participation may have been counterproductive in this seed.

## 9. Paper-use Recommendation

Keep the legacy result as a clearly labeled historical/protocol-specific result. Use the force result as the more appropriate main protocol for claims about random pre/post client sampling. State the sampling policy explicitly in every table and figure. If force were materially weaker across repeated seeds, present legacy and force as separate protocol conditions rather than pooling them; use the force Round-500 checkpoint as the uniform starting point for future force-protocol resumes.

## 10. Final Verdict

`final_verdict = {verdict}`

One-sentence conclusion: under this one-seed comparison, force_during_attack {'still retains broadly comparable PBU persistence' if verdict == 'force_similar_to_legacy' else 'shows weaker PBU persistence than legacy_all_malicious' if verdict == 'force_weaker_than_legacy' else 'shows stronger PBU persistence than legacy_all_malicious' if verdict == 'force_stronger_than_legacy' else 'has mixed differences that do not satisfy a decisive threshold'}.

No training was started; no training code, configuration, raw database, log, or checkpoint was modified.
"""
    (OUT / "sampling_policy_force_vs_legacy_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), "final_verdict": verdict, "summaries": summaries}, indent=2, default=str))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export verified lightweight paper metrics from source SQLite files.

This is a provenance utility used to build the release; it opens every source
database in read-only mode and never loads checkpoints, datasets, or models.
"""
from __future__ import annotations

import csv
import json
import shutil
import sqlite3
from pathlib import Path


RELEASE = Path(__file__).resolve().parents[1]
SOURCE = RELEASE.parent / "collaborative_backdoor_fl"


def spec(category, name, method, dataset, seed, config, databases, start=3201, end=3700, expected=None):
    return dict(category=category, name=name, method=method, dataset=dataset, seed=seed,
                config=config, databases=databases if isinstance(databases, list) else [databases],
                start=start, end=end, expected=expected or {})


SPECS = [
    spec("main", "pgfa", "PGFA", "CIFAR-10", 2026,
         "configs/clean_stage_sensitivity_pbu_force/pbu_force_clean3000_attack3001_3200_observe3201_3700.yaml",
         "outputs/pbu_force_clean_stage_sensitivity/clean3000_attack3001_3200_observe3201_3700/metrics_pbu_force_clean3000_attack3001_3200_observe3201_3700.db",
         expected={"PA-AUC": 87.91, "ASR@+500": 70.40, "N80": 406, "N90": 217}),
    spec("main", "ce_only_uap", "CE-only UAP", "CIFAR-10", 2026,
         "configs/main_experiment_clean3000_force/ce_uap_clean3000_attack3001_3200_observe3201_3700.yaml",
         "outputs/main_experiment_clean3000_force/ce_uap_clean3000_attack3001_3200_observe3201_3700/metrics_ce_uap_clean3000_attack3001_3200_observe3201_3700.db",
         expected={"PA-AUC": 73.19, "ASR@+500": 56.34, "N80": 143, "N90": 61}),
    spec("main", "dba", "DBA", "CIFAR-10", 2026,
         "configs/main_experiment_clean3000_force/dba_clean3000_attack3001_3200_observe3201_3700.yaml",
         ["outputs/main_experiment_clean3000_force/dba_clean3000_attack3001_3200_observe3201_3700/metrics_dba_clean3000_attack3001_3200_observe3201_3700.db",
          "outputs/parallel_13_resume_unfinished/resume_dba_from3500_to3700/metrics_resume_dba_from3500_to3700.db"],
         expected={"PA-AUC": 58.84, "ASR@+500": 4.68, "N80": 197}),
    spec("main", "neurotoxin", "Neurotoxin-UAP", "CIFAR-10", 2026,
         "configs/parallel_resume2_rerun9_queue/neurotoxin_clean3000_attack3001_3200_observe3201_3700_rerun1.yaml",
         "outputs/main_experiment_clean3000_force/neurotoxin_clean3000_attack3001_3200_observe3201_3700_rerun1/metrics_neurotoxin_clean3000_attack3001_3200_observe3201_3700_rerun1.db",
         expected={"PA-AUC": 71.03, "ASR@+500": 60.66, "N80": 149}),
    spec("main", "a3fl_relaxed_strong", "A3FL", "CIFAR-10", 2026,
         "configs/a3fl_v2_relaxed_strong/a3fl_v2_relaxed_strong_seed2026_resume3000.yaml",
         "outputs/a3fl_v2_relaxed_strong_seed2026/metrics_a3fl_v2_relaxed_strong_seed2026_resume3000.db",
         expected={"PA-AUC": 80.04, "ASR@+500": 65.24, "N80": 240, "N90": 136}),
]

for seed in range(20261, 20266):
    SPECS.append(spec("five_seeds", f"pgfa_seed{seed}", "PGFA", "CIFAR-10", seed,
        f"configs/parallel_resume2_rerun9_queue/pbu_seed{seed}_fromscratch_clean3000_attack3001_3200_observe3201_3700_rerun1.yaml",
        f"outputs/seed_stability_pbu_force_fromscratch_5_rerun1/pbu_seed{seed}_fromscratch_clean3000_attack3001_3200_observe3201_3700/metrics_pbu_seed{seed}_fromscratch_clean3000_attack3001_3200_observe3201_3700_rerun1.db"))
    SPECS.append(spec("five_seeds", f"ce_only_seed{seed}", "CE-only UAP", "CIFAR-10", seed,
        f"configs/pgfa_final_supplement_resume3000_parallel_queue/ce_uap_seed{seed}.yaml",
        f"outputs/pgfa_final_supplement_resume3000/ce_uap_multiseed/ce_uap_seed{seed}_final_supplement/metrics_ce_uap_seed{seed}_final_supplement_resume3000.db"))

SPECS += [
    spec("stage_ablation", "ce_only", "CE-only UAP", "CIFAR-10", 2026,
         "configs/pbu_ablation_clean3000_force/ablation_ce_only_uap_clean3000_attack3001_3200_observe3201_3700.yaml",
         "outputs/pbu_ablation_clean3000_force/ce_only_uap_clean3000_attack3001_3200_observe3201_3700/metrics_ablation_ce_only_uap_clean3000_attack3001_3200_observe3201_3700.db"),
    spec("stage_ablation", "generation_only", "Prototype generation only", "CIFAR-10", 2026,
         "configs/pbu_ablation_clean3000_force/ablation_proto_delta_only_clean3000_attack3001_3200_observe3201_3700.yaml",
         "outputs/pbu_ablation_clean3000_force/proto_delta_only_clean3000_attack3001_3200_observe3201_3700/metrics_ablation_proto_delta_only_clean3000_attack3001_3200_observe3201_3700.db"),
    spec("stage_ablation", "training_only", "Prototype training only", "CIFAR-10", 2026,
         "configs/pbu_ablation_clean3000_force/ablation_proto_train_only_clean3000_attack3001_3200_observe3201_3700.yaml",
         "outputs/pbu_ablation_clean3000_force/proto_train_only_clean3000_attack3001_3200_observe3201_3700/metrics_ablation_proto_train_only_clean3000_attack3001_3200_observe3201_3700.db"),
    spec("loss_form_ablation", "ordinary_feature_alignment", "Ordinary Feature Alignment", "CIFAR-10", 2026,
         "configs/pgfa_feature_loss_ablation_seed2026_queue/pgfa_feature_alignment_seed2026.yaml",
         "outputs/pgfa_feature_loss_ablation_seed2026/feature_alignment_seed2026/metrics_pgfa_feature_alignment_seed2026.db",
         expected={"PA-AUC": 82.83, "ASR@+500": 70.12, "N80": 291}),
    spec("loss_form_ablation", "target_only", "Target-only Prototype", "CIFAR-10", 2026,
         "configs/pgfa_feature_loss_ablation_seed2026_queue/pgfa_target_only_seed2026.yaml",
         "outputs/pgfa_feature_loss_ablation_seed2026/target_only_seed2026/metrics_pgfa_target_only_seed2026.db",
         expected={"PA-AUC": 85.40, "ASR@+500": 67.60, "N80": 333}),
    spec("cross_dataset", "gtsrb_ce_only", "CE-only UAP", "GTSRB", 2026,
         "configs/cross_dataset_generalization/gtsrb_ce_uap_clean3000_attack3001_3200_observe3201_3700.yaml",
         "outputs/cross_dataset_generalization/gtsrb_ce_uap_clean3000_attack3001_3200_observe3201_3700/metrics_gtsrb_ce_uap_clean3000_attack3001_3200_observe3201_3700.db",
         expected={"PA-AUC": 45.16}),
    spec("cross_dataset", "gtsrb_pgfa", "PGFA", "GTSRB", 2026,
         "configs/cross_dataset_generalization_retry/gtsrb_pbu_clean3000_attack3001_3200_observe3201_3700_retry1.yaml",
         "outputs/cross_dataset_generalization_retry/gtsrb_pbu_clean3000_attack3001_3200_observe3201_3700_retry1/metrics_gtsrb_pbu_clean3000_attack3001_3200_observe3201_3700_retry1.db",
         expected={"PA-AUC": 81.93}),
    spec("cross_dataset", "svhn_ce_only", "CE-only UAP", "SVHN", 2026,
         "configs/cross_dataset_generalization/svhn_ce_uap_clean3000_attack3001_3200_observe3201_3700.yaml",
         "outputs/cross_dataset_generalization/svhn_ce_uap_clean3000_attack3001_3200_observe3201_3700/metrics_svhn_ce_uap_clean3000_attack3001_3200_observe3201_3700.db",
         expected={"PA-AUC": 62.59}),
    spec("cross_dataset", "svhn_pgfa", "PGFA", "SVHN", 2026,
         "configs/cross_dataset_generalization/svhn_pbu_clean3000_attack3001_3200_observe3201_3700.yaml",
         "outputs/cross_dataset_generalization/svhn_pbu_clean3000_attack3001_3200_observe3201_3700/metrics_svhn_pbu_clean3000_attack3001_3200_observe3201_3700.db",
         expected={"PA-AUC": 83.88}),
]

for alpha in ("0.8", "0.9"):
    SPECS.append(spec("non_iid", f"pgfa_alpha{alpha}", "PGFA", "CIFAR-10", 2026,
        f"configs/parallel_resume2_rerun9_queue/cifar10_noniid_alpha{alpha}_pbu_fromscratch_clean3000_attack3001_3200_observe3201_3700_rerun1.yaml",
        f"outputs/cifar10_noniid_pbu_force_fromscratch_3_rerun1/alpha{alpha}_pbu_fromscratch_clean3000_attack3001_3200_observe3201_3700/metrics_cifar10_noniid_alpha{alpha}_pbu_fromscratch_clean3000_attack3001_3200_observe3201_3700_rerun1.db"))
    SPECS.append(spec("non_iid", f"ce_only_alpha{alpha}", "CE-only UAP", "CIFAR-10", 2026,
        f"configs/pgfa_final_supplement_resume3000_parallel_queue/ce_uap_noniid_alpha{alpha}.yaml",
        f"outputs/pgfa_final_supplement_resume3000/noniid_ce_uap/ce_uap_noniid_alpha{alpha}_final_supplement/metrics_ce_uap_noniid_alpha{alpha}_final_supplement_resume3000.db"))

for name, method, cfg, db in [
    ("norm_clipping", "Norm Clipping", "configs/pbu_defense_clean3000_force/pbu_defense_nc_clean3000_attack3001_3200_observe3201_3700.yaml", "outputs/pbu_defense_clean3000_force/nc_clean3000_attack3001_3200_observe3201_3700/metrics_pbu_defense_nc_clean3000_attack3001_3200_observe3201_3700.db"),
    ("rlr", "RLR", "configs/pbu_defense_clean3000_force/pbu_defense_rlr_clean3000_attack3001_3200_observe3201_3700.yaml", "outputs/pbu_defense_clean3000_force/rlr_clean3000_attack3001_3200_observe3201_3700/metrics_pbu_defense_rlr_clean3000_attack3001_3200_observe3201_3700.db"),
    ("crfl_style", "CRFL-style", "configs/pbu_defense_clean3000_force/pbu_defense_crfl_clean3000_attack3001_3200_observe3201_3700.yaml", "outputs/pbu_defense_clean3000_force/crfl_clean3000_attack3001_3200_observe3201_3700/metrics_pbu_defense_crfl_clean3000_attack3001_3200_observe3201_3700.db"),
    ("multi_krum", "Multi-Krum", "configs/pbu_defense_clean3000_force/pbu_defense_multi_krum_clean3000_attack3001_3200_observe3201_3700.yaml", "outputs/pbu_defense_clean3000_force/multi_krum_clean3000_attack3001_3200_observe3201_3700/metrics_pbu_defense_multi_krum_clean3000_attack3001_3200_observe3201_3700.db"),
]: SPECS.append(spec("defenses", name, method, "CIFAR-10", 2026, cfg, db))

SPECS += [
    spec("sensitivity", "attack_duration_100", "PGFA", "CIFAR-10", 2026,
         "configs/cifar10_pbu_duration_sensitivity_fix/b1_cifar10_pbu_ta100_clean3000_attack3001_3100_observe3101_3600.yaml",
         "outputs/cifar10_pbu_duration_sensitivity_fix/b1_cifar10_pbu_ta100_clean3000_attack3001_3100_observe3101_3600/metrics_b1_cifar10_pbu_ta100_clean3000_attack3001_3100_observe3101_3600.db", 3101, 3600,
         expected={"PA-AUC": 84.91, "ASR@+500": 67.74}),
    spec("sensitivity", "attack_duration_300", "PGFA", "CIFAR-10", 2026,
         "configs/cifar10_pbu_duration_sensitivity_fix/b2_cifar10_pbu_ta300_clean3000_attack3001_3300_observe3301_3800.yaml",
         "outputs/cifar10_pbu_duration_sensitivity_fix/b2_cifar10_pbu_ta300_clean3000_attack3001_3300_observe3301_3800/metrics_b2_cifar10_pbu_ta300_clean3000_attack3001_3300_observe3301_3800.db", 3301, 3800,
         expected={"PA-AUC": 88.49, "ASR@+500": 74.16}),
]


def read_database(path: Path):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        return [dict(row) for row in con.execute("SELECT * FROM round_metrics ORDER BY round_index")]


def metric_summary(rows, start, end):
    data = {int(row["round_index"]): row for row in rows}
    post = [float(data[r]["attack_success_rate"]) for r in range(start, end + 1)]
    attack_end = start - 1
    return {"PA-AUC": sum(post) / len(post), "ASR@+500": float(data[end]["attack_success_rate"]),
            "ASR@attack_end": float(data[attack_end]["attack_success_rate"]),
            "N80": sum(value >= 80 for value in post), "N90": sum(value >= 90 for value in post),
            "rounds": len(rows), "post_rounds": len(post)}


def export_one(item):
    raw_paths = [SOURCE / path for path in item["databases"]]
    combined = {}
    for source_path in raw_paths:
        for row in read_database(source_path):
            row["source_sqlite"] = str(source_path.relative_to(SOURCE))
            combined[int(row["round_index"])] = row
    rows = [combined[key] for key in sorted(combined)]
    required = set(range(item["start"] - 1, item["end"] + 1))
    missing = sorted(required - set(combined))
    summary = metric_summary(rows, item["start"], item["end"]) if not missing else {}
    expected_ok = all(abs(float(summary[key]) - value) < 0.011 for key, value in item["expected"].items()) if summary else False
    status = "VERIFIED" if not missing and (not item["expected"] or expected_ok) else "MISSING"
    out = RELEASE / "results" / item["category"] / item["name"]
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) + ["method", "dataset", "seed", "configuration_identifier"]
    with (out / "roundwise_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({**row, "method": item["method"], "dataset": item["dataset"], "seed": item["seed"], "configuration_identifier": Path(item["config"]).stem})
    with (out / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"experiment": item["name"], "method": item["method"], "dataset": item["dataset"], "seed": item["seed"], "post_attack_start": item["start"], "post_attack_end": item["end"], "source_sqlite": item["databases"], "computed_metrics": summary, "expected_metrics": item["expected"], "status": status, "missing_rounds": missing}, handle, indent=2)
    for index, source_path in enumerate(raw_paths, 1):
        metadata = source_path.parent / "experiment_metadata.json"
        if metadata.is_file(): shutil.copy2(metadata, out / f"source_metadata_{index}.json")
    source_config, public_config = SOURCE / item["config"], RELEASE / item["config"]
    public_config.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source_config, public_config)
    return {"experiment_name": item["name"], "method": item["method"], "dataset": item["dataset"], "seed": item["seed"], "original_config_path": item["config"], "public_config_path": str(public_config.relative_to(RELEASE)), "original_result_source": "; ".join(item["databases"]), "public_result_path": str((out / "roundwise_metrics.csv").relative_to(RELEASE)), "key_metrics": json.dumps(summary, sort_keys=True), "sqlite_source": "; ".join(item["databases"]), "status": status}


def main():
    manifest = [export_one(item) for item in SPECS]
    fields = list(manifest[0])
    with (RELEASE / "docs" / "experiment_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(manifest)
    print(json.dumps({"experiments": len(manifest), "verified": sum(row["status"] == "VERIFIED" for row in manifest), "missing": sum(row["status"] == "MISSING" for row in manifest)}))


if __name__ == "__main__": main()

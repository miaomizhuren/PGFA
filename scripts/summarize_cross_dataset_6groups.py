"""Export the five completed cross-dataset runs plus CIFAR-10 Full PBU.

This is an offline SQLite-metrics summarizer.  It neither loads model
checkpoints nor starts training.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "cross_dataset_generalization_6groups_summary"
POST_START, POST_END = 3201, 3700
KEY_ROUNDS = (3200, 3201, 3300, 3400, 3500, 3600, 3700)
SPECS = (
    ("CIFAR-10", "PBU", "proto_uap", ROOT / "outputs/pbu_force_clean_stage_sensitivity/clean3000_attack3001_3200_observe3201_3700"),
    ("CIFAR-100", "CE-UAP", "ce_uap", ROOT / "outputs/cross_dataset_generalization/cifar100_ce_uap_clean3000_attack3001_3200_observe3201_3700"),
    ("CIFAR-100", "PBU", "proto_uap", ROOT / "outputs/cross_dataset_generalization/cifar100_pbu_clean3000_attack3001_3200_observe3201_3700"),
    ("GTSRB", "CE-UAP", "ce_uap", ROOT / "outputs/cross_dataset_generalization/gtsrb_ce_uap_clean3000_attack3001_3200_observe3201_3700"),
    ("SVHN", "CE-UAP", "ce_uap", ROOT / "outputs/cross_dataset_generalization/svhn_ce_uap_clean3000_attack3001_3200_observe3201_3700"),
    ("SVHN", "PBU", "proto_uap", ROOT / "outputs/cross_dataset_generalization/svhn_pbu_clean3000_attack3001_3200_observe3201_3700"),
)


def metrics_db(directory: Path) -> Path:
    found = sorted(directory.glob("*.db"))
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one metrics DB in {directory}, found {found}")
    return found[0]


def read_rounds(db_path: Path) -> dict[int, dict[str, float]]:
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT round_index, clean_accuracy, attack_success_rate FROM round_metrics "
            "WHERE round_index BETWEEN ? AND ? ORDER BY round_index",
            (3001, POST_END),
        ).fetchall()
    result = {
        int(round_index): {"clean_acc": float(clean_acc), "asr": float(asr)}
        for round_index, clean_acc, asr in rows
    }
    expected = set(range(3001, POST_END + 1))
    if set(result) != expected:
        raise RuntimeError(f"{db_path} does not contain the full 3001--3700 curve")
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    curves: dict[str, dict[int, dict[str, float]]] = {}
    summary_rows: list[dict] = []
    key_rows: list[dict] = []

    for dataset, display_name, method, directory in SPECS:
        run_id = f"{dataset} / {display_name}"
        rounds = read_rounds(metrics_db(directory))
        curves[run_id] = rounds
        post_asr = [rounds[round_index]["asr"] for round_index in range(POST_START, POST_END + 1)]
        summary_rows.append({
            "dataset": dataset,
            "method": display_name,
            "attack_method": method,
            "output_dir": str(directory.relative_to(ROOT)),
            "attack_start": 3001,
            "attack_end": 3200,
            "post_attack_start": POST_START,
            "final_round": POST_END,
            "clean_acc_at_attack_end": rounds[3200]["clean_acc"],
            "asr_at_attack_end": rounds[3200]["asr"],
            "clean_acc_final": rounds[3700]["clean_acc"],
            "asr_post_1": rounds[3201]["asr"],
            "asr_post_100": rounds[3300]["asr"],
            "asr_post_200": rounds[3400]["asr"],
            "asr_post_300": rounds[3500]["asr"],
            "asr_post_400": rounds[3600]["asr"],
            "asr_post_500": rounds[3700]["asr"],
            "post_attack_auc_3201_3700": sum(post_asr),
            "asr_ge_80_duration_rounds": sum(value >= 80.0 for value in post_asr),
            "asr_ge_50_duration_rounds": sum(value >= 50.0 for value in post_asr),
            "mean_post_asr": sum(post_asr) / len(post_asr),
            "max_post_asr": max(post_asr),
            "min_post_asr": min(post_asr),
            "notes": "Completed metrics DB; no checkpoint/model data read.",
        })
        for round_index in KEY_ROUNDS:
            key_rows.append({
                "dataset": dataset,
                "method": display_name,
                "global_round": round_index,
                "relative_post_round": round_index - 3200 if round_index >= POST_START else 0,
                "clean_acc": rounds[round_index]["clean_acc"],
                "asr": rounds[round_index]["asr"],
            })

    curve_rows: list[dict] = []
    for relative_post_round, global_round in enumerate(range(POST_START, POST_END + 1), start=1):
        row = {"relative_post_round": relative_post_round, "global_round": global_round}
        for run_id, rounds in curves.items():
            row[f"{run_id}_clean_acc"] = rounds[global_round]["clean_acc"]
            row[f"{run_id}_asr"] = rounds[global_round]["asr"]
        curve_rows.append(row)

    write_csv(OUTPUT / "cross_dataset_6groups_summary.csv", summary_rows)
    write_csv(OUTPUT / "cross_dataset_6groups_key_rounds.csv", key_rows)
    write_csv(OUTPUT / "cross_dataset_6groups_post_attack_curves.csv", curve_rows)

    report = [
        "# Six-Group Cross-Dataset Summary",
        "",
        "This offline aggregation contains the five completed cross-dataset runs and the completed CIFAR-10 Full PBU main-experiment run.",
        "",
        "Excluded: GTSRB PBU, because its metrics database has no completed round rows after the round-3001 CUDA failure.",
        "",
        "All rows use the common attack window 3001--3200 and post-attack window 3201--3700.",
        "",
        "## Included runs",
        "",
        "| Dataset | Method | Final clean accuracy | ASR at post-round 500 | Post-attack ASR AUC |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        report.append(
            f"| {row['dataset']} | {row['method']} | {row['clean_acc_final']:.2f} | "
            f"{row['asr_post_500']:.2f} | {row['post_attack_auc_3201_3700']:.2f} |"
        )
    report.extend([
        "",
        "## Files",
        "",
        "- `cross_dataset_6groups_summary.csv`: one row per experiment with persistence metrics.",
        "- `cross_dataset_6groups_key_rounds.csv`: clean accuracy and ASR at attack end and selected post-attack rounds.",
        "- `cross_dataset_6groups_post_attack_curves.csv`: 500 aligned post-attack rounds for all six runs.",
        "",
        "No training was started and no existing output, checkpoint, or metrics database was modified.",
    ])
    (OUTPUT / "cross_dataset_6groups_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

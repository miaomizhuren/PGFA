#!/usr/bin/env python3
"""Summarize completed or partial PBU poisoning-window experiments."""

import csv
import sqlite3
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/window_search"
OUTPUT = ROOT / "outputs/window_search/window_search_summary.csv"
NAMES = [f"w{index:02d}_start{start}_end{end}" for index, start, end in (
    (1, 60, 110), (2, 40, 90), (3, 50, 100), (4, 70, 120),
    (5, 80, 130), (6, 60, 130), (7, 80, 150),
)]
FIELDS = [
    "experiment_id", "attack_start", "attack_end", "attack_length",
    "post_attack_start", "pa_auc_start", "pa_auc_end", "CleanACC@250",
    "ASR@attack_end", "ASR@post_attack_start", "ASR@150", "ASR@180",
    "ASR@200", "ASR@250", "PA-AUC", "Half-life", "PSNR", "SSIM",
    "L_inf", "mechanism_fixed_margin_mean", "mechanism_dynamic_margin_mean",
]


def value(record, key):
    return "" if record is None or record.get(key) is None else record[key]


def post_attack_auc(records, start, end):
    points = [(round_index, records[round_index]["asr"]) for round_index in sorted(records)
              if start <= round_index <= end and records[round_index]["asr"] is not None]
    if len(points) < 2 or points[0][0] != start or points[-1][0] != end:
        return ""
    area = sum((right[0] - left[0]) * (left[1] + right[1]) / 2 for left, right in zip(points, points[1:]))
    return area / (end - start)


def mean_margin(path, column):
    if not path.exists():
        return ""
    with path.open(newline="", encoding="utf-8") as stream:
        values = [float(row[column]) for row in csv.DictReader(stream)
                  if column in row and row[column] not in (None, "")]
    return sum(values) / len(values) if values else ""


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in NAMES:
        config_path = CONFIG_DIR / f"{name}.yaml"
        row = {field: "" for field in FIELDS}
        row["experiment_id"] = name
        if not config_path.exists():
            rows.append(row)
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        attack, evaluation = config["attack"], config.get("evaluation", {})
        start = attack.get("start_round", attack.get("attack_start_round"))
        end = attack.get("end_round", attack.get("attack_end_round"))
        row.update({"attack_start": start, "attack_end": end, "attack_length": end - start + 1,
                    "post_attack_start": evaluation.get("post_attack_start"),
                    "pa_auc_start": evaluation.get("pa_auc_start"), "pa_auc_end": evaluation.get("pa_auc_end")})
        output_dir = ROOT / config["experiment"]["output_dir"]
        db_path = output_dir / config.get("tracking", {}).get("database_name", "metrics.db")
        records = {}
        if db_path.exists():
            try:
                with sqlite3.connect(db_path) as connection:
                    for item in connection.execute("SELECT round_index, clean_accuracy, attack_success_rate, psnr, ssim, linf FROM round_metrics"):
                        records[item[0]] = {"clean": item[1], "asr": item[2], "psnr": item[3], "ssim": item[4], "linf": item[5]}
            except sqlite3.DatabaseError:
                records = {}
        end_record, post_record, last_record = records.get(end), records.get(end + 1), records.get(250)
        row.update({"CleanACC@250": value(last_record, "clean"), "ASR@attack_end": value(end_record, "asr"),
                    "ASR@post_attack_start": value(post_record, "asr"), "ASR@150": value(records.get(150), "asr"),
                    "ASR@180": value(records.get(180), "asr"), "ASR@200": value(records.get(200), "asr"),
                    "ASR@250": value(last_record, "asr"), "PA-AUC": post_attack_auc(records, end + 1, 250),
                    "PSNR": value(last_record, "psnr"), "SSIM": value(last_record, "ssim"), "L_inf": value(last_record, "linf")})
        baseline = value(end_record, "asr")
        if baseline != "":
            row["Half-life"] = next((round_index for round_index in sorted(records)
                                      if round_index >= end + 1 and records[round_index]["asr"] is not None and records[round_index]["asr"] <= baseline / 2), "")
        margin_path = output_dir / "feature_margins.csv"
        row["mechanism_fixed_margin_mean"] = mean_margin(margin_path, "fixed_margin")
        row["mechanism_dynamic_margin_mean"] = mean_margin(margin_path, "dynamic_margin")
        rows.append(row)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()

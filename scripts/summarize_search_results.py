#!/usr/bin/env python3
"""Summarize quick-search round metrics, preferring SQLite then training logs."""

from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/search"
LOG_DIR = ROOT / "experiments/logs/search2"
SUMMARY_DIR = ROOT / "experiments/summaries/search2"
ROUNDS = (100, 110, 120, 130, 140, 150)
FIELDS = ["experiment_name", "config_path", "local_epochs", "training_cost", "lambda_delta", "lambda_train", "source_margin", "poison_ratio", "steps", "alpha", "temperature", *[f"CleanACC@{r}" for r in (100, 120, 130, 150)], *[f"ASR@{r}" for r in ROUNDS], "PSNR", "SSIM", "L_inf", "post_attack_auc_100_150", "decay_slope_100_150", "half_life_round", "status"]
LOG_PATTERN = re.compile(r"Round\s+(\d+).*?Clean ACC:\s+([\d.]+)%.*?ASR:\s+([\d.]+|N/A)(?:.*?PSNR:\s+([\d.]+)\s+dB)?(?:.*?SSIM:\s+([\d.]+))?(?:.*?L_inf:\s+([\d.]+))?")


def value(metrics: dict[int, dict], round_index: int, field: str):
    item = metrics.get(round_index, {}).get(field)
    return "" if item is None else item


def mean_auc(metrics: dict[int, dict], start: int, end: int) -> float | str:
    points = [(r, metrics[r]["asr"]) for r in sorted(metrics) if start <= r <= end and metrics[r].get("asr") is not None]
    if len(points) < 2:
        return ""
    area = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(points, points[1:]))
    return area / (points[-1][0] - points[0][0])


def slope(metrics: dict[int, dict], start: int, end: int) -> float | str:
    first, last = value(metrics, start, "asr"), value(metrics, end, "asr")
    return "" if first == "" or last == "" else (last - first) / (end - start)


def sqlite_metrics(path: Path) -> dict[int, dict]:
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(round_metrics)")}
        required = {"round_index", "clean_accuracy", "attack_success_rate"}
        if not required <= columns:
            raise ValueError("round_metrics schema lacks required columns")
        selected = ["round_index", "clean_accuracy", "attack_success_rate"] + [c for c in ("psnr", "ssim", "linf") if c in columns]
        rows = connection.execute(f"SELECT {', '.join(selected)} FROM round_metrics ORDER BY round_index").fetchall()
    return {row[0]: dict(zip(["round", "clean", "asr", *selected[3:]], row)) for row in rows}


def log_metrics(path: Path) -> dict[int, dict]:
    metrics = {}
    for match in LOG_PATTERN.finditer(path.read_text(encoding="utf-8", errors="replace")):
        r, clean, asr, psnr, ssim, linf = match.groups()
        metrics[int(r)] = {"round": int(r), "clean": float(clean), "asr": None if asr == "N/A" else float(asr), "psnr": float(psnr) if psnr else None, "ssim": float(ssim) if ssim else None, "linf": float(linf) if linf else None}
    return metrics


def latest_metric(metrics: dict[int, dict], key: str):
    values = [metrics[r].get(key) for r in sorted(metrics, reverse=True) if metrics[r].get(key) is not None]
    return values[0] if values else ""


def summarize(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    name = config["experiment"]["name"]
    adv = config["attack"]["trigger"]["adversarial"]
    proto = adv["prototype_contrastive"]
    local_epochs = config["federated"]["local_epochs"]
    row = {"experiment_name": name, "config_path": str(config_path.relative_to(ROOT)), "local_epochs": local_epochs, "training_cost": "higher" if local_epochs == 3 else "standard", "lambda_delta": proto["lambda_delta"], "lambda_train": proto["lambda_train"], "source_margin": proto["source_margin"], "poison_ratio": config["attack"]["poison_ratio"], "steps": adv["steps"], "alpha": adv["alpha"], "temperature": proto["temperature"], "status": "not_started"}
    db = ROOT / config["experiment"]["output_dir"] / config["tracking"]["database_name"]
    try:
        metrics = sqlite_metrics(db)
        row["status"] = "complete" if metrics and max(metrics) >= config["federated"]["num_rounds"] else "running"
    except (sqlite3.Error, OSError, ValueError):
        log = LOG_DIR / f"{name}.log"
        metrics = log_metrics(log) if log.exists() else {}
        if metrics:
            row["status"] = "complete" if max(metrics) >= config["federated"]["num_rounds"] else "running_log_fallback"
        elif log.exists() and "FAILED exit_code=" in log.read_text(encoding="utf-8", errors="replace"):
            row["status"] = "failed"
    for r in (100, 120, 130, 150): row[f"CleanACC@{r}"] = value(metrics, r, "clean")
    for r in ROUNDS: row[f"ASR@{r}"] = value(metrics, r, "asr")
    row["PSNR"], row["SSIM"], row["L_inf"] = latest_metric(metrics, "psnr"), latest_metric(metrics, "ssim"), latest_metric(metrics, "linf")
    row["post_attack_auc_100_150"] = mean_auc(metrics, 100, 150)
    row["decay_slope_100_150"] = slope(metrics, 100, 150)
    row["half_life_round"] = next((r for r in sorted(metrics) if r >= 100 and metrics[r].get("asr") is not None and metrics[r]["asr"] < 50), ">150" if metrics and max(metrics) >= 150 else "")
    return row


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows = [summarize(path) for path in sorted(CONFIG_DIR.glob("*.yaml"))]
    with (SUMMARY_DIR / "search_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    lines = ["# Quick search summary", "", "`post_attack_auc_100_150` is the trapezoid-integrated mean ASR (%), so it is comparable with point ASRs.", "", "| Experiment | Status | Clean@150 | ASR@120 | ASR@130 | ASR@150 | AUC@100–150 |", "|---|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['experiment_name']} | {r['status']} | {r.get('CleanACC@150','')} | {r.get('ASR@120','')} | {r.get('ASR@130','')} | {r.get('ASR@150','')} | {r.get('post_attack_auc_100_150','')} |" for r in rows]
    (SUMMARY_DIR / "search_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

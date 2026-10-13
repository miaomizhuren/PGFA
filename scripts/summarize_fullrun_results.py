#!/usr/bin/env python3
"""Summarize 200-round experiments from SQLite, with log fallback."""
from __future__ import annotations
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_search_results as common

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR, LOG_DIR, SUMMARY_DIR = ROOT / "configs/fullrun", ROOT / "experiments/logs/fullrun", ROOT / "experiments/summaries/fullrun"
ROUNDS = (100, 110, 120, 130, 140, 150, 160, 170, 180, 200)

def auc(metrics, end): return common.mean_auc(metrics, 100, end)
def summarize(path):
    config = common.yaml.safe_load(path.read_text(encoding="utf-8")); name = config["experiment"]["name"]
    local_epochs = config["federated"]["local_epochs"]
    row = {"experiment_name": name, "config_path": str(path.relative_to(ROOT)), "local_epochs": local_epochs, "training_cost": "higher" if local_epochs == 3 else "standard"}
    db = ROOT / config["experiment"]["output_dir"] / config["tracking"]["database_name"]
    try: metrics = common.sqlite_metrics(db); row["status"] = "complete" if metrics and max(metrics) >= 200 else "running"
    except (common.sqlite3.Error, OSError, ValueError):
        log = LOG_DIR / f"{name}.log"; metrics = common.log_metrics(log) if log.exists() else {}; row["status"] = "complete" if metrics and max(metrics) >= 200 else ("running_log_fallback" if metrics else "not_started")
    for r in (100, 120, 130, 150, 180, 200): row[f"CleanACC@{r}"] = common.value(metrics, r, "clean")
    for r in ROUNDS: row[f"ASR@{r}"] = common.value(metrics, r, "asr")
    row.update({"PSNR": common.latest_metric(metrics, "psnr"), "SSIM": common.latest_metric(metrics, "ssim"), "L_inf": common.latest_metric(metrics, "linf"), "post_attack_auc_100_180": auc(metrics, 180), "post_attack_auc_100_200": auc(metrics, 200), "half_life_round": next((r for r in sorted(metrics) if r >= 100 and metrics[r].get("asr") is not None and metrics[r]["asr"] < 50), ">200" if metrics and max(metrics) >= 200 else ""), "decay_slope_100_150": common.slope(metrics, 100, 150), "decay_slope_100_180": common.slope(metrics, 100, 180)})
    return row
def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True); rows = [summarize(p) for p in sorted(CONFIG_DIR.glob("*.yaml"))]
    fields = ["experiment_name", "config_path", "local_epochs", "training_cost", *[f"CleanACC@{r}" for r in (100,120,130,150,180,200)], *[f"ASR@{r}" for r in ROUNDS], "PSNR", "SSIM", "L_inf", "post_attack_auc_100_180", "post_attack_auc_100_200", "half_life_round", "decay_slope_100_150", "decay_slope_100_180", "status"]
    with (SUMMARY_DIR / "fullrun_summary.csv").open("w", newline="", encoding="utf-8") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (SUMMARY_DIR / "fullrun_summary.md").write_text("# Full-run summary\n\n" + "\n".join(f"- {r['experiment_name']}: {r['status']}" for r in rows) + "\n", encoding="utf-8")
if __name__ == "__main__": main()

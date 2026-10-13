#!/usr/bin/env python3
"""Rank completed quick-search configurations and flag quality risks."""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "experiments/summaries/search2"

def number(row, key):
    try: return float(row[key])
    except (KeyError, ValueError, TypeError): return None

def main() -> None:
    source = DIR / "search_summary.csv"
    if not source.exists(): raise SystemExit(f"Missing {source}; run summarize_search_results.py first.")
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    candidates = []
    for row in rows:
        needed = [number(row, f"ASR@{r}") for r in (120, 130, 150)] + [number(row, "CleanACC@150"), number(row, "post_attack_auc_100_150")]
        if row.get("status") != "complete" or any(x is None for x in needed): continue
        asr120, asr130, asr150, clean150, auc = needed
        score = .20 * asr120 + .25 * asr130 + .30 * asr150 + .15 * auc + .10 * clean150
        risks = []
        if clean150 < 78: score -= (78 - clean150) * 2; risks.append("high_risk:CleanACC@150<78")
        if (psnr := number(row, "PSNR")) is not None and psnr < 30: risks.append("stealth_risk:PSNR<30")
        if (ssim := number(row, "SSIM")) is not None and ssim < .95: risks.append("stealth_risk:SSIM<0.95")
        if asr120 < 70: risks.append("early_collapse:ASR@120<70")
        if number(row, "local_epochs") == 3: risks.append("higher_training_cost:local_epochs=3")
        row["score"], row["risk_flags"] = round(score, 6), "; ".join(risks) or "none"; candidates.append(row)
    candidates.sort(key=lambda r: r["score"], reverse=True)
    selected = candidates[:3]
    fields = ["rank", "experiment_name", "config_path", "local_epochs", "training_cost", "score", "risk_flags", "CleanACC@150", "ASR@120", "ASR@130", "ASR@150", "post_attack_auc_100_150", "PSNR", "SSIM", "L_inf"]
    for index, row in enumerate(selected, 1): row["rank"] = index
    with (DIR / "top_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(selected)
    lines = ["# Full-run candidates", "", "Only completed runs with all scoring values are eligible.", "", "| Rank | Experiment | Score | Risks |", "|---:|---|---:|---|"]
    lines += [f"| {r['rank']} | {r['experiment_name']} | {r['score']:.3f} | {r['risk_flags']} |" for r in selected]
    (DIR / "top_candidates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__": main()

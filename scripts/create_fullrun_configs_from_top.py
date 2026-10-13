#!/usr/bin/env python3
"""Create non-overwriting 200-round configs from ranked quick-search candidates."""
from __future__ import annotations
import argparse, csv
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--top", type=Path, default=ROOT / "experiments/summaries/search2/top_candidates.csv"); parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args(); top = args.top if args.top.is_absolute() else ROOT / args.top
    rows = list(csv.DictReader(top.open(encoding="utf-8")))[:args.limit]
    if not rows: raise SystemExit("No candidates found; summarize and select completed quick-search runs first.")
    destination = ROOT / "configs/fullrun"; destination.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = ROOT / row["config_path"]
        config = yaml.safe_load(source.read_text(encoding="utf-8")); name = f"fullrun_{row['experiment_name']}"
        if "local_epochs" not in config.get("federated", {}): raise SystemExit(f"Candidate lacks federated.local_epochs: {source}")
        if (destination / f"{name}.yaml").exists():
            name = f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        config["experiment"]["name"] = name; config["experiment"]["output_dir"] = f"./outputs/{name}"; config["federated"]["num_rounds"] = 200; config["tracking"]["database_name"] = f"metrics_{name}.db"
        target = destination / f"{name}.yaml"
        target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(f"CREATED {target.relative_to(ROOT)}")

if __name__ == "__main__": main()

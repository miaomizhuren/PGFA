#!/usr/bin/env python3
"""Measure target/source feature geometry for saved mechanism250 checkpoints."""

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from attacks.triggers import apply_trigger
from data.dataset import load_cifar10
from models.resnet18 import build_resnet18


CONFIG_DIR = ROOT / "configs/paper100_mechanism250"
SUMMARY_DIR = ROOT / "experiments/summaries/paper100_mechanism250"
OUTPUT_CSV = SUMMARY_DIR / "feature_mechanism250_summary.csv"
DELTA_NAMES = ("uap_delta.pt", "delta.pt", "adversarial_delta.pt")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="List inputs without loading data or models.")
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional cap for test samples in each source class.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def device_from(value):
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)


def load_configs():
    configs = []
    for path in sorted(CONFIG_DIR.glob("mech250_*.yaml")):
        with path.open(encoding="utf-8") as handle:
            configs.append((path, yaml.safe_load(handle)))
    if len(configs) != 3:
        raise RuntimeError(f"expected 3 mechanism configs in {CONFIG_DIR}, found {len(configs)}")
    return configs


def expected_paths(config):
    exp = config["experiment"]
    output_dir = ROOT / exp["output_dir"]
    rounds = config["checkpoint"]["save_rounds"]
    return output_dir, [(round_index, output_dir / "checkpoints" / f"global_round_{round_index}.pt") for round_index in rounds]


def find_delta(output_dir, round_index):
    checkpoints = output_dir / "checkpoints"
    candidates = [checkpoints / f"delta_round_{round_index}.pt"]
    candidates += [checkpoints / name for name in DELTA_NAMES]
    candidates += [output_dir / name for name in DELTA_NAMES]
    candidates += sorted(output_dir.rglob(f"*delta*{round_index}*.pt"))
    candidates += [path for name in DELTA_NAMES for path in output_dir.rglob(name)]
    return next((path for path in candidates if path.is_file()), None)


def round_metrics(output_dir, database_name):
    database = output_dir / database_name
    if not database.is_file():
        return {}
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT round_index, clean_accuracy, attack_success_rate FROM round_metrics").fetchall()
    return {row[0]: {"clean_acc": row[1], "asr": row[2]} for row in rows}


def feature_vectors(model, images, feature_layer):
    return F.normalize(model.forward_features(images, feature_layer), dim=1)


def calculate_round(model, loader, delta, target, feature_layer, max_per_class, device):
    prototype_sums, prototype_counts = {}, {}
    triggered_sums, source_sums, source_counts = {}, {}, {}
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            clean = feature_vectors(model, images, feature_layer)
            triggered = feature_vectors(model, apply_trigger(images, trigger_type="adversarial", perturbation=delta), feature_layer)
            for label in labels.unique().tolist():
                mask = labels == label
                if max_per_class is not None and prototype_counts.get(label, 0) >= max_per_class:
                    continue
                if max_per_class is not None:
                    available = max_per_class - prototype_counts.get(label, 0)
                    indices = mask.nonzero(as_tuple=False).flatten()[:available]
                    mask = torch.zeros_like(mask, dtype=torch.bool)
                    mask[indices] = True
                count = int(mask.sum())
                prototype_sums[label] = prototype_sums.get(label, 0) + clean[mask].sum(0)
                prototype_counts[label] = prototype_counts.get(label, 0) + count
                if label != target:
                    triggered_sums[label] = triggered_sums.get(label, 0) + triggered[mask].sum(0)
                    source_sums[label] = source_sums.get(label, 0) + clean[mask].sum(0)
                    source_counts[label] = source_counts.get(label, 0) + count
    prototypes = {label: F.normalize(total, dim=0) for label, total in prototype_sums.items()}
    results = []
    for source, count in sorted(source_counts.items()):
        triggered_mean = F.normalize(triggered_sums[source] / count, dim=0)
        target_sim = float(torch.dot(triggered_mean, prototypes[target]))
        source_sim = float(torch.dot(triggered_mean, prototypes[source]))
        results.append((source, count, target_sim, source_sim, target_sim - source_sim))
    return results


def main():
    args = parse_args()
    configs = load_configs()
    if args.dry_run:
        for config_path, config in configs:
            output_dir, checkpoints = expected_paths(config)
            print(f"{config['experiment']['name']} | config={config_path.relative_to(ROOT)}")
            for round_index, checkpoint in checkpoints:
                print(f"  round {round_index}: checkpoint={checkpoint.relative_to(ROOT)} delta-search={output_dir.relative_to(ROOT)}/checkpoints/delta_round_{round_index}.pt, common delta names recursively")
        return

    device = device_from(args.device)
    rows = []
    for _, config in configs:
        experiment = config["experiment"]
        attack = config["attack"]
        output_dir, checkpoints = expected_paths(config)
        metrics = round_metrics(output_dir, config.get("tracking", {}).get("database_name", "metrics.db"))
        _, test_dataset = load_cifar10(ROOT / config["dataset"]["root"], download=False)
        loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0)
        feature_layer = attack["trigger"]["adversarial"]["prototype_contrastive"]["feature_layer"]
        target = attack["target_label"]
        for round_index, checkpoint_path in checkpoints:
            delta_path = find_delta(output_dir, round_index)
            if not checkpoint_path.is_file() or delta_path is None:
                print(f"SKIP {experiment['name']} round {round_index}: checkpoint or delta missing")
                continue
            model = build_resnet18(config["dataset"]["num_classes"]).to(device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            delta = torch.load(delta_path, map_location=device).to(device=device, dtype=torch.float32)
            for source, count, target_sim, source_sim, margin in calculate_round(model, loader, delta, target, feature_layer, args.max_per_class, device):
                metric = metrics.get(round_index, {})
                rows.append({"experiment_id": experiment["name"], "method": experiment["name"], "round": round_index, "feature_layer": feature_layer, "target_label": target, "source_label": source, "num_samples": count, "target_sim": target_sim, "source_sim": source_sim, "feature_margin": margin, "asr": metric.get("asr"), "clean_acc": metric.get("clean_acc"), "checkpoint_path": str(checkpoint_path.relative_to(ROOT)), "delta_path": str(delta_path.relative_to(ROOT))})
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["experiment_id", "method", "round", "feature_layer", "target_label", "source_label", "num_samples", "target_sim", "source_sim", "feature_margin", "asr", "clean_acc", "checkpoint_path", "delta_path"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} rows: {OUTPUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Construct and audit only the clean parallel balanced-Dirichlet partitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import load_dataset
from data.partition import partition_dirichlet_balanced


def merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = raw.pop("base_config", None)
    if base is None:
        return raw
    base_path = Path(base)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return merge(load_config(base_path), raw)


def partition_hash(partitions: dict[int, list[int]]) -> tuple[str, dict[int, str]]:
    whole = hashlib.sha256()
    clients: dict[int, str] = {}
    for client_id in sorted(partitions):
        encoded = f"{client_id}:" + ",".join(map(str, sorted(partitions[client_id]))) + "\n"
        whole.update(encoded.encode("ascii"))
        clients[client_id] = hashlib.sha256(encoded.encode("ascii")).hexdigest()
    return whole.hexdigest(), clients


def audit(config_path: Path) -> dict:
    config = load_config(config_path)
    partition, dataset = config["partition"], config["dataset"]
    if partition.get("type") != "dirichlet_balanced":
        raise ValueError(f"{config_path}: expected dirichlet_balanced")
    summary_dir = ROOT / partition["summary_dir"]
    train_dataset, _ = load_dataset(dataset["name"], ROOT / dataset["root"], dataset.get("download", False))
    labels = np.asarray(train_dataset.targets, dtype=int)
    clients, classes = int(partition["num_clients"]), int(dataset["num_classes"])
    quota, alpha, seed = int(partition["samples_per_client"]), float(partition["alpha"]), int(config["experiment"]["seed"])
    args = (labels, clients, alpha, seed, quota, int(partition.get("min_classes_per_client", 1)), int(partition.get("max_retries", 100)))
    partitions = partition_dirichlet_balanced(*args)
    if partitions != partition_dirichlet_balanced(*args):
        raise AssertionError(f"{config_path}: partition is not deterministic")
    all_indices = [index for client_id in range(clients) for index in partitions[client_id]]
    if len(partitions) != clients or any(len(partitions[c]) != quota for c in range(clients)):
        raise AssertionError(f"{config_path}: every client must receive exactly {quota} samples")
    if len(all_indices) != len(labels) or len(set(all_indices)) != len(labels) or set(all_indices) != set(range(len(labels))):
        raise AssertionError(f"{config_path}: partition has duplicate or missing indices")
    counts = np.vstack([np.bincount(labels[partitions[c]], minlength=classes) for c in range(clients)])
    expected = np.bincount(labels, minlength=classes)
    if not np.array_equal(counts.sum(axis=0), expected):
        raise AssertionError(f"{config_path}: class inventories do not match CIFAR-10")
    if not np.all(counts.sum(axis=1) == quota):
        raise AssertionError(f"{config_path}: client quotas changed during allocation")
    whole_hash, client_hashes = partition_hash(partitions)
    probability = counts / counts.sum(axis=1, keepdims=True)
    entropy = -np.sum(np.where(probability > 0, probability * np.log(np.where(probability > 0, probability, 1.0)), 0.0), axis=1)
    max_prop = probability.max(axis=1)
    effective = np.exp(entropy)
    kl = math.log(classes) - entropy
    fields = ["client_id", "num_samples", "num_classes"] + [f"class_{value}_count" for value in range(classes)] + ["is_malicious_client", "client_partition_hash"]
    malicious = {0, 1, 2}
    rows = [{"client_id": c, "num_samples": int(counts[c].sum()), "num_classes": int(np.count_nonzero(counts[c])), **{f"class_{value}_count": int(counts[c, value]) for value in range(classes)}, "is_malicious_client": c in malicious, "client_partition_hash": client_hashes[c]} for c in range(clients)]
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "partition_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    record = {
        "setting": config["experiment"]["name"], "alpha": alpha, "seed": seed, "config": str(config_path.relative_to(ROOT)), "summary_dir": str(summary_dir.relative_to(ROOT)),
        "num_clients": clients, "samples_per_client": quota, "total_samples": len(all_indices), "min_samples": int(counts.sum(axis=1).min()), "max_samples": int(counts.sum(axis=1).max()), "mean_samples": float(counts.sum(axis=1).mean()),
        "min_nonzero_classes": int((counts > 0).sum(axis=1).min()), "max_nonzero_classes": int((counts > 0).sum(axis=1).max()), "mean_nonzero_classes": float((counts > 0).sum(axis=1).mean()),
        **{f"class_{value}_total": int(counts[:, value].sum()) for value in range(classes)},
        **{f"client_{value}_samples": int(counts[value].sum()) for value in sorted(malicious)}, **{f"client_{value}_nonzero_classes": int((counts[value] > 0).sum()) for value in sorted(malicious)},
        "mean_entropy": float(entropy.mean()), "mean_max_class_proportion": float(max_prop.mean()), "mean_effective_classes": float(effective.mean()), "mean_kl_to_uniform": float(kl.mean()), "partition_hash": whole_hash,
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="configs/final_noniid_balanced_pbu_seed2026_parallel_clean")
    args = parser.parse_args()
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = ROOT / config_dir
    configs = sorted(config_dir.glob("final_noniid_balanced_a*_seed2026_parallel.yaml"))
    if len(configs) != 4:
        raise SystemExit(f"Expected exactly four clean parallel configs; found {len(configs)}")
    records = sorted((audit(path) for path in configs), key=lambda row: row["alpha"], reverse=True)
    overview_dir = ROOT / "outputs/final_noniid_balanced_parallel_clean/partition_summaries"
    overview_dir.mkdir(parents=True, exist_ok=True)
    overview = overview_dir / "final_balanced_noniid_parallel_clean_overview.csv"
    with overview.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    entropy = [row["mean_entropy"] for row in records]
    maximum = [row["mean_max_class_proportion"] for row in records]
    effective = [row["mean_effective_classes"] for row in records]
    kl = [row["mean_kl_to_uniform"] for row in records]
    trend_ok = all(entropy[i] > entropy[i + 1] and maximum[i] < maximum[i + 1] and effective[i] > effective[i + 1] and kl[i] < kl[i + 1] for i in range(3))
    print(f"Wrote {overview.relative_to(ROOT)}")
    for row in records:
        print(f"alpha={row['alpha']:.1f} hash={row['partition_hash']} samples={row['min_samples']}/{row['max_samples']}/{row['mean_samples']:.1f} classes={row['min_nonzero_classes']}-{row['max_nonzero_classes']} entropy={row['mean_entropy']:.6f} maxprop={row['mean_max_class_proportion']:.6f} effective={row['mean_effective_classes']:.6f} kl={row['mean_kl_to_uniform']:.6f}")
    print(f"heterogeneity_trend_alpha_0.9_to_0.6={'PASS' if trend_ok else 'FAIL'}")
    if not trend_ok:
        raise SystemExit("Heterogeneity trend is not monotonic")


if __name__ == "__main__":
    main()

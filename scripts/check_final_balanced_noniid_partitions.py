#!/usr/bin/env python3
"""Construct and audit the final balanced-Dirichlet CIFAR-10 partitions only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_config = config.pop("base_config", None)
    if base_config is None:
        return config
    base_path = Path(base_config)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return deep_merge(load_config(base_path), config)


def partition_hash(partitions: dict[int, list[int]]) -> tuple[str, dict[int, str]]:
    digest = hashlib.sha256()
    client_hashes: dict[int, str] = {}
    for client_id in sorted(partitions):
        indices = sorted(int(index) for index in partitions[client_id])
        encoded = f"{client_id}:" + ",".join(map(str, indices)) + "\n"
        digest.update(encoded.encode("ascii"))
        client_hashes[client_id] = hashlib.sha256(encoded.encode("ascii")).hexdigest()
    return digest.hexdigest(), client_hashes


def entropy_metrics(counts: np.ndarray) -> tuple[float, float, float, float]:
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    positive = probabilities > 0
    log_probabilities = np.zeros_like(probabilities)
    log_probabilities[positive] = np.log(probabilities[positive])
    entropy = -np.sum(probabilities * log_probabilities, axis=1)
    max_proportion = probabilities.max(axis=1)
    effective_classes = np.exp(entropy)
    uniform_entropy = math.log(counts.shape[1])
    kl_uniform = uniform_entropy - entropy
    return entropy, max_proportion, effective_classes, kl_uniform


def audit_config(config_path: Path, output_dir: Path) -> dict:
    config = load_config(config_path)
    partition = config["partition"]
    dataset = config["dataset"]
    if partition.get("type") != "dirichlet_balanced":
        raise ValueError(f"{config_path}: partition.type must be dirichlet_balanced")
    train_dataset, _ = load_dataset(
        dataset["name"], ROOT / dataset["root"], dataset.get("download", False)
    )
    labels = np.asarray(getattr(train_dataset, "targets"), dtype=int)
    num_clients = int(partition["num_clients"])
    num_classes = int(dataset["num_classes"])
    samples_per_client = int(partition["samples_per_client"])
    alpha = float(partition["alpha"])
    seed = int(config["experiment"]["seed"])
    partitions = partition_dirichlet_balanced(
        labels, num_clients, alpha, seed, samples_per_client,
        int(partition.get("min_classes_per_client", 1)), int(partition.get("max_retries", 100)),
    )
    # Determinism is a protocol requirement, not merely an implementation detail.
    repeat = partition_dirichlet_balanced(
        labels, num_clients, alpha, seed, samples_per_client,
        int(partition.get("min_classes_per_client", 1)), int(partition.get("max_retries", 100)),
    )
    if partitions != repeat:
        raise AssertionError(f"{config_path}: same seed/alpha did not reproduce the partition")
    all_indices = [index for client in range(num_clients) for index in partitions[client]]
    if len(partitions) != num_clients or any(len(partitions[client]) != samples_per_client for client in range(num_clients)):
        raise AssertionError(f"{config_path}: client row-sum check failed")
    if len(all_indices) != len(labels) or len(set(all_indices)) != len(labels) or set(all_indices) != set(range(len(labels))):
        raise AssertionError(f"{config_path}: partition has duplicates or missing indices")
    counts = np.vstack([np.bincount(labels[partitions[client]], minlength=num_classes) for client in range(num_clients)])
    expected_per_class = np.bincount(labels, minlength=num_classes)
    if not np.all(counts.sum(axis=1) == samples_per_client):
        raise AssertionError(f"{config_path}: row count matrix check failed")
    if not np.array_equal(counts.sum(axis=0), expected_per_class):
        raise AssertionError(f"{config_path}: class inventory check failed")
    if len(labels) != num_clients * samples_per_client:
        raise AssertionError(f"{config_path}: total quota does not equal dataset size")
    whole_hash, client_hashes = partition_hash(partitions)
    entropy, max_prop, effective, kl = entropy_metrics(counts.astype(float))
    malicious_ids = {int(value) for value in config["attack"]["malicious_client_ids"]}
    rows = []
    for client_id in range(num_clients):
        rows.append({
            "client_id": client_id,
            "num_samples": int(counts[client_id].sum()),
            "num_classes": int(np.count_nonzero(counts[client_id])),
            **{f"class_{class_id}_count": int(counts[client_id, class_id]) for class_id in range(num_classes)},
            "is_malicious_client": client_id in malicious_ids,
            "partition_hash_or_client_hash": client_hashes[client_id],
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"alpha{alpha:.1f}_partition_summary.csv"
    fields = ["client_id", "num_samples", "num_classes"] + [f"class_{class_id}_count" for class_id in range(num_classes)] + ["is_malicious_client", "partition_hash_or_client_hash"]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    malicious_rows = [rows[client_id] for client_id in sorted(malicious_ids)]
    def client_hist(client_id: int) -> str:
        return json.dumps([int(value) for value in counts[client_id]], separators=(",", ":"))
    return {
        "setting": config["experiment"]["name"], "alpha": alpha, "seed": seed,
        "num_clients": num_clients, "num_classes": num_classes,
        "target_samples_per_client": samples_per_client, "total_samples": len(all_indices),
        "min_samples": int(counts.sum(axis=1).min()), "max_samples": int(counts.sum(axis=1).max()),
        "mean_samples": float(counts.sum(axis=1).mean()), "std_samples": float(counts.sum(axis=1).std()),
        "min_classes": int((counts > 0).sum(axis=1).min()), "max_classes": int((counts > 0).sum(axis=1).max()),
        "mean_classes": float((counts > 0).sum(axis=1).mean()), "std_classes": float((counts > 0).sum(axis=1).std()),
        **{f"class_{class_id}_total": int(counts[:, class_id].sum()) for class_id in range(num_classes)},
        **{f"malicious_{client_id}_samples": int(counts[client_id].sum()) for client_id in (0, 1, 2)},
        **{f"malicious_{client_id}_num_classes": int((counts[client_id] > 0).sum()) for client_id in (0, 1, 2)},
        **{f"malicious_{client_id}_class_hist": client_hist(client_id) for client_id in (0, 1, 2)},
        "malicious_avg_samples": float(np.mean([row["num_samples"] for row in malicious_rows])),
        "malicious_avg_classes": float(np.mean([row["num_classes"] for row in malicious_rows])),
        "mean_entropy": float(entropy.mean()), "std_entropy": float(entropy.std()),
        "mean_max_class_proportion": float(max_prop.mean()), "mean_effective_classes": float(effective.mean()),
        "mean_kl_to_uniform": float(kl.mean()),
        "malicious_mean_entropy": float(entropy[[0, 1, 2]].mean()),
        "malicious_mean_max_class_proportion": float(max_prop[[0, 1, 2]].mean()),
        "malicious_mean_effective_classes": float(effective[[0, 1, 2]].mean()),
        "malicious_mean_kl_to_uniform": float(kl[[0, 1, 2]].mean()),
        "partition_hash": whole_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_dir = args.config_dir if args.config_dir.is_absolute() else ROOT / args.config_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    configs = sorted(config_dir.glob("final_noniid_balanced_a*_seed2026.yaml"), reverse=True)
    if len(configs) != 4:
        raise SystemExit(f"Expected exactly four final configs, found {len(configs)}")
    records = [audit_config(path, output_dir) for path in configs]
    records.sort(key=lambda record: record["alpha"], reverse=True)
    fields = list(records[0])
    overview = output_dir / "final_balanced_noniid_partition_overview.csv"
    with overview.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    entropy = [record["mean_entropy"] for record in records]
    dominant = [record["mean_max_class_proportion"] for record in records]
    effective = [record["mean_effective_classes"] for record in records]
    kl = [record["mean_kl_to_uniform"] for record in records]
    trend_ok = all(entropy[index] > entropy[index + 1] for index in range(3)) and all(
        dominant[index] < dominant[index + 1] for index in range(3)
    ) and all(effective[index] > effective[index + 1] for index in range(3)) and all(
        kl[index] < kl[index + 1] for index in range(3)
    )
    print(f"Wrote {overview.relative_to(ROOT)}")
    for record in records:
        print(
            f"alpha={record['alpha']:.1f} hash={record['partition_hash']} "
            f"samples={record['min_samples']}/{record['max_samples']}/{record['mean_samples']:.1f} "
            f"classes={record['min_classes']}-{record['max_classes']} "
            f"entropy={record['mean_entropy']:.6f} maxprop={record['mean_max_class_proportion']:.6f} "
            f"effective={record['mean_effective_classes']:.6f} kl={record['mean_kl_to_uniform']:.6f}"
        )
    print(f"heterogeneity_trend_alpha_0.9_to_0.6={'PASS' if trend_ok else 'NOT_MONOTONIC'}")
    if not trend_ok:
        raise SystemExit("Heterogeneity trend was not monotonic; outputs were retained for audit.")


if __name__ == "__main__":
    main()

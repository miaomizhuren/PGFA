"""Construct and audit balanced-quantity Dirichlet CIFAR-10 partitions.

This script only loads the training labels and writes partition summaries; it
does not construct a model or start federated training.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import load_dataset
from data.partition import partition_dirichlet_balanced


CONFIG_DIR = ROOT / "configs/cifar10_noniid_balanced_pbu_force_fromscratch_3"
OUTPUT_DIR = ROOT / "outputs/cifar10_noniid_balanced_pbu_force_fromscratch_3/partition_summaries"


def audit_config(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    partition = config["partition"]
    experiment = config["experiment"]
    dataset_config = config["dataset"]
    if partition["type"] != "dirichlet_balanced":
        raise ValueError(f"{config_path}: expected partition.type=dirichlet_balanced")
    dataset_root = ROOT / dataset_config["root"]
    train_dataset, _ = load_dataset(dataset_config["name"], dataset_root, dataset_config.get("download", False))
    labels = np.asarray(getattr(train_dataset, "targets"))
    partitions = partition_dirichlet_balanced(
        labels=labels,
        num_clients=int(partition["num_clients"]),
        alpha=float(partition["alpha"]),
        seed=int(experiment["seed"]),
        samples_per_client=int(partition["samples_per_client"]),
        min_classes_per_client=int(partition.get("min_classes_per_client", 1)),
        max_retries=int(partition.get("max_retries", 100)),
    )

    all_indices = [index for indices in partitions.values() for index in indices]
    if len(partitions) != int(partition["num_clients"]):
        raise AssertionError("wrong number of clients")
    if len(all_indices) != len(labels) or len(set(all_indices)) != len(labels):
        raise AssertionError("partition has missing or duplicate indices")
    if set(all_indices) != set(range(len(labels))):
        raise AssertionError("partition indices do not cover the dataset")

    malicious_ids = set(config["attack"]["malicious_client_ids"])
    rows = []
    for client_id in range(int(partition["num_clients"])):
        indices = partitions[client_id]
        counts = np.bincount(labels[indices], minlength=int(dataset_config["num_classes"]))
        rows.append({
            "client_id": client_id,
            "num_samples": len(indices),
            "num_classes": int(np.count_nonzero(counts)),
            **{f"class_{class_id}_count": int(counts[class_id]) for class_id in range(len(counts))},
            "is_malicious_client": client_id in malicious_ids,
        })

    alpha = float(partition["alpha"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / f"alpha{alpha:.1f}_partition_summary.csv"
    fields = ["client_id", "num_samples", "num_classes"] + [f"class_{i}_count" for i in range(int(dataset_config["num_classes"]))] + ["is_malicious_client"]
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    sample_counts = np.asarray([row["num_samples"] for row in rows], dtype=float)
    class_counts = np.asarray([row["num_classes"] for row in rows], dtype=float)
    malicious_rows = [row for row in rows if row["is_malicious_client"]]
    return {
        "setting": config["experiment"]["name"],
        "alpha": alpha,
        "num_clients": len(rows),
        "target_samples_per_client": int(partition["samples_per_client"]),
        "min_samples": int(sample_counts.min()),
        "max_samples": int(sample_counts.max()),
        "mean_samples": float(sample_counts.mean()),
        "std_samples": float(sample_counts.std()),
        "min_classes": int(class_counts.min()),
        "max_classes": int(class_counts.max()),
        "mean_classes": float(class_counts.mean()),
        "std_classes": float(class_counts.std()),
        **{f"malicious_{client_id}_samples": next(row["num_samples"] for row in rows if row["client_id"] == client_id) for client_id in (0, 1, 2)},
        **{f"malicious_{client_id}_num_classes": next(row["num_classes"] for row in rows if row["client_id"] == client_id) for client_id in (0, 1, 2)},
        "malicious_avg_samples": float(np.mean([row["num_samples"] for row in malicious_rows])),
        "malicious_avg_classes": float(np.mean([row["num_classes"] for row in malicious_rows])),
    }


def main() -> None:
    records = [audit_config(path) for path in sorted(CONFIG_DIR.glob("*.yaml"), reverse=True)]
    fields = list(records[0])
    overview_path = OUTPUT_DIR / "balanced_noniid_partition_overview.csv"
    with overview_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    report_path = OUTPUT_DIR / "balanced_noniid_partition_audit.md"
    report_lines = [
        "# Balanced Dirichlet Non-IID partition audit",
        "",
        "This dry-run constructed CIFAR-10 training partitions only; no model training was started.",
        "Every setting covers all 50,000 training indices exactly once, with no duplicate indices.",
        "",
        "| alpha | clients | samples/client (min–max) | sample std | classes/client (min–max) | mean classes | malicious clients 0/1/2 |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        report_lines.append(
            f"| {record['alpha']:.1f} | {record['num_clients']} | "
            f"{record['min_samples']}–{record['max_samples']} | {record['std_samples']:.1f} | "
            f"{record['min_classes']}–{record['max_classes']} | {record['mean_classes']:.2f} | "
            f"{record['malicious_0_samples']}/{record['malicious_1_samples']}/{record['malicious_2_samples']} samples; "
            f"{record['malicious_0_num_classes']}/{record['malicious_1_num_classes']}/{record['malicious_2_num_classes']} classes |"
        )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {overview_path.relative_to(ROOT)}")
    print(f"Wrote {report_path.relative_to(ROOT)}")
    for record in records:
        print(
            f"alpha={record['alpha']:.1f}: samples min/max/mean/std="
            f"{record['min_samples']}/{record['max_samples']}/"
            f"{record['mean_samples']:.1f}/{record['std_samples']:.1f}; "
            f"classes min/max/mean={record['min_classes']}/{record['max_classes']}/{record['mean_classes']:.2f}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only availability check for cross-dataset PBU target prototypes.

This script resolves each YAML inheritance chain, reads local dataset labels, and
recreates the configured IID index partition.  It does not construct a model,
load a checkpoint, create a DataLoader, retrieve images, or start training.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import load_dataset
from data.partition import partition_iid
from train import load_config


CONFIG_DIR = PROJECT_ROOT / "configs" / "cross_dataset_generalization"
CONFIGS = sorted(CONFIG_DIR.glob("*_pbu_clean3000_attack3001_3200_observe3201_3700.yaml"))


def main() -> None:
    if len(CONFIGS) != 3:
        raise RuntimeError(f"expected exactly three cross-dataset PBU configs, found {CONFIGS}")
    print("dataset,malicious_client_id,target_label,target_class_count,prototype_available")
    for config_path in CONFIGS:
        config = load_config(config_path)
        dataset_config = config["dataset"]
        partition_config = config["partition"]
        attack_config = config["attack"]
        prototype_config = attack_config["trigger"]["adversarial"]["prototype_contrastive"]
        if partition_config.get("type", "iid") != "iid":
            raise ValueError("this static check implements the configured IID partition only")
        if prototype_config["prototype_source"] != "single_malicious":
            raise ValueError("prototype availability check requires prototype_source=single_malicious")

        # With no explicit prototype_client_id, the forced malicious clients are
        # selected in ascending client-id order; the first client constructs the
        # shared local UAP/prototype bank at the attack start.
        source_client_id = prototype_config.get("prototype_client_id")
        if source_client_id is None:
            source_client_id = min(attack_config["malicious_client_ids"])

        train_dataset, _ = load_dataset(
            name=dataset_config["name"],
            root=PROJECT_ROOT / dataset_config["root"],
            download=False,
        )
        labels = getattr(train_dataset, "targets", None)
        if labels is None:
            raise ValueError(f"{dataset_config['name']} train dataset exposes no targets")
        partitions = partition_iid(
            len(train_dataset), partition_config["num_clients"], config["experiment"]["seed"]
        )
        target_label = int(attack_config["target_label"])
        target_count = sum(int(labels[index]) == target_label for index in partitions[source_client_id])
        available = target_count >= int(prototype_config["min_samples_per_class"])
        print(
            f"{dataset_config['name']},{source_client_id},{target_label},"
            f"{target_count},{available}"
        )


if __name__ == "__main__":
    main()

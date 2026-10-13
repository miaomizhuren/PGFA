#!/usr/bin/env python3
"""Dry-run the actual Server.select_clients policy without training or data loading."""

from __future__ import annotations

import random
import sys
import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attacks.config import AttackConfig
from federated.server import Server


DEFAULT_CONFIG_PATH = ROOT / (
    "configs/cifar10_pbu_bs64_fromscratch_allE3_force_attack_only_"
    "clean500_attack501_700_stop701_1000.yaml"
)
ROUNDS = [1, 100, 500, 501, 600, 700, 701, 800, 1000]
MALICIOUS_IDS = {0, 1, 2}


def make_server(policy: str, seed: int, config: dict) -> Server:
    """Construct only the fields read by Server.select_clients."""
    attack = config["attack"]
    instance = Server.__new__(Server)
    instance.clients = [SimpleNamespace(client_id=client_id) for client_id in range(100)]
    instance.clients_per_round = config["federated"]["clients_per_round"]
    instance.rng = random.Random(seed)
    instance.attack_config = AttackConfig(
        enabled=attack["enabled"],
        malicious_client_ids=attack["malicious_client_ids"],
        num_malicious_clients=attack["num_malicious_clients"],
        selection_policy=policy,
        min_malicious_per_attack_round=attack["min_malicious_per_attack_round"],
        attack_start_round=attack["start_round"],
        attack_end_round=attack["end_round"],
    )
    return instance


def record(server: Server, round_index: int) -> dict:
    selected_ids = [client.client_id for client in server.select_clients(round_index)]
    selected_malicious = sorted(MALICIOUS_IDS.intersection(selected_ids))
    mode, _ = server._selection_mode(round_index)
    return {
        "round": round_index,
        "attack_active": server.attack_config.is_attack_round(round_index),
        "selection_policy": server.attack_config.selection_policy,
        "selected_clients": selected_ids,
        "selected_malicious_clients": selected_malicious,
        "all_012_present": MALICIOUS_IDS.issubset(selected_ids),
        "mode": mode,
    }


def print_record(item: dict) -> None:
    print(
        "round={round} attack_active={attack_active} "
        "selection_policy={selection_policy} mode={mode} "
        "selected_clients={selected_clients} "
        "selected_malicious_clients={selected_malicious_clients} "
        "all_012_present={all_012_present}".format(**item)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["partition"]["num_clients"] == 100
    assert config["federated"]["clients_per_round"] == 10
    assert config["training"]["batch_size"] == 64
    assert config["federated"]["local_epochs"] == 3
    assert config["federated"].get("malicious_local_epochs", 3) == 3
    assert config["federated"]["total_rounds"] == 1000
    assert config["attack"]["enabled"] is True
    assert config["attack"]["method"] == "proto_uap"
    assert config["attack"]["start_round"] == 501
    assert config["attack"]["end_round"] == 700
    assert config["attack"]["selection_policy"] == "force_during_attack"
    assert config.get("resume", {}).get("enabled", False) is False
    assert "checkpoint_path" not in config.get("resume", {})
    assert {500, 700, 1000}.issubset(set(config["checkpoint"]["save_rounds"]))
    print(f"config={config_path}")

    print("=== legacy_all_malicious: historical compatibility check ===")
    legacy = make_server("legacy_all_malicious", config["experiment"]["seed"], config)
    for round_index in ROUNDS:
        item = record(legacy, round_index)
        print_record(item)
        assert item["all_012_present"], "legacy policy must include 0/1/2 every round"
        assert item["mode"] == "legacy_all_malicious"

    print("=== force_during_attack: phase-aware check ===")
    forced = make_server("force_during_attack", config["experiment"]["seed"], config)
    for round_index in ROUNDS:
        item = record(forced, round_index)
        print_record(item)
        if item["attack_active"]:
            assert item["all_012_present"], "attack rounds must include 0/1/2"
            assert item["mode"] == "force_malicious_during_attack"
        else:
            assert item["mode"] == "random_all_clients"

    print("=== force_during_attack: non-attack seed sweep ===")
    for round_index in (1, 100, 500, 701, 800, 1000):
        all_present_count = 0
        for seed in range(10):
            item = record(make_server("force_during_attack", seed, config), round_index)
            all_present_count += int(item["all_012_present"])
        print(
            f"round={round_index} seeds=0..9 all_012_present_count="
            f"{all_present_count}/10 mode=random_all_clients"
        )
        assert all_present_count < 10, "non-attack sampling must not fix 0/1/2"

    print("PASS: dry-run checked only sampling; no model, dataset, checkpoint, or training was used.")


if __name__ == "__main__":
    main()

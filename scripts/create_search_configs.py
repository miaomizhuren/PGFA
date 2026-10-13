#!/usr/bin/env python3
"""Create isolated, non-overwriting quick-search YAML configurations."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "configs/cifar10_adv_proto_train_layer4_persistent_baseline.yaml"
DESTINATION = ROOT / "configs/search"

# A constrained, representative 24-run design.  The e3 subset is capped at 8
# because it has the highest training cost.  Unlisted values stay at the
# persistent-baseline setting.
CASES = [
    ("search2_q01_e1_control", {"local_epochs": 1}),
    ("search2_q02_e1_lambda_6_3", {"local_epochs": 1, "lambda_delta": 6.0, "lambda_train": 3.0}),
    ("search2_q03_e1_lambda_12_7", {"local_epochs": 1, "lambda_delta": 12.0, "lambda_train": 7.0}),
    ("search2_q04_e1_poison_0_60", {"local_epochs": 1, "poison_ratio": 0.60}),
    ("search2_q05_e1_poison_0_80", {"local_epochs": 1, "poison_ratio": 0.80}),
    ("search2_q06_e1_margin_0_5", {"local_epochs": 1, "source_margin": 0.5}),
    ("search2_q07_e1_temp_0_02", {"local_epochs": 1, "temperature": 0.02}),
    ("search2_q08_e1_steps_100", {"local_epochs": 1, "steps": 100, "alpha": 0.005}),
    ("search2_q09_e2_control", {"local_epochs": 2}),
    ("search2_q10_e2_lambda_6_3", {"local_epochs": 2, "lambda_delta": 6.0, "lambda_train": 3.0}),
    ("search2_q11_e2_lambda_12_7", {"local_epochs": 2, "lambda_delta": 12.0, "lambda_train": 7.0}),
    ("search2_q12_e2_lambda_15_9", {"local_epochs": 2, "lambda_delta": 15.0, "lambda_train": 9.0}),
    ("search2_q13_e2_poison_0_80", {"local_epochs": 2, "poison_ratio": 0.80}),
    ("search2_q14_e2_margin_1_0", {"local_epochs": 2, "source_margin": 1.0}),
    ("search2_q15_e2_temp_0_005", {"local_epochs": 2, "temperature": 0.005}),
    ("search2_q16_e2_steps_150", {"local_epochs": 2, "steps": 150, "alpha": 0.004}),
    ("search2_q17_e3_control", {"local_epochs": 3}),
    ("search2_q18_e3_lambda_12_7", {"local_epochs": 3, "lambda_delta": 12.0, "lambda_train": 7.0}),
    ("search2_q19_e3_strong_p80_m1", {"local_epochs": 3, "lambda_delta": 15.0, "lambda_train": 9.0, "poison_ratio": 0.80, "source_margin": 1.0}),
    ("search2_q20_e3_poison_0_80", {"local_epochs": 3, "poison_ratio": 0.80}),
    ("search2_q21_e3_margin_1_temp_005", {"local_epochs": 3, "source_margin": 1.0, "temperature": 0.005}),
    ("search2_q22_e3_temp_0_02", {"local_epochs": 3, "temperature": 0.02}),
    ("search2_q23_e3_steps_100", {"local_epochs": 3, "steps": 100, "alpha": 0.005}),
    ("search2_q24_e3_steps_150", {"local_epochs": 3, "steps": 150, "alpha": 0.004}),
]


def apply_case(config: dict, name: str, values: dict) -> dict:
    result = deepcopy(config)
    result["experiment"]["name"] = name
    result["experiment"]["output_dir"] = f"./outputs/{name}"
    result["federated"]["num_rounds"] = 150
    result["tracking"]["database_name"] = f"metrics_{name}.db"
    attack = result["attack"]
    adversarial = attack["trigger"]["adversarial"]
    prototype = adversarial["prototype_contrastive"]
    for key, value in values.items():
        if key == "local_epochs":
            result["federated"][key] = value
        elif key == "poison_ratio":
            attack[key] = value
        elif key in {"steps", "alpha"}:
            adversarial[key] = value
        else:
            prototype[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    args = parser.parse_args()
    template_path = args.template if args.template.is_absolute() else ROOT / args.template
    with template_path.open(encoding="utf-8") as handle:
        template = yaml.safe_load(handle)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, values in CASES:
        target = DESTINATION / f"{name}.yaml"
        if target.exists():
            print(f"SKIP existing: {target.relative_to(ROOT)}")
            continue
        with target.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(apply_case(template, name, values), handle, sort_keys=False)
        print(f"CREATED {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

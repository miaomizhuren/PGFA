#!/usr/bin/env python3
"""Read-only feature-persistence analysis for existing CIFAR-10 UAP runs.

The script never invokes training and writes only to its requested analysis
directory.  It compares current-checkpoint penultimate features at fixed test
indices, while obtaining clean accuracy and ASR from existing SQLite metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.triggers import apply_trigger
from data.dataset import load_dataset
from models.resnet18 import build_resnet18


METHODS = {
    "CE-UAP": {
        "root": "outputs/main_experiment_clean3000_force/ce_uap_clean3000_attack3001_3200_observe3201_3700",
        "config": "configs/main_experiment_clean3000_force/ce_uap_clean3000_attack3001_3200_observe3201_3700.yaml",
        "metrics": "metrics_ce_uap_clean3000_attack3001_3200_observe3201_3700.db",
    },
    "PBU": {
        "root": "outputs/pbu_force_clean_stage_sensitivity/clean3000_attack3001_3200_observe3201_3700",
        "config": "configs/clean_stage_sensitivity_pbu_force/pbu_force_clean3000_attack3001_3200_observe3201_3700.yaml",
        "metrics": "metrics_pbu_force_clean3000_attack3001_3200_observe3201_3700.db",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--include-rounds", nargs="+", type=int, default=[3200, 3300, 3400, 3500, 3700])
    parser.add_argument("--include-optional-3600", action="store_true")
    parser.add_argument("--output-dir", default="outputs/feature_persistence_analysis")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write only fixed indices.")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configured_paths(rounds: list[int]) -> dict[str, dict]:
    paths: dict[str, dict] = {}
    for method, spec in METHODS.items():
        root = resolve_path(spec["root"])
        checkpoints = {round_index: root / "checkpoints" / f"global_round_{round_index}.pt" for round_index in rounds}
        paths[method] = {
            **spec,
            "root_path": root,
            "config_path": resolve_path(spec["config"]),
            "delta_path": root / "checkpoints" / "delta_round_3200.pt",
            "metrics_path": root / spec["metrics"],
            "checkpoints": checkpoints,
        }
    return paths


def validate_inputs(paths: dict[str, dict], rounds: list[int]) -> None:
    for method, spec in paths.items():
        required = [spec["root_path"], spec["config_path"], spec["delta_path"], spec["metrics_path"], *spec["checkpoints"].values()]
        missing = [str(path) for path in required if not path.is_file() and not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"{method} is missing required inputs: {missing}")
        with sqlite3.connect(f"file:{spec['metrics_path']}?mode=ro", uri=True) as connection:
            columns = {row[1] for row in connection.execute("pragma table_info(round_metrics)")}
            needed = {"round_index", "clean_accuracy", "attack_success_rate"}
            if not needed <= columns:
                raise ValueError(f"{method} round_metrics lacks {sorted(needed - columns)}")
            present = {row[0] for row in connection.execute(
                "select round_index from round_metrics where round_index in (%s)" % ",".join("?" * len(rounds)), rounds)}
            if set(rounds) - present:
                raise ValueError(f"{method} metrics missing rounds {sorted(set(rounds) - present)}")


def choose_fixed_indices(labels: list[int], target: int, per_class: int, seed: int) -> tuple[dict[str, list[int]], list[int]]:
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[int]] = {}
    trigger_indices: list[int] = []
    for class_id in range(10):
        candidates = np.asarray([index for index, label in enumerate(labels) if label == class_id], dtype=np.int64)
        chosen = rng.choice(candidates, size=min(per_class, len(candidates)), replace=False)
        selected = sorted(int(index) for index in chosen)
        by_class[str(class_id)] = selected
        if class_id != target:
            trigger_indices.extend(selected)
    return by_class, sorted(trigger_indices)


def write_fixed_indices(output_dir: Path, prototypes: dict[str, list[int]], triggers: list[int], args: argparse.Namespace) -> None:
    fixed = output_dir / "fixed_indices"
    fixed.mkdir(parents=True, exist_ok=True)
    (fixed / "prototype_indices_by_class.json").write_text(
        json.dumps({"seed": args.seed, "samples_per_class": args.samples_per_class, "indices_by_class": prototypes}, indent=2), encoding="utf-8")
    (fixed / "trigger_eval_indices.json").write_text(
        json.dumps({"seed": args.seed, "target_label": args.target_label, "indices": triggers}, indent=2), encoding="utf-8")


def read_metrics(path: Path, rounds: list[int]) -> dict[int, tuple[float | None, float | None]]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "select round_index, clean_accuracy, attack_success_rate from round_metrics where round_index in (%s)" % ",".join("?" * len(rounds)), rounds).fetchall()
    return {int(round_index): (clean, asr) for round_index, clean, asr in rows}


def state_dict_from_checkpoint(checkpoint: object) -> dict:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a model state dictionary")
    for key in ("model_state_dict", "state_dict", "global_model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    if not all(isinstance(key, str) for key in checkpoint):
        raise TypeError("checkpoint state dictionary keys must be strings")
    return {key.removeprefix("module."): value for key, value in checkpoint.items()}


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_resnet18(num_classes=10)
    model.load_state_dict(state_dict_from_checkpoint(checkpoint), strict=True)
    return model.to(device).eval()


def load_delta(path: Path, device: torch.device) -> torch.Tensor:
    delta = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(delta, torch.Tensor):
        raise TypeError(f"delta is not a tensor: {path}")
    if delta.ndim not in (3, 4) or (delta.ndim == 4 and delta.shape[0] != 1):
        raise ValueError(f"delta must be [C,H,W] or [1,C,H,W], got {tuple(delta.shape)}")
    return delta.to(device=device, dtype=torch.float32)


def extract_features(model: torch.nn.Module, dataset, indices: list[int], batch_size: int, device: torch.device, delta: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
    all_features, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if delta is not None:
                # Reuses attacks.triggers.apply_trigger: normalized-space additive UAP, no extra clamp.
                images = apply_trigger(images, trigger_type="adversarial", perturbation=delta.to(dtype=images.dtype))
            features = model.forward_features(images, feature_layer="penultimate")
            if features.ndim != 2:
                features = torch.flatten(functional.adaptive_avg_pool2d(features, 1), 1)
            all_features.append(functional.normalize(features, p=2, dim=1).cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_features), torch.cat(all_labels)


def compute_prototypes(model, dataset, prototype_indices: dict[str, list[int]], batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_indices = [index for values in prototype_indices.values() for index in values]
    features, labels = extract_features(model, dataset, all_indices, batch_size, device)
    prototypes = []
    for class_id in range(10):
        values = features[labels == class_id]
        if not len(values):
            raise ValueError(f"prototype reference set has no samples for class {class_id}")
        prototypes.append(functional.normalize(values.mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0))
    return torch.stack(prototypes), features, labels


def stats(values: torch.Tensor) -> tuple[float, float, float]:
    array = values.detach().cpu().numpy().astype(float)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return mean, std, std / math.sqrt(len(array)) if len(array) else float("nan")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_report(output_dir: Path, summary_rows: list[dict], paths: dict[str, dict], args: argparse.Namespace) -> None:
    table_fields = ["method", "round", "post_round", "trigger_to_target_proto_cos_mean", "feature_drift_mean", "target_source_margin_mean", "clean_target_proto_cos_mean", "clean_accuracy", "attack_success_rate"]
    lines = ["# Feature Persistence Mechanism Analysis", "", "## 1. Goal", "", "Assess whether triggered samples drift away from the current target-class representation after attack end.", "", "## 2. Inputs", ""]
    for method, spec in paths.items():
        lines.append(f"- {method}: checkpoints under `{spec['root']}/checkpoints`, attack-end delta `{spec['delta_path']}`, metrics `{spec['metrics_path']}`.")
    lines += ["", "## 3. Sample Selection", "", f"Fixed seed {args.seed}; up to {args.samples_per_class} clean CIFAR-10 test samples per class for prototypes and every non-target class for triggered evaluation. Indices are saved in `fixed_indices/` and shared by both methods.", "", "## 4. Feature Definition", "", "`CIFARResNet18.forward_features(feature_layer='penultimate')` is used: layer4 output is globally average pooled and flattened before FC. Features and current-checkpoint class prototypes are L2-normalized. The saved UAP is applied with `attacks.triggers.apply_trigger`, which adds the normalized-space perturbation without an extra clamp.", "", "## 5. Metric Definitions", "", "Target cosine is `cos(z, p_target)`; drift is `||z_r-z_3200||_2`; margin is `cos(z, p_target)-cos(z, p_source)`; clean control is target-class clean-feature cosine to the current target prototype.", "", "## 6. Summary Table", ""]
    lines.append("| " + " | ".join(table_fields) + " |")
    lines.append("|" + "|".join([" --- "] * len(table_fields)) + "|")
    for row in summary_rows:
        lines.append("| " + " | ".join(str(row[field]) for field in table_fields) + " |")
    lines += ["", "## 7. Key Findings", "", "Interpret the trajectories in `feature_persistence_summary.csv`: higher target cosine and margin, together with lower drift, are consistent with stronger persistence. The report intentionally makes no claim beyond the computed values.", "", "## 8. Paper-ready Wording", "", "在固定的非目标测试样本和逐 checkpoint 重建的类别原型下，我们比较了停攻后的触发样本表征。若 PBU 在后攻击轮次保持更高的触发样本—目标原型余弦相似度与目标—源原型间隔，并呈现较小的归一化特征漂移，则该现象与原型约束有助于缓解良性训练导致的表征遗忘相一致；该结论描述相关性，不单独证明因果机制。", "", "## 9. Files Generated", "", "- `feature_persistence_summary.csv`", "- `feature_persistence_sample_level.csv`", "- `feature_persistence_by_source_class.csv`", "- `feature_persistence_metrics_joined.csv`", "- `fixed_indices/prototype_indices_by_class.json`", "- `fixed_indices/trigger_eval_indices.json`", ""]
    (output_dir / "feature_persistence_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.samples_per_class <= 0 or not 0 <= args.target_label < 10:
        raise ValueError("batch size and samples per class must be positive; target label must be in [0,9]")
    rounds = sorted(set(args.include_rounds + ([3600] if args.include_optional_3600 else [])))
    if 3200 not in rounds:
        raise ValueError("round 3200 is required as the normalized-drift reference")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    output_dir = resolve_path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    paths = configured_paths(rounds); validate_inputs(paths, rounds)
    _, test_dataset = load_dataset("cifar10", PROJECT_ROOT / "datasets", download=False)
    labels = [int(value) for value in test_dataset.targets]
    prototype_indices, trigger_indices = choose_fixed_indices(labels, args.target_label, args.samples_per_class, args.seed)
    write_fixed_indices(output_dir, prototype_indices, trigger_indices, args)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "dataset_size": len(test_dataset), "prototype_samples": sum(map(len, prototype_indices.values())), "trigger_eval_samples": len(trigger_indices), "rounds": rounds, "output_dir": str(output_dir)}, indent=2))
        return
    device = resolve_device(args.device)
    summary_rows, sample_rows, class_rows, joined_rows = [], [], [], []
    for method, spec in paths.items():
        metrics = read_metrics(spec["metrics_path"], rounds)
        delta = load_delta(spec["delta_path"], device)
        reference_features = None
        for round_index in rounds:
            model = load_model(spec["checkpoints"][round_index], device)
            prototypes, prototype_features, prototype_labels = compute_prototypes(model, test_dataset, prototype_indices, args.batch_size, device)
            trigger_features, trigger_labels = extract_features(model, test_dataset, trigger_indices, args.batch_size, device, delta)
            target_cos = trigger_features @ prototypes[args.target_label]
            source_cos = (trigger_features * prototypes[trigger_labels]).sum(dim=1)
            margins = target_cos - source_cos
            if round_index == 3200:
                reference_features = trigger_features.clone()
            if reference_features is None:
                raise RuntimeError("round 3200 reference features were not computed")
            drift = torch.linalg.vector_norm(trigger_features - reference_features, dim=1)
            clean_values = prototype_features[prototype_labels == args.target_label] @ prototypes[args.target_label]
            post_round = round_index - 3200
            target_mean, target_std, target_se = stats(target_cos); drift_mean, drift_std, drift_se = stats(drift)
            margin_mean, margin_std, margin_se = stats(margins); clean_mean, clean_std, clean_se = stats(clean_values)
            clean_accuracy, attack_success_rate = metrics[round_index]
            common = {"method": method, "round": round_index, "post_round": post_round, "target_label": args.target_label}
            summary_rows.append({**common, "num_trigger_samples": len(trigger_indices), "num_proto_samples": sum(map(len, prototype_indices.values())), "trigger_to_target_proto_cos_mean": target_mean, "trigger_to_target_proto_cos_std": target_std, "trigger_to_target_proto_cos_se": target_se, "feature_drift_mean": drift_mean, "feature_drift_std": drift_std, "feature_drift_se": drift_se, "target_source_margin_mean": margin_mean, "target_source_margin_std": margin_std, "target_source_margin_se": margin_se, "clean_target_proto_cos_mean": clean_mean, "clean_target_proto_cos_std": clean_std, "clean_target_proto_cos_se": clean_se, "clean_accuracy": clean_accuracy, "attack_success_rate": attack_success_rate, "checkpoint_path": str(spec["checkpoints"][round_index]), "delta_path": str(spec["delta_path"])})
            for index, label, tcos, scos, margin, drift_value in zip(trigger_indices, trigger_labels.tolist(), target_cos.tolist(), source_cos.tolist(), margins.tolist(), drift.tolist()):
                sample_rows.append({**common, "sample_idx": index, "true_label": label, "target_cos": tcos, "source_cos": scos, "margin": margin, "feature_drift": drift_value, "predicted_label_if_available": ""})
            for class_id in sorted(set(trigger_labels.tolist())):
                mask = trigger_labels == class_id
                class_rows.append({**common, "source_class": class_id, "num_samples": int(mask.sum()), "target_cos_mean": stats(target_cos[mask])[0], "target_cos_std": stats(target_cos[mask])[1], "margin_mean": stats(margins[mask])[0], "margin_std": stats(margins[mask])[1], "feature_drift_mean": stats(drift[mask])[0], "feature_drift_std": stats(drift[mask])[1]})
            joined_rows.append({**common, "clean_accuracy": clean_accuracy, "attack_success_rate": attack_success_rate, "trigger_to_target_proto_cos_mean": target_mean, "feature_drift_mean": drift_mean, "target_source_margin_mean": margin_mean, "clean_target_proto_cos_mean": clean_mean})
            del model
            if device.type == "cuda": torch.cuda.empty_cache()
    write_csv(output_dir / "feature_persistence_summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(output_dir / "feature_persistence_sample_level.csv", sample_rows, list(sample_rows[0]))
    write_csv(output_dir / "feature_persistence_by_source_class.csv", class_rows, list(class_rows[0]))
    write_csv(output_dir / "feature_persistence_metrics_joined.csv", joined_rows, list(joined_rows[0]))
    make_report(output_dir, summary_rows, paths, args)
    print(json.dumps({"device": str(device), "feature_dim": 512, "trigger_eval_samples": len(trigger_indices), "prototype_samples": sum(map(len, prototype_indices.values())), "metrics_joined": True, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()

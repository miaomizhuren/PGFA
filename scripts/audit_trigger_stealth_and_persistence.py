#!/usr/bin/env python3
"""Read-only CIFAR-10 trigger stealth and persistence audit.

This script never trains, loads models, or modifies an existing experiment.  It
only reads YAML, SQLite, saved deltas and the local CIFAR-10 test batch, then
writes a new audit directory.
"""
from __future__ import annotations

import csv
import random
import sqlite3
from pathlib import Path

import numpy as np
import torch
import yaml
from skimage.metrics import structural_similarity
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/final_trigger_stealth_audit_20260924"
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)[None, :, None, None]
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)[None, :, None, None]
THRESHOLD = 1.0 / 255.0
SEED, N, TARGET = 20260924, 1000, 0


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_yaml(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def rows_from_db(path: Path) -> dict[int, float]:
    with sqlite3.connect(path) as conn:
        return {int(r): float(a) for r, a in conn.execute(
            "SELECT round_index, attack_success_rate FROM round_metrics "
            "WHERE attack_success_rate IS NOT NULL")}


def extract_tensor(path: Path) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.Tensor):
        tensor = payload
    elif isinstance(payload, dict):
        tensor = next((v for v in payload.values() if isinstance(v, torch.Tensor)), None)
        if tensor is None:
            raise ValueError(f"No tensor found in {path}")
    else:
        raise ValueError(f"Unsupported delta payload in {path}: {type(payload)}")
    return tensor.detach().cpu().float().numpy()


def uap_apply(raw: np.ndarray, delta_path: Path) -> np.ndarray:
    delta = extract_tensor(delta_path)
    if delta.ndim == 3:
        delta = delta[None]
    normalized = (raw - MEAN) / STD
    return np.clip(normalized + delta, -np.inf, np.inf) * STD + MEAN


def dba_apply(raw: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    # This exactly mirrors dba_component_masks(..., full_trigger=True) for the
    # final corner_blocks DBA3 configuration, but in raw space: the source
    # applies normalize_pixel_value(1.0), which inverses to raw RGB white.
    dba = config["attack"]["dba"]
    size, count = dba["subtrigger_size"], dba["num_parts"]
    height, width = raw.shape[-2:]
    candidates = [(height - size, 0), (height - size, width - size), (0, width - size), (0, 0)]
    mask = np.zeros((height, width), dtype=bool)
    for top, left in candidates[:count]:
        mask[top:top + size, left:left + size] = True
    triggered = raw.copy()
    triggered[:, :, mask] = 1.0
    return triggered, mask


def visual_metrics(clean: np.ndarray, triggered_unclipped: np.ndarray) -> dict:
    triggered = np.clip(triggered_unclipped, 0.0, 1.0)
    diff = np.abs(triggered - clean)
    mse = np.mean((triggered - clean) ** 2, axis=(1, 2, 3))
    psnr = np.where(mse == 0, np.inf, 10.0 * np.log10(1.0 / mse))
    ssims = [structural_similarity(c.transpose(1, 2, 0), t.transpose(1, 2, 0),
                                   channel_axis=-1, data_range=1.0)
             for c, t in zip(clean, triggered)]
    per_image_linf = diff.max(axis=(1, 2, 3))
    # Ratio of modified RGB elements.  For this DBA layout it equals the
    # spatial-pixel ratio because all three RGB channels are overwritten.
    return {"PSNR_mean": float(np.mean(psnr)), "SSIM_mean": float(np.mean(ssims)),
            "Raw_Linf_global": float(diff.max()), "Raw_Linf_mean": float(np.mean(per_image_linf)),
            "Modified_pixel_ratio_mean": float(np.mean(diff > THRESHOLD))}


def persistence(db_paths: list[Path]) -> tuple[dict, str]:
    values: dict[int, float] = {}
    for db in db_paths:
        values.update(rows_from_db(db))  # continuation is deliberately later
    missing = [r for r in range(3201, 3701) if r not in values]
    if missing:
        raise RuntimeError(f"Missing post-attack rounds: {missing[:10]} ...")
    post = np.array([values[r] for r in range(3201, 3701)], dtype=float)
    return ({"PA_AUC_1_500": float(post.mean()), "ASR_post_500": float(values[3700]),
             "N80": int((post >= 80).sum()), "N90": int((post >= 90).sum())},
            "; ".join(rel(p) for p in db_paths))


def method_specs() -> list[dict]:
    main = ROOT / "outputs/main_experiment_clean3000_force"
    return [
        {"Method": "ce_only_uap", "Display_name": "CE-only UAP", "kind": "uap",
         "config": ROOT / "configs/main_experiment_clean3000_force/ce_uap_clean3000_attack3001_3200_observe3201_3700.yaml",
         "output": main / "ce_uap_clean3000_attack3001_3200_observe3201_3700"},
        {"Method": "dba", "Display_name": "DBA", "kind": "dba",
         "config": ROOT / "configs/main_experiment_clean3000_force/dba_clean3000_attack3001_3200_observe3201_3700.yaml",
         "output": main / "dba_clean3000_attack3001_3200_observe3201_3700",
         "continuation": ROOT / "outputs/parallel_13_resume_unfinished/resume_dba_from3500_to3700"},
        {"Method": "neurotoxin", "Display_name": "Neurotoxin", "kind": "uap",
         "config": ROOT / "configs/parallel_resume2_rerun9_queue/neurotoxin_clean3000_attack3001_3200_observe3201_3700_rerun1.yaml",
         "output": main / "neurotoxin_clean3000_attack3001_3200_observe3201_3700_rerun1"},
        {"Method": "a3fl", "Display_name": "A3FL", "kind": "uap",
         "config": ROOT / "configs/main_experiment_clean3000_force/a3fl_clean3000_attack3001_3200_observe3201_3700.yaml",
         "output": main / "a3fl_clean3000_attack3001_3200_observe3201_3700"},
        {"Method": "pgfa", "Display_name": "PGFA (Ours)", "kind": "uap",
         "config": ROOT / "configs/clean_stage_sensitivity_pbu_force/pbu_force_clean3000_attack3001_3200_observe3201_3700.yaml",
         "output": ROOT / "outputs/pbu_force_clean_stage_sensitivity/clean3000_attack3001_3200_observe3201_3700"},
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = CIFAR10(ROOT / "datasets", train=False, download=False)
    labels = np.asarray(dataset.targets)
    candidates = np.flatnonzero(labels != TARGET)
    rng = np.random.default_rng(SEED)
    selected = rng.choice(candidates, size=N, replace=False)
    clean = np.stack([np.asarray(dataset.data[i], dtype=np.float32).transpose(2, 0, 1) / 255.0 for i in selected])
    records, details = [], []
    for spec in method_specs():
        cfg, attack = load_yaml(spec["config"]), None
        attack = cfg["attack"]
        db = spec["output"] / cfg["tracking"]["database_name"]
        dbs = [db]
        if "continuation" in spec:
            dbs.append(spec["continuation"] / "metrics_resume_dba_from3500_to3700.db")
        persist, trajectory = persistence(dbs)
        if spec["kind"] == "uap":
            trigger_path = spec["output"] / "checkpoints/delta_round_3200.pt"
            metrics = visual_metrics(clean, uap_apply(clean, trigger_path))
            if spec["Method"] == "a3fl":
                trigger_type = "fixed shared adaptive UAP (final attack-end delta)"
                special = "A3FL UAP, fixed after build; visual metrics use delta_round_3200.pt."
            elif spec["Method"] == "neurotoxin":
                trigger_type = "fixed shared CE-only UAP + Neurotoxin gradient mask"
                special = "mask_ratio=0.05; gradient mask changes parameter updates, not the visual trigger."
            elif spec["Method"] == "pgfa":
                trigger_type, special = "fixed shared prototype-guided UAP", "PBU is the legacy implementation label."
            else:
                trigger_type, special = "fixed shared CE-only UAP", ""
            dba_extra = {"DBA_num_parts": "", "DBA_subtrigger_size": "", "DBA_full_trigger_size_or_bbox": "", "DBA_full_trigger_modified_pixels": "", "DBA_full_trigger_modified_pixel_ratio_theoretical": "", "DBA_trigger_location": "", "DBA_pattern_summary": ""}
        else:
            triggered, mask = dba_apply(clean, cfg)
            metrics = visual_metrics(clean, triggered)
            trigger_path = None
            trigger_type = "DBA3 full fixed white patch (union of three sub-triggers)"
            special = "Evaluation uses full_trigger=True; patch is set to raw white through normalized application."
            pixels = int(mask.sum())
            dba_extra = {"DBA_num_parts": 3, "DBA_subtrigger_size": "5x5", "DBA_full_trigger_size_or_bbox": "3 disjoint 5x5 blocks",
                         "DBA_full_trigger_modified_pixels": pixels,
                         "DBA_full_trigger_modified_pixel_ratio_theoretical": pixels / (32 * 32),
                         "DBA_trigger_location": "bottom-left, bottom-right, top-right", "DBA_pattern_summary": "corner_blocks; raw RGB white"}
        record = {"Method": spec["Method"], "Display_name": spec["Display_name"], "Trigger_type": trigger_type,
                  "Trigger_source": "saved delta" if trigger_path else "YAML + attacks/dba.py deterministic layout",
                  "Trigger_path": rel(trigger_path) if trigger_path else "Not applicable (no saved patch file)",
                  "Target_label": attack["target_label"], "Num_malicious": attack["num_malicious_clients"],
                  "Poison_ratio": attack["poison_ratio"], "Attack_window": "3001-3200", **metrics, "Modified_pixel_ratio_threshold": THRESHOLD,
                  **persist, "Trajectory_source": trajectory, "Config_path": rel(spec["config"]), "Notes": special,
                  "Mask_ratio_or_selection_ratio": attack.get("neurotoxin", {}).get("mask_ratio", ""),
                  "Trigger_update_rule": ("fixed after build" if spec["Method"] == "a3fl" else ""), **dba_extra}
        records.append(record)
        details.append(f"## {spec['Display_name']}\n\n- Read final config: `{record['Config_path']}`.\n- Searched trigger evidence: final output checkpoint directory; selected `{record['Trigger_path']}`. DBA has no candidate trigger file because its trigger is deterministic source/YAML geometry.\n- Adopted trigger: {trigger_type}. Reason: {record['Trigger_source']}.\n- Application: {'normalized tensor + saved delta, inverse-normalized then clipped to [0,1]' if spec['kind'] == 'uap' else 'union DBA mask applied as raw-white pixels (equivalent to source normalized white patch), then clipped to [0,1]'}.\n- Visual: PSNR {metrics['PSNR_mean']:.4f}, SSIM {metrics['SSIM_mean']:.6f}, raw L∞ global {metrics['Raw_Linf_global']:.6f}, modified ratio {metrics['Modified_pixel_ratio_mean']:.6f}.\n- Persistence: PA-AUC {persist['PA_AUC_1_500']:.4f}%, ASR@+500 {persist['ASR_post_500']:.4f}%, N80={persist['N80']}, N90={persist['N90']}.\n- Notes: {special or 'None.'}\n")
    fields = list(records[0])
    with (OUT / "trigger_stealth_persistence_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    table_fields = ["Display_name", "Trigger_type", "PSNR_mean", "SSIM_mean", "Raw_Linf_global", "Modified_pixel_ratio_mean", "PA_AUC_1_500", "ASR_post_500", "N80", "N90"]
    table = "| " + " | ".join(table_fields) + " |\n|" + "|".join(["---"] * len(table_fields)) + "|\n"
    table += "\n".join("| " + " | ".join(str(r[k]) for k in table_fields) + " |" for r in records)
    summary = "# Trigger Stealth and Persistence Audit\n\n## Scope\n\nRead-only CIFAR-10 main-experiment audit. No training, checkpoint modification, GPU use, or original-result modification occurred.\n\n## Unified Evaluation Protocol\n\nCIFAR-10 test non-target subset: N=1000, target label=0, seed=20260924. Metrics compare clean and triggered raw RGB [0,1] images after clipping. PSNR and SSIM are per-image means (SSIM: `skimage.metrics.structural_similarity`, channel_axis=-1, data_range=1); raw L∞ is global max; modified-pixel ratio is the mean fraction of changed RGB elements using `abs(diff) > 1/255`. PA-AUC is the mean ASR (%) over rounds 3201--3700; N80/N90 are counts at thresholds 80/90%.\n\n## Summary Table\n\n" + table + "\n\n## Important Notes\n\nDBA is a local patch, so its PSNR/SSIM may look favorable; modified-pixel ratio is the more direct coverage comparison. UAP variants usually alter nearly every pixel element. Neurotoxin's mask is an optimization/update mask rather than a visual trigger. A3FL here is the project's adaptive-UAP implementation, not a claim of a complete original-paper patch-trigger reproduction.\n\n## Missing or Uncertain Fields\n\nNone for the requested five final paths. DBA has no serialized patch because its deterministic patch is defined by YAML and `attacks/dba.py`.\n"
    (OUT / "trigger_stealth_persistence_summary.md").write_text(summary)
    (OUT / "per_method_trigger_audit.md").write_text("# Per-method Trigger Audit\n\n" + "\n".join(details) + "\n## Evidence searched\n\nFor each method the script reads the selected final YAML, its listed SQLite database(s), and `checkpoints/delta_round_3200.pt` where applicable. DBA has no delta by design; its full evaluation trigger is reconstructed from the final YAML and `attacks/dba.py`. DBA's continuation database intentionally overwrites overlapping 3501--3590 rows and supplies 3591--3700.\n")
    print(f"Wrote {len(records)} method records to {OUT}")


if __name__ == "__main__":
    main()

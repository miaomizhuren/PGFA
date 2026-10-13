"""Offline CE-UAP versus PBU feature-margin analysis from saved checkpoints.

The script intentionally refuses to substitute checkpoints across methods or
rounds.  Run the generated recovery script first if its precheck reports a
missing checkpoint.
"""
from pathlib import Path
import sqlite3
import sys
import torch
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.dataset import load_dataset
from models.resnet18 import build_resnet18
from attacks.triggers import apply_trigger
OUT = ROOT / "outputs/saved_results/mechanism_feature_margin_20260911"
ROUNDS = [150, 210, 270, 330, 350]
SOURCES = {
    "CE-UAP": ROOT / "outputs/final_ablation_60_150_r350/abl01_ce_only_s60_e150_r350",
    "PBU": ROOT / "outputs/mechanism_feature_margin_60_150_r350/mech_pbu_s60_e150_r350_saveckpt",
}

def checkpoint(source, round_):
    path = source / "checkpoints" / f"global_round_{round_}.pt"
    return path if path.exists() else None

def fixed_delta(source):
    # The trigger is generated once and retained.  The attack-end snapshot is
    # therefore the canonical fixed UAP used for all post-attack checkpoints.
    path = source / "checkpoints" / "delta_round_150.pt"
    return path if path.exists() else None

def metric_rows(source):
    dbs = list(source.glob("*.db"))
    if not dbs:
        return {}
    with sqlite3.connect(dbs[0]) as conn:
        rows = conn.execute("select round_index, clean_accuracy, attack_success_rate from round_metrics").fetchall()
    return {round_: (clean, asr) for round_, clean, asr in rows}

def load_state(model, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state)
    return model.eval()

def main():
    missing = {method: [round_ for round_ in ROUNDS if checkpoint(source, round_) is None] for method, source in SOURCES.items()}
    missing = {method: rounds for method, rounds in missing.items() if rounds}
    if missing:
        details = "; ".join(f"{method}: {rounds}" for method, rounds in missing.items())
        raise RuntimeError(f"Required checkpoints are missing ({details}). Run scripts/run_mechanism_feature_margin_missing_ckpt.sh, then rerun this analysis.")
    deltas = {method: fixed_delta(source) for method, source in SOURCES.items()}
    if any(path is None for path in deltas.values()):
        raise RuntimeError("Missing attack-end delta_round_150.pt; analysis cannot use a method-matched fixed UAP.")
    banks = {
        "CE-UAP": SOURCES["PBU"] / "artifacts/ce_uap_prototype_bank_reconstructed_round60.pt",
        "PBU": SOURCES["PBU"] / "checkpoints/prototype_bank_round_150.pt",
    }
    if not all(path.exists() for path in banks.values()):
        raise RuntimeError("A trusted prototype bank is missing; run scripts/reconstruct_mechanism_prototype_bank.py first.")
    _, testset = load_dataset("cifar10", ROOT / "datasets", download=False)
    loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    for method, source in SOURCES.items():
        bank = torch.load(banks[method], map_location=device, weights_only=False)
        prototypes = F.normalize(bank["prototypes"].to(device), dim=1)
        delta = torch.load(deltas[method], map_location=device, weights_only=False).to(device)
        metrics = metric_rows(source)
        for round_ in ROUNDS:
            model = load_state(build_resnet18(10).to(device), checkpoint(source, round_), device)
            targets, sources, norms, count = [], [], [], 0
            with torch.no_grad():
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    keep = labels.ne(0)
                    if not keep.any(): continue
                    features = model.forward_features(apply_trigger(images[keep], "adversarial", perturbation=delta), feature_layer="layer4").flatten(1)
                    norms.append(features.norm(dim=1).cpu())
                    features = F.normalize(features, dim=1)
                    targets.append((features * prototypes[0]).sum(1).cpu())
                    sources.append((features * prototypes[labels[keep]]).sum(1).cpu())
                    count += int(keep.sum())
            target, source_sim, norm = torch.cat(targets), torch.cat(sources), torch.cat(norms)
            clean, asr = metrics[round_]
            records.append({"method": method, "round": round_, "relative_post_round": round_ - 150, "clean_acc": clean, "asr": asr, "S_target_mean": target.mean().item(), "S_target_std": target.std(unbiased=False).item(), "S_source_mean": source_sim.mean().item(), "S_source_std": source_sim.std(unbiased=False).item(), "M_target_minus_source": (target-source_sim).mean().item(), "feature_norm_mean": norm.mean().item(), "num_samples": count, "checkpoint_path": str(checkpoint(source, round_)), "uap_path": str(deltas[method]), "prototype_source": "client0", "prototype_path_or_status": str(banks[method])})
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records); frame.to_csv(OUT / "FEATURE_MARGIN_CE_vs_PBU.csv", index=False)
    frame[["relative_post_round","round","method","M_target_minus_source","S_target_mean","S_source_mean","asr","clean_acc"]].to_csv(OUT / "FEATURE_MARGIN_CURVE_DATA.csv", index=False)
    summary = []
    for method, group in frame.groupby("method", sort=False):
        values = group.set_index("relative_post_round")
        m0, m200 = values.loc[0,"M_target_minus_source"], values.loc[200,"M_target_minus_source"]
        summary.append({"method":method, **{f"M@{x}":values.loc[x,"M_target_minus_source"] for x in [0,60,120,180,200]}, "M_drop_0_to_200":m0-m200, "M_retention_200_vs_0":m200/m0 if m0 else np.nan, **{f"ASR@{x}":values.loc[x,"asr"] for x in [0,60,120,180,200]}, "PA-AUC_0:200":np.trapezoid(values["asr"], values.index)/200, "source_dir":str(SOURCES[method])})
    pd.DataFrame(summary).to_csv(OUT / "FEATURE_MARGIN_SUMMARY.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5,4.5))
    for method, group in frame.groupby("method", sort=False): ax.plot(group["relative_post_round"], group["M_target_minus_source"], marker="o", linewidth=2, label=method)
    ax.set(title="Target-over-source feature margin", xlabel="Rounds After Attack Termination", ylabel="Feature Margin M", xticks=[0,60,120,180,200]); ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    figures=OUT/"figures"; figures.mkdir(exist_ok=True)
    for suffix in ["pdf","png","svg"]: fig.savefig(figures/f"feature_margin_ce_vs_pbu.{suffix}", dpi=300 if suffix=="png" else None)
    print(OUT / "FEATURE_MARGIN_CE_vs_PBU.csv")

if __name__ == "__main__":
    main()

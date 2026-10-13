"""Strictly reconstruct the CE-UAP analysis-only prototype bank at round 60."""
from pathlib import Path
import json
import sys
import torch
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from torch.utils.data import DataLoader, Subset
from data.dataset import load_dataset, CIFAR10_MEAN, CIFAR10_STD
from data.partition import partition_iid
from models.resnet18 import build_resnet18
from attacks.prototype import collect_class_statistics, aggregate_class_statistics

SOURCE = ROOT / "outputs/final_ablation_60_150_r350/abl01_ce_only_s60_e150_r350"
CONFIG = ROOT / "configs/final_ablation_60_150_r350/abl01_ce_only_s60_e150_r350.yaml"
OUT = ROOT / "outputs/mechanism_feature_margin_60_150_r350/mech_pbu_s60_e150_r350_saveckpt/artifacts"

def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    seed, client_id, batches = cfg["experiment"]["seed"], 0, cfg["attack"]["trigger"]["adversarial"]["prototype_contrastive"]["center_batches"]
    train, _ = load_dataset("cifar10", ROOT / "datasets", download=False)
    indices = partition_iid(len(train), 100, seed)[client_id]
    model = build_resnet18(10)
    checkpoint = SOURCE / "checkpoints/global_round_60.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
    stats = collect_class_statistics(model, DataLoader(Subset(train, indices), batch_size=128, shuffle=True, num_workers=0), torch.device("cpu"), "layer4", batches)
    bank, details = aggregate_class_statistics([stats], 0, 1, 10, "skip")
    if bank is None: raise RuntimeError("Unable to construct a valid client-0 prototype bank.")
    bank.update({"temperature": 0.01, "mode": "target_vs_source", "feature_layer": "layer4", "source_margin": 0.7, "prototype_version": "analysis_reconstructed"})
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "ce_uap_prototype_bank_reconstructed_round60.pt"
    torch.save(bank, destination)
    metadata = {"reconstructed": True, "partition_seed": seed, "partition_type": "iid", "prototype_client_id": client_id, "num_samples_per_class_used": details["class_counts"], "class_counts": details["class_counts"], "feature_layer": "layer4", "center_batches": batches, "model_checkpoint_used": str(checkpoint), "normalization": {"mean": CIFAR10_MEAN, "std": CIFAR10_STD}, "warnings": ["CE-UAP bank is recomputed for analysis only; it was not used in CE-UAP training."]}
    (OUT / "ce_uap_prototype_bank_reconstruction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(destination)
if __name__ == "__main__": main()

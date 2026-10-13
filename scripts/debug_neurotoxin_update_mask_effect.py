"""Data-free, one-step check of Neurotoxin gradient versus update-delta masking."""
from pathlib import Path
import copy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from attacks.neurotoxin import (
    apply_neurotoxin_mask,
    apply_neurotoxin_update_mask,
    build_neurotoxin_mask,
)
from models.resnet18 import build_resnet18


torch.manual_seed(2026)
MODEL = build_resnet18(10).cpu()
PREVIOUS = {name: parameter.detach().clone() for name, parameter in MODEL.named_parameters()}
CURRENT = {
    name: parameter.detach().clone() + torch.randn_like(parameter) * 0.01
    for name, parameter in MODEL.named_parameters()
}
MASKS, INFO = build_neurotoxin_mask(PREVIOUS, CURRENT, MODEL.named_parameters(), 0.05)
IMAGES = torch.randn(4, 3, 32, 32)
LABELS = torch.tensor([0, 1, 2, 3])


def one_step(apply_mode):
    local = copy.deepcopy(MODEL).train()
    optimizer = torch.optim.SGD(local.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    optimizer.zero_grad()
    nn.CrossEntropyLoss()(local(IMAGES), LABELS).backward()
    if apply_mode == "gradient":
        apply_neurotoxin_mask(local.named_parameters(), MASKS)
    optimizer.step()
    if apply_mode == "update_delta":
        apply_neurotoxin_update_mask(local, MODEL, MASKS)
    return local


def summarize(local):
    global_parameters = dict(MODEL.named_parameters())
    total_before_sq = total_after_sq = masked_sq = kept_sq = 0.0
    masked_total = masked_nonzero = 0
    layer_stats = {}
    for name, parameter in local.named_parameters():
        delta = (parameter.detach() - global_parameters[name].detach()).cpu()
        mask = MASKS[name].cpu().bool()
        # "Before" is the unconstrained delta from an identical local step.
        layer = name.split(".")[0]
        stat = layer_stats.setdefault(layer, [0.0, 0.0])
        stat[1] += float(torch.linalg.vector_norm(delta).square())
        masked_delta = delta[~mask]
        kept_delta = delta[mask]
        masked_sq += float(torch.linalg.vector_norm(masked_delta).square())
        kept_sq += float(torch.linalg.vector_norm(kept_delta).square())
        masked_total += masked_delta.numel()
        masked_nonzero += int((masked_delta.abs() > 1e-12).sum())
        total_after_sq += float(torch.linalg.vector_norm(delta).square())
    return {
        "total_update_norm": total_after_sq ** 0.5,
        "masked_delta_norm": masked_sq ** 0.5,
        "kept_delta_norm": kept_sq ** 0.5,
        "masked_nonzero_fraction": masked_nonzero / masked_total if masked_total else 0.0,
        "per_layer_after_sq": layer_stats,
    }


# First calculate an unconstrained baseline to report per-layer before norms.
BASELINE = one_step("none")
GRADIENT = one_step("gradient")
UPDATE_DELTA = one_step("update_delta")
baseline_parameters = dict(BASELINE.named_parameters())
global_parameters = dict(MODEL.named_parameters())
for stats in (summarize(GRADIENT), summarize(UPDATE_DELTA)):
    for name, parameter in baseline_parameters.items():
        layer = name.split(".")[0]
        stats["per_layer_after_sq"].setdefault(layer, [0.0, 0.0])[0] += float(
            torch.linalg.vector_norm((parameter.detach() - global_parameters[name].detach()).cpu()).square()
        )

gradient_stats = summarize(GRADIENT)
update_stats = summarize(UPDATE_DELTA)
for stats in (gradient_stats, update_stats):
    for name, parameter in baseline_parameters.items():
        layer = name.split(".")[0]
        stats["per_layer_after_sq"].setdefault(layer, [0.0, 0.0])[0] += float(
            torch.linalg.vector_norm((parameter.detach() - global_parameters[name].detach()).cpu()).square()
        )

out = Path("outputs/debug_neurotoxin_mask")
out.mkdir(parents=True, exist_ok=True)
lines = [
    "mask_ratio=0.05",
    f"total_update_norm_before_mask={summarize(BASELINE)['total_update_norm']:.12g}",
    f"gradient_mode_total_update_norm_after_mask={gradient_stats['total_update_norm']:.12g}",
    f"gradient_mode_masked_positions_delta_norm={gradient_stats['masked_delta_norm']:.12g}",
    f"gradient_mode_kept_positions_delta_norm={gradient_stats['kept_delta_norm']:.12g}",
    f"gradient_mode_fraction_masked_positions_nonzero={gradient_stats['masked_nonzero_fraction']:.12g}",
    f"update_delta_mode_total_update_norm_after_mask={update_stats['total_update_norm']:.12g}",
    f"update_delta_mode_masked_positions_delta_norm={update_stats['masked_delta_norm']:.12g}",
    f"update_delta_mode_kept_positions_delta_norm={update_stats['kept_delta_norm']:.12g}",
    f"update_delta_mode_fraction_masked_positions_nonzero={update_stats['masked_nonzero_fraction']:.12g}",
]
for layer in ("conv1", "layer1", "layer2", "layer3", "layer4", "fc"):
    layer_masked = sum(values["masked"] for name, values in INFO["per_layer"].items() if name.split(".")[0] == layer)
    layer_total = sum(values["total"] for name, values in INFO["per_layer"].items() if name.split(".")[0] == layer)
    before_sq = gradient_stats["per_layer_after_sq"].get(layer, [0.0, 0.0])[0]
    gradient_sq = gradient_stats["per_layer_after_sq"].get(layer, [0.0, 0.0])[1]
    update_sq = update_stats["per_layer_after_sq"].get(layer, [0.0, 0.0])[1]
    lines.append(f"{layer}_delta_norm_before_mask={before_sq ** 0.5:.12g}")
    lines.append(f"{layer}_delta_norm_gradient_mode={gradient_sq ** 0.5:.12g}")
    lines.append(f"{layer}_delta_norm_update_delta_mode={update_sq ** 0.5:.12g}")
    lines.append(f"{layer}_masked_ratio={(layer_masked / layer_total) if layer_total else 0.0:.12g}")
    lines.append(f"{layer}_keep_ratio={(1 - layer_masked / layer_total) if layer_total else 1.0:.12g}")
(out / "neurotoxin_update_mask_effect.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

assert gradient_stats["masked_nonzero_fraction"] > 0.0, "gradient mode unexpectedly produced exact update masking"
assert update_stats["masked_delta_norm"] < 1e-10
assert update_stats["masked_nonzero_fraction"] == 0.0
print("NEUROTOXIN_UPDATE_MASK_EFFECT_PASS")

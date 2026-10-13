"""Gradient masks for the Neurotoxin-style UAP baseline."""

import math
import torch


def build_neurotoxin_mask(previous_state, current_state, named_parameters, ratio):
    """Mask the globally most-active trainable coordinates (0=masked)."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be in [0, 1]")
    parameters = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
    if previous_state is None or current_state is None:
        return None, {"usable_previous_global_delta": False, "masked_parameter_fraction": 0.0, "masked_coordinate_count": 0, "global_delta_l2": None}
    magnitudes, entries = [], []
    for name, parameter in parameters:
        if name not in previous_state or name not in current_state:
            raise KeyError(f"missing trainable parameter in global snapshot: {name}")
        delta = (current_state[name].detach().cpu().float() - previous_state[name].detach().cpu().float()).abs().reshape(-1)
        magnitudes.append(delta); entries.append((name, parameter, delta.numel()))
    flat = torch.cat(magnitudes) if magnitudes else torch.empty(0)
    count = int(math.ceil(flat.numel() * ratio))
    mask_flat = torch.ones_like(flat)
    if count:
        # stable sort makes ties deterministic in named-parameter/flatten order.
        top = torch.argsort(flat, descending=True, stable=True)[:count]
        mask_flat[top] = 0
    masks, offset, per_layer = {}, 0, {}
    for name, parameter, numel in entries:
        masks[name] = mask_flat[offset:offset + numel].reshape_as(parameter).to(parameter.device, parameter.dtype)
        masked = int((masks[name] == 0).sum().item())
        per_layer[name] = {"total": numel, "kept": numel - masked, "masked": masked,
                           "keep_ratio": (numel - masked) / numel, "masked_ratio": masked / numel}
        offset += numel
    masked_fraction = count / flat.numel() if flat.numel() else 0.0
    return masks, {"usable_previous_global_delta": True, "masked_parameter_fraction": masked_fraction,
                   "kept_parameter_fraction": 1.0 - masked_fraction,
                   "masked_coordinate_count": count, "kept_coordinate_count": flat.numel() - count,
                   "total_trainable_params": flat.numel(), "selected_direction": "high_gradient",
                   "per_layer": per_layer, "global_delta_l2": float(torch.linalg.vector_norm(flat))}


def apply_neurotoxin_mask(named_parameters, masks):
    """Apply precomputed masks after backward and before the local SGD step."""
    if masks is None:
        return
    for name, parameter in named_parameters:
        if parameter.grad is not None and name in masks:
            parameter.grad.mul_(masks[name].to(parameter.grad.device, parameter.grad.dtype))


def apply_neurotoxin_update_mask(local_model, global_model, masks):
    """Project a malicious local update onto the kept Neurotoxin coordinates.

    This optional mode is deliberately applied after all local optimizer steps:
    each trainable local parameter becomes ``global + mask * (local-global)``.
    It therefore guarantees zero uploaded *trainable-parameter* delta at every
    masked coordinate, including coordinates changed by optimizer weight decay.
    """
    if masks is None:
        return
    global_parameters = dict(global_model.named_parameters())
    with torch.no_grad():
        for name, local_parameter in local_model.named_parameters():
            if name not in masks:
                continue
            global_parameter = global_parameters[name].to(
                device=local_parameter.device,
                dtype=local_parameter.dtype,
            )
            mask = masks[name].to(
                device=local_parameter.device,
                dtype=local_parameter.dtype,
            )
            local_parameter.copy_(global_parameter + mask * (local_parameter - global_parameter))

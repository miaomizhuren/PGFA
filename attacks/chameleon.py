"""Core, UAP-free Chameleon adaptation and projection primitives."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def snapshot_named_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return detached parameter copies for Chameleons global-model reference.

    The official Helper.model_dist_norm iterates model.named_parameters.
    Deliberately do not use state_dict here: BatchNorm running statistics are
    buffers, not model parameters, and are outside this constraint.
    """
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


@torch.no_grad()
def project_named_parameters_to_l2_ball(
    model: torch.nn.Module,
    reference_parameters: dict[str, torch.Tensor],
    *,
    radius: float,
) -> dict[str, float | int | bool]:
    """Apply the official Chameleon post-step model-distance projection.

    training.py calls model_dist_norm after every optimizer step and, when the
    global L2 distance is above s_norm, scales every named parameters
    displacement from the rounds global model by one common factor. This
    mirrors that behavior for image models: all named parameters participate,
    whereas BatchNorm running buffers do not.

    Although the classifier is frozen in adaptation, it is intentionally
    retained in this all-parameter definition. Its displacement is zero in
    that stage. During projection this also preserves the official behavior of
    shrinking the encoders Stage-1 displacement if the whole model is outside
    the radius.
    """
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or radius <= 0:
        raise ValueError("radius must be a positive number")

    named_parameters = dict(model.named_parameters())
    if set(named_parameters) != set(reference_parameters):
        missing_reference = sorted(set(named_parameters) - set(reference_parameters))
        extra_reference = sorted(set(reference_parameters) - set(named_parameters))
        raise ValueError(
            "reference parameter names must exactly match the model; "
            f"missing={missing_reference}, extra={extra_reference}"
        )

    first_parameter = next(iter(named_parameters.values()))
    squared_distance = torch.zeros((), device=first_parameter.device, dtype=torch.float64)
    for name, parameter in named_parameters.items():
        reference = reference_parameters[name].to(device=parameter.device, dtype=parameter.dtype)
        squared_distance.add_((parameter.detach() - reference).to(torch.float64).square().sum())
    distance_before = float(torch.sqrt(squared_distance).item())
    if not torch.isfinite(torch.tensor(distance_before)):
        raise FloatingPointError("non-finite Chameleon model distance")

    projected = distance_before > float(radius)
    if projected:
        scale = float(radius) / distance_before
        for name, parameter in named_parameters.items():
            reference = reference_parameters[name].to(device=parameter.device, dtype=parameter.dtype)
            parameter.copy_(reference + scale * (parameter - reference))
        distance_after = float(radius)
    else:
        distance_after = distance_before

    if not all(torch.isfinite(parameter).all() for parameter in named_parameters.values()):
        raise FloatingPointError("non-finite parameter after Chameleon projection")
    return {
        "distance_before": distance_before,
        "distance_after": distance_after,
        "projected": projected,
        "parameter_tensors": len(named_parameters),
    }


def weighted_supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    target_label: int,
    temperature: float = 0.07,
    target_anchor_weight: float = 6.0,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Compute Chameleon's weighted SupCon loss on normalized features.

    Samples of an equal *training* label are positives.  Target-label anchors
    (both poisoned examples and clean facilitators) receive the official
    multiplicative positive-pair weight.  An anchor without a positive pair
    contributes exactly zero and remains in the batch mean, matching the
    original loss' zero-mask semantics while avoiding ``0 * -inf`` for a
    final one-sample batch.
    """
    if features.ndim != 2:
        raise ValueError("features must have shape [batch, feature_dim]")
    if labels.ndim != 1 or labels.size(0) != features.size(0):
        raise ValueError("labels must have shape [batch]")
    if temperature <= 0 or target_anchor_weight <= 0:
        raise ValueError("temperature and target_anchor_weight must be positive")

    batch_size = features.size(0)
    normalized = functional.normalize(features, dim=1)
    target_anchors = labels.eq(target_label)
    if batch_size < 2:
        return normalized.sum() * 0.0, {
            "valid_anchors": 0,
            "target_anchors": int(target_anchors.sum().item()),
            "target_valid_anchors": 0,
        }

    logits = normalized @ normalized.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    nonself = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)
    positive = labels.view(-1, 1).eq(labels.view(1, -1)) & nonself
    positive_counts = positive.sum(dim=1)
    valid = positive_counts.gt(0)

    masked_logits = logits.masked_fill(~nonself, float("-inf"))
    log_prob = logits - torch.logsumexp(masked_logits, dim=1, keepdim=True)
    anchor_weight = torch.where(
        target_anchors,
        torch.full_like(labels, float(target_anchor_weight), dtype=features.dtype),
        torch.ones_like(labels, dtype=features.dtype),
    )
    positive_log_prob = torch.where(positive, log_prob, torch.zeros_like(log_prob)).sum(dim=1)
    per_anchor = torch.zeros(batch_size, dtype=features.dtype, device=features.device)
    per_anchor[valid] = -anchor_weight[valid] * positive_log_prob[valid] / positive_counts[valid]
    return per_anchor.mean(), {
        "valid_anchors": int(valid.sum().item()),
        "target_anchors": int(target_anchors.sum().item()),
        "target_valid_anchors": int((valid & target_anchors).sum().item()),
    }


def set_chameleon_adaptation_trainability(model: torch.nn.Module) -> None:
    """Enable encoder parameters and freeze only the classifier parameters."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("fc."))


def set_chameleon_projection_trainability(model: torch.nn.Module) -> None:
    """Freeze encoder parameters while deliberately keeping train-mode BN buffers live."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("fc."))


def restore_all_trainability(model: torch.nn.Module) -> None:
    """Restore a detached local copy to the default fully trainable state."""
    for parameter in model.parameters():
        parameter.requires_grad_(True)

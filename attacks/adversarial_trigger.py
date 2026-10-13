"""Shared, targeted adversarial triggers for controlled FL experiments."""

import numbers

import torch
import torch.nn.functional as functional


def compute_prototype_loss(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    class_ids: list[int],
    target_proto_index: int | None,
    temperature: float,
    mode: str,
    source_labels: torch.Tensor | None = None,
    source_margin: float = 0.2,
    valid_class_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Compute target-vs-all or target-vs-source prototype regularization."""
    if prototypes.ndim != 2:
        raise ValueError("prototypes must have shape [num_prototypes, feature_dim]")
    if len(class_ids) != prototypes.size(0):
        raise ValueError("class_ids must contain one entry per prototype row")
    if target_proto_index is None:
        return None
    target_proto_index = int(target_proto_index)
    if target_proto_index < 0 or target_proto_index >= prototypes.size(0):
        return None

    if valid_class_mask is not None:
        valid_class_mask = valid_class_mask.to(device=prototypes.device, dtype=torch.bool)
        if valid_class_mask.ndim != 1:
            raise ValueError("valid_class_mask must be one-dimensional")

        # Aggregated prototype banks are full class-indexed tensors, whereas
        # legacy-local banks are compact tensors whose ``class_ids`` may skip
        # classes.  A legacy-local bank still records its availability mask in
        # the full class space.  Convert either representation to a mask over
        # prototype rows before indexing the compact tensor.
        if valid_class_mask.numel() == prototypes.size(0):
            row_mask = valid_class_mask
        else:
            class_id_tensor = torch.as_tensor(
                class_ids, device=prototypes.device, dtype=torch.long
            )
            if class_id_tensor.numel() and (
                int(class_id_tensor.min().item()) < 0
                or int(class_id_tensor.max().item()) >= valid_class_mask.numel()
            ):
                raise ValueError(
                    "full-space valid_class_mask does not cover every class_id"
                )
            row_mask = valid_class_mask[class_id_tensor]

        if not bool(row_mask[target_proto_index]):
            return None
        indices = row_mask.nonzero(as_tuple=False).flatten()
        target_proto_index = int((indices == target_proto_index).nonzero(as_tuple=False)[0].item())
        prototypes = prototypes[indices]
        class_ids = [class_ids[index] for index in indices.cpu().tolist()]
    normalized_features = functional.normalize(features, dim=1)
    normalized_prototypes = functional.normalize(prototypes.detach(), dim=1)
    if mode == "target_vs_all":
        logits = normalized_features @ normalized_prototypes.T / temperature
        targets = torch.full(
            (features.size(0),), target_proto_index, device=features.device, dtype=torch.long
        )
        return functional.cross_entropy(logits, targets)
    if mode != "target_vs_source":
        raise ValueError("mode must be 'target_vs_all' or 'target_vs_source'")
    if source_labels is None:
        raise ValueError("source_labels are required for target_vs_source")

    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
    target_class_id = class_ids[target_proto_index]
    valid_positions = [
        index
        for index, label in enumerate(source_labels.detach().cpu().tolist())
        if int(label) != target_class_id and int(label) in class_to_index
    ]
    if not valid_positions:
        return None
    positions = torch.tensor(valid_positions, device=features.device, dtype=torch.long)
    source_indices = torch.tensor(
        [class_to_index[int(source_labels[index].item())] for index in valid_positions],
        device=features.device,
        dtype=torch.long,
    )
    selected_features = normalized_features[positions]
    target_prototype = normalized_prototypes[target_proto_index].unsqueeze(0)
    source_prototypes = normalized_prototypes[source_indices]
    sim_target = (selected_features * target_prototype).sum(dim=1)
    sim_source = (selected_features * source_prototypes).sum(dim=1)
    return functional.relu(source_margin + sim_source - sim_target).mean()


def compute_feature_regularization(
    features: torch.Tensor,
    variant: str,
    prototypes: torch.Tensor | None = None,
    class_ids: list[int] | None = None,
    target_proto_index: int | None = None,
    temperature: float = 0.2,
    mode: str = "target_vs_all",
    source_labels: torch.Tensor | None = None,
    source_margin: float = 0.2,
    valid_class_mask: torch.Tensor | None = None,
    target_reference_feature: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Compute the requested PGFA feature regularizer without changing margin."""
    if variant == "margin":
        if prototypes is None or class_ids is None:
            return None
        return compute_prototype_loss(features, prototypes, class_ids, target_proto_index, temperature, mode, source_labels, source_margin, valid_class_mask)
    if variant == "target_only":
        if prototypes is None or target_proto_index is None or target_proto_index < 0 or target_proto_index >= prototypes.size(0):
            return None
        reference = prototypes[target_proto_index]
    elif variant == "feature_alignment":
        if target_reference_feature is None:
            return None
        # Banks intentionally cache the fixed local reference on CPU.  Move a
        # detached view only at use time so the UAP and local-training paths
        # share device/dtype with differentiable poisoned features.
        reference = target_reference_feature.detach().to(
            device=features.device, dtype=features.dtype
        )
    else:
        raise ValueError("variant must be margin, target_only, or feature_alignment")
    if reference.ndim != 1 or reference.numel() != features.size(1):
        raise ValueError("reference feature dimension must match features")
    normalized_features = functional.normalize(features, dim=1)
    normalized_reference = functional.normalize(reference.detach(), dim=0)
    return (1.0 - (normalized_features * normalized_reference.unsqueeze(0)).sum(dim=1)).mean()


def generate_adversarial_trigger(
    model: torch.nn.Module,
    images: torch.Tensor,
    target_label: int,
    epsilon: float,
    steps: int,
    alpha: float,
) -> torch.Tensor:
    """Generate one targeted L-infinity-bounded perturbation for an image batch.

    The returned perturbation has shape ``[1, C, H, W]`` so it can be broadcast
    to every image in the supplied batch.  ``epsilon`` and ``alpha`` are in the
    same (usually normalized) input space as ``images``.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor")
    if images.ndim != 4:
        raise ValueError("images must have shape [N, C, H, W]")
    if images.shape[0] == 0:
        raise ValueError("images must contain at least one image")
    if not images.is_floating_point():
        raise TypeError("images must use a floating-point dtype")
    if isinstance(target_label, bool) or not isinstance(target_label, numbers.Integral):
        raise TypeError("target_label must be an integer")
    if isinstance(steps, bool) or not isinstance(steps, numbers.Integral):
        raise TypeError("steps must be an integer")
    if steps <= 0:
        raise ValueError("steps must be positive")
    for name, value in (("epsilon", epsilon), ("alpha", alpha)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be a real number")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    epsilon = float(epsilon)
    alpha = float(alpha)
    delta = torch.empty(
        (1, *images.shape[1:]), device=images.device, dtype=images.dtype
    ).uniform_(-epsilon, epsilon)

    was_training = model.training
    model.eval()
    try:
        for _ in range(steps):
            delta.requires_grad_(True)
            logits = model(images + delta)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                raise ValueError("model must return logits with shape [N, num_classes]")
            if target_label < 0 or target_label >= logits.shape[1]:
                raise ValueError("target_label is outside the model's class range")

            targets = torch.full(
                (images.shape[0],), target_label, device=images.device, dtype=torch.long
            )
            # Targeted PGD minimizes the loss of the desired target class.
            loss = functional.cross_entropy(logits, targets)
            gradient = torch.autograd.grad(loss, delta, only_inputs=True)[0]
            delta = (delta - alpha * gradient.sign()).detach().clamp_(-epsilon, epsilon)
    finally:
        model.train(was_training)

    return delta.to(device=images.device, dtype=images.dtype)


def generate_universal_adversarial_trigger(
    model: torch.nn.Module,
    dataloader,
    target_label: int,
    epsilon: float,
    steps: int,
    alpha: float,
    prototype_contrastive_enabled: bool = False,
    prototype_lambda: float = 0.1,
    prototype_temperature: float = 0.2,
    prototype_center_batches: int = 5,
    prototype_mode: str = "target_vs_all",
    prototype_min_samples_per_class: int = 1,
    exclude_target_in_uap: bool = True,
    prototype_feature_layer: str = "penultimate",
    prototype_source_margin: float = 0.2,
    prototype_loss_variant: str = "margin",
    fixed_target_reference_feature: torch.Tensor | None = None,
    return_prototypes: bool = False,
    prototype_bank: dict | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict | None]:
    """Generate one targeted perturbation shared across batches in ``dataloader``.

    The default CE-only path preserves the original targeted UAP objective.
    When enabled, prototype contrastive regularization aligns triggered
    penultimate features with the clean target-class prototype.  Set
    ``return_prototypes`` to cache the clean prototype bank for later local
    malicious-client training without changing the default return type.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not hasattr(dataloader, "__iter__"):
        raise TypeError("dataloader must be iterable")
    if isinstance(target_label, bool) or not isinstance(target_label, numbers.Integral):
        raise TypeError("target_label must be an integer")
    if isinstance(steps, bool) or not isinstance(steps, numbers.Integral):
        raise TypeError("steps must be an integer")
    if steps <= 0:
        raise ValueError("steps must be positive")
    for name, value in (("epsilon", epsilon), ("alpha", alpha)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be a real number")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if prototype_temperature <= 0:
        raise ValueError("prototype_temperature must be positive")
    if prototype_center_batches == 0 or prototype_center_batches < -1:
        raise ValueError("prototype_center_batches must be positive or -1 (full local loader)")
    if prototype_min_samples_per_class <= 0:
        raise ValueError("prototype_min_samples_per_class must be positive")
    if prototype_loss_variant not in ("margin", "target_only", "feature_alignment"):
        raise ValueError("prototype_loss_variant must be margin, target_only, or feature_alignment")
    if prototype_mode not in ("target_vs_all", "target_vs_source"):
        raise ValueError("prototype_mode must be 'target_vs_all' or 'target_vs_source'")
    if prototype_source_margin < 0:
        raise ValueError("prototype_source_margin must be non-negative")
    if prototype_feature_layer not in ("layer3", "layer4", "avgpool", "penultimate"):
        raise ValueError(
            "prototype_feature_layer must be 'layer3', 'layer4', 'avgpool', or 'penultimate'"
        )

    try:
        reference_batch = next(iter(dataloader))
    except StopIteration as error:
        raise ValueError("dataloader must contain at least one batch") from error
    if not isinstance(reference_batch, (tuple, list)) or not reference_batch:
        raise ValueError("dataloader batches must contain an image tensor")
    reference_images = reference_batch[0]
    if not isinstance(reference_images, torch.Tensor):
        raise TypeError("dataloader batches must contain images as a torch.Tensor")
    if reference_images.ndim != 4 or reference_images.shape[0] == 0:
        raise ValueError("images must have non-empty shape [N, C, H, W]")
    if not reference_images.is_floating_point():
        raise TypeError("images must use a floating-point dtype")

    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else reference_images.device
    dtype = reference_images.dtype
    epsilon = float(epsilon)
    alpha = float(alpha)
    delta = torch.empty(
        (1, *reference_images.shape[1:]), device=device, dtype=dtype
    ).uniform_(-epsilon, epsilon)

    was_training = model.training
    model.eval()
    try:
        use_prototype_contrastive = prototype_contrastive_enabled
        build_prototype_bank = prototype_contrastive_enabled or return_prototypes
        prototypes = None
        prototype_labels: list[int] = []
        target_prototype_index = None
        target_reference_feature = (fixed_target_reference_feature.detach().to(device=device, dtype=dtype) if fixed_target_reference_feature is not None else None)
        if prototype_bank is not None:
            prototypes = prototype_bank["prototypes"].to(device=device, dtype=dtype).detach()
            prototype_labels = list(prototype_bank["class_ids"])
            target_prototype_index = int(prototype_bank["target_proto_index"])
        elif build_prototype_bank:
            if not hasattr(model, "forward_features"):
                print(
                    "[UAP-PROTO] warning=forward_features_missing; falling back to CE-only",
                    flush=True,
                )
                use_prototype_contrastive = False
            else:
                class_sums: dict[int, torch.Tensor] = {}
                class_counts: dict[int, int] = {}
                with torch.no_grad():
                    for batch_index, batch in enumerate(dataloader):
                        if prototype_center_batches != -1 and batch_index >= prototype_center_batches:
                            break
                        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                            raise ValueError("dataloader batches must contain images and labels")
                        images, labels = batch[0], batch[1]
                        if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
                            raise TypeError("dataloader batches must contain tensor images and labels")
                        features = model.forward_features(
                            images.to(device=device, dtype=dtype),
                            feature_layer=prototype_feature_layer,
                        )
                        target_mask = labels == target_label
                        if target_reference_feature is None and target_mask.any():
                            target_reference_feature = features[target_mask.to(device)][0].detach().clone()
                        for class_label in labels.unique().tolist():
                            class_label = int(class_label)
                            mask = labels == class_label
                            selected = features[mask.to(device)].detach()
                            class_sums[class_label] = class_sums.get(
                                class_label, torch.zeros_like(selected[0])
                            ) + selected.sum(dim=0)
                            class_counts[class_label] = class_counts.get(class_label, 0) + selected.size(0)

                prototype_labels = sorted(
                    label
                    for label, count in class_counts.items()
                    if count >= prototype_min_samples_per_class
                )
                if target_label not in prototype_labels:
                    print(
                        "[UAP-PROTO] warning=target_prototype_unavailable; falling back to CE-only",
                        flush=True,
                    )
                    use_prototype_contrastive = False
                else:
                    prototypes = torch.stack(
                        [class_sums[label] / class_counts[label] for label in prototype_labels]
                    ).detach()
                    prototypes = functional.normalize(prototypes, dim=1)
                    target_prototype_index = prototype_labels.index(target_label)
                    if use_prototype_contrastive:
                        print("[UAP-PROTO] enabled=True", flush=True)
                        print(
                            f"[UAP-PROTO] lambda={prototype_lambda}, "
                            f"temperature={prototype_temperature}, "
                            f"center_batches={prototype_center_batches}",
                            flush=True,
                        )
                        print(f"[UAP-PROTO] available_classes={prototype_labels}", flush=True)
                        print(f"[UAP-PROTO] target_label={target_label}", flush=True)
                        print(f"[UAP-PROTO] prototype_shape={tuple(prototypes.shape)}", flush=True)
                        print(
                            f"[UAP-PROTO] exclude_target_in_uap={exclude_target_in_uap}",
                            flush=True,
                        )
        if not use_prototype_contrastive:
            print("[UAP] mode=ce_only", flush=True)
            print(f"[UAP] epsilon={epsilon}, alpha={alpha}, steps={steps}", flush=True)

        # A later batch can contain a class absent from the fixed local
        # prototype bank. Warn once and use target_vs_all for that batch.
        source_fallback_warned = False
        for _ in range(steps):
            delta.requires_grad_(True)
            accumulated_gradient = torch.zeros_like(delta)
            batch_count = 0
            ce_loss_total = 0.0
            proto_loss_total = 0.0
            for batch in dataloader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise ValueError("dataloader batches must contain images and labels")
                images, labels = batch[0], batch[1]
                if not isinstance(images, torch.Tensor) or images.ndim != 4:
                    raise ValueError("images must have shape [N, C, H, W]")
                if images.shape[1:] != delta.shape[1:]:
                    raise ValueError("all dataloader images must share one shape")
                images = images.to(device=device, dtype=dtype)
                labels = labels.to(device=device)
                source_labels = labels
                if use_prototype_contrastive and exclude_target_in_uap:
                    non_target_mask = labels != target_label
                    images = images[non_target_mask]
                    source_labels = source_labels[non_target_mask]
                if images.size(0) == 0:
                    continue

                logits = model(images + delta)
                if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                    raise ValueError("model must return logits with shape [N, num_classes]")
                if target_label < 0 or target_label >= logits.shape[1]:
                    raise ValueError("target_label is outside the model's class range")
                targets = torch.full(
                    (images.shape[0],), target_label, device=device, dtype=torch.long
                )
                ce_loss = functional.cross_entropy(logits, targets)
                loss = ce_loss
                proto_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
                if use_prototype_contrastive:
                    triggered_features = model.forward_features(
                        images + delta,
                        feature_layer=prototype_feature_layer,
                    )
                    proto_loss = compute_feature_regularization(
                        triggered_features, prototype_loss_variant, prototypes, prototype_labels, target_prototype_index,
                        prototype_temperature, prototype_mode, source_labels, prototype_source_margin,
                        (prototype_bank or {}).get("valid_class_mask"), target_reference_feature,
                    )
                    if proto_loss is None and prototype_loss_variant != "margin":
                        raise RuntimeError(f"{prototype_loss_variant} requested but its required target reference is unavailable")
                    if proto_loss is None:
                        if not source_fallback_warned:
                            print(
                                "[UAP-PROTO] warning=source_prototype_unavailable; "
                                "falling back to target_vs_all for this batch",
                                flush=True,
                            )
                            source_fallback_warned = True
                        proto_loss = compute_prototype_loss(
                            triggered_features,
                            prototypes,
                            prototype_labels,
                            target_prototype_index,
                            prototype_temperature,
                            "target_vs_all",
                            valid_class_mask=(prototype_bank or {}).get("valid_class_mask"),
                        )
                    if proto_loss is None:
                        # A zero-filled missing target prototype is an
                        # explicit placeholder, never a contrastive anchor.
                        proto_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
                    loss = ce_loss + prototype_lambda * proto_loss
                accumulated_gradient.add_(
                    torch.autograd.grad(loss, delta, only_inputs=True)[0]
                )
                batch_count += 1
                ce_loss_total += ce_loss.item()
                proto_loss_total += proto_loss.item()

            if batch_count == 0:
                raise ValueError("dataloader contains no usable UAP-generation samples")
            delta = (
                delta - alpha * accumulated_gradient.sign()
            ).detach().clamp_(-epsilon, epsilon)
            if use_prototype_contrastive and (_ == 0 or _ == steps - 1):
                print(
                    f"[UAP-PROTO] step {_ + 1:03d} "
                    f"ce_loss={ce_loss_total / batch_count:.6f} "
                    f"proto_loss={proto_loss_total / batch_count:.6f} "
                    f"total_loss={(ce_loss_total + prototype_lambda * proto_loss_total) / batch_count:.6f}",
                    flush=True,
                )
        if use_prototype_contrastive:
            print(f"[UAP-PROTO] delta_linf={delta.abs().max().item():.6f}", flush=True)
            print(f"[UAP-PROTO] delta_mean_abs={delta.abs().mean().item():.6f}", flush=True)
    finally:
        model.train(was_training)

    final_delta = delta.to(device=device, dtype=dtype)
    prototype_bank = None
    if prototypes is not None:
        prototype_bank = {
            "prototypes": prototypes.detach().cpu(),
            "class_ids": list(prototype_labels),
            "target_proto_index": int(target_prototype_index),
            "temperature": float(prototype_temperature),
            "mode": prototype_mode,
            "feature_layer": prototype_feature_layer,
            "source_margin": float(prototype_source_margin),
            "loss_variant": prototype_loss_variant,
            "target_reference_feature": target_reference_feature.detach().cpu() if target_reference_feature is not None else None,
            "valid_class_mask": torch.tensor(
                [label in prototype_labels for label in range(getattr(model.fc, "out_features", len(prototype_labels)))],
                dtype=torch.bool,
            ),
            "missing_classes": [
                label for label in range(getattr(model.fc, "out_features", len(prototype_labels)))
                if label not in prototype_labels
            ],
            "samples_per_class": [
                int(class_counts.get(label, 0)) for label in range(getattr(model.fc, "out_features", len(prototype_labels)))
            ],
        }
    if return_prototypes:
        return final_delta, prototype_bank
    return final_delta

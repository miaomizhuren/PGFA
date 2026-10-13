"""Model evaluation utilities for clean accuracy, ASR, and trigger stealth."""

import copy
import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from attacks.triggers import apply_patch_trigger, apply_trigger
from attacks.dba import apply_dba_trigger
from attacks.chameleon_trigger import apply_chameleon_type1_trigger
from data.dataset import CIFAR10_MEAN, CIFAR10_STD


def _crfl_defended_predictions(model: nn.Module, images: torch.Tensor, defense_config: dict, batch_index: int) -> torch.Tensor:
    """Vote over temporary Gaussian-perturbed model copies without mutating model."""
    count = defense_config["num_perturbed_models"]
    sigma = defense_config["sigma"]
    seed = defense_config["noise_seed"] + batch_index * count
    votes = []
    for copy_index in range(count):
        temporary_model = copy.deepcopy(model).to(images.device)
        temporary_model.eval()
        generator = torch.Generator(device=images.device).manual_seed(seed + copy_index)
        with torch.no_grad():
            for parameter in temporary_model.parameters():
                if parameter.is_floating_point():
                    noise = torch.randn(parameter.shape, device=parameter.device, dtype=parameter.dtype, generator=generator) * sigma
                    parameter.add_(noise)
            votes.append(temporary_model(images).argmax(dim=1))
    stacked = torch.stack(votes)
    # torch.mode deterministically returns the smallest label among equal maxima.
    return torch.mode(stacked, dim=0).values


def _defended_predictions(model: nn.Module, images: torch.Tensor, defense_config, batch_index: int) -> torch.Tensor:
    if defense_config and defense_config.get("name") == "crfl" and defense_config.get("smoothing_enabled", False):
        return _crfl_defended_predictions(model, images, defense_config, batch_index)
    return model(images).argmax(dim=1)


def _denormalize(
    images: torch.Tensor,
    mean_values=CIFAR10_MEAN,
    std_values=CIFAR10_STD,
) -> torch.Tensor:
    """Convert normalized RGB tensors into pixels in [0, 1]."""
    mean_values = CIFAR10_MEAN if mean_values is None else mean_values
    std_values = CIFAR10_STD if std_values is None else std_values
    mean = torch.tensor(mean_values, device=images.device, dtype=images.dtype)
    std = torch.tensor(std_values, device=images.device, dtype=images.dtype)
    return (images * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)).clamp(0.0, 1.0)


def compute_stealth_metrics(
    original_images: torch.Tensor,
    triggered_images: torch.Tensor,
    delta: torch.Tensor,
    mean_values=CIFAR10_MEAN,
    std_values=CIFAR10_STD,
) -> dict:
    """Return PSNR, mean SSIM, and L-infinity norm for a trigger batch."""
    if not isinstance(original_images, torch.Tensor):
        raise TypeError("original_images must be a torch.Tensor")
    if not isinstance(triggered_images, torch.Tensor):
        raise TypeError("triggered_images must be a torch.Tensor")
    if not isinstance(delta, torch.Tensor):
        raise TypeError("delta must be a torch.Tensor")
    if original_images.ndim != 4 or triggered_images.ndim != 4:
        raise ValueError("image tensors must have shape [N, C, H, W]")
    if original_images.shape != triggered_images.shape:
        raise ValueError("original_images and triggered_images must have the same shape")
    if original_images.shape[0] == 0:
        raise ValueError("image tensors must contain at least one sample")

    try:
        from skimage.metrics import structural_similarity
    except ImportError as error:
        raise ImportError(
            "compute_stealth_metrics requires scikit-image; install it with "
            "'pip install scikit-image'."
        ) from error

    originals = _denormalize(original_images.detach(), mean_values, std_values).cpu()
    triggered = _denormalize(triggered_images.detach(), mean_values, std_values).cpu()
    mean_squared_error = torch.mean((originals - triggered) ** 2).item()
    psnr = float("inf") if mean_squared_error == 0.0 else 10.0 * math.log10(1.0 / mean_squared_error)

    ssim_values = [
        structural_similarity(
            original.permute(1, 2, 0).numpy(),
            triggered_image.permute(1, 2, 0).numpy(),
            channel_axis=-1,
            data_range=1.0,
        )
        for original, triggered_image in zip(originals, triggered)
    ]
    return {
        "psnr": float(psnr),
        "ssim": float(sum(ssim_values) / len(ssim_values)),
        "linf": float((triggered - originals).abs().max().item()),
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    defense_config: dict | None = None,
) -> dict:
    """Evaluate a classification model and return loss and clean accuracy."""
    if len(dataloader) == 0:
        raise ValueError("dataloader must not be empty")

    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = _defended_predictions(model, images, defense_config, batch_index)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            correct_predictions += (predictions == labels).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("dataloader produced no samples")

    return {
        "test_loss": total_loss / total_samples,
        "clean_accuracy": 100.0 * correct_predictions / total_samples,
        "num_samples": total_samples,
        "crfl_smoothing_enabled": bool(defense_config and defense_config.get("name") == "crfl" and defense_config.get("smoothing_enabled", False)),
    }


def evaluate_backdoor_asr(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    target_label: int,
    patch_size: int = 3,
    pixel_value: float = 1.0,
    trigger_type: str = "patch",
    attack_config=None,
    round_index: int | None = None,
    defense_config: dict | None = None,
) -> dict:
    """Evaluate ASR on non-target samples using a patch or adversarial trigger."""
    if len(test_loader) == 0:
        raise ValueError("test_loader must not be empty")
    if type(target_label) is not int:
        raise TypeError("target_label must be an integer")
    if target_label < 0:
        raise ValueError("target_label must be non-negative")
    if trigger_type not in ("patch", "adversarial", "chameleon", "focused_flip"):
        raise ValueError("trigger_type must be 'patch', 'adversarial', or 'chameleon'")

    if trigger_type == "adversarial":
        # The shared perturbation is created only by malicious-client training.
        # Evaluation must not create or mutate it, otherwise pre-attack metrics
        # would no longer represent the absence of an adversarial trigger.
        from federated.malicious_client import MaliciousClient

        if MaliciousClient.shared_adversarial_delta is None:
            return {
                "attack_success_rate": None,
                "num_attack_samples": None,
                "num_successful_attacks": None,
            }

    model = model.to(device)
    model.eval()

    successful_attacks = 0
    total_attack_samples = 0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            non_target_mask = labels != target_label
            if not non_target_mask.any().item():
                continue

            non_target_images = images[non_target_mask]
            if getattr(attack_config, "method", None) == "dba":
                if total_attack_samples == 0:
                    print(f"[DBAEvalDebug] round={round_index} using_full_trigger=True "
                          f"num_parts={attack_config.dba_num_parts} target_label={target_label} "
                          "expected_num_attack_samples=9000", flush=True)
                triggered_images = apply_dba_trigger(
                    non_target_images, 0, attack_config.dba_num_parts,
                    patch_size=attack_config.patch_size, pixel_value=attack_config.pixel_value,
                    full_trigger=True, pattern=attack_config.dba_pattern,
                    subtrigger_size=attack_config.dba_subtrigger_size,
                    normalized_trigger=attack_config.dba_normalized_trigger,
                )
            elif trigger_type == "chameleon":
                triggered_images = apply_chameleon_type1_trigger(non_target_images)
            elif trigger_type == "focused_flip":
                from federated.focused_flip_client import FocusedFlipClient
                from attacks.focused_flip import apply_pattern
                pattern = FocusedFlipClient.evaluation_pattern()
                if pattern is None:
                    return {"attack_success_rate": None, "num_attack_samples": None, "num_successful_attacks": None}
                mask = torch.zeros_like(pattern); mask[:, 3:6, 23:26] = 1
                triggered_images = apply_pattern(non_target_images, pattern.to(non_target_images), mask.to(non_target_images))
            elif trigger_type == "patch":
                trigger_indices = torch.arange(
                    non_target_images.size(0),
                    dtype=torch.long,
                    device=device,
                )
                triggered_images = apply_patch_trigger(
                    non_target_images,
                    trigger_indices,
                    patch_size=getattr(attack_config, "patch_size", patch_size),
                    pixel_value=getattr(attack_config, "pixel_value", pixel_value),
                )
            else:
                delta = MaliciousClient.shared_adversarial_delta.to(
                    device=device,
                    dtype=non_target_images.dtype,
                )
                triggered_images = apply_trigger(
                    non_target_images,
                    trigger_type="adversarial",
                    perturbation=delta,
                    clamp_to_valid_rgb=(
                        attack_config.method == "a3fl"
                        and attack_config.a3fl_clip_to_valid_rgb
                    ),
                )
            predictions = _defended_predictions(model, triggered_images, defense_config, batch_index)

            successful_attacks += (predictions == target_label).sum().item()
            total_attack_samples += predictions.numel()

    if total_attack_samples == 0:
        raise ValueError("No non-target samples are available for ASR evaluation.")

    if getattr(attack_config, "method", None) == "neurotoxin_uap":
        print(
            f"[NeurotoxinEvalDebug] round={round_index} using_fixed_uap=True "
            f"target_label={target_label} num_attack_samples={total_attack_samples} "
            f"attack_active={attack_config.is_attack_round(round_index)}",
            flush=True,
        )

    return {
        "attack_success_rate": 100.0 * successful_attacks / total_attack_samples,
        "num_attack_samples": total_attack_samples,
        "num_successful_attacks": successful_attacks,
        "crfl_smoothing_enabled": bool(defense_config and defense_config.get("name") == "crfl" and defense_config.get("smoothing_enabled", False)),
    }


def evaluate_feature_margins(model, test_loader, device, attack_config, fixed_prototype_bank,
                             max_batches=10, compute_fixed=True, compute_dynamic=True):
    """Measure poisoned non-target feature target-over-source margins.

    Centroids are recomputed from clean evaluation data only; all computations
    are inference-only and bounded by ``max_batches``.
    """
    if not hasattr(model, "forward_features") or fixed_prototype_bank is None:
        return {}
    from federated.malicious_client import MaliciousClient
    if MaliciousClient.shared_adversarial_delta is None:
        return {}
    layer = fixed_prototype_bank["feature_layer"]
    target = attack_config.target_label
    model.eval()
    clean_sums, clean_counts, poisoned_features, source_labels = {}, {}, [], []
    num_eval_samples = num_source_samples = num_target_excluded = 0
    with torch.no_grad():
        for index, (images, labels) in enumerate(test_loader):
            if index >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            num_eval_samples += labels.numel()
            clean_features = model.forward_features(images, feature_layer=layer)
            for label in labels.unique().tolist():
                label = int(label); values = clean_features[labels == label]
                clean_sums[label] = clean_sums.get(label, torch.zeros_like(values[0])) + values.sum(0)
                clean_counts[label] = clean_counts.get(label, 0) + values.size(0)
            mask = labels != target
            num_target_excluded += int((~mask).sum().item())
            if mask.any():
                num_source_samples += int(mask.sum().item())
                triggered = apply_trigger(
                    images[mask],
                    "adversarial",
                    perturbation=MaliciousClient.shared_adversarial_delta.to(device),
                    clamp_to_valid_rgb=(
                        attack_config.method == "a3fl"
                        and attack_config.a3fl_clip_to_valid_rgb
                    ),
                )
                poisoned_features.append(model.forward_features(triggered, feature_layer=layer))
                source_labels.append(labels[mask])
    if not poisoned_features:
        return {}
    features, sources = torch.cat(poisoned_features), torch.cat(source_labels)
    features = torch.nn.functional.normalize(features, dim=1)
    result = {"num_eval_samples": num_eval_samples, "num_source_samples": num_source_samples,
              "num_target_excluded": num_target_excluded}
    def margin_from_bank(prototypes, class_ids, prefix):
        # Prototype banks are stored on CPU.  Move every tensor participating
        # in indexed margin computation to the evaluation device together.
        prototypes = torch.as_tensor(prototypes, device=device).detach()
        class_ids_tensor = torch.as_tensor(
            class_ids, device=prototypes.device, dtype=torch.long
        )
        class_id_values = [int(label) for label in class_ids_tensor.detach().cpu().tolist()]
        lookup = {label: idx for idx, label in enumerate(class_id_values)}
        if target not in lookup:
            return
        valid = torch.as_tensor(
            [int(x) in lookup for x in sources.detach().cpu().tolist()],
            device=features.device,
            dtype=torch.bool,
        )
        if not valid.any(): return
        valid_mask = None
        if prefix == "fixed":
            valid_mask = fixed_prototype_bank.get("valid_class_mask")
        if valid_mask is not None:
            keep = torch.as_tensor(
                valid_mask, device=prototypes.device, dtype=torch.bool
            )
            if keep.numel() != prototypes.size(0):
                raise ValueError("valid_class_mask must match the prototype bank length")
            original_indices = keep.nonzero(as_tuple=False).flatten()
            if not bool(keep[lookup[target]]):
                return
            class_ids_tensor = class_ids_tensor[keep]
            prototypes = prototypes[keep]
            lookup = {
                int(label): position
                for position, label in enumerate(class_ids_tensor.detach().cpu().tolist())
            }
        p = torch.nn.functional.normalize(prototypes, dim=1)
        values, labels = features[valid], sources[valid]
        target_sim = (values * p[lookup[target]]).sum(1)
        source_sim = torch.stack([(values[i] * p[lookup[int(label)]]).sum() for i, label in enumerate(labels)])
        result[f"{prefix}_target_sim"] = float(target_sim.mean().item())
        result[f"{prefix}_source_sim"] = float(source_sim.mean().item())
        result[f"{prefix}_margin"] = float((target_sim - source_sim).mean().item())
    if compute_fixed:
        margin_from_bank(fixed_prototype_bank["prototypes"], fixed_prototype_bank["class_ids"], "fixed")
    if compute_dynamic:
        ids = sorted(clean_sums)
        prototypes = torch.stack([clean_sums[label] / clean_counts[label] for label in ids])
        margin_from_bank(prototypes, ids, "dynamic")
    return result

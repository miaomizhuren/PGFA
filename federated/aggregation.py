"""Server-side aggregation and aggregation-only defense implementations."""

from typing import Any

import torch


def fedavg(
    client_states: list[dict[str, torch.Tensor]],
    client_num_samples: list[int],
    aggregation_weighting: str = "sample_weighted",
    bn_buffer_policy: str = "aggregate",
    global_state: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Aggregate client state dictionaries with explicit FedAvg buffer policies."""
    if not isinstance(client_states, list):
        raise TypeError("client_states must be a list")
    if not isinstance(client_num_samples, list):
        raise TypeError("client_num_samples must be a list")
    if not client_states:
        raise ValueError("client_states must not be empty")
    if not client_num_samples:
        raise ValueError("client_num_samples must not be empty")
    if len(client_states) != len(client_num_samples):
        raise ValueError("client_states and client_num_samples must have the same length")
    if aggregation_weighting not in {"sample_weighted", "equal_client"}:
        raise ValueError("aggregation_weighting must be 'sample_weighted' or 'equal_client'")
    if bn_buffer_policy not in {"aggregate", "keep_global"}:
        raise ValueError("bn_buffer_policy must be 'aggregate' or 'keep_global'")
    if bn_buffer_policy == "keep_global" and global_state is None:
        raise ValueError("global_state is required when bn_buffer_policy='keep_global'")

    for num_samples in client_num_samples:
        if type(num_samples) is not int:
            raise TypeError("each client sample count must be an integer")
        if num_samples <= 0:
            raise ValueError("each client sample count must be positive")

    for state in client_states:
        if not isinstance(state, dict):
            raise TypeError("each client state must be a dictionary")
        for key, value in state.items():
            if not isinstance(key, str):
                raise TypeError("state dictionary keys must be strings")
            if not isinstance(value, torch.Tensor):
                raise TypeError("state dictionary values must be tensors")

    reference_state = client_states[0]
    reference_keys = set(reference_state)
    for state in client_states[1:]:
        if set(state) != reference_keys:
            raise ValueError("all client state dictionaries must have identical keys")
        for key in reference_state:
            reference_tensor = reference_state[key]
            tensor = state[key]
            if tensor.shape != reference_tensor.shape:
                raise ValueError(f"tensor shape mismatch for key: {key}")
            if tensor.dtype != reference_tensor.dtype:
                raise ValueError(f"tensor dtype mismatch for key: {key}")
            if tensor.device != reference_tensor.device:
                raise ValueError(f"tensor device mismatch for key: {key}")

    if global_state is not None and set(global_state) != reference_keys:
        raise ValueError("global_state keys must match client state dictionaries")
    total_samples = sum(client_num_samples)
    weights = (
        [num_samples / total_samples for num_samples in client_num_samples]
        if aggregation_weighting == "sample_weighted"
        else [1.0 / len(client_states)] * len(client_states)
    )
    aggregated_state: dict[str, torch.Tensor] = {}
    for key, reference_tensor in reference_state.items():
        is_bn_buffer = key.endswith((".running_mean", ".running_var", ".num_batches_tracked"))
        if is_bn_buffer and bn_buffer_policy == "keep_global":
            aggregated_state[key] = global_state[key].clone()
            continue
        if not reference_tensor.is_floating_point():
            # Preserves the historical FedAvg behavior for integer buffers,
            # including BatchNorm's num_batches_tracked.
            aggregated_state[key] = reference_tensor.clone()
            continue

        aggregated_tensor = torch.zeros_like(reference_tensor)
        for state, weight in zip(client_states, weights):
            aggregated_tensor = aggregated_tensor + state[key] * weight
        aggregated_state[key] = aggregated_tensor

    return aggregated_state


def validate_defense_config(config: dict | None) -> dict[str, Any]:
    """Normalize modern ``name/params`` YAML and retain legacy compatibility."""
    config = config or {}
    params = dict(config.get("params", {}))
    for key in ("clip_norm", "trim_ratio", "noise_std", "eps", "f", "m", "clustering", "num_clusters", "clip_mode", "threshold", "server_lr", "tie_policy", "sigma", "num_perturbed_models", "smoothing_enabled", "vote_rule", "noise_seed", "strict_validity"):
        if key in config and key not in params:
            params[key] = config[key]
    name = str(config.get("name", config.get("method", "none"))).lower()
    if name == "fedavg":
        name = "none"
    if name == "krum":
        name = "multi_krum"
        params.setdefault("m", 1)
    enabled = bool(config.get("enabled", False))
    if not enabled:
        name = "none"
    allowed = {"none", "norm_clipping", "trimmed_mean", "multi_krum", "flame", "gaussian_noise", "rlr", "crfl"}
    if name not in allowed:
        raise ValueError(f"defense.name must be one of {sorted(allowed)}")
    start_round = config.get("start_round")
    end_round = config.get("end_round")
    if start_round is not None and (type(start_round) is not int or start_round <= 0):
        raise ValueError("defense.start_round must be a positive integer or omitted")
    if end_round is not None and (type(end_round) is not int or end_round <= 0):
        raise ValueError("defense.end_round must be a positive integer or omitted")
    if start_round is not None and end_round is not None and start_round > end_round:
        raise ValueError("defense.start_round must not exceed defense.end_round")
    result: dict[str, Any] = {"enabled": enabled, "name": name, "method": name, "params": params, "start_round": start_round, "end_round": end_round, "scope": str(config.get("scope", "server_aggregation"))}
    if name == "norm_clipping":
        result["clip_norm"] = float(params.get("clip_norm", 0.0))
        if result["clip_norm"] <= 0:
            raise ValueError("defense.params.clip_norm must be positive")
    elif name == "trimmed_mean":
        result["trim_ratio"] = float(params.get("trim_ratio", 0.0))
        if not 0 <= result["trim_ratio"] < .5:
            raise ValueError("defense.params.trim_ratio must be in [0, 0.5)")
    elif name == "multi_krum":
        result["f"] = int(params.get("f", 0))
        result["m"] = params.get("m")
        result["strict_validity"] = bool(params.get("strict_validity", False))
        if result["f"] < 0 or (result["m"] is not None and int(result["m"]) <= 0):
            raise ValueError("Multi-Krum f must be non-negative and m must be positive")
    elif name == "rlr":
        result.update({"threshold": int(params.get("threshold", 4)), "server_lr": float(params.get("server_lr", 1.0)), "tie_policy": str(params.get("tie_policy", "negative"))})
        if result["threshold"] < 0 or result["server_lr"] < 0:
            raise ValueError("RLR threshold and server_lr must be non-negative")
        if result["tie_policy"] not in {"negative", "positive"}:
            raise ValueError("RLR tie_policy must be 'negative' or 'positive'")
    elif name == "crfl":
        result.update({"clip_norm": float(params.get("clip_norm", 0.0)), "sigma": float(params.get("sigma", 0.001)), "num_perturbed_models": int(params.get("num_perturbed_models", 5)), "smoothing_enabled": bool(params.get("smoothing_enabled", True)), "vote_rule": str(params.get("vote_rule", "majority")), "noise_seed": int(params.get("noise_seed", 2026))})
        if result["clip_norm"] <= 0 or result["sigma"] < 0 or result["num_perturbed_models"] <= 0:
            raise ValueError("CRFL clip_norm must be positive, sigma non-negative, and num_perturbed_models positive")
        if result["vote_rule"] != "majority":
            raise ValueError("CRFL vote_rule must be 'majority'")
    elif name == "flame":
        result.update({"noise_std": float(params.get("noise_std", 0.001)), "eps": float(params.get("eps", 1e-12)), "clustering": str(params.get("clustering", "cosine_median")), "clip_mode": str(params.get("clip_mode", "median_norm"))})
        if result["noise_std"] < 0 or result["eps"] <= 0:
            raise ValueError("FLAME noise_std must be non-negative and eps must be positive")
    elif name == "gaussian_noise":
        result["noise_std"] = float(params.get("noise_std", 0.0))
    return result


def defense_active_this_round(defense: dict[str, Any], round_index: int | None) -> bool:
    """Return whether a normalized defense is active in the current round."""
    if not defense["enabled"]:
        return False
    if round_index is None:
        return True
    start_round, end_round = defense["start_round"], defense["end_round"]
    return (start_round is None or round_index >= start_round) and (end_round is None or round_index <= end_round)


def _client_updates(client_states, global_state, names):
    return [{name: state[name] - global_state[name] for name in names} for state in client_states]


def _flatten(update, names):
    return torch.cat([update[name].detach().float().reshape(-1) for name in names])


def aggregate_with_defense(client_states, client_num_samples, global_state, trainable_names, defense, client_ids=None, return_diagnostics=False, aggregation_weighting="sample_weighted", bn_buffer_policy="aggregate", round_index=None):
    """Aggregate states; defenses only alter server-side parameter aggregation."""
    defense = validate_defense_config(defense)
    result = fedavg(
        client_states, client_num_samples,
        aggregation_weighting=aggregation_weighting,
        bn_buffer_policy=bn_buffer_policy,
        global_state=global_state,
    )
    names = [name for name in trainable_names if global_state[name].is_floating_point()]
    client_ids = list(range(len(client_states))) if client_ids is None else list(client_ids)
    if len(client_ids) != len(client_states):
        raise ValueError("client_ids must align with client_states")
    active = defense_active_this_round(defense, round_index)
    diagnostics: dict[str, Any] = {"defense_enabled": defense["enabled"], "defense_name": defense["name"], "defense_active_this_round": active, "defense_start_round": defense["start_round"], "defense_end_round": defense["end_round"], "defense_params": defense["params"], "selected_client_ids": client_ids, "aggregation_weighting": aggregation_weighting, "bn_buffer_policy": bn_buffer_policy}
    if defense["name"] == "none" or not active:
        return (result, diagnostics) if return_diagnostics else result
    updates = _client_updates(client_states, global_state, names)
    vectors = [_flatten(update, names) for update in updates]
    norms = [float(torch.linalg.vector_norm(vector)) for vector in vectors]

    weights = ([count / sum(client_num_samples) for count in client_num_samples]
               if aggregation_weighting == "sample_weighted"
               else [1.0 / len(client_states)] * len(client_states))
    if defense["name"] == "norm_clipping":
        clip_norm = defense["clip_norm"]
        scales = [min(1.0, clip_norm / (norm + 1e-12)) for norm in norms]
        for name in names:
            result[name] = global_state[name] + sum(update[name] * scale * weight for update, scale, weight in zip(updates, scales, weights))
        diagnostics.update({"clip_norm": clip_norm, "num_clipped": sum(scale < 1.0 for scale in scales), "clipped_clients": sum(scale < 1.0 for scale in scales), "max_update_norm": max(norms), "median_update_norm": float(torch.tensor(norms).median()), "max_norm": max(norms), "median_norm": float(torch.tensor(norms).median())})
    elif defense["name"] == "rlr":
        reversed_elements, element_count, agreements = 0, 0, []
        for name in names:
            stacked = torch.stack([update[name] for update in updates])
            vote = stacked.sign().sum(dim=0)
            agreement = vote.abs()
            direction = torch.where(agreement >= defense["threshold"], torch.ones_like(agreement), -torch.ones_like(agreement))
            if defense["tie_policy"] == "positive":
                direction = torch.where(agreement == 0, torch.ones_like(direction), direction)
            average = sum(update[name] * weight for update, weight in zip(updates, weights))
            result[name] = global_state[name] + defense["server_lr"] * direction * average
            reversed_elements += int((direction < 0).sum().item())
            element_count += direction.numel()
            agreements.append(agreement.detach().float().reshape(-1))
        agreement_values = torch.cat(agreements)
        diagnostics.update({"threshold": defense["threshold"], "server_lr": defense["server_lr"], "tie_policy": defense["tie_policy"], "fraction_reversed": reversed_elements / element_count, "mean_agreement": float(agreement_values.mean()), "min_agreement": float(agreement_values.min()), "max_agreement": float(agreement_values.max())})
    elif defense["name"] == "crfl":
        clip_norm = defense["clip_norm"]
        scales = [min(1.0, clip_norm / (norm + 1e-12)) for norm in norms]
        noise_squares, noise_count = 0.0, 0
        generator = torch.Generator(device=global_state[names[0]].device).manual_seed(defense["noise_seed"] + int(round_index or 0))
        for name in names:
            aggregate = sum(update[name] * scale * weight for update, scale, weight in zip(updates, scales, weights))
            noise = torch.randn(aggregate.shape, device=aggregate.device, dtype=aggregate.dtype, generator=generator) * defense["sigma"]
            result[name] = global_state[name] + aggregate + noise
            noise_squares += float(noise.float().pow(2).sum())
            noise_count += noise.numel()
        diagnostics.update({"clip_norm": clip_norm, "sigma": defense["sigma"], "num_clipped": sum(scale < 1.0 for scale in scales), "max_update_norm": max(norms), "median_update_norm": float(torch.tensor(norms).median()), "noise_norm": noise_squares ** .5, "mean_noise_std": (noise_squares / noise_count) ** .5})
    elif defense["name"] == "trimmed_mean":
        trim_k = int(len(client_states) * defense["trim_ratio"])
        if 2 * trim_k >= len(client_states):
            raise ValueError("trim_ratio removes all client updates")
        for name in names:
            stacked = torch.stack([update[name] for update in updates])
            result[name] = global_state[name] + stacked.sort(dim=0).values[trim_k:len(client_states) - trim_k].mean(dim=0)
        diagnostics.update({"trim_ratio": defense["trim_ratio"], "trim_k": trim_k, "kept": len(client_states) - 2 * trim_k})
    elif defense["name"] == "multi_krum":
        n, f = len(client_states), defense["f"]
        if n <= 2 * f + 2:
            if defense["strict_validity"]:
                raise ValueError(f"Multi-Krum invalid: n={n} must satisfy n > 2f + 2 for f={f}")
            diagnostics.update({"warning": "n <= 2f + 2; fell back to FedAvg", "n": n, "f": f, "strict_validity": False})
        else:
            nearest_count = n - f - 2
            distances = torch.cdist(torch.stack(vectors), torch.stack(vectors), p=2).pow(2)
            scores = []
            for index in range(n):
                others = torch.cat((distances[index, :index], distances[index, index + 1:]))
                scores.append(float(torch.topk(others, nearest_count, largest=False).values.sum()))
            m = min(int(defense["m"] if defense["m"] is not None else nearest_count), n)
            selected = sorted(range(n), key=lambda index: (scores[index], index))[:m]
            for name in names:
                result[name] = global_state[name] + sum(updates[index][name] for index in selected) / len(selected)
            diagnostics.update({"n": n, "f": f, "m": m, "nearest_count": nearest_count, "strict_validity": defense["strict_validity"], "selected_krum_indices": selected, "selected_client_ids": [client_ids[i] for i in selected], "rejected_client_ids": [client_ids[i] for i in range(n) if i not in selected], "scores": scores, "krum_scores_summary": {"min": min(scores), "max": max(scores), "mean": sum(scores) / len(scores)}})
    elif defense["name"] == "flame":
        matrix = torch.stack(vectors)
        mean_vector = matrix.mean(dim=0)
        similarity = torch.nn.functional.cosine_similarity(matrix, mean_vector.unsqueeze(0), dim=1, eps=defense["eps"])
        median = float(torch.median(similarity))
        cluster_zero = [index for index, value in enumerate(similarity.tolist()) if value >= median]
        cluster_one = [index for index, value in enumerate(similarity.tolist()) if value < median]
        if not cluster_zero or not cluster_one:
            kept, kept_cluster, cluster_sizes = list(range(len(client_states))), 0, [len(client_states), 0]
        elif len(cluster_zero) > len(cluster_one):
            kept, kept_cluster, cluster_sizes = cluster_zero, 0, [len(cluster_zero), len(cluster_one)]
        elif len(cluster_one) > len(cluster_zero):
            kept, kept_cluster, cluster_sizes = cluster_one, 1, [len(cluster_zero), len(cluster_one)]
        else:
            mean_norms = [sum(norms[i] for i in cluster) / len(cluster) for cluster in (cluster_zero, cluster_one)]
            kept_cluster = 0 if mean_norms[0] <= mean_norms[1] else 1
            kept, cluster_sizes = (cluster_zero, cluster_one)[kept_cluster], [len(cluster_zero), len(cluster_one)]
        clip_norm = float(torch.tensor([norms[index] for index in kept]).median())
        scales = {index: min(1.0, clip_norm / (norms[index] + defense["eps"])) for index in kept}
        for name in names:
            aggregate = sum(updates[index][name] * scales[index] for index in kept) / len(kept)
            result[name] = global_state[name] + aggregate + torch.randn_like(aggregate) * defense["noise_std"]
        diagnostics.update({"cluster_sizes": cluster_sizes, "kept_cluster": kept_cluster, "kept_client_ids": [client_ids[i] for i in kept], "removed_client_ids": [client_ids[i] for i in range(len(client_states)) if i not in kept], "clip_norm": clip_norm, "noise_std": defense["noise_std"]})
    elif defense["name"] == "gaussian_noise":
        for name in names:
            result[name] = result[name] + torch.randn_like(result[name]) * defense["noise_std"]
    return (result, diagnostics) if return_diagnostics else result

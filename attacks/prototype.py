"""CPU-safe class prototype statistics and aggregation helpers."""

import torch
import torch.nn.functional as functional


def enforce_strict_prototype_coverage(bank, *, num_classes, dataset, prototype_source_client, center_batches):
    """Fail before attack training when a strict single-client bank is incomplete."""
    valid_mask = bank.get("valid_class_mask") if bank else None
    valid_classes = int(valid_mask.sum().item()) if isinstance(valid_mask, torch.Tensor) else len(bank.get("class_ids", [])) if bank else 0
    missing = bank.get("missing_classes", [i for i in range(num_classes) if not valid_mask[i]]) if bank else list(range(num_classes))
    if valid_classes < num_classes:
        raise RuntimeError(
            "strict_prototype_coverage failed: "
            f"dataset={dataset}, num_classes={num_classes}, valid_classes={valid_classes}, "
            f"missing_classes={missing}, prototype_source_client={prototype_source_client}, "
            f"center_batches={center_batches}. Use center_batches=-1 (full local loader) "
            "or choose another prototype_source_client."
        )
    return valid_classes, missing


def collect_class_statistics(model, dataloader, device, feature_layer, max_batches):
    """Return detached CPU ``feature_sums`` and integer ``class_counts``."""
    if not hasattr(model, "forward_features"):
        raise ValueError("prototype collection requires model.forward_features")
    sums, counts = {}, {}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch_index, (images, labels) in enumerate(dataloader):
                if max_batches != -1 and batch_index >= max_batches:
                    break
                features = model.forward_features(images.to(device), feature_layer=feature_layer)
                for label in labels.unique().tolist():
                    label = int(label)
                    values = features[labels.to(device) == label].detach().sum(dim=0).cpu()
                    sums[label] = sums.get(label, torch.zeros_like(values)) + values
                    counts[label] = counts.get(label, 0) + int((labels == label).sum().item())
    finally:
        model.train(was_training)
    return {"feature_sums": sums, "class_counts": counts}


def aggregate_class_statistics(statistics, target_label, min_samples_per_class,
                               num_classes, missing_class_strategy="skip"):
    """Count-weight aggregate clean client statistics into an L2-normalized bank."""
    if missing_class_strategy not in {"skip", "zero"}:
        raise ValueError("missing_class_strategy must be 'skip' or 'zero'")
    global_sums, counts, per_client_class_counts = {}, {}, []
    for item in statistics:
        per_client_class_counts.append({int(k): int(v) for k, v in item["class_counts"].items()})
        for label, value in item["feature_sums"].items():
            global_sums[label] = global_sums.get(label, torch.zeros_like(value)) + value.detach().cpu()
        for label, count in item["class_counts"].items():
            counts[label] = counts.get(label, 0) + int(count)
    valid_ids = [label for label in range(num_classes) if counts.get(label, 0) >= min_samples_per_class]
    missing_classes = [label for label in range(num_classes) if label not in valid_ids]
    if not global_sums:
        return None, {"class_counts": counts, "missing_classes": missing_classes, "classes_covered": valid_ids}
    feature_shape = next(iter(global_sums.values())).shape
    # Both policies retain a full class-indexed bank and use the mask to make
    # absent classes unavailable to every loss/metric.  ``zero`` explicitly
    # requests zero placeholders; ``skip`` treats those placeholders as
    # non-prototypes and never exposes them through the valid mask.
    class_ids = list(range(num_classes))
    prototypes = [functional.normalize((global_sums[label] / counts[label]).unsqueeze(0), dim=1).squeeze(0)
                  if label in valid_ids else torch.zeros(feature_shape)
                  for label in class_ids]
    valid_mask = torch.tensor([label in valid_ids for label in class_ids], dtype=torch.bool)
    if not prototypes:
        return None, {"class_counts": counts, "missing_classes": missing_classes, "classes_covered": valid_ids}
    return {"prototypes": torch.stack(prototypes).detach().cpu(), "class_ids": class_ids,
            "valid_class_mask": valid_mask.detach().cpu(),
            "target_proto_index": class_ids.index(target_label) if target_label in class_ids else None,
            "class_counts": dict(counts), "missing_classes": missing_classes,
            "per_client_class_counts": per_client_class_counts}, {
        "class_counts": dict(counts), "per_client_class_counts": per_client_class_counts,
        "missing_classes": missing_classes, "classes_covered": valid_ids,
    }

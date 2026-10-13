"""Deterministic Distributed Backdoor Attack layouts and trigger application."""

import torch

from attacks.triggers import normalize_pixel_value


def _block_origins(height, width, size, num_components, pattern):
    if pattern == "corner_blocks":
        candidates = [(height - size, 0), (height - size, width - size), (0, width - size), (0, 0)]
    elif pattern == "diagonal":
        candidates = [(round(index * (height - size) / max(num_components - 1, 1)),
                       round(index * (width - size) / max(num_components - 1, 1)))
                      for index in range(num_components)]
    elif pattern == "grid":
        columns = max(1, int(num_components ** 0.5 + 0.999))
        rows = (num_components + columns - 1) // columns
        row_slots = [round(index * (height - size) / max(rows - 1, 1)) for index in range(rows)]
        col_slots = [round(index * (width - size) / max(columns - 1, 1)) for index in range(columns)]
        candidates = [(row_slots[index // columns], col_slots[index % columns]) for index in range(num_components)]
    else:
        raise ValueError("unsupported DBA pattern")
    return candidates[:num_components]


def dba_component_masks(height, width, patch_size, num_components, device=None,
                        pattern="full_patch_split", subtrigger_size=None):
    """Build one mask per participant; full_patch_split retains legacy DBA."""
    subtrigger_size = patch_size if subtrigger_size is None else subtrigger_size
    if min(height, width, patch_size, subtrigger_size, num_components) <= 0:
        raise ValueError("invalid DBA geometry")
    masks = torch.zeros((num_components, 1, height, width), dtype=torch.bool, device=device)
    if pattern == "full_patch_split":
        if patch_size > min(height, width) or num_components > patch_size * patch_size:
            raise ValueError("invalid DBA full_patch_split geometry")
        for index, (row, column) in enumerate(
                ([(height - patch_size + row, width - patch_size + column)
                 for row in range(patch_size) for column in range(patch_size)])):
            masks[index % num_components, 0, row, column] = True
        return masks
    if pattern not in {"corner_blocks", "grid", "diagonal"}:
        raise ValueError("unsupported DBA pattern")
    if subtrigger_size > min(height, width):
        raise ValueError("subtrigger_size cannot exceed image dimensions")
    for component, (top, left) in enumerate(_block_origins(
            height, width, subtrigger_size, num_components, pattern)):
        masks[component, 0, top:top + subtrigger_size, left:left + subtrigger_size] = True
    if int(masks.sum()) != int(masks.any(dim=0).sum()):
        raise ValueError("DBA sub-trigger layout overlaps")
    return masks


def apply_dba_trigger(images, component_index, num_components, patch_size=3,
                      pixel_value=1.0, full_trigger=False, pattern="full_patch_split",
                      subtrigger_size=None, normalized_trigger=True):
    """Apply one participant sub-trigger or the union used for DBA evaluation."""
    masks = dba_component_masks(images.shape[-2], images.shape[-1], patch_size, num_components,
                                images.device, pattern, subtrigger_size)
    if not full_trigger and not 0 <= component_index < num_components:
        raise ValueError("invalid DBA component index")
    selected = masks.any(dim=0) if full_trigger else masks[component_index]
    value = (normalize_pixel_value(pixel_value, images.device, images.dtype)
             if normalized_trigger else torch.as_tensor(pixel_value, device=images.device, dtype=images.dtype))
    return torch.where(selected.unsqueeze(0), value, images)

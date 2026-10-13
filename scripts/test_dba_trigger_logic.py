"""Data-free DBA checks for legacy and enhanced component layouts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from attacks.dba import apply_dba_trigger, dba_component_masks


def assert_layout(size, pattern):
    masks = dba_component_masks(32, 32, 3, 3, pattern=pattern, subtrigger_size=size)
    assert masks.shape == (3, 1, 32, 32)
    assert int(masks.sum()) == int(masks.any(dim=0).sum()), "sub-triggers overlap"
    assert all(int(mask.sum()) == size * size for mask in masks)
    coordinates = [torch.nonzero(mask, as_tuple=False) for mask in masks]
    assert len({tuple(points.flatten().tolist()) for points in coordinates}) == 3
    assert all(int(points[:, -2].min()) >= 0 and int(points[:, -2].max()) < 32 and
               int(points[:, -1].min()) >= 0 and int(points[:, -1].max()) < 32
               for points in coordinates)

    images = torch.zeros(4, 3, 32, 32)
    parts = [apply_dba_trigger(images, index, 3, patch_size=3, pattern=pattern,
                               subtrigger_size=size, normalized_trigger=True)
             for index in range(3)]
    full = apply_dba_trigger(images, 0, 3, patch_size=3, full_trigger=True,
                             pattern=pattern, subtrigger_size=size, normalized_trigger=True)
    active = (full != images).any(dim=1)[0]
    assert int(active.sum()) == 3 * size * size
    assert torch.isfinite(full).all() and float(full.abs().max()) < 10.0

    # The malicious-client writeback must leave the clean subset byte-identical.
    poison_indices = torch.tensor([1, 3])
    only_poisoned = images.clone()
    only_poisoned[poison_indices] = apply_dba_trigger(
        images[poison_indices], 0, 3, patch_size=3, pattern=pattern,
        subtrigger_size=size, normalized_trigger=True,
    )
    assert torch.equal(only_poisoned[0], images[0]) and torch.equal(only_poisoned[2], images[2])
    assert not torch.equal(only_poisoned[1], images[1]) and not torch.equal(only_poisoned[3], images[3])
    return [int(mask.sum()) for mask in masks], int(active.sum())


legacy = dba_component_masks(32, 32, 3, 3)
assert int(legacy.sum()) == 9 and int(legacy.any(dim=0).sum()) == 9
summaries = {f"corner_blocks_size{size}": assert_layout(size, "corner_blocks") for size in (5, 6)}

out = Path("outputs/debug_dba_trigger")
out.mkdir(parents=True, exist_ok=True)
(out / "dba_trigger_layout.txt").write_text(
    "legacy_full_patch_split_pixels=9\n" +
    "\n".join(f"{name}: part_pixels={parts}, full_pixels={full}" for name, (parts, full) in summaries.items()) + "\n",
    encoding="utf-8",
)
print("DBA_TRIGGER_LOGIC_PASS")

"""Data-free semantic check for the Neurotoxin gradient mask."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from attacks.neurotoxin import build_neurotoxin_mask, apply_neurotoxin_mask

model = nn.Sequential(nn.Linear(4, 2, bias=False), nn.BatchNorm1d(2))
previous = {name: value.detach().clone() for name, value in model.state_dict().items()}
current = {name: value.detach().clone() for name, value in model.state_dict().items()}
current['0.weight'][0, 0] += 10
current['0.weight'][0, 1] += 5
masks, info = build_neurotoxin_mask(previous, current, model.named_parameters(), 0.25)
assert masks['0.weight'][0, 0] == 0 and masks['0.weight'][0, 1] == 0
for _, parameter in model.named_parameters():
    parameter.grad = torch.ones_like(parameter)
apply_neurotoxin_mask(model.named_parameters(), masks)
assert model[0].weight.grad[0, 0] == 0 and model[0].weight.grad[1, 1] == 1
assert all('running_' not in name for name in masks)
assert info['selected_direction'] == 'high_gradient'
assert info['kept_coordinate_count'] + info['masked_coordinate_count'] == info['total_trainable_params']

out = Path('outputs/debug_neurotoxin_mask')
out.mkdir(parents=True, exist_ok=True)
lines = [
    f"total_trainable_params={info['total_trainable_params']}",
    f"kept_params={info['kept_coordinate_count']}",
    f"masked_params={info['masked_coordinate_count']}",
    f"keep_ratio={info['kept_parameter_fraction']:.6f}",
    f"masked_ratio={info['masked_parameter_fraction']:.6f}",
    "selected_direction=high_gradient",
    "whether_bn_running_stats_included=False",
    "whether_gradients_zeroed_correctly=True",
]
for name, values in info['per_layer'].items():
    lines.append(f"per_layer_keep_ratio[{name}]={values['keep_ratio']:.6f}")
    lines.append(f"per_layer_masked_ratio[{name}]={values['masked_ratio']:.6f}")
(out / 'neurotoxin_mask_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('NEUROTOXIN_MASK_LOGIC_PASS')

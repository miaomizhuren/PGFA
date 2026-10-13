"""Focused-Flip (F3BA) primitives adapted from the AAAI-23 reference code.

Only Conv2d parameters named ``conv*`` and the terminal ``fc.weight`` receive
movement masks.  Residual downsample shortcuts are intentionally excluded:
the reference implementation's name-based rule does not classify their
``downsample.0.weight`` keys as convolution targets and supplies no activation
mapping for them.
"""
from __future__ import annotations

import copy
import time
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from data.dataset import CIFAR10_MEAN, CIFAR10_STD


def normalized_bounds(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(CIFAR10_MEAN, device=device, dtype=dtype).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device, dtype=dtype).view(3, 1, 1)
    return (0 - mean) / std, (1 - mean) / std


def make_pattern(device: torch.device, dtype: torch.dtype, seed: int = 111) -> tuple[torch.Tensor, torch.Tensor]:
    """Official CIFAR PatternSynthesizer geometry: random quantized 3x3 at (3,23)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # PatternSynthesizer uses one 3x3 tensor; assignment broadcasts it to RGB.
    raw = (torch.rand((3, 3), generator=generator) * 255).floor() / 255
    pattern = torch.zeros((3, 32, 32), device=device, dtype=dtype)
    mask = torch.zeros_like(pattern)
    pattern[:, 3:6, 23:26] = (raw.to(device=device, dtype=dtype) - torch.tensor(CIFAR10_MEAN, device=device, dtype=dtype).view(3,1,1)) / torch.tensor(CIFAR10_STD, device=device, dtype=dtype).view(3,1,1)
    mask[:, 3:6, 23:26] = 1
    return pattern, mask


def apply_pattern(images: torch.Tensor, pattern: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (1 - mask.to(images)) * images + mask.to(images) * pattern.to(images)


def _flip_filter_as_trigger(kernel: torch.Tensor, signal: torch.Tensor | None, pattern: torch.Tensor, flip_factor: float) -> torch.Tensor:
    source = pattern[:, 3:6, 23:26] if signal is None else signal
    resized = F.interpolate(source.unsqueeze(0), size=kernel.shape[-2:], mode="nearest")[0]
    pmin, pmax = resized.min(), resized.max()
    if torch.equal(pmin, pmax):
        return kernel.clone()
    scaled = (resized - pmin) / (pmax - pmin) * (kernel.max() - kernel.min()) + kernel.min()
    out = torch.sign(scaled) * kernel.abs()
    out[torch.sign(scaled) != torch.sign(kernel)] *= flip_factor
    return out


class FocusedFlip:
    """Stateful official-style movement selection, trigger optimisation and flipping."""
    def __init__(self, conv_rate: float = .02, fc_rate: float = .001, flip_factor: float = 1., trigger_step: float = .1, trigger_epochs: int = 2) -> None:
        self.conv_rate, self.fc_rate, self.flip_factor = conv_rate, fc_rate, flip_factor
        self.trigger_step, self.trigger_epochs = trigger_step, trigger_epochs
        self.previous_global: dict[str, torch.Tensor] | None = None
        self.pattern: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None
        self.last_diagnostics: dict = {}

    @staticmethod
    def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
        return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}

    def initialize_or_select(self, global_model: nn.Module) -> OrderedDict[str, torch.Tensor] | None:
        current = self._snapshot(global_model)
        if self.previous_global is None:
            self.previous_global = current
            self.last_diagnostics = {"history_initialized": True, "candidate_layers": {}, "reason": "official first-handcraft initialization; no R2999 model fabricated"}
            return None
        masks: OrderedDict[str, torch.Tensor] = OrderedDict(); detail = {}
        for name, now_cpu in current.items():
            if "conv" in name and name.endswith("weight"):
                rate = self.conv_rate
            elif name == "fc.weight":
                rate = self.fc_rate
            else:
                continue
            old_cpu = self.previous_global[name]
            score = (now_cpu - old_cpu) * now_cpu
            flat = score.flatten(); kth = int(flat.numel() * rate)
            threshold = torch.sort(flat, descending=False).values[kth]
            mask = (score < threshold).to(dtype=now_cpu.dtype)
            masks[name] = mask
            detail[name] = {"parameter_count": int(flat.numel()), "candidate_count": int(mask.sum().item()), "candidate_ratio": float(mask.float().mean().item()), "score_min": float(score.min().item()), "score_max": float(score.max().item())}
        self.previous_global = current
        self.last_diagnostics = {"history_initialized": False, "candidate_layers": detail}
        return masks

    @staticmethod
    def _activation_module(model: nn.Module, name: str) -> nn.Module | None:
        modules = dict(model.named_modules())
        if name == "conv1.weight": return modules.get("relu")
        if name.endswith(".conv1.weight"): return modules.get(name[:-len(".conv1.weight")] + ".relu")
        if name.endswith(".conv2.weight"): return modules.get(name[:-len(".conv2.weight")])
        return None

    @staticmethod
    def _activation_difference(
        model: nn.Module, module: nn.Module, loader, device: torch.device,
        pattern: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Average triggered-minus-clean activation over the complete handcraft loader."""
        values: list[torch.Tensor] = []
        handle = module.register_forward_hook(lambda _m, _i, output: values.append(output.detach()))
        total: torch.Tensor | None = None
        batches = 0
        try:
            with torch.no_grad():
                for images, _labels in loader:
                    images = images.to(device)
                    values.clear()
                    model(apply_pattern(images, pattern, mask))
                    triggered = values[-1].mean(0)
                    values.clear()
                    model(images)
                    clean = values[-1].mean(0)
                    difference = triggered - clean
                    total = difference if total is None else total + difference
                    batches += 1
        finally:
            handle.remove()
        if total is None:
            raise ValueError("F3BA handcraft loader must contain at least one batch")
        return total / batches

    def _set_first_layer(self, model: nn.Module, masks: OrderedDict[str, torch.Tensor]) -> int:
        if "conv1.weight" not in masks: return 0
        p = dict(model.named_parameters())["conv1.weight"]
        candidate = masks["conv1.weight"].to(p)
        crafted = torch.stack([_flip_filter_as_trigger(p[i].detach(), None, self.pattern, self.flip_factor) for i in range(p.shape[0])])
        before = p.detach().clone(); p.data.copy_(candidate * crafted + (1-candidate)*p)
        return int((torch.sign(before) != torch.sign(p.detach())).sum().item())

    def optimize_trigger(
        self, model: nn.Module, masks: OrderedDict[str, torch.Tensor], loader,
        device: torch.device, official_bn_mode: bool = False,
    ) -> dict:
        if self.pattern is None or self.mask is None: self.pattern, self.mask = make_pattern(device, next(model.parameters()).dtype)
        pattern = self.pattern.detach().clone().requires_grad_(True); original = copy.deepcopy(model.state_dict()); base_pattern = pattern.detach().clone(); losses=[]; started=time.perf_counter(); steps=0
        low, high = normalized_bounds(device, pattern.dtype)
        try:
            for _ in range(self.trigger_epochs):
                for images, _labels in loader:
                    images=images.to(device); self.pattern=pattern.detach(); self._set_first_layer(model,masks)
                    # Official trigger_loss enters train mode for each trigger
                    # batch and does not restore eval before injection.
                    if official_bn_mode:
                        model.train()
                    triggered=apply_pattern(images, pattern, self.mask)
                    clean = model.relu(model.bn1(model.conv1(images))).mean((0,1))
                    poisoned = model.relu(model.bn1(model.conv1(triggered))).mean((0,1))
                    loss=((poisoned-clean)**2).sum(); grad,=torch.autograd.grad(loss,pattern)
                    with torch.no_grad():
                        pattern.add_(self.trigger_step*grad)
                        pattern.copy_(torch.maximum(torch.minimum(pattern, high), low))
                        pattern.copy_(self.mask * pattern + (1 - self.mask) * base_pattern)
                    losses.append(float(loss.item())); steps+=1; model.zero_grad(set_to_none=True)
        finally: model.load_state_dict(original)
        self.pattern=pattern.detach(); return {"trigger_optimization_steps":steps,"trigger_optimization_time_sec":time.perf_counter()-started,"trigger_loss_mean":sum(losses)/len(losses) if losses else None}

    def inject(self, model: nn.Module, masks: OrderedDict[str, torch.Tensor], loader, device: torch.device) -> dict:
        flips={}; signal=None
        for name, mask in masks.items():
            if name == "fc.weight": continue  # official code stops at the terminal n_labels classifier
            module=self._activation_module(model,name)
            if module is None: continue
            p=dict(model.named_parameters())[name]; before=p.detach().clone(); crafted=torch.stack([_flip_filter_as_trigger(p[i].detach(),signal,self.pattern,self.flip_factor) for i in range(p.shape[0])]); m=mask.to(p); p.data.copy_(m*crafted+(1-m)*p)
            flips[name]=int((torch.sign(before)!=torch.sign(p.detach())).sum().item())
            signal = self._activation_difference(
                model, module, loader, device, self.pattern, self.mask,
            )
        self.last_diagnostics["sign_flips"]=flips; self.last_diagnostics["uncovered_layers"]=["downsample.* (no official layer2module mapping)","fc.weight terminal classifier (official inject_handcrafted_neurons stops at n_labels)"]
        return self.last_diagnostics

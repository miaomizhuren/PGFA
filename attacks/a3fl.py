"""A3FL-style adaptive universal adversarial trigger optimization.

The temporary adversarial model is deliberately isolated from the supplied
global model.  It simulates trigger unlearning while the universal additive
delta is optimized to remain targeted on both models.
"""

from __future__ import annotations

import copy
import numbers

import torch
import torch.nn.functional as functional

from attacks.triggers import clamp_normalized_cifar10_rgb


class A3FLTriggerOptimizer:
    """Build or refresh a targeted A3FL-style universal additive perturbation."""

    def __init__(
        self,
        *,
        epsilon: float,
        steps: int,
        alpha: float,
        lambda_adv: float = 1.0,
        lambda_mode: str = "fixed",
        lambda0: float | None = None,
        lambda_min: float = 0.0,
        lambda_max: float | None = None,
        adv_model_steps: int = 1,
        adv_model_lr: float = 0.01,
        adv_model_momentum: float = 0.0,
        adv_model_weight_decay: float = 0.0,
        outer_steps: int | None = None,
        k_trigger: int = 1,
        adv_updates_per_outer: int | None = None,
        freeze_adversarial_during_delta: bool = False,
        source: str = "non_target",
        clip_to_valid_rgb: bool = False,
    ) -> None:
        for name, value in (("epsilon", epsilon), ("alpha", alpha), ("lambda_adv", lambda_adv), ("adv_model_lr", adv_model_lr), ("adv_model_momentum", adv_model_momentum), ("adv_model_weight_decay", adv_model_weight_decay), ("lambda_min", lambda_min)):
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or value < 0:
                raise ValueError(f"{name} must be a non-negative real number")
        for name, value in (("steps", steps), ("adv_model_steps", adv_model_steps), ("k_trigger", k_trigger)):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if outer_steps is not None and (isinstance(outer_steps, bool) or not isinstance(outer_steps, numbers.Integral) or outer_steps <= 0):
            raise ValueError("outer_steps must be a positive integer or None")
        if adv_updates_per_outer is not None and (isinstance(adv_updates_per_outer, bool) or not isinstance(adv_updates_per_outer, numbers.Integral) or adv_updates_per_outer <= 0):
            raise ValueError("adv_updates_per_outer must be a positive integer or None")
        if lambda_mode not in {"fixed", "dynamic_similarity"}:
            raise ValueError("lambda_mode must be fixed or dynamic_similarity")
        if lambda0 is not None and (isinstance(lambda0, bool) or not isinstance(lambda0, numbers.Real) or lambda0 < 0):
            raise ValueError("lambda0 must be a non-negative real or None")
        if lambda_max is not None and (isinstance(lambda_max, bool) or not isinstance(lambda_max, numbers.Real) or lambda_max < lambda_min):
            raise ValueError("lambda_max must be a real >= lambda_min or None")
        if source != "non_target":
            raise ValueError("A3FL-style UAP supports source='non_target' only")
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.alpha = float(alpha)
        self.lambda_adv = float(lambda_adv)
        self.lambda_mode = lambda_mode
        self.lambda0 = float(lambda_adv if lambda0 is None else lambda0)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(self.lambda0 if lambda_max is None else lambda_max)
        self.adv_model_steps = int(adv_model_steps)
        self.adv_model_lr = float(adv_model_lr)
        self.adv_model_momentum = float(adv_model_momentum)
        self.adv_model_weight_decay = float(adv_model_weight_decay)
        self.outer_steps = int(outer_steps) if outer_steps is not None else int(steps)
        self.k_trigger = int(k_trigger)
        self.adv_updates_per_outer = int(adv_model_steps if adv_updates_per_outer is None else adv_updates_per_outer)
        self.freeze_adversarial_during_delta = bool(freeze_adversarial_during_delta)
        self.clip_to_valid_rgb = bool(clip_to_valid_rgb)
        self.last_lambda_t = self.lambda_adv
        self.last_similarity = None
        self.last_similarity_warning = None
        self.source = source

    def _build_adversarial_model(self, global_model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
        """Return a temporary trainable clone; never mutate ``global_model``."""
        return copy.deepcopy(global_model).to(device)

    @staticmethod
    def apply_trigger(
        images: torch.Tensor,
        delta: torch.Tensor,
        clip_to_valid_rgb: bool = False,
    ) -> torch.Tensor:
        """Apply a [C,H,W] universal delta in the project's normalized input space."""
        if delta.ndim != 3 or tuple(delta.shape) != tuple(images.shape[1:]):
            raise ValueError("delta must have shape [C, H, W] matching images")
        triggered = images + delta.unsqueeze(0)
        return clamp_normalized_cifar10_rgb(triggered) if clip_to_valid_rgb else triggered

    def _optimize_delta_step(
        self,
        global_model: torch.nn.Module,
        adversarial_model: torch.nn.Module,
        images: torch.Tensor,
        target_label: int,
        delta: torch.Tensor,
        lambda_t: float,
    ) -> torch.Tensor:
        """Take one targeted projected-gradient step on global + adversarial CE."""
        delta = delta.detach().requires_grad_(True)
        targets = torch.full((images.size(0),), target_label, device=images.device, dtype=torch.long)
        poisoned = self.apply_trigger(images, delta, self.clip_to_valid_rgb)
        loss = functional.cross_entropy(global_model(poisoned), targets)
        loss = loss + lambda_t * functional.cross_entropy(adversarial_model(poisoned), targets)
        gradient = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        return (delta - self.alpha * gradient.sign()).detach().clamp_(-self.epsilon, self.epsilon)

    def _dynamic_lambda(self, global_model: torch.nn.Module, adversarial_model: torch.nn.Module) -> float:
        """Return fixed or parameter-cosine weighted adversarial loss strength."""
        if self.lambda_mode == "fixed":
            self.last_similarity = None
            self.last_similarity_warning = None
            self.last_lambda_t = self.lambda_adv
            return self.last_lambda_t
        global_vector = torch.cat([p.detach().reshape(-1).float() for p in global_model.parameters()])
        adversarial_vector = torch.cat([p.detach().reshape(-1).float() for p in adversarial_model.parameters()])
        similarity = functional.cosine_similarity(global_vector, adversarial_vector, dim=0)
        if not torch.isfinite(similarity):
            self.last_similarity = 0.0
            self.last_similarity_warning = "nonfinite_cosine_fallback_zero"
        else:
            self.last_similarity = float(similarity.clamp(0.0, 1.0).item())
            self.last_similarity_warning = None
        self.last_lambda_t = float(min(self.lambda_max, max(self.lambda_min, self.lambda0 * self.last_similarity)))
        return self.last_lambda_t

    def _update_adversarial_model_step(
        self,
        adversarial_model: torch.nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        delta: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Let the temporary model minimize true-label CE on triggered images."""
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(
            adversarial_model(self.apply_trigger(images, delta.detach(), self.clip_to_valid_rgb)), labels
        )
        loss.backward()
        optimizer.step()

    def build_delta(
        self,
        global_model: torch.nn.Module,
        dataloader,
        target_label: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Backward-compatible zero-initialized delta construction entry point."""
        return self.refresh_delta(
            global_model=global_model,
            dataloader=dataloader,
            target_label=target_label,
            device=device,
            initial_delta=None,
        )

    def refresh_delta(
        self,
        global_model: torch.nn.Module,
        dataloader,
        target_label: int,
        device: torch.device,
        initial_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Refresh a CPU ``[C,H,W]`` delta without changing global weights.

        ``initial_delta`` supports adaptive A3FL-v2 refreshes: it is projected
        into the configured L-infinity ball before re-optimizing against the
        current global model and a temporary adversarial-model deep copy.
        """
        if not isinstance(global_model, torch.nn.Module):
            raise TypeError("global_model must be a torch.nn.Module")
        if not hasattr(dataloader, "__iter__"):
            raise TypeError("dataloader must be iterable")
        iterator = iter(dataloader)
        first_images = None
        first_labels = None
        # Find a non-target source batch; this realizes the stated source policy.
        for _ in range(max(1, len(dataloader) if hasattr(dataloader, "__len__") else 1)):
            try:
                images, labels = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                images, labels = next(iterator)
            mask = labels != target_label
            if bool(mask.any()):
                first_images, first_labels = images[mask], labels[mask]
                break
        if first_images is None:
            raise ValueError("A3FL delta construction requires at least one non-target sample")
        if first_images.ndim != 4:
            raise ValueError("images must have shape [N,C,H,W]")

        global_model = global_model.to(device)
        original_training = global_model.training
        original_requires_grad = [parameter.requires_grad for parameter in global_model.parameters()]
        adversarial_model = self._build_adversarial_model(global_model, device)
        adversarial_model.train()
        optimizer = torch.optim.SGD(adversarial_model.parameters(), lr=self.adv_model_lr, momentum=self.adv_model_momentum, weight_decay=self.adv_model_weight_decay)
        expected_shape = tuple(first_images.shape[1:])
        if initial_delta is None:
            delta = torch.zeros(expected_shape, device=device, dtype=first_images.dtype)
        else:
            if not isinstance(initial_delta, torch.Tensor):
                raise TypeError("initial_delta must be a tensor or None")
            if initial_delta.ndim != 3 or tuple(initial_delta.shape) != expected_shape:
                raise ValueError("initial_delta must have shape [C, H, W] matching images")
            delta = initial_delta.detach().to(device=device, dtype=first_images.dtype)
            delta = delta.clamp_(-self.epsilon, self.epsilon)
        global_model.eval()
        for parameter in global_model.parameters():
            parameter.requires_grad_(False)
        try:
            pending = (first_images, first_labels)
            for _ in range(self.outer_steps):
                images, labels = pending
                pending = None
                images = images.to(device)
                labels = labels.to(device)
                if self.freeze_adversarial_during_delta:
                    for parameter in adversarial_model.parameters():
                        parameter.requires_grad_(False)
                for _ in range(self.k_trigger):
                    lambda_t = self._dynamic_lambda(global_model, adversarial_model)
                    delta = self._optimize_delta_step(global_model, adversarial_model, images, target_label, delta, lambda_t)
                if self.freeze_adversarial_during_delta:
                    for parameter in adversarial_model.parameters():
                        parameter.requires_grad_(True)
                for _ in range(self.adv_updates_per_outer):
                    self._update_adversarial_model_step(adversarial_model, images, labels, delta, optimizer)
                if pending is None:
                    try:
                        next_images, next_labels = next(iterator)
                    except StopIteration:
                        iterator = iter(dataloader)
                        next_images, next_labels = next(iterator)
                    mask = next_labels != target_label
                    if bool(mask.any()):
                        pending = (next_images[mask], next_labels[mask])
                    else:
                        pending = (images.detach().cpu(), labels.detach().cpu())
        finally:
            for parameter, requires_grad in zip(global_model.parameters(), original_requires_grad):
                parameter.requires_grad_(requires_grad)
            global_model.train(original_training)
        return delta.detach().cpu()

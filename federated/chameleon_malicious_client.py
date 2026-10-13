"""Independent PGFA-protocol Chameleon malicious client."""

from __future__ import annotations

import copy

import torch
from torch import nn

from attacks.chameleon import (
    project_named_parameters_to_l2_ball,
    snapshot_named_parameters,
    restore_all_trainability,
    set_chameleon_adaptation_trainability,
    set_chameleon_projection_trainability,
    weighted_supervised_contrastive_loss,
)
from attacks.chameleon_trigger import poison_chameleon_batch
from attacks.config import AttackConfig
from federated.client import Client


class ChameleonMaliciousClient(Client):
    """Run Chameleon only in configured attack rounds; otherwise train benignly."""

    is_malicious = True

    def __init__(self, *args, attack_config: AttackConfig, **kwargs) -> None:
        if not isinstance(attack_config, AttackConfig):
            raise TypeError("attack_config must be an AttackConfig instance")
        super().__init__(*args, **kwargs)
        self.attack_config = attack_config

    def _benign_result(self, global_model: nn.Module, round_index: int, effective_local_epochs: int | None) -> dict:
        result = super().local_train(
            global_model,
            round_index=round_index,
            effective_local_epochs=effective_local_epochs,
        )
        result.update({
            "is_malicious": True,
            "attack_active": False,
            "num_poisoned_samples": 0,
            "adaptation_loss_avg": None,
            "projection_loss_avg": None,
            "supcon_valid_anchors": 0,
            "supcon_target_valid_anchors": 0,
            "stage1_distance_projection_count": 0,
            "stage2_distance_projection_count": 0,
            "stage1_distance_before_max": None,
            "stage1_distance_after_max": None,
            "stage2_distance_before_max": None,
            "stage2_distance_after_max": None,
        })
        return result

    def local_train(
        self,
        global_model: nn.Module,
        round_index: int,
        neurotoxin_masks=None,
        effective_local_epochs: int | None = None,
    ) -> dict:
        if type(round_index) is not int or round_index <= 0:
            raise ValueError("round_index must be a positive integer")
        attack_active = (
            self.attack_config.is_attack_round(round_index)
            and self.attack_config.is_malicious_client(self.client_id)
        )
        if not attack_active:
            return self._benign_result(global_model, round_index, effective_local_epochs)
        if neurotoxin_masks is not None:
            raise ValueError("Chameleon does not support Neurotoxin masks")
        if effective_local_epochs is not None and effective_local_epochs != self.attack_config.chameleon_total_epochs:
            raise ValueError("Chameleon attack rounds require the configured 2+1 local-epoch budget")

        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()
        if not hasattr(local_model, "forward_features") or not hasattr(local_model, "fc"):
            raise TypeError("Chameleon requires a model with forward_features() and fc")

        cfg = self.attack_config
        adaptation_losses, projection_losses = [], []
        reference_parameters = snapshot_named_parameters(global_model)
        stage1_projection_events, stage2_projection_events = [], []
        poisoned_samples = 0
        valid_anchors = target_valid_anchors = 0
        try:
            set_chameleon_adaptation_trainability(local_model)
            adaptation_optimizer = torch.optim.SGD(
                (parameter for parameter in local_model.parameters() if parameter.requires_grad),
                lr=cfg.chameleon_adaptation_lr,
                momentum=cfg.chameleon_momentum,
                weight_decay=cfg.chameleon_weight_decay,
            )
            for _ in range(cfg.chameleon_adaptation_epochs):
                for images, labels in self.train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    poisoned_images, poisoned_labels, poison_indices = poison_chameleon_batch(
                        images, labels,
                        target_label=cfg.target_label,
                        poison_per_batch=cfg.chameleon_poison_per_batch,
                        training=True,
                        noise_std=cfg.chameleon_noise_std,
                    )
                    adaptation_optimizer.zero_grad()
                    features = local_model.forward_features(poisoned_images)
                    loss, stats = weighted_supervised_contrastive_loss(
                        features,
                        poisoned_labels,
                        target_label=cfg.target_label,
                        temperature=cfg.chameleon_temperature,
                        target_anchor_weight=cfg.chameleon_target_anchor_weight,
                    )
                    loss.backward()
                    adaptation_optimizer.step()
                    if cfg.chameleon_distance_constraint_enabled:
                        stage1_projection_events.append(project_named_parameters_to_l2_ball(
                            local_model, reference_parameters,
                            radius=cfg.chameleon_distance_constraint_radius,
                        ))
                    adaptation_losses.append(float(loss.detach().item()))
                    poisoned_samples += int(poison_indices.numel())
                    valid_anchors += stats["valid_anchors"]
                    target_valid_anchors += stats["target_valid_anchors"]

            set_chameleon_projection_trainability(local_model)
            projection_optimizer = torch.optim.SGD(
                (parameter for parameter in local_model.parameters() if parameter.requires_grad),
                lr=cfg.chameleon_projection_lr,
                momentum=cfg.chameleon_momentum,
                weight_decay=cfg.chameleon_weight_decay,
            )
            criterion = nn.CrossEntropyLoss()
            for _ in range(cfg.chameleon_projection_epochs):
                for images, labels in self.train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    poisoned_images, poisoned_labels, poison_indices = poison_chameleon_batch(
                        images, labels,
                        target_label=cfg.target_label,
                        poison_per_batch=cfg.chameleon_poison_per_batch,
                        training=True,
                        noise_std=cfg.chameleon_noise_std,
                    )
                    projection_optimizer.zero_grad()
                    loss = criterion(local_model(poisoned_images), poisoned_labels)
                    loss.backward()
                    projection_optimizer.step()
                    if cfg.chameleon_distance_constraint_enabled:
                        stage2_projection_events.append(project_named_parameters_to_l2_ball(
                            local_model, reference_parameters,
                            radius=cfg.chameleon_distance_constraint_radius,
                        ))
                    projection_losses.append(float(loss.detach().item()))
                    poisoned_samples += int(poison_indices.numel())
        finally:
            restore_all_trainability(local_model)

        state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in local_model.state_dict().items()
        }
        combined_losses = adaptation_losses + projection_losses
        return {
            "client_id": self.client_id,
            "state_dict": state_dict,
            "num_samples": len(self.local_dataset),
            "train_loss": sum(combined_losses) / len(combined_losses),
            "is_malicious": True,
            "attack_active": True,
            "num_poisoned_samples": poisoned_samples,
            "adaptation_loss_avg": sum(adaptation_losses) / len(adaptation_losses),
            "projection_loss_avg": sum(projection_losses) / len(projection_losses),
            "supcon_valid_anchors": valid_anchors,
            "supcon_target_valid_anchors": target_valid_anchors,
            "stage1_distance_projection_count": sum(event["projected"] for event in stage1_projection_events),
            "stage2_distance_projection_count": sum(event["projected"] for event in stage2_projection_events),
            "stage1_distance_before_max": max((event["distance_before"] for event in stage1_projection_events), default=None),
            "stage1_distance_after_max": max((event["distance_after"] for event in stage1_projection_events), default=None),
            "stage2_distance_before_max": max((event["distance_before"] for event in stage2_projection_events), default=None),
            "stage2_distance_after_max": max((event["distance_after"] for event in stage2_projection_events), default=None),
        }

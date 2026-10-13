"""Malicious federated client supporting patch and adversarial triggers."""

import copy

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import Dataset

from attacks.adversarial_trigger import (
    compute_prototype_loss,
    compute_feature_regularization,
    generate_universal_adversarial_trigger,
)
from attacks.config import AttackConfig
from attacks.a3fl import A3FLTriggerOptimizer
from attacks.poisoning import (
    choose_non_target_indices,
    relabel_poisoned_samples,
    select_poison_indices,
)
from attacks.prototype import collect_class_statistics, enforce_strict_prototype_coverage
from attacks.triggers import apply_trigger
from attacks.dba import apply_dba_trigger, dba_component_masks
from attacks.neurotoxin import apply_neurotoxin_mask, apply_neurotoxin_update_mask
from federated.client import Client


class MaliciousClient(Client):
    """A client that applies its configured trigger only during active rounds."""

    # Stores the adversarial trigger shared by every malicious client in an
    # experiment, modelling a single federated adversarial perturbation.
    shared_adversarial_delta = None
    shared_adversarial_delta_round = None
    shared_a3fl_lambda_t = None
    shared_a3fl_similarity = None
    shared_a3fl_similarity_warning = None
    shared_prototype_bank = None
    prototype_bank_warning_emitted = False
    prototype_source_warning_emitted = False

    @classmethod
    def reset_shared_delta(cls) -> None:
        """Clear the experiment-level shared adversarial perturbation cache."""
        cls.shared_adversarial_delta = None
        cls.shared_adversarial_delta_round = None
        cls.shared_a3fl_lambda_t = None
        cls.shared_a3fl_similarity = None
        cls.shared_a3fl_similarity_warning = None
        cls.shared_prototype_bank = None
        cls.prototype_bank_warning_emitted = False
        cls.prototype_source_warning_emitted = False

    @classmethod
    def should_refresh_a3fl_delta(cls, attack_config: AttackConfig, round_index: int) -> bool:
        """Return whether the shared A3FL delta must be refreshed this attack round."""
        if cls.shared_adversarial_delta is None:
            return True
        policy = attack_config.a3fl_refresh_policy
        if policy == "fixed":
            return False
        if policy == "every_round":
            return cls.shared_adversarial_delta_round != round_index
        if policy == "periodic":
            return (round_index - attack_config.attack_start_round) % attack_config.a3fl_refresh_period == 0
        raise ValueError(f"unsupported A3FL refresh policy: {policy}")

    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        indices: list[int],
        batch_size: int,
        local_epochs: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        num_workers: int,
        device: torch.device,
        attack_config: AttackConfig,
    ) -> None:
        """Initialize the base client and its attack configuration."""
        if not isinstance(attack_config, AttackConfig):
            raise TypeError("attack_config must be an AttackConfig instance")

        super().__init__(
            client_id=client_id,
            dataset=dataset,
            indices=indices,
            batch_size=batch_size,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            num_workers=num_workers,
            device=device,
        )
        self.attack_config = attack_config

    def _get_dataset_name(self) -> str:
        """Resolve a diagnostic-only dataset name without assuming self.dataset."""
        name = getattr(self, "dataset_name", None)
        if name:
            return str(name)
        config = getattr(self, "config", None)
        if isinstance(config, dict):
            dataset = config.get("dataset", {})
            if isinstance(dataset, dict) and dataset.get("name"):
                return str(dataset["name"])
        dataset = getattr(getattr(self, "train_loader", None), "dataset", None)
        # Client loaders wrap the original dataset in torch.utils.data.Subset.
        while dataset is not None and hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        if dataset is not None:
            return str(getattr(dataset, "name", getattr(dataset, "split", type(dataset).__name__)))
        return "configured_dataset"

    def compute_local_prototype_statistics(self, model: nn.Module) -> dict:
        """Expose malicious-local clean feature sums/counts to the server only."""
        return collect_class_statistics(
            model, self.train_loader, self.device,
            self.attack_config.prototype_feature_layer,
            self.attack_config.prototype_center_batches,
        )

    def _fixed_local_target_reference(self, model: nn.Module) -> tuple[torch.Tensor, dict]:
        """Use lowest original local target index; never test or other-client data."""
        for position, original_index in sorted(enumerate(self.indices), key=lambda item: item[1]):
            image, label = self.local_dataset[position]
            if int(label) == self.attack_config.target_label:
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    feature = model.forward_features(
                        image.unsqueeze(0).to(self.device),
                        feature_layer=self.attack_config.prototype_feature_layer,
                    )[0].detach()
                model.train(was_training)
                return feature, {"reference_client_id": self.client_id,
                                 "reference_sample_index": original_index,
                                 "reference_label": int(label),
                                 "reference_selection_rule": "lowest_original_local_index_target"}
        raise RuntimeError("feature_alignment requires a local clean target-class reference sample")

    def local_train(
        self,
        global_model: nn.Module,
        round_index: int,
        neurotoxin_masks=None,
        effective_local_epochs: int | None = None,
    ) -> dict:
        """Train locally, applying the configured attack only when active."""
        if type(round_index) is not int:
            raise TypeError("round_index must be an integer")
        if round_index <= 0:
            raise ValueError("round_index must be positive")
        if effective_local_epochs is None:
            effective_local_epochs = self.local_epochs
        if type(effective_local_epochs) is not int or effective_local_epochs <= 0:
            raise ValueError("effective_local_epochs must be a positive integer or None")

        attack_active = (
            self.attack_config.is_attack_round(round_index)
            and self.attack_config.is_malicious_client(self.client_id)
        )
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        # Defaults retain the existing AttackConfig/patch-only behavior.  The
        # optional attributes allow adversarial settings without changing the
        # FedAvg-facing client interface.
        trigger_type = getattr(self.attack_config, "trigger_type", "patch")
        if trigger_type not in ("patch", "adversarial"):
            raise ValueError("trigger_type must be 'patch' or 'adversarial'")

        shared_delta = None
        if attack_active and trigger_type == "adversarial":
            a3fl_refresh_needed = (
                self.attack_config.method == "a3fl"
                and MaliciousClient.should_refresh_a3fl_delta(self.attack_config, round_index)
            )
            if MaliciousClient.shared_adversarial_delta is None or a3fl_refresh_needed:
                configured_source = self.attack_config.prototype_client_id
                if (self.attack_config.prototype_contrastive_enabled
                        and self.attack_config.prototype_source == "single_malicious"
                        and configured_source is not None
                        and self.client_id != configured_source):
                    # The configured anchor, not the first selected malicious
                    # client, must construct the shared UAP/prototype bank.
                    print(
                        f"[UAPGeneration] deferred_to_prototype_source_client={configured_source}",
                        flush=True,
                    )
                    # This selected client precedes the configured source in
                    # the serial round. It must not enter poisoning with a
                    # nonexistent shared delta; client1 will create it later
                    # in this round and all clients use it thereafter.
                    attack_active = False
                    shared_delta = None
                elif self.attack_config.method == "a3fl":
                    optimizer = A3FLTriggerOptimizer(
                        epsilon=self.attack_config.adversarial_epsilon,
                        steps=self.attack_config.adversarial_steps,
                        alpha=self.attack_config.adversarial_alpha,
                        lambda_adv=self.attack_config.a3fl_lambda_adv,
                        lambda_mode=self.attack_config.a3fl_lambda_mode,
                        lambda0=self.attack_config.a3fl_lambda0,
                        lambda_min=self.attack_config.a3fl_lambda_min,
                        lambda_max=self.attack_config.a3fl_lambda_max,
                        adv_model_steps=self.attack_config.a3fl_adv_model_steps,
                        adv_model_lr=self.attack_config.a3fl_adv_model_lr,
                        adv_model_momentum=self.attack_config.a3fl_adv_model_momentum,
                        adv_model_weight_decay=self.attack_config.a3fl_adv_model_weight_decay,
                        outer_steps=self.attack_config.a3fl_outer_steps,
                        k_trigger=self.attack_config.a3fl_k_trigger,
                        adv_updates_per_outer=self.attack_config.a3fl_adv_updates_per_outer,
                        freeze_adversarial_during_delta=self.attack_config.a3fl_freeze_adversarial_during_delta,
                        source=self.attack_config.a3fl_source,
                        clip_to_valid_rgb=self.attack_config.a3fl_clip_to_valid_rgb,
                    )
                    previous_delta = (MaliciousClient.shared_adversarial_delta
                                      if self.attack_config.a3fl_warm_start_delta else None)
                    shared_delta = optimizer.refresh_delta(
                        global_model=global_model,
                        dataloader=self.train_loader,
                        target_label=self.attack_config.target_label,
                        device=self.device,
                        initial_delta=previous_delta,
                    ).to(device=self.device, dtype=next(local_model.parameters()).dtype)
                    MaliciousClient.shared_adversarial_delta = shared_delta
                    MaliciousClient.shared_adversarial_delta_round = round_index
                    MaliciousClient.shared_a3fl_lambda_t = optimizer.last_lambda_t
                    MaliciousClient.shared_a3fl_similarity = optimizer.last_similarity
                    MaliciousClient.shared_a3fl_similarity_warning = optimizer.last_similarity_warning
                    print(
                        f"[A3FL{self.attack_config.a3fl_variant}Refresh] "
                        f"round={round_index} attack_active=True method=a3fl "
                        f"variant={self.attack_config.a3fl_variant} "
                        f"refresh_policy={self.attack_config.a3fl_refresh_policy} "
                        f"warm_start_delta={self.attack_config.a3fl_warm_start_delta} "
                        f"delta_refreshed=True delta_source_round={round_index} "
                        f"delta_linf={shared_delta.abs().max().item():.6f} linf={shared_delta.abs().max().item():.6f} "
                        f"lambda_adv={self.attack_config.a3fl_lambda_adv} "
                        f"lambda_mode={self.attack_config.a3fl_lambda_mode} lambda0={self.attack_config.a3fl_lambda0} "
                        f"lambda_t={optimizer.last_lambda_t} similarity_theta_adv_global={optimizer.last_similarity} "
                        f"similarity_warning={optimizer.last_similarity_warning} outer_steps={optimizer.outer_steps} "
                        f"k_trigger={optimizer.k_trigger} adv_update_per_outer={optimizer.adv_updates_per_outer} "
                        f"adv_model_steps={self.attack_config.a3fl_adv_model_steps} "
                        f"adv_lr={self.attack_config.a3fl_adv_model_lr} adv_momentum={self.attack_config.a3fl_adv_model_momentum} "
                        f"adv_weight_decay={self.attack_config.a3fl_adv_model_weight_decay} delta_alpha={self.attack_config.adversarial_alpha} epsilon={self.attack_config.adversarial_epsilon} "
                        f"delta_steps={self.attack_config.a3fl_delta_steps} "
                        f"temporary_adversarial_model=True shared_delta=True",
                        flush=True,
                    )
                else:
                    fixed_reference = None
                    if self.attack_config.prototype_loss_variant == "feature_alignment":
                        fixed_reference, reference_metadata = self._fixed_local_target_reference(local_model)
                        print(f"[FeatureAlignment] {reference_metadata}", flush=True)
                    request_prototypes = self.attack_config.prototype_contrastive_enabled and MaliciousClient.shared_prototype_bank is None and (self.attack_config.prototype_use_in_delta_generation or self.attack_config.prototype_use_in_malicious_training)
                    prototype_bank = MaliciousClient.shared_prototype_bank
                    generated_trigger = generate_universal_adversarial_trigger(model=local_model, dataloader=self.train_loader, target_label=self.attack_config.target_label, epsilon=self.attack_config.adversarial_epsilon, steps=self.attack_config.adversarial_steps, alpha=self.attack_config.adversarial_alpha, prototype_contrastive_enabled=(self.attack_config.prototype_contrastive_enabled and self.attack_config.prototype_use_in_delta_generation), prototype_lambda=self.attack_config.prototype_lambda_delta, prototype_temperature=self.attack_config.prototype_temperature, prototype_center_batches=self.attack_config.prototype_center_batches, prototype_mode=self.attack_config.prototype_mode, prototype_min_samples_per_class=self.attack_config.prototype_min_samples_per_class, exclude_target_in_uap=self.attack_config.exclude_target_in_uap, prototype_feature_layer=self.attack_config.prototype_feature_layer, prototype_source_margin=self.attack_config.prototype_source_margin, prototype_loss_variant=self.attack_config.prototype_loss_variant, fixed_target_reference_feature=fixed_reference, return_prototypes=request_prototypes, prototype_bank=prototype_bank)
                    if request_prototypes:
                        shared_delta, prototype_bank = generated_trigger
                    MaliciousClient.shared_prototype_bank = prototype_bank
                    if prototype_bank is not None:
                        if self.attack_config.strict_prototype_coverage:
                            valid_classes, missing = enforce_strict_prototype_coverage(
                                prototype_bank, num_classes=self.attack_config.num_classes,
                                dataset=self._get_dataset_name(),
                                prototype_source_client=self.attack_config.prototype_client_id or self.client_id,
                                center_batches=self.attack_config.prototype_center_batches,
                            )
                            print("[GTSRB Prototype Bank]", flush=True)
                            print(f"prototype_source_client = {self.attack_config.prototype_client_id or self.client_id}", flush=True)
                            print(f"valid_classes = {valid_classes}/{self.attack_config.num_classes}", flush=True)
                            print(f"missing_classes = {missing}", flush=True)
                            print(f"samples_per_class = {prototype_bank.get('samples_per_class', [])}", flush=True)
                        print(
                            "[PrototypeTrain] Cached prototype bank: "
                            f"num_classes={len(prototype_bank['class_ids'])}, "
                            f"feature_dim={prototype_bank['prototypes'].shape[1]}, "
                            f"target_proto_index={prototype_bank['target_proto_index']}",
                            flush=True,
                        )
                    else:
                        shared_delta = generated_trigger
                    MaliciousClient.shared_adversarial_delta = shared_delta
                    print(
                    "[Adversarial Trigger Generated]\n"
                    f"delta max: {shared_delta.max().item():.6f}\n"
                    f"delta min: {shared_delta.min().item():.6f}\n"
                    f"delta mean: {shared_delta.mean().item():.6f}",
                    flush=True,
                )
            else:
                shared_delta = MaliciousClient.shared_adversarial_delta
                if shared_delta.device != self.device:
                    shared_delta = shared_delta.to(self.device)
                    MaliciousClient.shared_adversarial_delta = shared_delta

        if attack_active:
            print(f"[MaliciousLocalTraining] round={round_index} client_id={self.client_id}", flush=True)

        training_prototype_enabled = (
            attack_active
            and self.attack_config.prototype_contrastive_enabled
            and self.attack_config.prototype_use_in_malicious_training
            and self.attack_config.prototype_lambda_train > 0
        )
        if (not attack_active and self.attack_config.prototype_contrastive_enabled
                and self.attack_config.is_malicious_client(self.client_id)
                and round_index == self.attack_config.attack_end_round + 1):
            print(f"[PBUTrainDebug] round={round_index} client_id={self.client_id} "
                  "attack_active=False benign_ce_only no_poison=True no_uap=True no_proto_loss=True",
                  flush=True)
        if (not attack_active and self.attack_config.method == "dba"
                and self.attack_config.is_malicious_client(self.client_id)
                and round_index == self.attack_config.attack_end_round + 1):
            print(f"[DBATrainDebug] round={round_index} client_id={self.client_id} "
                  "attack_active=False benign_ce_only no_poison=True no_trigger=True",
                  flush=True)
        if (not attack_active and self.attack_config.method == "a3fl"
                and self.attack_config.is_malicious_client(self.client_id)
                and round_index == self.attack_config.attack_end_round + 1):
            print(f"[A3FL{self.attack_config.a3fl_variant}Post] round={round_index} client_id={self.client_id} "
                  f"attack_active=False refresh=False last_delta_round={MaliciousClient.shared_adversarial_delta_round} "
                  "benign_ce_only=True no_poison=True no_delta_training=True",
                  flush=True)
        if training_prototype_enabled and MaliciousClient.shared_prototype_bank is None:
            if not MaliciousClient.prototype_bank_warning_emitted:
                print(
                    "[PrototypeTrain] warning=prototype_bank_unavailable; "
                    "training-stage prototype loss will be skipped",
                    flush=True,
                )
                MaliciousClient.prototype_bank_warning_emitted = True

        criterion = nn.CrossEntropyLoss(reduction="none")
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        total_loss = 0.0
        total_samples = 0
        total_ce_loss = 0.0
        total_proto_loss = 0.0
        proto_train_batches = 0
        proto_train_samples = 0
        num_poisoned_samples = 0
        adversarial_trigger_logged = False
        for _ in range(effective_local_epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                original_labels = labels.clone()
                poisoned_images_after_delta = None
                poison_indices = None

                if attack_active:
                    candidate_indices = choose_non_target_indices(
                        labels,
                        self.attack_config.target_label,
                    )
                    if candidate_indices.numel() > 0:
                        local_poison_indices = select_poison_indices(
                            batch_size=candidate_indices.numel(),
                            poison_ratio=self.attack_config.poison_ratio,
                            device=self.device,
                        )
                        poison_indices = candidate_indices[local_poison_indices]
                        if poison_indices.numel() > 0:
                            if self.attack_config.method == "dba":
                                component_index = self.attack_config.malicious_client_ids.index(self.client_id)
                                poisoned_images = images.clone()
                                poisoned_images[poison_indices] = apply_dba_trigger(
                                    images[poison_indices], component_index, self.attack_config.dba_num_parts,
                                    patch_size=self.attack_config.patch_size,
                                    pixel_value=self.attack_config.pixel_value,
                                    pattern=self.attack_config.dba_pattern,
                                    subtrigger_size=self.attack_config.dba_subtrigger_size,
                                    normalized_trigger=self.attack_config.dba_normalized_trigger,
                                )
                                images = poisoned_images
                                trigger_pixels = int(dba_component_masks(
                                    images.shape[-2], images.shape[-1], self.attack_config.patch_size,
                                    self.attack_config.dba_num_parts, images.device,
                                    self.attack_config.dba_pattern,
                                    self.attack_config.dba_subtrigger_size,
                                )[component_index].sum())
                                print(f"[DBATrainDebug] round={round_index} client_id={self.client_id} "
                                      f"part_id={component_index} num_poisoned={poison_indices.numel()} "
                                      f"num_clean={images.size(0) - poison_indices.numel()} "
                                      f"poison_ratio={self.attack_config.poison_ratio} "
                                      f"subtrigger_size={self.attack_config.dba_subtrigger_size} "
                                      f"poison_loss_weight={self.attack_config.dba_poison_loss_weight} "
                                      "trigger_only_on_poison=True "
                                      f"target_label={self.attack_config.target_label} "
                                      f"trigger_pixels={trigger_pixels} attack_active=True", flush=True)
                            elif trigger_type == "patch":
                                images = apply_trigger(
                                    images,
                                    trigger_type="patch",
                                    poison_indices=poison_indices,
                                    patch_size=self.attack_config.patch_size,
                                    pixel_value=self.attack_config.pixel_value,
                                )
                            else:
                                if not adversarial_trigger_logged:
                                    print("[Using Adversarial Trigger]", flush=True)
                                    adversarial_trigger_logged = True
                                poisoned_images = images.clone()
                                poisoned_images[poison_indices] = apply_trigger(
                                    images[poison_indices],
                                    trigger_type="adversarial",
                                    perturbation=shared_delta,
                                    clamp_to_valid_rgb=(
                                        self.attack_config.method == "a3fl"
                                        and self.attack_config.a3fl_clip_to_valid_rgb
                                    ),
                                )
                                images = poisoned_images
                                poisoned_images_after_delta = images[poison_indices]
                            labels = relabel_poisoned_samples(
                                labels,
                                poison_indices,
                                self.attack_config.target_label,
                            )
                            num_poisoned_samples += int(poison_indices.numel())

                optimizer.zero_grad()
                use_full_batch_features = (
                    training_prototype_enabled
                    and MaliciousClient.shared_prototype_bank is not None
                    and hasattr(local_model, "forward_features")
                    and isinstance(getattr(local_model, "fc", None), nn.Module)
                )
                if use_full_batch_features:
                    prototype_bank = MaliciousClient.shared_prototype_bank
                    feature_layer = prototype_bank["feature_layer"]
                    features = local_model.forward_features(
                        images,
                        feature_layer=feature_layer,
                    )
                    # Preserve the established layer4/avgpool/penultimate path,
                    # but layer3 features (256-D) cannot enter the 512-D fc head.
                    if features.size(1) == local_model.fc.in_features:
                        outputs = local_model.fc(features)
                    else:
                        outputs = local_model(images)
                else:
                    features = None
                    outputs = local_model(images)
                per_sample_ce = criterion(outputs, labels)
                ce_loss = per_sample_ce.mean()
                loss = ce_loss
                if (self.attack_config.method == "dba" and attack_active
                        and poison_indices is not None and poison_indices.numel() > 0):
                    poison_mask = torch.zeros(labels.size(0), dtype=torch.bool, device=self.device)
                    poison_mask[poison_indices] = True
                    poison_loss = per_sample_ce[poison_mask].mean()
                    clean_loss = per_sample_ce[~poison_mask].mean() if (~poison_mask).any() else None
                    if self.attack_config.dba_poison_ce_only:
                        loss = poison_loss
                    elif clean_loss is None:
                        loss = poison_loss
                    else:
                        loss = clean_loss + self.attack_config.dba_poison_loss_weight * poison_loss
                proto_loss = None
                if (
                    training_prototype_enabled
                    and poisoned_images_after_delta is not None
                    and MaliciousClient.shared_prototype_bank is not None
                    and features is not None
                ):
                    prototype_bank = MaliciousClient.shared_prototype_bank
                    prototypes = prototype_bank["prototypes"].to(
                        device=self.device,
                        dtype=poisoned_images_after_delta.dtype,
                    ).detach()
                    if features.size(1) != prototypes.size(1):
                        raise ValueError(
                            "prototype feature dimension does not match the cached bank: "
                            f"{features.size(1)} != {prototypes.size(1)}"
                        )
                    proto_loss = compute_feature_regularization(
                        features[poison_indices], prototype_bank.get("loss_variant", "margin"), prototypes,
                        prototype_bank["class_ids"], prototype_bank["target_proto_index"], prototype_bank["temperature"],
                        prototype_bank["mode"], original_labels[poison_indices],
                        prototype_bank.get("source_margin", self.attack_config.prototype_source_margin),
                        prototype_bank.get("valid_class_mask"), prototype_bank.get("target_reference_feature"),
                    )
                    if (
                        proto_loss is None
                        and prototype_bank["mode"] == "target_vs_source"
                        and not MaliciousClient.prototype_source_warning_emitted
                    ):
                        print(
                            "[PrototypeTrain] warning=source_prototype_unavailable; "
                            "skipping prototype loss for this batch",
                            flush=True,
                        )
                        MaliciousClient.prototype_source_warning_emitted = True
                    if proto_loss is not None:
                        loss = (
                            ce_loss
                            + self.attack_config.prototype_lambda_train * proto_loss
                        )
                loss.backward()
                if (self.attack_config.method == "neurotoxin_uap" and attack_active
                        and self.attack_config.neurotoxin_mask_apply_mode == "gradient"):
                    apply_neurotoxin_mask(local_model.named_parameters(), neurotoxin_masks)
                optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_ce_loss += ce_loss.item() * batch_size
                total_samples += batch_size
                if proto_loss is not None:
                    poisoned_count = poisoned_images_after_delta.size(0)
                    total_proto_loss += proto_loss.item() * poisoned_count
                    proto_train_samples += poisoned_count
                    proto_train_batches += 1

        if (attack_active and self.attack_config.prototype_contrastive_enabled
                and round_index % self.attack_config.prototype_debug_interval == 0):
            print(
                "[PBUTrainDebug] "
                f"round={round_index} client_id={self.client_id} "
                f"num_poisoned={num_poisoned_samples} "
                f"num_valid_proto_samples={proto_train_samples} "
                f"ce_loss={total_ce_loss / total_samples:.6f} "
                f"proto_loss={(total_proto_loss / proto_train_samples) if proto_train_samples else None} "
                f"lambda_train={self.attack_config.prototype_lambda_train} "
                f"total_loss={total_loss / total_samples:.6f} "
                f"prototype_source={self.attack_config.prototype_source} "
                f"prototype_version={MaliciousClient.shared_prototype_bank.get('prototype_version') if MaliciousClient.shared_prototype_bank else None}",
                flush=True,
            )
        if (self.attack_config.method == "neurotoxin_uap" and attack_active
                and self.attack_config.neurotoxin_mask_apply_mode == "update_delta"):
            apply_neurotoxin_update_mask(local_model, global_model, neurotoxin_masks)
        if self.attack_config.method == "neurotoxin_uap" and attack_active:
            total_masked = sum(int((mask == 0).sum().item()) for mask in (neurotoxin_masks or {}).values())
            total_mask_entries = sum(mask.numel() for mask in (neurotoxin_masks or {}).values())
            print(f"[NeurotoxinDebug] round={round_index} client_id={self.client_id} mask_enabled=True "
                  f"mask_ratio={self.attack_config.neurotoxin_mask_ratio} "
                  f"effective_keep_ratio={(1 - total_masked / total_mask_entries) if total_mask_entries else 1.0:.6f} "
                  f"mask_apply_mode={self.attack_config.neurotoxin_mask_apply_mode} "
                  f"selected_direction=high_gradient num_poisoned={num_poisoned_samples} "
                  f"uap_fixed={MaliciousClient.shared_adversarial_delta is not None}", flush=True)

        state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in local_model.state_dict().items()
        }
        return {
            "client_id": self.client_id,
            "state_dict": state_dict,
            "num_samples": len(self.local_dataset),
            "train_loss": total_loss / total_samples,
            "ce_loss_avg": total_ce_loss / total_samples,
            "proto_train_loss_avg": (
                total_proto_loss / proto_train_samples
                if proto_train_samples > 0
                else None
            ),
            "proto_train_batches": proto_train_batches,
            "lambda_train": self.attack_config.prototype_lambda_train,
            "is_malicious": True,
            "attack_active": attack_active,
            "num_poisoned_samples": num_poisoned_samples,
        }

"""Configuration structures for federated backdoor attack experiments."""

from dataclasses import dataclass, field


@dataclass
class AttackConfig:
    """Store validated settings for a potential federated backdoor attack."""

    enabled: bool = False
    target_label: int = 0
    malicious_client_ids: list[int] = field(default_factory=list)
    num_malicious_clients: int | None = None
    selection_policy: str = "legacy_all_malicious"
    min_malicious_per_attack_round: int = 1
    prototype_build_policy: str = "legacy_local"
    poison_ratio: float = 0.2
    attack_start_round: int = 1
    attack_end_round: int | None = None
    trigger_type: str = "patch"
    patch_size: int = 3
    pixel_value: float = 1.0
    adversarial_epsilon: float = 8 / 255
    adversarial_steps: int = 10
    adversarial_alpha: float = 2 / 255
    prototype_contrastive_enabled: bool = False
    prototype_lambda: float = 0.1
    prototype_use_in_delta_generation: bool = False
    prototype_use_in_malicious_training: bool = False
    prototype_lambda_delta: float = 0.1
    prototype_lambda_train: float = 0.0
    prototype_temperature: float = 0.2
    prototype_source_margin: float = 0.2
    prototype_center_batches: int = 5
    prototype_mode: str = "target_vs_all"
    prototype_loss_variant: str = "margin"
    prototype_feature_layer: str = "penultimate"
    prototype_min_samples_per_class: int = 1
    prototype_source: str = "single_malicious"
    prototype_client_id: int | None = None
    prototype_aggregate: bool = False
    prototype_freeze_after_build: bool = True
    prototype_missing_class_strategy: str = "skip"
    prototype_refresh_policy: str = "fixed"
    prototype_refresh_during: str = "attack_only"
    prototype_debug_interval: int = 10
    num_classes: int = 10
    exclude_target_in_uap: bool = True
    strict_prototype_coverage: bool = False
    method: str | None = None
    neurotoxin_enabled: bool = False
    neurotoxin_mask_ratio: float = 0.0
    neurotoxin_mask_apply_mode: str = "gradient"
    dba_enabled: bool = False
    dba_num_parts: int | None = None
    dba_pattern: str = "full_patch_split"
    dba_subtrigger_size: int | None = None
    dba_full_trigger_size: int | None = None
    dba_normalized_trigger: bool = True
    dba_poison_loss_weight: float = 1.0
    dba_poison_ce_only: bool = False
    dba_eval_full_trigger: bool = True
    a3fl_enabled: bool = False
    a3fl_variant: str = "v2"
    # Preserve v2's existing refresh behavior unless a config opts out.
    a3fl_warm_start_delta: bool = True
    a3fl_delta_steps: int | None = None
    a3fl_lambda_adv: float = 1.0
    a3fl_lambda_mode: str = "fixed"
    a3fl_lambda0: float | None = None
    a3fl_lambda_min: float = 0.0
    a3fl_lambda_max: float | None = None
    a3fl_adv_model_steps: int = 1
    a3fl_adv_model_lr: float = 0.01
    a3fl_adv_model_momentum: float = 0.0
    a3fl_adv_model_weight_decay: float = 0.0
    a3fl_outer_steps: int | None = None
    a3fl_k_trigger: int = 1
    a3fl_adv_updates_per_outer: int | None = None
    a3fl_freeze_adversarial_during_delta: bool = False
    a3fl_trigger_scope: str = "uap"
    a3fl_refresh_policy: str = "fixed"
    a3fl_refresh_period: int = 1
    a3fl_use_current_global: bool = True
    a3fl_use_adversarial_model: bool = True
    a3fl_freeze_delta_after_attack: bool = True
    a3fl_save_delta_every_refresh: bool = False
    a3fl_use_last_attack_delta_for_post_eval: bool = True
    a3fl_post_eval_delta_round: int | None = None
    a3fl_freeze_after_build: bool = True
    a3fl_build_once_at_attack_start: bool = True
    a3fl_share_delta_across_malicious_clients: bool = True
    a3fl_source: str = "non_target"
    # Disabled by default to preserve all historical A3FL runs. When enabled
    # for CIFAR-10, inputs are clipped to the normalized interval for raw RGB [0, 1].
    a3fl_clip_to_valid_rgb: bool = False
    chameleon_adaptation_epochs: int = 2
    chameleon_projection_epochs: int = 1
    chameleon_adaptation_lr: float = 0.005
    chameleon_projection_lr: float = 0.01
    chameleon_momentum: float = 0.9
    chameleon_weight_decay: float = 0.0005
    chameleon_temperature: float = 0.07
    chameleon_target_anchor_weight: float = 6.0
    chameleon_poison_per_batch: int = 7
    chameleon_noise_std: float = 0.01
    chameleon_distance_constraint_enabled: bool = False
    chameleon_distance_constraint_radius: float = 5.0
    f3ba_conv_rate: float = 0.02
    f3ba_fc_rate: float = 0.001
    f3ba_flip_factor: float = 1.0
    f3ba_trigger_step: float = 0.1
    f3ba_trigger_epochs: int = 2
    f3ba_normal_scale: float = 0.6
    f3ba_backdoor_scale: float = 0.4
    f3ba_kernel_selection: str = "movement"
    f3ba_bn_mode: str = "legacy_eval"

    def __post_init__(self) -> None:
        """Validate attack configuration values after initialization."""
        if type(self.target_label) is not int:
            raise TypeError("target_label must be an integer")
        if self.target_label < 0:
            raise ValueError("target_label must be non-negative")
        if not isinstance(self.malicious_client_ids, list):
            raise TypeError("malicious_client_ids must be a list")
        if any(type(client_id) is not int for client_id in self.malicious_client_ids):
            raise TypeError("malicious_client_ids must contain only integers")
        if any(client_id < 0 for client_id in self.malicious_client_ids):
            raise ValueError("malicious client IDs must be non-negative")
        if len(set(self.malicious_client_ids)) != len(self.malicious_client_ids):
            raise ValueError("malicious_client_ids must not contain duplicates")
        if self.num_malicious_clients is None:
            self.num_malicious_clients = len(self.malicious_client_ids)
        if self.num_malicious_clients != len(self.malicious_client_ids):
            raise ValueError("num_malicious_clients must equal len(malicious_client_ids)")
        if self.selection_policy not in {"legacy_all_malicious", "force_during_attack", "random_only"}:
            raise ValueError("unsupported malicious client selection_policy")
        if self.min_malicious_per_attack_round <= 0:
            raise ValueError("min_malicious_per_attack_round must be positive")
        if self.prototype_build_policy not in {"legacy_local", "force_at_attack_start", "server_aggregate"}:
            raise ValueError("unsupported prototype_build_policy")
        if isinstance(self.poison_ratio, bool) or not isinstance(
            self.poison_ratio, (int, float)
        ):
            raise TypeError("poison_ratio must be a number")
        if not 0.0 <= self.poison_ratio <= 1.0:
            raise ValueError("poison_ratio must be between 0.0 and 1.0")
        if type(self.attack_start_round) is not int:
            raise TypeError("attack_start_round must be an integer")
        if self.attack_start_round <= 0:
            raise ValueError("attack_start_round must be positive")
        if self.attack_end_round is not None:
            if type(self.attack_end_round) is not int:
                raise TypeError("attack_end_round must be an integer or None")
            if self.attack_end_round <= 0:
                raise ValueError("attack_end_round must be positive")
            if self.attack_end_round < self.attack_start_round:
                raise ValueError(
                    "attack_end_round cannot be earlier than attack_start_round"
                )
        if self.trigger_type not in ("patch", "adversarial", "chameleon", "focused_flip"):
            raise ValueError("trigger_type must be 'patch', 'adversarial', 'chameleon', or 'focused_flip'")
        if self.prototype_temperature <= 0:
            raise ValueError("prototype_temperature must be positive")
        if self.prototype_lambda_delta < 0:
            raise ValueError("prototype_lambda_delta must be non-negative")
        if self.prototype_lambda_train < 0:
            raise ValueError("prototype_lambda_train must be non-negative")
        if self.prototype_center_batches == 0 or self.prototype_center_batches < -1:
            raise ValueError("prototype_center_batches must be positive or -1 (full local loader)")
        if self.prototype_min_samples_per_class <= 0:
            raise ValueError("prototype_min_samples_per_class must be positive")
        if self.prototype_source not in {"single_malicious", "multi_malicious", "all_clients"}:
            raise ValueError("prototype_source must be single_malicious, multi_malicious, or all_clients")
        if self.prototype_build_policy == "server_aggregate" and (
                self.prototype_source not in {"multi_malicious", "all_clients"} or not self.prototype_aggregate):
            raise ValueError("server_aggregate requires multi_malicious/all_clients source and aggregate_prototypes=true")
        if self.prototype_client_id is not None:
            if type(self.prototype_client_id) is not int or self.prototype_client_id < 0:
                raise ValueError("prototype_client_id must be a non-negative integer or None")
            if self.prototype_source == "single_malicious" and self.prototype_client_id not in self.malicious_client_ids:
                raise ValueError("prototype_client_id must belong to malicious_client_ids")
        if self.prototype_missing_class_strategy not in {"skip", "zero"}:
            raise ValueError("prototype_missing_class_strategy must be skip or zero")
        if self.prototype_refresh_policy != "fixed":
            raise ValueError("only fixed prototype banks are supported by the main code")
        if self.prototype_refresh_during not in {"attack_only", "all_rounds"}:
            raise ValueError("prototype_refresh_during must be attack_only or all_rounds")
        if type(self.prototype_debug_interval) is not int or self.prototype_debug_interval <= 0:
            raise ValueError("prototype_debug_interval must be positive")
        if type(self.num_classes) is not int or self.num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if self.prototype_source_margin < 0:
            raise ValueError("prototype_source_margin must be non-negative")
        if self.prototype_mode not in ("target_vs_all", "target_vs_source"):
            raise ValueError(
                "prototype_mode must be 'target_vs_all' or 'target_vs_source'"
            )
        if self.prototype_loss_variant not in {"margin", "target_only", "feature_alignment"}:
            raise ValueError("prototype_loss_variant must be margin, target_only, or feature_alignment")
        if self.prototype_feature_layer not in (
            "layer3", "layer4", "avgpool", "penultimate"
        ):
            raise ValueError(
                "prototype_feature_layer must be 'layer3', 'layer4', 'avgpool', or 'penultimate'"
            )
        inferred_method = (
            "patch" if self.trigger_type == "patch" else
            "chameleon" if self.trigger_type == "chameleon" else
            "focused_flip" if self.trigger_type == "focused_flip" else
            "proto_uap" if self.prototype_contrastive_enabled else "ce_uap"
        )
        self.method = self.method or inferred_method
        if self.method not in {"patch", "ce_uap", "proto_uap", "neurotoxin_uap", "dba", "a3fl", "chameleon", "focused_flip"}:
            raise ValueError("unsupported attack.method")
        if not 0.0 <= self.neurotoxin_mask_ratio <= 1.0:
            raise ValueError("neurotoxin.mask_ratio must be between 0 and 1")
        if self.neurotoxin_mask_apply_mode not in {"gradient", "update_delta"}:
            raise ValueError("neurotoxin.mask_apply_mode must be gradient or update_delta")
        if self.method in {"ce_uap", "proto_uap", "neurotoxin_uap", "a3fl"} and self.trigger_type != "adversarial":
            raise ValueError("UAP methods require trigger.type='adversarial'")
        if self.method in {"patch", "dba"} and self.trigger_type != "patch":
            raise ValueError("patch and dba methods require trigger.type='patch'")
        if self.method == "chameleon" and self.trigger_type != "chameleon":
            raise ValueError("Chameleon requires trigger.type='chameleon'")
        if self.method == "focused_flip" and self.trigger_type != "focused_flip":
            raise ValueError("focused_flip requires trigger.type='focused_flip'")
        if self.method in {"ce_uap", "neurotoxin_uap", "a3fl", "patch", "dba", "chameleon"} and self.prototype_contrastive_enabled:
            raise ValueError("prototype_contrastive is only valid for attack.method='proto_uap'")
        if self.method == "proto_uap" and not self.prototype_contrastive_enabled:
            raise ValueError("proto_uap requires prototype_contrastive.enabled=true")
        if self.method != "neurotoxin_uap" and self.neurotoxin_enabled:
            raise ValueError("neurotoxin is only valid for attack.method='neurotoxin_uap'")
        if self.method != "dba" and self.dba_enabled:
            raise ValueError("dba settings require attack.method='dba'")
        if self.method != "a3fl" and self.a3fl_enabled:
            raise ValueError("a3fl settings require attack.method='a3fl'")
        if self.method == "a3fl":
            if not self.a3fl_enabled:
                raise ValueError("attack.method='a3fl' requires a3fl.enabled=true")
            if self.a3fl_lambda_adv < 0 or self.a3fl_adv_model_lr < 0:
                raise ValueError("a3fl lambda_adv and adv_model_lr must be non-negative")
            if self.a3fl_variant not in {"v2", "strong_v3", "faithful_uap_v4"}:
                raise ValueError("a3fl.variant must be v2, strong_v3, or faithful_uap_v4")
            if type(self.a3fl_warm_start_delta) is not bool:
                raise TypeError("a3fl.warm_start_delta must be a boolean")
            if type(self.a3fl_clip_to_valid_rgb) is not bool:
                raise TypeError("a3fl.clip_to_valid_rgb must be a boolean")
            if self.a3fl_delta_steps is None:
                self.a3fl_delta_steps = self.adversarial_steps
            if type(self.a3fl_delta_steps) is not int or self.a3fl_delta_steps <= 0:
                raise ValueError("a3fl.delta_steps must be a positive integer")
            if self.a3fl_delta_steps != self.adversarial_steps:
                raise ValueError("a3fl.delta_steps must match the adversarial UAP steps")
            if type(self.a3fl_adv_model_steps) is not int or self.a3fl_adv_model_steps <= 0:
                raise ValueError("a3fl.adv_model_steps must be a positive integer")
            if self.a3fl_lambda_mode not in {"fixed", "dynamic_similarity"}:
                raise ValueError("a3fl.lambda_mode must be fixed or dynamic_similarity")
            if self.a3fl_lambda0 is None:
                self.a3fl_lambda0 = self.a3fl_lambda_adv
            if self.a3fl_lambda0 < 0 or self.a3fl_lambda_min < 0:
                raise ValueError("a3fl lambda0 and lambda_min must be non-negative")
            if self.a3fl_lambda_max is None:
                self.a3fl_lambda_max = self.a3fl_lambda0
            if self.a3fl_lambda_max < self.a3fl_lambda_min:
                raise ValueError("a3fl.lambda_max must be >= lambda_min")
            if self.a3fl_adv_model_momentum < 0 or self.a3fl_adv_model_weight_decay < 0:
                raise ValueError("a3fl adversarial optimizer momentum and weight decay must be non-negative")
            if self.a3fl_outer_steps is not None and (type(self.a3fl_outer_steps) is not int or self.a3fl_outer_steps <= 0):
                raise ValueError("a3fl.outer_steps must be a positive integer or None")
            if type(self.a3fl_k_trigger) is not int or self.a3fl_k_trigger <= 0:
                raise ValueError("a3fl.k_trigger must be a positive integer")
            if self.a3fl_adv_updates_per_outer is not None and (type(self.a3fl_adv_updates_per_outer) is not int or self.a3fl_adv_updates_per_outer <= 0):
                raise ValueError("a3fl.adv_update_per_outer must be a positive integer or None")
            if self.a3fl_variant == "faithful_uap_v4" and self.a3fl_lambda_mode != "dynamic_similarity":
                raise ValueError("faithful_uap_v4 requires lambda_mode=dynamic_similarity")
            if self.a3fl_trigger_scope != "uap" or self.a3fl_source != "non_target":
                raise ValueError("A3FL-style baseline requires trigger_scope='uap' and source='non_target'")
            if self.a3fl_refresh_policy not in {"fixed", "every_round", "periodic"}:
                raise ValueError("a3fl.refresh_policy must be fixed, every_round, or periodic")
            if type(self.a3fl_refresh_period) is not int or self.a3fl_refresh_period <= 0:
                raise ValueError("a3fl.refresh_period must be a positive integer")
            if self.a3fl_refresh_policy == "every_round" and self.a3fl_refresh_period != 1:
                raise ValueError("a3fl.refresh_period must be 1 when refresh_policy='every_round'")
            if self.a3fl_refresh_policy == "fixed" and (
                    not self.a3fl_freeze_after_build or not self.a3fl_build_once_at_attack_start):
                raise ValueError("legacy fixed A3FL requires freeze_after_build=true and build_once_at_attack_start=true")
            if not self.a3fl_use_current_global or not self.a3fl_use_adversarial_model:
                raise ValueError("A3FL requires current global and temporary adversarial model objectives")
            if not self.a3fl_freeze_delta_after_attack or not self.a3fl_use_last_attack_delta_for_post_eval:
                raise ValueError("A3FL requires freezing and evaluating the last attack delta after attack end")
            if self.a3fl_post_eval_delta_round is not None and self.a3fl_post_eval_delta_round != self.attack_end_round:
                raise ValueError("a3fl.post_eval_delta_round must equal attack.end_round")
            if not self.a3fl_share_delta_across_malicious_clients:
                raise ValueError("A3FL-style baseline requires a shared malicious-client delta")
        if self.method == "chameleon":
            if (self.chameleon_adaptation_epochs, self.chameleon_projection_epochs) not in {(2, 1), (1, 2)}:
                raise ValueError("Chameleon supports only 2+1 or diagnostic 1+2 local epochs")
            for name, value in {
                "adaptation_lr": self.chameleon_adaptation_lr,
                "projection_lr": self.chameleon_projection_lr,
                "momentum": self.chameleon_momentum,
                "weight_decay": self.chameleon_weight_decay,
                "temperature": self.chameleon_temperature,
                "target_anchor_weight": self.chameleon_target_anchor_weight,
                "noise_std": self.chameleon_noise_std,
            }.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(f"chameleon.{name} must be a non-negative number")
            if self.chameleon_adaptation_lr <= 0 or self.chameleon_projection_lr <= 0:
                raise ValueError("Chameleon learning rates must be positive")
            if self.chameleon_temperature <= 0 or self.chameleon_target_anchor_weight <= 0:
                raise ValueError("Chameleon temperature and target_anchor_weight must be positive")
            if type(self.chameleon_poison_per_batch) is not int or self.chameleon_poison_per_batch <= 0:
                raise ValueError("chameleon.poison_per_batch must be a positive integer")
            if type(self.chameleon_distance_constraint_enabled) is not bool:
                raise TypeError("chameleon.distance_constraint.enabled must be a boolean")
            if isinstance(self.chameleon_distance_constraint_radius, bool) or not isinstance(self.chameleon_distance_constraint_radius, (int, float)) or self.chameleon_distance_constraint_radius <= 0:
                raise ValueError("chameleon.distance_constraint.radius must be a positive number")
        if self.method == "focused_flip":
            if self.f3ba_kernel_selection != "movement":
                raise ValueError("focused_flip faithfully supports only kernel_selection='movement'")
            for name, value in {"conv_rate": self.f3ba_conv_rate, "fc_rate": self.f3ba_fc_rate, "flip_factor": self.f3ba_flip_factor, "trigger_step": self.f3ba_trigger_step, "normal_scale": self.f3ba_normal_scale, "backdoor_scale": self.f3ba_backdoor_scale}.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(f"f3ba.{name} must be a non-negative number")
            if not 0 < self.f3ba_conv_rate <= 1 or not 0 < self.f3ba_fc_rate <= 1:
                raise ValueError("f3ba conv_rate and fc_rate must be in (0, 1]")
            if type(self.f3ba_trigger_epochs) is not int or self.f3ba_trigger_epochs <= 0:
                raise ValueError("f3ba.trigger_epochs must be a positive integer")
            if self.f3ba_trigger_step != 0.1:
                raise ValueError("focused_flip requires trigger_step=0.1")
            if self.f3ba_normal_scale + self.f3ba_backdoor_scale <= 0:
                raise ValueError("f3ba normal_scale and backdoor_scale must have positive total weight")
            if self.f3ba_bn_mode not in {"legacy_eval", "official_compatible"}:
                raise ValueError("f3ba.bn_mode must be legacy_eval or official_compatible")
        if self.method == "dba":
            if not self.dba_enabled:
                raise ValueError("attack.method='dba' requires dba.enabled=true")
            self.dba_num_parts = self.dba_num_parts or len(self.malicious_client_ids)
            self.dba_subtrigger_size = self.dba_subtrigger_size or self.patch_size
            if self.dba_num_parts != len(self.malicious_client_ids):
                raise ValueError("dba.num_parts must equal len(malicious_client_ids)")
            if self.dba_pattern not in {"corner_blocks", "grid", "diagonal", "full_patch_split"}:
                raise ValueError("unsupported dba.pattern")
            if type(self.dba_subtrigger_size) is not int or self.dba_subtrigger_size <= 0:
                raise ValueError("dba.subtrigger_size must be a positive integer")
            if self.dba_full_trigger_size is not None and (
                    type(self.dba_full_trigger_size) is not int or self.dba_full_trigger_size <= 0):
                raise ValueError("dba.full_trigger_size must be a positive integer or None")
            if self.dba_poison_loss_weight <= 0:
                raise ValueError("dba.poison_loss_weight must be positive")
            if not self.dba_eval_full_trigger:
                raise ValueError("DBA evaluation must use the full composed trigger")

    def is_attack_round(self, round_index: int) -> bool:
        """Return whether the attack is active during the given round."""
        if not self.enabled:
            return False
        if round_index < self.attack_start_round:
            return False
        if self.attack_end_round is not None and round_index > self.attack_end_round:
            return False
        return True

    @property
    def chameleon_total_epochs(self) -> int:
        """Return Chameleon's fixed PGFA attack-round local budget."""
        return self.chameleon_adaptation_epochs + self.chameleon_projection_epochs

    def is_malicious_client(self, client_id: int) -> bool:
        """Return whether a client ID is configured as malicious."""
        return client_id in self.malicious_client_ids

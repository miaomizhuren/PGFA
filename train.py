"""Entry point for locally configured federated learning experiments."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from attacks.config import AttackConfig
from data.dataset import get_dataset_spec, load_dataset
from data.partition import partition_dirichlet, partition_dirichlet_balanced, partition_iid
from experiment_tracking.recorder import ExperimentRecorder
from federated.client import Client
from federated.malicious_client import MaliciousClient
from federated.chameleon_malicious_client import ChameleonMaliciousClient
from federated.focused_flip_client import FocusedFlipClient
from federated.server import Server
from federated.aggregation import validate_defense_config
from models.resnet18 import build_resnet18
from utils.seed import set_seed
from utils.config_validator import validate_experiment_protocol


def resolve_device(device_type: str) -> torch.device:
    """Resolve a configured device type to an available PyTorch device."""
    if device_type == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_type == "cpu":
        return torch.device("cpu")
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    raise ValueError(f"unsupported device type: {device_type}")


def parse_args() -> argparse.Namespace:
    """Parse the optional experiment configuration path."""
    parser = argparse.ArgumentParser(description="Run a local-dataset FedAvg experiment.")
    parser.add_argument(
        "--config",
        default="configs/cifar10_fedavg.yaml",
        help="YAML configuration path, relative to the project root by default.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without loading data or training.")
    parser.add_argument(
        "--stop-after-round",
        type=int,
        default=None,
        help="Optional execution-only round cap for an isolated diagnostic run; does not alter YAML protocol fields.",
    )
    return parser.parse_args()


def _merge_config(base: dict, override: dict) -> dict:
    """Recursively merge a compact diagnostic config over its base YAML."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path) -> dict:
    """Load YAML, optionally inheriting one local ``base_config`` YAML file."""
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise TypeError("configuration root must be a mapping")
    base_value = config.pop("base_config", None)
    if base_value is None:
        return config
    base_path = Path(base_value)
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    if not base_path.is_file():
        raise FileNotFoundError(f"base_config not found: {base_path}")
    return _merge_config(load_config(base_path.resolve()), config)


def main() -> None:
    """Run the configured local-dataset FedAvg experiment."""
    project_root = Path(__file__).resolve().parent
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_config(config_path)

    experiment_config = config["experiment"]
    dataset_config = config["dataset"]
    partition_config = config["partition"]
    training_config = config["training"]
    federated_config = config["federated"]
    tracking_config = config.get("tracking", {})
    checkpoint_config = config.get("checkpoint", {})
    attack_section = config.get("attack", {})
    visualization_config = config.get("visualization", {})
    trigger_section = attack_section.get("trigger", {})
    adversarial_section = trigger_section.get("adversarial", {})
    prototype_section = adversarial_section.get("prototype_contrastive", {})
    method = attack_section.get("method")
    neurotoxin_section = attack_section.get("neurotoxin", {})
    dba_section = attack_section.get("dba", {})
    a3fl_section = attack_section.get("a3fl", {})
    chameleon_section = attack_section.get("chameleon", {})
    chameleon_distance_constraint_section = chameleon_section.get("distance_constraint", {})
    f3ba_section = attack_section.get("focused_flip", {})
    a3fl_epsilon = a3fl_section.get("epsilon", adversarial_section.get("epsilon", 8 / 255))
    a3fl_steps = a3fl_section.get("delta_steps", a3fl_section.get("steps", adversarial_section.get("steps", 10)))
    a3fl_alpha = a3fl_section.get("alpha", adversarial_section.get("alpha", 2 / 255))
    defense_config = validate_defense_config(config.get("defense"))
    if (attack_section.get("enabled", False) and "target_label" not in attack_section
            and "target_label" not in trigger_section):
        raise ValueError("enabled attack requires explicit YAML field attack.target_label")
    attack_config = AttackConfig(
        enabled=attack_section.get("enabled", False),
        target_label=attack_section.get("target_label", trigger_section.get("target_label", 0)),
        malicious_client_ids=attack_section.get("malicious_client_ids", []),
        num_malicious_clients=attack_section.get("num_malicious_clients"),
        selection_policy=attack_section.get("selection_policy", "legacy_all_malicious"),
        min_malicious_per_attack_round=attack_section.get("min_malicious_per_attack_round", 1),
        prototype_build_policy=attack_section.get("prototype_build_policy", "legacy_local"),
        poison_ratio=attack_section.get("poison_ratio", 0.2),
        attack_start_round=attack_section.get("start_round", attack_section.get("attack_start_round", 1)),
        attack_end_round=attack_section.get("end_round", attack_section.get("attack_end_round")),
        trigger_type=trigger_section.get("type", "patch"),
        patch_size=trigger_section.get("patch_size", 3),
        pixel_value=dba_section.get("trigger_value", trigger_section.get("pixel_value", 1.0)),
        adversarial_epsilon=a3fl_epsilon if method == "a3fl" else adversarial_section.get("epsilon", 8 / 255),
        adversarial_steps=a3fl_steps if method == "a3fl" else adversarial_section.get("steps", 10),
        adversarial_alpha=a3fl_alpha if method == "a3fl" else adversarial_section.get("alpha", 2 / 255),
        prototype_contrastive_enabled=prototype_section.get("enabled", False),
        prototype_lambda=prototype_section.get("lambda", 0.1),
        prototype_use_in_delta_generation=prototype_section.get(
            "use_in_delta_generation", prototype_section.get("enabled", False)
        ),
        prototype_use_in_malicious_training=prototype_section.get(
            "use_in_malicious_training", False
        ),
        prototype_lambda_delta=prototype_section.get(
            "lambda_delta", prototype_section.get("lambda", 0.1)
        ),
        prototype_lambda_train=prototype_section.get("lambda_train", 0.0),
        prototype_temperature=prototype_section.get("temperature", 0.2),
        prototype_source_margin=prototype_section.get("source_margin", 0.2),
        prototype_center_batches=prototype_section.get("center_batches", 5),
        prototype_mode=prototype_section.get("mode", "target_vs_all"),
        prototype_loss_variant=prototype_section.get("loss_variant", "margin"),
        prototype_feature_layer=prototype_section.get("feature_layer", "penultimate"),
        prototype_min_samples_per_class=prototype_section.get("min_samples_per_class", 1),
        prototype_source=prototype_section.get("prototype_source", "single_malicious"),
        prototype_client_id=prototype_section.get("prototype_client_id"),
        prototype_aggregate=prototype_section.get("aggregate_prototypes", False),
        prototype_freeze_after_build=prototype_section.get("freeze_after_build", True),
        prototype_missing_class_strategy=prototype_section.get("missing_class_strategy", "skip"),
        prototype_refresh_policy=prototype_section.get("refresh_policy", "fixed"),
        prototype_refresh_during=prototype_section.get("refresh_during", "attack_only"),
        prototype_debug_interval=prototype_section.get("debug_interval", 10),
        num_classes=dataset_config.get("num_classes", 10),
        exclude_target_in_uap=prototype_section.get("exclude_target_in_uap", True),
        strict_prototype_coverage=prototype_section.get("strict_prototype_coverage", False),
        method=method,
        neurotoxin_enabled=neurotoxin_section.get("enabled", method == "neurotoxin_uap"),
        neurotoxin_mask_ratio=neurotoxin_section.get("mask_ratio", 0.0),
        neurotoxin_mask_apply_mode=neurotoxin_section.get("mask_apply_mode", "gradient"),
        dba_enabled=dba_section.get("enabled", method == "dba"),
        # ``num_sub_triggers`` is the explicit DBA4-facing spelling; retain
        # ``num_parts`` so all legacy three-client DBA configurations remain
        # valid and behaviorally unchanged.
        dba_num_parts=dba_section.get("num_sub_triggers", dba_section.get("num_parts")),
        dba_pattern=dba_section.get("pattern", "full_patch_split"),
        dba_subtrigger_size=dba_section.get("subtrigger_size"),
        dba_full_trigger_size=dba_section.get("full_trigger_size"),
        dba_normalized_trigger=dba_section.get("normalized_trigger", True),
        dba_poison_loss_weight=dba_section.get("poison_loss_weight", 1.0),
        dba_poison_ce_only=dba_section.get("poison_ce_only", False),
        dba_eval_full_trigger=dba_section.get("eval_full_trigger", True),
        a3fl_enabled=a3fl_section.get("enabled", method == "a3fl"),
        a3fl_variant=a3fl_section.get("variant", "v2"),
        a3fl_warm_start_delta=a3fl_section.get("warm_start_delta", True),
        a3fl_delta_steps=a3fl_section.get("delta_steps", a3fl_steps),
        a3fl_lambda_adv=a3fl_section.get("lambda_adv", 1.0),
        a3fl_lambda_mode=a3fl_section.get("lambda_mode", "fixed"),
        a3fl_lambda0=a3fl_section.get("lambda0"),
        a3fl_lambda_min=a3fl_section.get("lambda_min", 0.0),
        a3fl_lambda_max=a3fl_section.get("lambda_max"),
        a3fl_adv_model_steps=a3fl_section.get("adv_model_steps", 1),
        a3fl_adv_model_lr=a3fl_section.get("adv_lr", a3fl_section.get("adv_model_lr", 0.01)),
        a3fl_adv_model_momentum=a3fl_section.get("adv_momentum", 0.0),
        a3fl_adv_model_weight_decay=a3fl_section.get("adv_weight_decay", 0.0),
        a3fl_outer_steps=a3fl_section.get("outer_steps"),
        a3fl_k_trigger=a3fl_section.get("k_trigger", 1),
        a3fl_adv_updates_per_outer=a3fl_section.get("adv_update_per_outer"),
        a3fl_freeze_adversarial_during_delta=a3fl_section.get("freeze_adversarial_during_delta", False),
        a3fl_trigger_scope=a3fl_section.get("trigger_scope", "uap"),
        a3fl_refresh_policy=a3fl_section.get("refresh_policy", "fixed"),
        a3fl_refresh_period=a3fl_section.get("refresh_period", 1),
        a3fl_use_current_global=a3fl_section.get("use_current_global", True),
        a3fl_use_adversarial_model=a3fl_section.get("use_adversarial_model", True),
        a3fl_freeze_delta_after_attack=a3fl_section.get("freeze_delta_after_attack", True),
        a3fl_save_delta_every_refresh=a3fl_section.get("save_delta_every_refresh", False),
        a3fl_use_last_attack_delta_for_post_eval=a3fl_section.get("use_last_attack_delta_for_post_eval", True),
        a3fl_post_eval_delta_round=a3fl_section.get("post_eval_delta_round"),
        a3fl_freeze_after_build=a3fl_section.get("freeze_after_build", True),
        a3fl_build_once_at_attack_start=a3fl_section.get("build_once_at_attack_start", True),
        a3fl_share_delta_across_malicious_clients=a3fl_section.get("share_delta_across_malicious_clients", True),
        a3fl_source=a3fl_section.get("source", "non_target"),
        a3fl_clip_to_valid_rgb=a3fl_section.get("clip_to_valid_rgb", False),
        chameleon_adaptation_epochs=chameleon_section.get("adaptation_epochs", 2),
        chameleon_projection_epochs=chameleon_section.get("projection_epochs", 1),
        chameleon_adaptation_lr=chameleon_section.get("adaptation_lr", 0.005),
        chameleon_projection_lr=chameleon_section.get("projection_lr", 0.01),
        chameleon_momentum=chameleon_section.get("momentum", 0.9),
        chameleon_weight_decay=chameleon_section.get("weight_decay", 0.0005),
        chameleon_temperature=chameleon_section.get("temperature", 0.07),
        chameleon_target_anchor_weight=chameleon_section.get("target_anchor_weight", 6.0),
        chameleon_poison_per_batch=chameleon_section.get("poison_per_batch", 7),
        chameleon_noise_std=chameleon_section.get("noise_std", 0.01),
        chameleon_distance_constraint_enabled=chameleon_distance_constraint_section.get("enabled", False),
        chameleon_distance_constraint_radius=chameleon_distance_constraint_section.get("radius", 5.0),
        f3ba_conv_rate=f3ba_section.get("conv_rate", 0.02),
        f3ba_fc_rate=f3ba_section.get("fc_rate", 0.001),
        f3ba_flip_factor=f3ba_section.get("flip_factor", 1.0),
        f3ba_trigger_step=f3ba_section.get("trigger_step", 0.1),
        f3ba_trigger_epochs=f3ba_section.get("trigger_epochs", 2),
        f3ba_normal_scale=f3ba_section.get("normal_scale", 0.6),
        f3ba_backdoor_scale=f3ba_section.get("backdoor_scale", 0.4),
        f3ba_kernel_selection=f3ba_section.get("kernel_selection", "movement"),
        f3ba_bn_mode=f3ba_section.get("bn_mode", "legacy_eval"),
    )
    total_rounds = validate_experiment_protocol(config, attack_config)
    base_local_epochs = federated_config["local_epochs"]
    configured_malicious_local_epochs = federated_config.get("malicious_local_epochs")
    malicious_local_epochs = (
        base_local_epochs
        if configured_malicious_local_epochs is None
        else configured_malicious_local_epochs
    )
    if type(malicious_local_epochs) is not int or malicious_local_epochs <= 0:
        raise ValueError("federated.malicious_local_epochs must be a positive integer")
    resume_config = config.get("resume", {})
    resume_checkpoint_value = resume_config.get("checkpoint_path")
    resume_delta_value = resume_config.get("adversarial_delta_path")
    resume_delta_round_value = resume_config.get("adversarial_delta_round")
    resume_f3ba_trigger_value = resume_config.get("f3ba_trigger_path")
    resume_round = int(resume_config.get("resume_round", 0))
    if resume_round < 0 or resume_round >= total_rounds:
        raise ValueError("resume.resume_round must be in [0, total_rounds)")
    resume_checkpoint_path = None
    if resume_checkpoint_value is not None:
        resume_checkpoint_path = Path(resume_checkpoint_value)
        if not resume_checkpoint_path.is_absolute():
            resume_checkpoint_path = project_root / resume_checkpoint_path
        resume_checkpoint_path = resume_checkpoint_path.resolve()
        if not resume_checkpoint_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_checkpoint_path}")
    elif resume_round:
        raise ValueError("resume.resume_round requires resume.checkpoint_path")
    resume_delta_path = None
    resume_delta_round = None
    if resume_delta_value is not None:
        resume_delta_path = Path(resume_delta_value)
        if not resume_delta_path.is_absolute():
            resume_delta_path = project_root / resume_delta_path
        resume_delta_path = resume_delta_path.resolve()
        if not resume_delta_path.is_file():
            raise FileNotFoundError(f"resume adversarial delta not found: {resume_delta_path}")
        if attack_config.trigger_type != "adversarial":
            raise ValueError("resume.adversarial_delta_path requires attack.trigger.type='adversarial'")
        if resume_delta_round_value is not None:
            if type(resume_delta_round_value) is not int or resume_delta_round_value <= 0:
                raise ValueError("resume.adversarial_delta_round must be a positive integer or omitted")
            resume_delta_round = resume_delta_round_value
    elif resume_delta_round_value is not None:
        raise ValueError("resume.adversarial_delta_round requires resume.adversarial_delta_path")
    resume_f3ba_trigger_path = None
    if attack_config.method == "focused_flip":
        if attack_config.attack_start_round <= resume_round < attack_config.attack_end_round:
            raise ValueError("F3BA mid-attack resume is unsupported because per-client movement/trigger state is not persisted")
        if resume_round > attack_config.attack_end_round and resume_f3ba_trigger_value is None:
            raise ValueError("post-attack F3BA resume requires resume.f3ba_trigger_path from round 3200")
    if resume_f3ba_trigger_value is not None:
        if attack_config.method != "focused_flip":
            raise ValueError("resume.f3ba_trigger_path requires attack.method='focused_flip'")
        resume_f3ba_trigger_path = Path(resume_f3ba_trigger_value)
        if not resume_f3ba_trigger_path.is_absolute():
            resume_f3ba_trigger_path = project_root / resume_f3ba_trigger_path
        resume_f3ba_trigger_path = resume_f3ba_trigger_path.resolve()
        if not resume_f3ba_trigger_path.is_file():
            raise FileNotFoundError(f"resume F3BA trigger not found: {resume_f3ba_trigger_path}")
    aggregation_weighting = str(federated_config.get("aggregation_weighting", "sample_weighted"))
    bn_buffer_policy = str(federated_config.get("bn_buffer_policy", "aggregate"))
    if aggregation_weighting not in {"sample_weighted", "equal_client"}:
        raise ValueError("federated.aggregation_weighting must be sample_weighted or equal_client")
    if bn_buffer_policy not in {"aggregate", "keep_global"}:
        raise ValueError("federated.bn_buffer_policy must be aggregate or keep_global")
    replay_client_sampling = bool(resume_config.get("replay_client_sampling", False))
    start_round = resume_round + 1
    replay_client_sampling_start_round = int(resume_config.get("replay_client_sampling_start_round", 1))
    if replay_client_sampling_start_round <= 0 or replay_client_sampling_start_round > start_round:
        raise ValueError("resume.replay_client_sampling_start_round must be in [1, start_round]")
    execution_end_round = total_rounds
    if args.stop_after_round is not None:
        if not start_round <= args.stop_after_round <= total_rounds:
            raise ValueError("--stop-after-round must be in [start_round, total_rounds]")
        execution_end_round = args.stop_after_round
    if args.dry_run:
        evaluation = config.get("evaluation", {})
        analysis = config.get("mechanism_analysis", {})
        is_uap_method = attack_config.method in {"ce_uap", "proto_uap", "neurotoxin_uap", "a3fl"}
        print(json.dumps({
            "config": str(config_path), "total_rounds": total_rounds,
            "execution_end_round": execution_end_round,
            "resume_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path else None,
            "resume_round": resume_round, "start_round": start_round,
            "resume_adversarial_delta": str(resume_delta_path) if resume_delta_path else None,
            "resume_adversarial_delta_round": resume_delta_round,
            "resume_f3ba_trigger": str(resume_f3ba_trigger_path) if resume_f3ba_trigger_path else None,
            "replay_client_sampling": replay_client_sampling,
            "replay_client_sampling_start_round": replay_client_sampling_start_round,
            "aggregation_weighting": aggregation_weighting, "bn_buffer_policy": bn_buffer_policy,
            "defense": defense_config,
            "defense_actual_active_rounds": (
                [defense_config["start_round"] or start_round, defense_config["end_round"] or total_rounds]
                if defense_config["enabled"] else []
            ),
            "attack_start": attack_config.attack_start_round, "attack_end": attack_config.attack_end_round,
            "post_attack_start": evaluation.get("post_attack_start"),
            "pa_auc_start": evaluation.get("pa_auc_start"), "pa_auc_end": evaluation.get("pa_auc_end"),
            "eval_rounds": evaluation.get("eval_rounds", []), "target_label": attack_config.target_label,
            "attack_method": attack_config.method,
            "focused_flip": ({
                "kernel_selection": attack_config.f3ba_kernel_selection,
                "conv_rate": attack_config.f3ba_conv_rate,
                "fc_rate": attack_config.f3ba_fc_rate,
                "flip_factor": attack_config.f3ba_flip_factor,
                "trigger_step": attack_config.f3ba_trigger_step,
                "trigger_epochs": attack_config.f3ba_trigger_epochs,
                "normal_scale": attack_config.f3ba_normal_scale,
                "backdoor_scale": attack_config.f3ba_backdoor_scale,
                "bn_mode": attack_config.f3ba_bn_mode,
                "first_attack_round_behavior": "initialize previous global only; no fabricated R2999 movement mask",
                "post_attack_behavior": "benign local training; frozen R3200 averaged trigger for evaluation",
            } if attack_config.method == "focused_flip" else None),
            "chameleon": ({
                "trigger": "type1_post_normalization_255",
                "adaptation_epochs": attack_config.chameleon_adaptation_epochs,
                "projection_epochs": attack_config.chameleon_projection_epochs,
                "adaptation_lr": attack_config.chameleon_adaptation_lr,
                "projection_lr": attack_config.chameleon_projection_lr,
                "momentum": attack_config.chameleon_momentum,
                "weight_decay": attack_config.chameleon_weight_decay,
                "temperature": attack_config.chameleon_temperature,
                "target_anchor_weight": attack_config.chameleon_target_anchor_weight,
                "poison_per_batch": attack_config.chameleon_poison_per_batch,
                "noise_std": attack_config.chameleon_noise_std,
                "distance_constraint_enabled": attack_config.chameleon_distance_constraint_enabled,
                "distance_constraint_radius": attack_config.chameleon_distance_constraint_radius,
            } if attack_config.method == "chameleon" else None),
            "dba_enabled": attack_config.dba_enabled,
            "dba_num_parts": attack_config.dba_num_parts if attack_config.method == "dba" else None,
            "dba_pattern": attack_config.dba_pattern if attack_config.method == "dba" else None,
            "dba_subtrigger_size": attack_config.dba_subtrigger_size if attack_config.method == "dba" else None,
            "dba_poison_loss_weight": attack_config.dba_poison_loss_weight if attack_config.method == "dba" else None,
            "dba_trigger_only_on_poison": True if attack_config.method == "dba" else None,
            "dba_expected_num_attack_samples": 9000 if attack_config.method == "dba" else None,
            "asr_expected_num_attack_samples": 9000 if attack_config.enabled else None,
            "post_attack_behavior": "benign_ce_only; no_poison; no_trigger; no_label_change",
            "dba_training_trigger": "per-malicious-client sub-trigger" if attack_config.method == "dba" else None,
            "dba_evaluation_trigger": "full trigger" if attack_config.method == "dba" else None,
            "neurotoxin_enabled": attack_config.neurotoxin_enabled,
            "neurotoxin_mask_enabled": (
                attack_config.neurotoxin_enabled
                if attack_config.method == "neurotoxin_uap" else None
            ),
            "neurotoxin_mask_ratio": attack_config.neurotoxin_mask_ratio,
            "neurotoxin_mask_apply_mode": attack_config.neurotoxin_mask_apply_mode if attack_config.method == "neurotoxin_uap" else None,
            "neurotoxin_selected_direction": "high_gradient (masked; low-gradient coordinates kept)" if attack_config.method == "neurotoxin_uap" else None,
            "neurotoxin_effective_keep_ratio": (1.0 - attack_config.neurotoxin_mask_ratio) if attack_config.method == "neurotoxin_uap" else None,
            "neurotoxin_uap": "one shared delta, generated on first attack-active malicious client and reused for ASR" if attack_config.method == "neurotoxin_uap" else None,
            "neurotoxin_uap_epsilon": attack_config.adversarial_epsilon if attack_config.method == "neurotoxin_uap" else None,
            "neurotoxin_uap_steps": attack_config.adversarial_steps if attack_config.method == "neurotoxin_uap" else None,
            "neurotoxin_uap_alpha": attack_config.adversarial_alpha if attack_config.method == "neurotoxin_uap" else None,
            "neurotoxin_asr_evaluation": "same fixed UAP is retained and used after attack_end" if attack_config.method == "neurotoxin_uap" else None,
            "a3fl_enabled": attack_config.a3fl_enabled if attack_config.method == "a3fl" else None,
            "a3fl_variant": attack_config.a3fl_variant if attack_config.method == "a3fl" else None,
            "a3fl_warm_start_delta": attack_config.a3fl_warm_start_delta if attack_config.method == "a3fl" else None,
            "a3fl_lambda_adv": attack_config.a3fl_lambda_adv if attack_config.method == "a3fl" else None,
            "a3fl_lambda_mode": attack_config.a3fl_lambda_mode if attack_config.method == "a3fl" else None,
            "a3fl_lambda0": attack_config.a3fl_lambda0 if attack_config.method == "a3fl" else None,
            "a3fl_lambda_min": attack_config.a3fl_lambda_min if attack_config.method == "a3fl" else None,
            "a3fl_lambda_max": attack_config.a3fl_lambda_max if attack_config.method == "a3fl" else None,
            "a3fl_adv_model_steps": attack_config.a3fl_adv_model_steps if attack_config.method == "a3fl" else None,
            "a3fl_adv_model_lr": attack_config.a3fl_adv_model_lr if attack_config.method == "a3fl" else None,
            "a3fl_adv_model_momentum": attack_config.a3fl_adv_model_momentum if attack_config.method == "a3fl" else None,
            "a3fl_adv_model_weight_decay": attack_config.a3fl_adv_model_weight_decay if attack_config.method == "a3fl" else None,
            "a3fl_outer_steps": attack_config.a3fl_outer_steps if attack_config.method == "a3fl" else None,
            "a3fl_delta_steps": attack_config.a3fl_delta_steps if attack_config.method == "a3fl" else None,
            "a3fl_k_trigger": attack_config.a3fl_k_trigger if attack_config.method == "a3fl" else None,
            "a3fl_adv_updates_per_outer": attack_config.a3fl_adv_updates_per_outer if attack_config.method == "a3fl" else None,
            "a3fl_refresh_policy": attack_config.a3fl_refresh_policy if attack_config.method == "a3fl" else None,
            "a3fl_refresh_period": attack_config.a3fl_refresh_period if attack_config.method == "a3fl" else None,
            "a3fl_freeze_delta_after_attack": attack_config.a3fl_freeze_delta_after_attack if attack_config.method == "a3fl" else None,
            "a3fl_save_delta_every_refresh": attack_config.a3fl_save_delta_every_refresh if attack_config.method == "a3fl" else None,
            "a3fl_clip_to_valid_rgb": attack_config.a3fl_clip_to_valid_rgb if attack_config.method == "a3fl" else None,
            "a3fl_uap": (
                "temporary adversarial model; shared delta refreshed every attack round from current global model"
                if attack_config.method == "a3fl" and attack_config.a3fl_refresh_policy == "every_round"
                else "temporary adversarial model; shared delta refreshed periodically from current global model"
                if attack_config.method == "a3fl" and attack_config.a3fl_refresh_policy == "periodic"
                else "temporary adversarial model; one fixed shared delta generated at first attack-active client"
                if attack_config.method == "a3fl" else None
            ),
            "uap_epsilon": attack_config.adversarial_epsilon if is_uap_method else None,
            "uap_steps": attack_config.adversarial_steps if is_uap_method else None,
            "uap_alpha": attack_config.adversarial_alpha if is_uap_method else None,
            "uap_asr_evaluation": (
                "last attack-round UAP is frozen and used after attack_end"
                if attack_config.method == "a3fl" else "same fixed UAP is retained and used after attack_end"
            ) if is_uap_method else None,
            "num_clients": partition_config["num_clients"], "clients_per_round": federated_config["clients_per_round"],
            "partition_type": partition_config.get("type", "iid"),
            "dirichlet_alpha": partition_config.get("alpha"),
            "samples_per_client": partition_config.get("samples_per_client"),
            "min_classes_per_client": partition_config.get("min_classes_per_client"),
            "min_samples_per_client": partition_config.get("min_samples_per_client"),
            "max_partition_retries": partition_config.get("max_retries"),
            "malicious_client_ids": attack_config.malicious_client_ids,
            "num_malicious_clients": attack_config.num_malicious_clients,
            "selection_policy": attack_config.selection_policy,
            "min_malicious_per_attack_round": attack_config.min_malicious_per_attack_round,
            "prototype_build_policy": attack_config.prototype_build_policy,
            "prototype_source": attack_config.prototype_source,
            "prototype_client_id": attack_config.prototype_client_id,
            "aggregate_prototypes": attack_config.prototype_aggregate,
            "contributing_clients": attack_config.malicious_client_ids if attack_config.prototype_aggregate else [],
            "freeze_after_build": attack_config.prototype_freeze_after_build,
            "refresh_policy": attack_config.prototype_refresh_policy,
            "refresh_during": attack_config.prototype_refresh_during,
            "feature_layer": attack_config.prototype_feature_layer,
            "prototype_contrastive_enabled": attack_config.prototype_contrastive_enabled,
            "prototype_use_in_delta_generation": attack_config.prototype_use_in_delta_generation,
            "prototype_use_in_malicious_training": attack_config.prototype_use_in_malicious_training,
            "prototype_lambda_delta": attack_config.prototype_lambda_delta,
            "prototype_lambda_train": attack_config.prototype_lambda_train,
            "prototype_min_samples_per_class": attack_config.prototype_min_samples_per_class,
            "prototype_source_margin": attack_config.prototype_source_margin,
            "prototype_temperature": attack_config.prototype_temperature,
            "uap_generation": (
                "shared A3FL delta refreshed according to a3fl.refresh_policy"
                if attack_config.enabled and attack_config.method == "a3fl"
                else "one shared delta, generated on first attack-active malicious client"
                if attack_config.enabled and is_uap_method else None
            ),
            "prototype_bank": (
                "fixed single build" if attack_config.prototype_refresh_policy == "fixed" else "refreshable"
            ) if attack_config.enabled and attack_config.prototype_contrastive_enabled else None,
            "poison_ratio": attack_config.poison_ratio,
            "local_epochs": base_local_epochs,
            "malicious_local_epochs": malicious_local_epochs,
            "client_lr": training_config["learning_rate"],
            "seed": experiment_config["seed"],
            "defense": {
                "enabled": defense_config["enabled"],
                "name": defense_config["name"],
                "start_round": defense_config["start_round"],
                "end_round": defense_config["end_round"],
                "params": defense_config["params"],
                "scope": defense_config["scope"],
            },
            "multi_krum": ({
                "f": defense_config["f"], "m": defense_config["m"],
                "n": federated_config["clients_per_round"],
                "nearest_count": federated_config["clients_per_round"] - defense_config["f"] - 2,
            } if defense_config["name"] == "multi_krum" else None),
            "flame": ({
                "clustering": defense_config["clustering"],
                "clip_mode": defense_config["clip_mode"],
                "noise_std": defense_config["noise_std"],
            } if defense_config["name"] == "flame" else None),
            "mechanism_analysis.enabled": analysis.get("enabled", False),
            "mechanism_analysis.compute_fixed_margin": analysis.get("compute_fixed_margin", True),
            "mechanism_analysis.compute_dynamic_margin": analysis.get("compute_dynamic_margin", True),
            "mechanism_analysis.max_batches": analysis.get("max_batches", 10),
        }, indent=2))
        return
    invalid_malicious_client_ids = [
        client_id
        for client_id in attack_config.malicious_client_ids
        if client_id >= partition_config["num_clients"]
    ]
    if invalid_malicious_client_ids:
        raise ValueError(
            "malicious_client_ids contain IDs outside the configured client range: "
            f"{invalid_malicious_client_ids}"
        )
    if attack_config.target_label < 0 or attack_config.target_label >= dataset_config["num_classes"]:
        raise ValueError(
            "target_label must be smaller than dataset.num_classes: "
            f"{attack_config.target_label} >= {dataset_config['num_classes']}"
        )
    dataset_root = Path(dataset_config["root"])
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    dataset_root = dataset_root.resolve()
    output_dir = Path(experiment_config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_rounds = set(checkpoint_config.get("save_rounds", [])) if checkpoint_config.get("enabled", False) else set()
    delta_save_rounds = set(checkpoint_config.get("delta_save_rounds", [])) if checkpoint_config.get("enabled", False) else set()
    checkpoint_dir = output_dir / "checkpoints"

    seed = experiment_config["seed"]
    set_seed(seed)
    device = resolve_device(config["device"]["type"])
    MaliciousClient.reset_shared_delta()
    FocusedFlipClient.reset_shared_state()
    print("[Train] Reset shared adversarial delta/prototype bank and F3BA trigger state.")
    if resume_delta_path is not None:
        resumed_delta = torch.load(resume_delta_path, map_location="cpu", weights_only=True)
        if not isinstance(resumed_delta, torch.Tensor):
            raise TypeError("resume adversarial delta must be a tensor")
        MaliciousClient.shared_adversarial_delta = resumed_delta.detach().cpu().clone()
        MaliciousClient.shared_adversarial_delta_round = resume_delta_round
        print(
            f"[Resume] Loaded fixed adversarial delta: {resume_delta_path} "
            f"source_round={resume_delta_round}",
            flush=True,
        )
    if resume_f3ba_trigger_path is not None:
        f3ba_payload = torch.load(resume_f3ba_trigger_path, map_location="cpu", weights_only=True)
        if not isinstance(f3ba_payload, dict) or not isinstance(f3ba_payload.get("pattern"), torch.Tensor):
            raise TypeError("resume F3BA trigger must be a mapping containing tensor key 'pattern'")
        frozen_round = f3ba_payload.get("frozen_round", attack_config.attack_end_round)
        if frozen_round != attack_config.attack_end_round:
            raise ValueError("resume F3BA trigger must be frozen at attack.end_round")
        FocusedFlipClient.restore_frozen_pattern(f3ba_payload["pattern"], frozen_round)
        print(f"[Resume] Loaded frozen F3BA trigger: {resume_f3ba_trigger_path}", flush=True)
    print(f"Device: {device}")
    print(f"Attack enabled: {attack_config.enabled}")
    print(
        f"Local epochs: base={base_local_epochs}, malicious_schedule={malicious_local_epochs}",
        flush=True,
    )
    if not attack_config.enabled:
        print(
            "[CleanFedAvg] attack disabled; all clients use CE-only local training; "
            "ASR/UAP/trigger/prototype paths are disabled.",
            flush=True,
        )
    if attack_config.enabled:
        attack_end_round = (
            attack_config.attack_end_round
            if attack_config.attack_end_round is not None
            else "end"
        )
        print(f"Target label: {attack_config.target_label}")
        print(f"Malicious clients: {attack_config.malicious_client_ids}")
        print(f"Poison ratio: {attack_config.poison_ratio}")
        print(
            f"Attack rounds: {attack_config.attack_start_round}-{attack_end_round}"
        )
        print(
            "Prototype contrastive: "
            f"{attack_config.prototype_contrastive_enabled}"
        )

    recorder = None
    if tracking_config.get("enabled", False):
        database_path = output_dir / tracking_config.get("database_name", "metrics.db")
        recorder = ExperimentRecorder(database_path)
        print(f"Metrics database: {database_path}")

    dataset_name = dataset_config.get("name", "cifar10").lower()
    dataset_spec = get_dataset_spec(dataset_name)
    if dataset_config["num_classes"] != dataset_spec["num_classes"]:
        raise ValueError(
            f"dataset.num_classes for {dataset_name} must be "
            f"{dataset_spec['num_classes']}"
        )
    train_dataset, test_dataset = load_dataset(
        name=dataset_name,
        root=dataset_root,
        download=dataset_config.get("download", False),
    )
    partition_type = partition_config.get("type", "iid")
    dataset_labels = getattr(train_dataset, "targets", None)
    if partition_type == "iid":
        client_partitions = partition_iid(len(train_dataset), partition_config["num_clients"], seed)
    elif partition_type == "dirichlet":
        if dataset_labels is None:
            raise ValueError("Dirichlet partition requires a dataset with labels/targets")
        client_partitions = partition_dirichlet(
            dataset_labels, partition_config["num_clients"], partition_config["alpha"], seed,
            partition_config.get("min_samples_per_client", 1), partition_config.get("max_retries", 100),
        )
    elif partition_type == "dirichlet_balanced":
        if dataset_labels is None:
            raise ValueError("balanced Dirichlet partition requires a dataset with labels/targets")
        client_partitions = partition_dirichlet_balanced(
            dataset_labels, partition_config["num_clients"], partition_config["alpha"], seed,
            partition_config["samples_per_client"],
            partition_config.get("min_classes_per_client", 1),
            partition_config.get("max_retries", 100),
        )
    else:
        raise ValueError("partition.type must be 'iid', 'dirichlet', or 'dirichlet_balanced'")
    partition_sizes = [len(indices) for indices in client_partitions.values()]
    print(f"Partition: type={partition_type}, min={min(partition_sizes)}, max={max(partition_sizes)}, mean={sum(partition_sizes) / len(partition_sizes):.2f}")
    partition_histograms = (
        {str(client_id): dict(sorted(Counter(dataset_labels[index] for index in indices).items())) for client_id, indices in client_partitions.items()}
        if dataset_labels is not None else None
    )
    # Persist a compact, paper-ready description of every non-IID split.  The
    # partition is deterministic for a given seed/alpha, so CE-UAP and PBU
    # safely write the same alpha file when they share the protocol.
    if partition_type in {"dirichlet", "dirichlet_balanced"} and partition_histograms is not None:
        summary_dir_value = partition_config.get("summary_dir")
        if summary_dir_value:
            summary_dir = Path(summary_dir_value)
            if not summary_dir.is_absolute():
                summary_dir = project_root / summary_dir
            summary_dir.mkdir(parents=True, exist_ok=True)
            alpha_value = float(partition_config["alpha"])
            alpha_tag = f"{alpha_value:.1f}"
            distribution_path = summary_dir / f"partition_alpha{alpha_tag}_label_distribution.csv"
            distribution_rows = []
            for client_id, histogram in partition_histograms.items():
                counts = [int(histogram.get(class_id, histogram.get(str(class_id), 0))) for class_id in range(dataset_config["num_classes"])]
                total = sum(counts)
                dominant_class = int(np.argmax(counts))
                distribution_rows.append({
                    "alpha": alpha_value, "client_id": int(client_id),
                    **{f"class_{class_id}": counts[class_id] for class_id in range(dataset_config["num_classes"])},
                    "total_samples": total, "dominant_class": dominant_class,
                    "dominant_ratio": (counts[dominant_class] / total) if total else 0.0,
                })
            distribution_rows.sort(key=lambda row: row["client_id"])
            distribution_fields = ["alpha", "client_id"] + [f"class_{i}" for i in range(dataset_config["num_classes"])] + ["total_samples", "dominant_class", "dominant_ratio"]
            with distribution_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=distribution_fields)
                writer.writeheader(); writer.writerows(distribution_rows)
            summary_path = summary_dir / "partition_alpha_summary.csv"
            summary_rows = []
            if summary_path.exists():
                with summary_path.open(newline="", encoding="utf-8") as stream:
                    summary_rows = list(csv.DictReader(stream))
            sample_counts = np.asarray([row["total_samples"] for row in distribution_rows], dtype=float)
            dominant_ratios = np.asarray([row["dominant_ratio"] for row in distribution_rows], dtype=float)
            summary_row = {
                "alpha": alpha_value, "num_clients": len(distribution_rows),
                "min_client_samples": int(sample_counts.min()), "max_client_samples": int(sample_counts.max()),
                "mean_client_samples": float(sample_counts.mean()), "std_client_samples": float(sample_counts.std()),
                "mean_dominant_ratio": float(dominant_ratios.mean()), "max_dominant_ratio": float(dominant_ratios.max()),
                "clients_missing_any_class_count": sum(any(row[f"class_{i}"] == 0 for i in range(dataset_config["num_classes"])) for row in distribution_rows),
                "notes": (
                    f"Balanced Dirichlet label-skew; seed={seed}; "
                    f"samples_per_client={partition_config['samples_per_client']}; "
                    f"min_classes_per_client={partition_config.get('min_classes_per_client', 1)}; "
                    f"max_retries={partition_config.get('max_retries', 100)}"
                    if partition_type == "dirichlet_balanced" else
                    f"Dirichlet label-skew; seed={seed}; "
                    f"min_samples_per_client={partition_config.get('min_samples_per_client', 1)}; "
                    f"max_retries={partition_config.get('max_retries', 100)}"
                ),
            }
            summary_rows = [row for row in summary_rows if float(row["alpha"]) != alpha_value] + [summary_row]
            summary_fields = list(summary_row)
            with summary_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=summary_fields)
                writer.writeheader(); writer.writerows(sorted(summary_rows, key=lambda row: float(row["alpha"])))
    metadata = {
        "method": attack_config.method, "seed": seed, "partition_type": partition_type,
        "chameleon": ({
            "trigger": "type1_post_normalization_255",
            "coordinates": [[29, 29], [30, 28], [28, 30], [30, 30]],
            "adaptation_epochs": attack_config.chameleon_adaptation_epochs,
            "projection_epochs": attack_config.chameleon_projection_epochs,
            "adaptation_lr": attack_config.chameleon_adaptation_lr,
            "projection_lr": attack_config.chameleon_projection_lr,
            "momentum": attack_config.chameleon_momentum,
            "weight_decay": attack_config.chameleon_weight_decay,
            "temperature": attack_config.chameleon_temperature,
            "target_anchor_weight": attack_config.chameleon_target_anchor_weight,
            "poison_per_batch": attack_config.chameleon_poison_per_batch,
            "noise_std": attack_config.chameleon_noise_std,
            "adaptation": "encoder_only_train_mode_bn",
            "distance_constraint_enabled": attack_config.chameleon_distance_constraint_enabled,
            "distance_constraint_radius": attack_config.chameleon_distance_constraint_radius,
            "projection": "classifier_only_train_mode_bn",
        } if attack_config.method == "chameleon" else None),
        "dataset": dataset_name, "num_classes": dataset_config["num_classes"],
        "dirichlet_alpha": partition_config.get("alpha"),
        "samples_per_client": partition_config.get("samples_per_client"),
        "min_classes_per_client": partition_config.get("min_classes_per_client"),
        "target_label": attack_config.target_label,
        "num_clients": partition_config["num_clients"], "clients_per_round": federated_config["clients_per_round"],
        "malicious_client_ids": attack_config.malicious_client_ids, "num_malicious_clients": len(attack_config.malicious_client_ids),
        "attack_start_round": attack_config.attack_start_round, "attack_end_round": attack_config.attack_end_round,
        "resume_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path else None,
        "resume_round": resume_round, "start_round": start_round,
        "resume_adversarial_delta": str(resume_delta_path) if resume_delta_path else None,
        "resume_adversarial_delta_round": resume_delta_round,
        "poison_ratio": attack_config.poison_ratio, "epsilon": attack_config.adversarial_epsilon,
        "steps": attack_config.adversarial_steps, "alpha": attack_config.adversarial_alpha,
        "lambda_delta": attack_config.prototype_lambda_delta, "lambda_train": attack_config.prototype_lambda_train,
        "center_batches": attack_config.prototype_center_batches, "feature_layer": attack_config.prototype_feature_layer,
        "neurotoxin_mask_ratio": attack_config.neurotoxin_mask_ratio,
        "neurotoxin_mask_apply_mode": attack_config.neurotoxin_mask_apply_mode,
        "a3fl_refresh_policy": attack_config.a3fl_refresh_policy if attack_config.method == "a3fl" else None,
        "a3fl_refresh_period": attack_config.a3fl_refresh_period if attack_config.method == "a3fl" else None,
        "a3fl_variant": attack_config.a3fl_variant if attack_config.method == "a3fl" else None,
        "a3fl_warm_start_delta": attack_config.a3fl_warm_start_delta if attack_config.method == "a3fl" else None,
        "a3fl_delta_steps": attack_config.a3fl_delta_steps if attack_config.method == "a3fl" else None,
        "a3fl_lambda_mode": attack_config.a3fl_lambda_mode if attack_config.method == "a3fl" else None,
        "a3fl_lambda0": attack_config.a3fl_lambda0 if attack_config.method == "a3fl" else None,
        "a3fl_lambda_min": attack_config.a3fl_lambda_min if attack_config.method == "a3fl" else None,
        "a3fl_lambda_max": attack_config.a3fl_lambda_max if attack_config.method == "a3fl" else None,
        "a3fl_adv_model_steps": attack_config.a3fl_adv_model_steps if attack_config.method == "a3fl" else None,
        "a3fl_adv_model_momentum": attack_config.a3fl_adv_model_momentum if attack_config.method == "a3fl" else None,
        "a3fl_adv_model_weight_decay": attack_config.a3fl_adv_model_weight_decay if attack_config.method == "a3fl" else None,
        "a3fl_outer_steps": attack_config.a3fl_outer_steps if attack_config.method == "a3fl" else None,
        "a3fl_k_trigger": attack_config.a3fl_k_trigger if attack_config.method == "a3fl" else None,
        "a3fl_adv_updates_per_outer": attack_config.a3fl_adv_updates_per_outer if attack_config.method == "a3fl" else None,
        "a3fl_lambda_adv": attack_config.a3fl_lambda_adv if attack_config.method == "a3fl" else None,
        "a3fl_post_eval_delta_round": attack_config.a3fl_post_eval_delta_round if attack_config.method == "a3fl" else None,
        "a3fl_freeze_delta_after_attack": attack_config.a3fl_freeze_delta_after_attack if attack_config.method == "a3fl" else None,
        "num_subtriggers": len(attack_config.malicious_client_ids) if attack_config.method == "dba" else None,
        "dba_client_to_subtrigger": ({str(client_id): index for index, client_id in enumerate(attack_config.malicious_client_ids)} if attack_config.method == "dba" else None),
        "partition_class_histograms": partition_histograms,
        "aggregation_weighting": aggregation_weighting,
        "bn_buffer_policy": bn_buffer_policy,
        "defense": defense_config,
        "defense_actual_active_rounds": (
            [max(start_round, defense_config["start_round"] or start_round), min(total_rounds, defense_config["end_round"] or total_rounds)]
            if defense_config["enabled"] else []
        ),
        "replay_client_sampling": replay_client_sampling,
        "replay_client_sampling_start_round": replay_client_sampling_start_round,
        "execution_end_round": execution_end_round,
        "resume_adversarial_delta": str(resume_delta_path) if resume_delta_path else None,
        "resume_f3ba_trigger": str(resume_f3ba_trigger_path) if resume_f3ba_trigger_path else None,
    }
    (output_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
    )

    global_model = build_resnet18(num_classes=dataset_config["num_classes"])
    if resume_checkpoint_path is not None:
        resumed_state = torch.load(resume_checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(resumed_state, dict):
            raise TypeError("resume checkpoint must contain a state_dict mapping")
        global_model.load_state_dict(resumed_state, strict=True)
        print(
            f"[Resume] Loaded global model checkpoint: {resume_checkpoint_path} "
            f"(completed round {resume_round}; next round {start_round})",
            flush=True,
        )
    clients = []
    malicious_client_count = 0
    for client_id, indices in client_partitions.items():
        client_kwargs = {
            "client_id": client_id,
            "dataset": train_dataset,
            "indices": indices,
            "batch_size": training_config["batch_size"],
            "local_epochs": federated_config["local_epochs"],
            "learning_rate": training_config["learning_rate"],
            "momentum": training_config["momentum"],
            "weight_decay": training_config["weight_decay"],
            "num_workers": training_config["num_workers"],
            "device": device,
        }
        if attack_config.enabled and attack_config.is_malicious_client(client_id):
            malicious_client_class = (ChameleonMaliciousClient if attack_config.method == "chameleon" else FocusedFlipClient if attack_config.method == "focused_flip" else MaliciousClient)
            clients.append(malicious_client_class(
                **client_kwargs,
                attack_config=attack_config,
            ))
            malicious_client_count += 1
        else:
            clients.append(Client(**client_kwargs))
    print(
        f"Clients initialized: {len(clients)} total, "
        f"{malicious_client_count} malicious"
    )
    server = Server(
        global_model=global_model,
        clients=clients,
        test_loader=test_loader,
        num_rounds=execution_end_round,
        clients_per_round=federated_config["clients_per_round"],
        device=device,
        seed=seed,
        start_round=start_round,
        attack_config=attack_config,
        visualize_round=(
            visualization_config.get("round")
            if visualization_config.get("enabled", False)
            else None
        ),
        visualization_num_samples=visualization_config.get("num_samples", 5),
        visualization_output_dir=output_dir,
        normalization_mean=dataset_spec["mean"],
        normalization_std=dataset_spec["std"],
        defense_config=defense_config,
        mechanism_analysis=config.get("mechanism_analysis", {}),
        malicious_local_epochs=malicious_local_epochs,
        aggregation_weighting=aggregation_weighting,
        bn_buffer_policy=bn_buffer_policy,
        replay_client_sampling=replay_client_sampling,
        replay_client_sampling_start_round=replay_client_sampling_start_round,
    )

    for result in server.run():
        if recorder is not None:
            recorder.record_round(result)
            if tracking_config.get("record_client_metrics", True):
                recorder.record_clients(
                    round_index=result["round"],
                    client_results=result["client_metrics"],
                )
        if attack_config.method == "focused_flip":
            f3ba_records = [metric for metric in result["client_metrics"] if "f3ba" in metric]
            if f3ba_records:
                with (output_dir / "f3ba_diagnostics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"round": result["round"], "attack_active": result["attack_active"], "clients": f3ba_records}, sort_keys=True) + "\n")
        if result.get("feature_margins") and config.get("mechanism_analysis", {}).get("save_csv", True):
            margin_path = output_dir / "feature_margins.csv"
            fieldnames = ["round", "method", "seed", "target_label", "clean_acc", "asr",
                          "fixed_target_sim", "fixed_source_sim", "fixed_margin",
                          "dynamic_target_sim", "dynamic_source_sim", "dynamic_margin",
                          "prototype_version", "feature_layer", "num_eval_samples",
                          "num_source_samples", "num_target_excluded"]
            row = {field: None for field in fieldnames}
            row.update({"round": result["round"], "method": attack_config.method, "seed": seed,
                   "target_label": attack_config.target_label, "clean_acc": result["clean_accuracy"],
                   "asr": result["attack_success_rate"], "prototype_version": server.prototype_version,
                   "feature_layer": attack_config.prototype_feature_layer, **result["feature_margins"]})
            write_header = not margin_path.exists()
            with margin_path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
                if write_header: writer.writeheader()
                writer.writerow(row)
        save_a3fl_refresh_delta = bool(
            attack_config.method == "a3fl"
            and attack_config.a3fl_save_delta_every_refresh
            and result.get("a3fl_delta_refreshed", False)
        )
        if result["round"] in checkpoint_rounds or result["round"] in delta_save_rounds or save_a3fl_refresh_delta:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            round_index = result["round"]
            if result["round"] in checkpoint_rounds and checkpoint_config.get("save_global_model", True):
                checkpoint_path = checkpoint_dir / f"global_round_{round_index}.pt"
                torch.save(
                    server.global_model.state_dict(),
                    checkpoint_path,
                )
                print(
                    f"[Checkpoint] Saved global model: round={round_index} path={checkpoint_path}",
                    flush=True,
                )
            if checkpoint_config.get("save_adversarial_delta", False) and (
                    result["round"] in checkpoint_rounds or result["round"] in delta_save_rounds or save_a3fl_refresh_delta):
                delta = MaliciousClient.shared_adversarial_delta
                if delta is not None:
                    torch.save(
                        delta.detach().cpu(),
                        checkpoint_dir / f"delta_round_{round_index}.pt",
                    )
                    print(
                        f"[Checkpoint] Saved adversarial delta: round={round_index} "
                        f"source_round={MaliciousClient.shared_adversarial_delta_round} "
                        f"path={checkpoint_dir / f'delta_round_{round_index}.pt'}",
                        flush=True,
                    )
            if attack_config.method == "focused_flip":
                pattern = FocusedFlipClient.evaluation_pattern()
                if pattern is not None:
                    torch.save({"pattern": pattern.detach().cpu(), "frozen_round": FocusedFlipClient.frozen_round, "mask_position": [3, 6, 23, 26]}, checkpoint_dir / f"f3ba_trigger_round_{round_index}.pt")
            if checkpoint_config.get("save_prototype_bank", False):
                bank = MaliciousClient.shared_prototype_bank
                if bank is not None:
                    serializable = {
                        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                        for key, value in bank.items()
                    }
                    torch.save(serializable, checkpoint_dir / f"prototype_bank_round_{round_index}.pt")
        log_message = (
            f"Round {result['round']} | "
            f"Train Loss: {result['average_train_loss']:.4f} | "
            f"Test Loss: {result['test_loss']:.4f} | "
            f"Clean ACC: {result['clean_accuracy']:.2f}%"
        )
        log_message += (f" | Selected malicious: {result['selected_malicious_clients']}"
                        f" ({result['num_selected_malicious']}) | Attack active: {result['attack_active']}")
        if result.get("attack_success_rate") is not None:
            log_message += f" | ASR: {result['attack_success_rate']:.2f}%"
        else:
            log_message += " | ASR: N/A"
        psnr = result.get("psnr")
        ssim = result.get("ssim")
        linf = result.get("linf")
        log_message += (
            f" | PSNR: {psnr:.2f} dB" if psnr is not None else " | PSNR: N/A"
        )
        log_message += f" | SSIM: {ssim:.3f}" if ssim is not None else " | SSIM: N/A"
        log_message += f" | L_inf: {linf:.3f}" if linf is not None else " | L_inf: N/A"
        print(log_message, flush=True)


if __name__ == "__main__":
    main()

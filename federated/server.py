"""Server-side coordination for benign serial FedAvg training."""

import random
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from attacks.triggers import apply_trigger
from attacks.dba import apply_dba_trigger
from attacks.chameleon_trigger import apply_chameleon_type1_trigger
from attacks.neurotoxin import build_neurotoxin_mask
from evaluation.evaluator import (
    compute_stealth_metrics,
    evaluate_backdoor_asr,
    evaluate_model,
    evaluate_feature_margins,
)
from federated.aggregation import aggregate_with_defense
from federated.malicious_client import MaliciousClient
from attacks.prototype import aggregate_class_statistics, collect_class_statistics, enforce_strict_prototype_coverage
from utils.trigger_visualizer import save_trigger_comparison


def _is_malicious_client(client) -> bool:
    """Use the common malicious-client marker without coupling attack classes."""
    return bool(getattr(client, "is_malicious", isinstance(client, MaliciousClient)))


class Server:
    """Coordinate client selection, aggregation, and global evaluation."""

    def __init__(
        self,
        global_model: nn.Module,
        clients: list,
        test_loader: DataLoader,
        num_rounds: int,
        clients_per_round: int,
        device: torch.device,
        seed: int,
        start_round: int = 1,
        attack_config=None,
        visualize_round: int | None = None,
        visualization_num_samples: int = 5,
        visualization_output_dir: str | Path = "outputs",
        normalization_mean=None,
        normalization_std=None,
        defense_config: dict | None = None,
        mechanism_analysis: dict | None = None,
        malicious_local_epochs: int | None = None,
        aggregation_weighting: str = "sample_weighted",
        bn_buffer_policy: str = "aggregate",
        replay_client_sampling: bool = False,
        replay_client_sampling_start_round: int = 1,
    ) -> None:
        """Initialize the server and deterministic client sampler."""
        if not isinstance(clients, list):
            raise TypeError("clients must be a list")
        if not clients:
            raise ValueError("clients must not be empty")
        if type(num_rounds) is not int:
            raise TypeError("num_rounds must be an integer")
        if num_rounds <= 0:
            raise ValueError("num_rounds must be positive")
        if type(clients_per_round) is not int:
            raise TypeError("clients_per_round must be an integer")
        if clients_per_round <= 0:
            raise ValueError("clients_per_round must be positive")
        if clients_per_round > len(clients):
            raise ValueError("clients_per_round cannot exceed the number of clients")
        if type(seed) is not int:
            raise TypeError("seed must be an integer")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if type(start_round) is not int or start_round <= 0:
            raise ValueError("start_round must be a positive integer")
        if start_round > num_rounds + 1:
            raise ValueError("start_round cannot be greater than num_rounds + 1")
        if type(replay_client_sampling_start_round) is not int or replay_client_sampling_start_round <= 0:
            raise ValueError("replay_client_sampling_start_round must be a positive integer")
        if replay_client_sampling_start_round > start_round:
            raise ValueError("replay_client_sampling_start_round cannot exceed start_round")
        if visualize_round is not None and (
            type(visualize_round) is not int or visualize_round <= 0
        ):
            raise ValueError("visualize_round must be a positive integer or None")
        if type(visualization_num_samples) is not int or visualization_num_samples <= 0:
            raise ValueError("visualization_num_samples must be a positive integer")
        if malicious_local_epochs is not None and (
            type(malicious_local_epochs) is not int or malicious_local_epochs <= 0
        ):
            raise ValueError("malicious_local_epochs must be a positive integer or None")

        self.global_model = global_model.to(device)
        self.clients = clients
        self.test_loader = test_loader
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        self.device = device
        self.seed = seed
        self.start_round = start_round
        self.attack_config = attack_config
        self.malicious_local_epochs = malicious_local_epochs
        self.visualize_round = visualize_round
        self.visualization_num_samples = visualization_num_samples
        self.visualization_output_dir = Path(visualization_output_dir)
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.rng = random.Random(seed)
        self.defense_config = defense_config or {"enabled": False, "method": "none"}
        self.aggregation_weighting = aggregation_weighting
        self.bn_buffer_policy = bn_buffer_policy
        if replay_client_sampling:
            for prior_round in range(replay_client_sampling_start_round, start_round):
                self.select_clients(prior_round)
        self.neurotoxin_enabled = getattr(attack_config, "method", None) == "neurotoxin_uap"
        self.previous_global_state = None
        self.current_global_state = None
        self.prototype_version = 0
        self.prototype_events = []
        self.mechanism_analysis = mechanism_analysis or {"enabled": False}
        self.fixed_anchor_prototype_bank = None
        images, _ = next(iter(self.test_loader))
        if images.size(0) == 0:
            raise ValueError("test_loader must provide at least one test image")
        self.stealth_images = images[:visualization_num_samples].to(self.device)

    def _apply_trigger_to_stealth_images(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Apply the configured trigger to the fixed stealth-evaluation images."""
        trigger_type = getattr(self.attack_config, "trigger_type", "patch")
        images = self.stealth_images
        if getattr(self.attack_config, "method", None) == "dba":
            triggered_images = apply_dba_trigger(
                images, 0, self.attack_config.dba_num_parts,
                patch_size=self.attack_config.patch_size,
                pixel_value=self.attack_config.pixel_value,
                full_trigger=True,
                pattern=self.attack_config.dba_pattern,
                subtrigger_size=self.attack_config.dba_subtrigger_size,
                normalized_trigger=self.attack_config.dba_normalized_trigger,
            )
            return images, triggered_images
        if trigger_type == "patch":
            trigger_indices = torch.arange(images.size(0), device=self.device)
            triggered_images = apply_trigger(
                images,
                trigger_type="patch",
                poison_indices=trigger_indices,
                patch_size=getattr(self.attack_config, "patch_size", 3),
                pixel_value=getattr(self.attack_config, "pixel_value", 1.0),
            )
            return images, triggered_images
        if trigger_type == "chameleon":
            return images, apply_chameleon_type1_trigger(images)
        if trigger_type == "focused_flip":
            from federated.focused_flip_client import FocusedFlipClient
            from attacks.focused_flip import apply_pattern
            pattern = FocusedFlipClient.evaluation_pattern()
            if pattern is None:
                return None
            mask = torch.zeros_like(pattern); mask[:, 3:6, 23:26] = 1
            return images, apply_pattern(images, pattern.to(images), mask.to(images))
        if trigger_type == "adversarial":
            delta = MaliciousClient.shared_adversarial_delta
            if delta is None:
                return None
            delta = delta.to(device=self.device, dtype=images.dtype)
            triggered_images = apply_trigger(
                images,
                trigger_type="adversarial",
                perturbation=delta,
                clamp_to_valid_rgb=(
                    self.attack_config.method == "a3fl"
                    and self.attack_config.a3fl_clip_to_valid_rgb
                ),
            )
            return images, triggered_images
        raise ValueError("trigger_type must be 'patch', 'adversarial', 'chameleon', or 'focused_flip'")

    def _compute_stealth_metrics(self) -> dict:
        """Measure trigger stealth on the same cached test images every round."""
        unavailable = {"psnr": None, "ssim": None, "linf": None}
        if self.attack_config is None or not self.attack_config.enabled:
            return unavailable
        trigger_images = self._apply_trigger_to_stealth_images()
        if trigger_images is None:
            return unavailable
        images, triggered_images = trigger_images
        return compute_stealth_metrics(
            images, triggered_images, triggered_images - images,
            self.normalization_mean, self.normalization_std,
        )

    def _effective_local_epochs(self, client, attack_active: bool) -> int:
        """Return the server-scheduled local epoch count for one client.

        This policy is intentionally independent of ``attack.method``.  A
        malicious client receives the configured override only during an
        attack-active round; all other client/round combinations use the
        client's normal local-epoch setting.
        """
        if type(attack_active) is not bool:
            raise TypeError("attack_active must be a boolean")
        base_local_epochs = client.local_epochs
        if attack_active and _is_malicious_client(client):
            if self.malicious_local_epochs is not None:
                return self.malicious_local_epochs
        return base_local_epochs

    def _visualize_trigger(self, round_index: int) -> dict:
        """Save and measure the configured trigger on one deterministic test batch."""
        trigger_images = self._apply_trigger_to_stealth_images()
        if trigger_images is None:
            return {"psnr": None, "ssim": None, "linf": None}
        images, triggered_images = trigger_images

        delta = triggered_images - images
        destination = save_trigger_comparison(
            images,
            triggered_images,
            delta,
            round_index=round_index,
            output_dir=self.visualization_output_dir,
            num_samples=self.visualization_num_samples,
            mean_values=self.normalization_mean,
            std_values=self.normalization_std,
        )
        metrics = compute_stealth_metrics(
            images, triggered_images, delta,
            self.normalization_mean, self.normalization_std,
        )
        print(
            f"Round {round_index} Trigger Visualization\n"
            f"PSNR: {metrics['psnr']:.2f} dB\n"
            f"SSIM: {metrics['ssim']:.3f}\n"
            f"L_inf: {metrics['linf']:.3f}\n"
            f"Saved: {destination}",
            flush=True,
        )
        return metrics

    def _selection_mode(self, round_index: int) -> tuple[str, list[int]]:
        """Describe the sampling branch used for a round without sampling clients."""
        if self.attack_config is None or not self.attack_config.enabled:
            return "random_all_clients", []

        malicious_ids = list(self.attack_config.malicious_client_ids)
        if self.attack_config.selection_policy == "legacy_all_malicious":
            return "legacy_all_malicious", malicious_ids
        if (
            self.attack_config.selection_policy == "force_during_attack"
            and self.attack_config.is_attack_round(round_index)
        ):
            return "force_malicious_during_attack", malicious_ids
        return "random_all_clients", []

    def select_clients(self, round_index: int | None = None) -> list:
        """Select clients according to the configured, phase-aware sampling policy.

        ``legacy_all_malicious`` retains the historical behavior of including
        every configured malicious client in every round.  In contrast,
        ``force_during_attack`` includes every configured malicious client only
        in attack-active rounds; all other rounds are an unbiased sample from
        the complete client population.
        """
        round_index = 1 if round_index is None else round_index
        if self.attack_config is None or not self.attack_config.enabled:
            return self.rng.sample(self.clients, self.clients_per_round)

        malicious_client_ids = set(self.attack_config.malicious_client_ids)
        malicious_clients = [
            client for client in self.clients if client.client_id in malicious_client_ids
        ]
        normal_clients = [
            client for client in self.clients if client.client_id not in malicious_client_ids
        ]

        if len(malicious_clients) != len(malicious_client_ids):
            raise ValueError("clients must include all configured malicious client IDs")

        selection_mode, forced_malicious_ids = self._selection_mode(round_index)
        if selection_mode == "random_all_clients":
            return self.rng.sample(self.clients, self.clients_per_round)

        # Both forcing modes intentionally include *all* configured malicious
        # clients.  This preserves legacy reproducibility and makes the new
        # force_during_attack policy unambiguous in attack-active rounds.
        num_normal_clients = self.clients_per_round - len(forced_malicious_ids)
        if num_normal_clients < 0 or num_normal_clients > len(normal_clients):
            raise ValueError("clients_per_round is incompatible with malicious selection")
        return malicious_clients + self.rng.sample(normal_clients, num_normal_clients)

    def _should_refresh_prototypes(self, round_index: int) -> bool:
        cfg = self.attack_config
        if not (cfg and cfg.prototype_contrastive_enabled):
            return False
        # The legacy single-client path intentionally builds in local UAP
        # generation, preserving historic experiments exactly.
        if cfg.prototype_source == "single_malicious" and cfg.prototype_build_policy == "legacy_local":
            return False
        if cfg.prototype_refresh_policy == "fixed":
            return round_index == cfg.attack_start_round
        if round_index < cfg.attack_start_round:
            return False
        if cfg.prototype_refresh_during == "attack_only" and not cfg.is_attack_round(round_index):
            return False
        if cfg.prototype_refresh_policy == "every_round":
            return True
        return ((round_index - cfg.attack_start_round) % cfg.prototype_refresh_interval) == 0

    def _refresh_prototype_bank(self, round_index: int, selected_clients: list) -> None:
        if not self._should_refresh_prototypes(round_index):
            return
        available = [c for c in selected_clients if _is_malicious_client(c)]
        if self.attack_config.prototype_source == "single_malicious" and self.attack_config.prototype_client_id is not None:
            sources = [client for client in self.clients if client.client_id == self.attack_config.prototype_client_id]
            if len(sources) != 1 or not _is_malicious_client(sources[0]):
                raise RuntimeError("configured prototype_client_id does not resolve to a malicious client")
        elif self.attack_config.prototype_source == "multi_malicious" and self.attack_config.prototype_aggregate:
            configured_ids = set(self.attack_config.malicious_client_ids)
            sources = [client for client in self.clients
                       if _is_malicious_client(client) and client.client_id in configured_ids]
        elif self.attack_config.prototype_source == "all_clients" and self.attack_config.prototype_aggregate:
            sources = list(self.clients)
        else:
            sources = available[:1]
            if self.attack_config.prototype_source == "single_malicious":
                print("[PrototypeBank] warning=prototype_client_id_unset; using first selected malicious client for legacy compatibility", flush=True)
            elif self.attack_config.prototype_source == "multi_malicious":
                print("[Prototype] warning=aggregate_prototypes_false; using first selected malicious client", flush=True)
        if not sources:
            message = "prototype construction requires a selected malicious client"
            if self.attack_config.prototype_build_policy == "force_at_attack_start":
                raise RuntimeError(message)
            print(f"[Prototype] warning={message}", flush=True)
            return
        stats = [
            client.compute_local_prototype_statistics(self.global_model)
            if _is_malicious_client(client) else collect_class_statistics(
                self.global_model, client.train_loader, client.device,
                self.attack_config.prototype_feature_layer,
                self.attack_config.prototype_center_batches,
            )
            for client in sources
        ]
        bank, details = aggregate_class_statistics(stats, self.attack_config.target_label,
            self.attack_config.prototype_min_samples_per_class, self.attack_config.num_classes,
            self.attack_config.prototype_missing_class_strategy)
        if bank is None:
            raise RuntimeError("target class is missing from malicious prototype statistics")
        bank.update({"temperature": self.attack_config.prototype_temperature,
                     "mode": self.attack_config.prototype_mode,
                     "feature_layer": self.attack_config.prototype_feature_layer,
                     "source_margin": self.attack_config.prototype_source_margin,
                     "version": self.prototype_version + 1,
                     "prototype_version": self.prototype_version + 1,
                     "used_malicious_client_ids": [client.client_id for client in sources]})
        if self.attack_config.strict_prototype_coverage:
            enforce_strict_prototype_coverage(
                bank, num_classes=self.attack_config.num_classes, dataset="configured_dataset",
                prototype_source_client=self.attack_config.prototype_client_id,
                center_batches=self.attack_config.prototype_center_batches,
            )
        MaliciousClient.shared_prototype_bank = bank
        if self.fixed_anchor_prototype_bank is None:
            self.fixed_anchor_prototype_bank = {
                key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                for key, value in bank.items()
            }
        self.prototype_version += 1
        event = {"round": round_index, "refresh_policy": self.attack_config.prototype_refresh_policy,
                 "prototype_version": self.prototype_version,
                 "used_malicious_client_ids": [c.client_id for c in sources],
                 "event_type": "prototype_bank_built", "prototype_source": self.attack_config.prototype_source,
                 "prototype_client_id": self.attack_config.prototype_client_id,
                 "aggregate_prototypes": self.attack_config.prototype_aggregate, **details}
        self.prototype_events.append(event)
        print("[PrototypeAggregation]", flush=True)
        print(f"round={round_index}", flush=True)
        print(f"source={self.attack_config.prototype_source}", flush=True)
        print(f"contributing_clients={[client.client_id for client in sources]}", flush=True)
        print(f"feature_layer={self.attack_config.prototype_feature_layer}", flush=True)
        print(f"center_batches={self.attack_config.prototype_center_batches}", flush=True)
        print(f"per_client_class_counts={details['per_client_class_counts']}", flush=True)
        print(f"aggregated_class_counts={details['class_counts']}", flush=True)
        print(f"missing_classes={details['missing_classes']}", flush=True)
        print(f"valid_class_mask={bank['valid_class_mask'].tolist()}", flush=True)
        print(f"prototype_version={self.prototype_version}", flush=True)
        print(f"freeze_after_build={self.attack_config.prototype_freeze_after_build}", flush=True)
        print(f"refresh_policy={self.attack_config.prototype_refresh_policy}", flush=True)

    def train_one_round(self, round_index: int) -> dict:
        """Train selected clients, aggregate their updates, and evaluate globally."""
        selected_clients = self.select_clients(round_index)
        attack_active = bool(
            self.attack_config is not None
            and self.attack_config.enabled
            and self.attack_config.is_attack_round(round_index)
        )
        if self.attack_config is not None and self.attack_config.enabled:
            selection_mode, forced_malicious_ids = self._selection_mode(round_index)
            selected_malicious = [
                client for client in selected_clients if _is_malicious_client(client)
            ]
            selected_benign = [
                client for client in selected_clients if not _is_malicious_client(client)
            ]
            benign_epochs = sorted({client.local_epochs for client in selected_benign})
            malicious_epochs = sorted({
                self._effective_local_epochs(client, attack_active)
                for client in selected_malicious
            })
            def format_epoch_values(values):
                if not values:
                    return "N/A"
                return values[0] if len(values) == 1 else values

            print(
                "[LocalEpochPolicy] "
                f"Round {round_index} | attack_active={attack_active} | "
                f"selection_policy={self.attack_config.selection_policy} | "
                f"sampling_mode={selection_mode} | "
                f"forced_malicious={forced_malicious_ids} | "
                f"base_local_epochs={format_epoch_values(benign_epochs)} | "
                f"malicious_local_epochs={self.malicious_local_epochs or format_epoch_values(benign_epochs)} | "
                f"malicious_effective_epochs={format_epoch_values(malicious_epochs)} | "
                f"selected_clients={[client.client_id for client in selected_clients]} | "
                f"selected_malicious={[client.client_id for client in selected_malicious]}",
                flush=True,
            )
        protocol_warning = None
        if (self.attack_config and round_index == self.attack_config.attack_start_round and
                self.attack_config.selection_policy == "random_only" and
                self.attack_config.prototype_build_policy == "legacy_local" and
                not any(_is_malicious_client(client) for client in selected_clients)):
            protocol_warning = ("attack_start selected no malicious client; prototype/UAP construction is delayed "
                                "until the first selected malicious client; this protocol is not strictly controlled")
            print(f"[ProtocolWarning] {protocol_warning}", flush=True)
        self._refresh_prototype_bank(round_index, selected_clients)
        neurotoxin_masks, neurotoxin_info = (None, None)
        if self.neurotoxin_enabled:
            neurotoxin_masks, neurotoxin_info = build_neurotoxin_mask(
                self.previous_global_state, self.current_global_state,
                self.global_model.named_parameters(), self.attack_config.neurotoxin_mask_ratio,
            )
            print(f"[Neurotoxin] usable_previous_global_delta={neurotoxin_info['usable_previous_global_delta']} masked_fraction={neurotoxin_info['masked_parameter_fraction']:.6f}", flush=True)
        client_results = [
            client.local_train(
                self.global_model,
                round_index=round_index,
                neurotoxin_masks=neurotoxin_masks,
                effective_local_epochs=self._effective_local_epochs(client, attack_active),
            )
            for client in selected_clients
        ]
        if (getattr(self.attack_config, "method", None) == "focused_flip"
                and round_index == self.attack_config.attack_end_round):
            from federated.focused_flip_client import FocusedFlipClient
            FocusedFlipClient.freeze_evaluation_pattern(round_index)
            print(
                f"[F3BA] Frozen averaged trigger after all malicious clients at round {round_index}",
                flush=True,
            )
        client_metrics = []
        for result in client_results:
            metric = {
                "client_id": result["client_id"],
                "num_samples": result["num_samples"],
                "train_loss": result["train_loss"],
                "is_malicious": result.get("is_malicious", False),
                "attack_active": result.get("attack_active", False),
                "num_poisoned_samples": result.get("num_poisoned_samples", 0),
            }
            for name in (
                "ce_loss_avg", "proto_train_loss_avg", "proto_train_batches",
                "adaptation_loss_avg", "projection_loss_avg",
                "supcon_valid_anchors", "supcon_target_valid_anchors",
            ):
                if name in result:
                    metric[name] = result[name]
            if "f3ba" in result:
                metric["f3ba"] = result["f3ba"]
            client_metrics.append(metric)

        client_states = [result["state_dict"] for result in client_results]
        client_num_samples = [result["num_samples"] for result in client_results]
        state_before_aggregation = self._parameter_snapshot() if self.neurotoxin_enabled else None
        # Client state dictionaries are intentionally returned on CPU.  Use a
        # detached CPU global snapshot for update arithmetic in defenses.
        global_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.global_model.state_dict().items()
        }
        aggregated_state, defense_diagnostics = aggregate_with_defense(
            client_states, client_num_samples, global_state,
            [n for n, _ in self.global_model.named_parameters()], self.defense_config,
            client_ids=[client.client_id for client in selected_clients],
            return_diagnostics=True,
            aggregation_weighting=self.aggregation_weighting,
            bn_buffer_policy=self.bn_buffer_policy,
            round_index=round_index,
        )
        self._log_defense_diagnostics(round_index, defense_diagnostics)
        self.global_model.load_state_dict(aggregated_state)
        if self.neurotoxin_enabled:
            self.previous_global_state = state_before_aggregation
            self.current_global_state = self._parameter_snapshot()

        evaluation_defense = self.defense_config if defense_diagnostics.get("defense_active_this_round") else None
        evaluation_result = evaluate_model(
            self.global_model,
            self.test_loader,
            self.device,
            defense_config=evaluation_defense,
        )
        asr_result = None
        if self.attack_config is not None and self.attack_config.enabled:
            asr_result = evaluate_backdoor_asr(
                self.global_model,
                self.test_loader,
                self.device,
                target_label=self.attack_config.target_label,
                trigger_type=getattr(self.attack_config, "trigger_type", "patch"),
                attack_config=self.attack_config,
                round_index=round_index,
                defense_config=evaluation_defense,
            )
        if evaluation_defense and evaluation_defense.get("name") == "crfl":
            print("[CRFLEval] "
                  f"round={round_index} crfl_smoothing_enabled={evaluation_defense['smoothing_enabled']} "
                  f"num_perturbed_models={evaluation_defense['num_perturbed_models']} "
                  f"sigma={evaluation_defense['sigma']} vote_rule={evaluation_defense['vote_rule']} "
                  f"clean_acc_defended={evaluation_result['clean_accuracy']:.4f} "
                  f"asr_defended={asr_result['attack_success_rate'] if asr_result else None}", flush=True)
        stealth_metrics = self._compute_stealth_metrics()
        feature_margins = {}
        analysis_rounds = self.mechanism_analysis.get("eval_rounds", [])
        if (self.mechanism_analysis.get("enabled", False) and round_index in analysis_rounds):
            feature_margins = evaluate_feature_margins(
                self.global_model, self.test_loader, self.device, self.attack_config,
                self.fixed_anchor_prototype_bank,
                max_batches=self.mechanism_analysis.get("max_batches", 10),
                compute_fixed=self.mechanism_analysis.get("compute_fixed_margin", True),
                compute_dynamic=self.mechanism_analysis.get("compute_dynamic_margin", True),
            )
        if (
            self.visualize_round == round_index
            and self.attack_config is not None
            and self.attack_config.enabled
        ):
            stealth_metrics = self._visualize_trigger(round_index)
        total_samples = sum(client_num_samples)
        average_train_loss = sum(
            result["train_loss"] * result["num_samples"]
            for result in client_results
        ) / total_samples
        if self.attack_config is not None and self.attack_config.method == "a3fl":
            delta = MaliciousClient.shared_adversarial_delta
            delta_linf = delta.detach().abs().max().item() if delta is not None else None
            attack_is_active = self.attack_config.is_attack_round(round_index)
            print(
                f"[A3FL{self.attack_config.a3fl_variant}Round] "
                f"round={round_index} attack_active={attack_is_active} method=a3fl "
                f"variant={self.attack_config.a3fl_variant} "
                f"refresh_policy={self.attack_config.a3fl_refresh_policy} "
                f"warm_start_delta={self.attack_config.a3fl_warm_start_delta} "
                f"delta_refreshed={MaliciousClient.shared_adversarial_delta_round == round_index} "
                f"delta_source_round={MaliciousClient.shared_adversarial_delta_round} "
                f"using_delta_round={MaliciousClient.shared_adversarial_delta_round} "
                f"post_uses_last_delta_round={MaliciousClient.shared_adversarial_delta_round if not attack_is_active else None} "
                f"delta_linf={delta_linf} linf={delta_linf} selected_clients={[client.client_id for client in selected_clients]} "
                f"adv_steps={self.attack_config.a3fl_adv_model_steps} lambda_adv={self.attack_config.a3fl_lambda_adv} delta_steps={self.attack_config.a3fl_delta_steps} "
                f"lambda_mode={self.attack_config.a3fl_lambda_mode} lambda0={self.attack_config.a3fl_lambda0} "
                f"lambda_t={MaliciousClient.shared_a3fl_lambda_t} similarity_theta_adv_global={MaliciousClient.shared_a3fl_similarity} "
                f"outer_steps={self.attack_config.a3fl_outer_steps} k_trigger={self.attack_config.a3fl_k_trigger} "
                f"adv_update_per_outer={self.attack_config.a3fl_adv_updates_per_outer or self.attack_config.a3fl_adv_model_steps} "
                f"adv_lr={self.attack_config.a3fl_adv_model_lr} adv_momentum={self.attack_config.a3fl_adv_model_momentum} "
                f"adv_weight_decay={self.attack_config.a3fl_adv_model_weight_decay} delta_alpha={self.attack_config.adversarial_alpha} epsilon={self.attack_config.adversarial_epsilon} "
                f"selected_malicious={sorted(client.client_id for client in selected_clients if _is_malicious_client(client))} "
                f"clean_acc={evaluation_result['clean_accuracy']:.4f} "
                f"asr={asr_result['attack_success_rate'] if asr_result is not None else None}",
                flush=True,
            )

        return {
            "round": round_index,
            "selected_clients": [client.client_id for client in selected_clients],
            "selected_malicious_clients": sorted(client.client_id for client in selected_clients if _is_malicious_client(client)),
            "num_selected_malicious": sum(_is_malicious_client(client) for client in selected_clients),
            "attack_active": bool(self.attack_config and self.attack_config.is_attack_round(round_index)),
            "a3fl_delta_refreshed": bool(
                self.attack_config
                and self.attack_config.method == "a3fl"
                and MaliciousClient.shared_adversarial_delta_round == round_index
            ),
            "a3fl_delta_source_round": (
                MaliciousClient.shared_adversarial_delta_round
                if self.attack_config is not None and self.attack_config.method == "a3fl" else None
            ),
            "aggregation_weighting": self.aggregation_weighting,
            "bn_buffer_policy": self.bn_buffer_policy,
            "average_train_loss": average_train_loss,
            "test_loss": evaluation_result["test_loss"],
            "clean_accuracy": evaluation_result["clean_accuracy"],
            "attack_success_rate": (
                asr_result["attack_success_rate"] if asr_result is not None else None
            ),
            "num_attack_samples": (
                asr_result["num_attack_samples"] if asr_result is not None else None
            ),
            "num_successful_attacks": (
                asr_result["num_successful_attacks"]
                if asr_result is not None
                else None
            ),
            "psnr": stealth_metrics["psnr"],
            "ssim": stealth_metrics["ssim"],
            "linf": stealth_metrics["linf"],
            "client_metrics": client_metrics,
            "stealth_metrics": stealth_metrics,
            "neurotoxin": neurotoxin_info,
            "prototype_event": self.prototype_events[-1] if self.prototype_events and self.prototype_events[-1]["round"] == round_index else None,
            "protocol_warning": protocol_warning,
            "feature_margins": feature_margins,
            "defense_diagnostics": defense_diagnostics,
        }

    def _log_defense_diagnostics(self, round_index: int, diagnostics: dict) -> None:
        """Emit parseable server-aggregation diagnostics for final defense analysis."""
        name = diagnostics.get("defense_name", "none")
        print("[Defense] "
              f"round={round_index} defense_enabled={diagnostics.get('defense_enabled')} "
              f"defense_name={name} defense_active_this_round={diagnostics.get('defense_active_this_round')} "
              f"defense_start_round={diagnostics.get('defense_start_round')} "
              f"defense_end_round={diagnostics.get('defense_end_round')} "
              f"defense_params={diagnostics.get('defense_params')}", flush=True)
        if not diagnostics.get("defense_active_this_round"):
            return
        if name == "norm_clipping":
            print("[NormClipping] "
                  f"round={round_index} clip_norm={diagnostics['clip_norm']} "
                  f"num_clipped={diagnostics['num_clipped']} "
                  f"max_update_norm={diagnostics['max_update_norm']:.6f} "
                  f"median_update_norm={diagnostics['median_update_norm']:.6f}", flush=True)
        elif name == "rlr":
            print("[RLR] "
                  f"round={round_index} threshold={diagnostics['threshold']} "
                  f"server_lr={diagnostics['server_lr']} fraction_reversed={diagnostics['fraction_reversed']:.6f} "
                  f"mean_agreement={diagnostics['mean_agreement']:.6f} "
                  f"min_agreement={diagnostics['min_agreement']:.6f} "
                  f"max_agreement={diagnostics['max_agreement']:.6f}", flush=True)
        elif name == "crfl":
            print("[CRFL] "
                  f"round={round_index} clip_norm={diagnostics['clip_norm']} sigma={diagnostics['sigma']} "
                  f"num_clipped={diagnostics['num_clipped']} max_update_norm={diagnostics['max_update_norm']:.6f} "
                  f"median_update_norm={diagnostics['median_update_norm']:.6f} noise_norm={diagnostics['noise_norm']:.6f}", flush=True)
        elif name == "trimmed_mean":
            print("[TrimmedMean] "
                  f"round={round_index} n={len(diagnostics['selected_client_ids'])} "
                  f"trim_ratio={diagnostics['trim_ratio']} trim_k={diagnostics['trim_k']} "
                  f"kept={diagnostics['kept']}", flush=True)
        elif name == "multi_krum":
            if diagnostics.get("warning"):
                print(f"[MultiKrum] round={round_index} warning={diagnostics['warning']}", flush=True)
            else:
                scores = diagnostics["scores"]
                summary = f"min={min(scores):.6f},max={max(scores):.6f}"
                print("[MultiKrum] "
                      f"round={round_index} n={diagnostics['n']} f={diagnostics['f']} "
                      f"m={diagnostics['m']} nearest_count={diagnostics['nearest_count']} "
                      f"selected_client_ids={diagnostics['selected_client_ids']} "
                      f"rejected_client_ids={diagnostics['rejected_client_ids']} "
                      f"scores_summary={summary}", flush=True)
        elif name == "flame":
            print("[FLAME] "
                  f"round={round_index} n={len(diagnostics['selected_client_ids'])} "
                  f"cluster_sizes={diagnostics['cluster_sizes']} "
                  f"kept_cluster={diagnostics['kept_cluster']} "
                  f"kept_client_ids={diagnostics['kept_client_ids']} "
                  f"removed_client_ids={diagnostics['removed_client_ids']} "
                  f"clip_norm={diagnostics['clip_norm']:.6f} "
                  f"noise_std={diagnostics['noise_std']}", flush=True)

    def _parameter_snapshot(self) -> dict[str, torch.Tensor]:
        return {name: parameter.detach().cpu().clone() for name, parameter in self.global_model.named_parameters()}

    def run(self) -> Iterator[dict]:
        """Run federated rounds and yield each result as soon as it completes."""
        for round_index in range(self.start_round, self.num_rounds + 1):
            result = self.train_one_round(round_index)
            yield result

"""Lightweight, data-free validation of experiment protocol invariants."""


def validate_experiment_protocol(config, attack_config):
    """Validate explicitly configured protocol fields without breaking legacy YAML."""
    federated = config["federated"]
    total_rounds = federated.get("total_rounds", federated.get("num_rounds"))
    if not isinstance(total_rounds, int) or total_rounds <= 0:
        raise ValueError("federated.total_rounds/num_rounds must be a positive integer")
    attack = config.get("attack", {})
    evaluation = config.get("evaluation", {})
    if attack_config.enabled:
        if "target_label" not in attack and "target_label" not in attack.get("trigger", {}):
            raise ValueError("enabled attack requires an explicit attack.target_label")
        if attack_config.attack_end_round is None:
            raise ValueError("enabled attack requires attack_end_round for protocol validation")
        if not attack_config.attack_start_round <= attack_config.attack_end_round < total_rounds:
            raise ValueError("attack.start_round <= attack.end_round < total_rounds is required")
        if len(attack_config.malicious_client_ids) != attack_config.num_malicious_clients:
            raise ValueError("malicious_client_ids and num_malicious_clients must agree")
        if attack_config.min_malicious_per_attack_round > federated["clients_per_round"]:
            raise ValueError("min_malicious_per_attack_round cannot exceed clients_per_round")
        if len(attack_config.malicious_client_ids) > federated["clients_per_round"]:
            raise ValueError("configured malicious clients cannot exceed clients_per_round")
        if attack_config.min_malicious_per_attack_round > len(attack_config.malicious_client_ids):
            raise ValueError("min_malicious_per_attack_round cannot exceed configured malicious clients")
        if (attack_config.selection_policy == "random_only" and
                attack_config.prototype_build_policy in {"force_at_attack_start", "server_aggregate"}):
            raise ValueError("random_only cannot guarantee attack-start prototype construction")
        if (attack_config.prototype_source == "single_malicious" and
                attack_config.prototype_client_id is not None and
                attack_config.prototype_client_id not in attack_config.malicious_client_ids):
            raise ValueError("prototype_client_id must belong to attack.malicious_client_ids")
    if evaluation:
        expected = attack_config.attack_end_round + 1 if attack_config.attack_end_round is not None else None
        if evaluation.get("post_attack_start") != expected:
            raise ValueError("evaluation.post_attack_start must equal attack.end_round + 1")
        # A post-attack-only AUC starts at end+1; final-table reporting also
        # legitimately includes the final attack round as relative time zero.
        if evaluation.get("pa_auc_start") not in {attack_config.attack_end_round, expected}:
            raise ValueError("evaluation.pa_auc_start must equal attack.end_round or attack.end_round + 1")
        if evaluation.get("pa_auc_end", total_rounds) > total_rounds:
            raise ValueError("evaluation.pa_auc_end cannot exceed total_rounds")
        if any(round_index > total_rounds for round_index in evaluation.get("eval_rounds", [])):
            raise ValueError("all evaluation.eval_rounds must be <= total_rounds")
    return total_rounds

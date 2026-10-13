"""Lightweight, data-free tests for EMA prototype refresh invariants."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from attacks.config import AttackConfig
from attacks.prototype import ema_update_prototype_bank
from federated.malicious_client import MaliciousClient
from federated.server import Server


old = {"prototypes": torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
       "valid_class_mask": torch.tensor([True, False, False]), "class_ids": [0, 1, 2]}
new = {"prototypes": torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]),
       "valid_class_mask": torch.tensor([True, True, False]), "class_ids": [0, 1, 2], "class_counts": {0: 5, 1: 5}}
updated, stats = ema_update_prototype_bank(old, new, beta=0.9)
expected = torch.tensor([0.9, 0.1]); expected /= expected.norm()
assert torch.allclose(updated["prototypes"][0], expected, atol=1e-6)
assert torch.isclose(updated["prototypes"][0].norm(), torch.tensor(1.0), atol=1e-6)
assert torch.equal(updated["prototypes"][1], torch.tensor([1.0, 0.0]))  # old invalid/new valid
assert not bool(updated["valid_class_mask"][2])  # both invalid
print("EMA_MATH_AND_MISSING_CLASS_PASS")

cfg = AttackConfig(enabled=True, target_label=0, malicious_client_ids=[0, 1, 2], num_malicious_clients=3,
                   selection_policy="legacy_all_malicious", prototype_build_policy="legacy_local",
                   attack_start_round=60, attack_end_round=150, trigger_type="adversarial",
                   prototype_contrastive_enabled=True, prototype_refresh_policy="ema",
                   prototype_refresh_interval=15, prototype_ema_beta=0.9)
server = Server.__new__(Server); server.attack_config = cfg
rounds = [round_index for round_index in range(1, 152) if server._should_refresh_prototypes(round_index)]
assert rounds == [75, 90, 105, 120, 135], rounds
print("EMA_SCHEDULE_PASS", rounds)

fake_delta = torch.tensor([1.0, 2.0])
MaliciousClient.shared_adversarial_delta = fake_delta
before = fake_delta.clone()
ema_update_prototype_bank(old, new, beta=0.9)
assert MaliciousClient.shared_adversarial_delta is fake_delta
assert torch.equal(MaliciousClient.shared_adversarial_delta, before)
print("DELTA_INVARIANT_PASS")

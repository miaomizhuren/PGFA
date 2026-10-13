import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from attacks.config import AttackConfig
from federated.server import Server


class ConstantClassModel(nn.Module):
    def __init__(self, predicted_class: int, num_classes: int = 2) -> None:
        super().__init__()
        self.predicted_class = predicted_class
        self.num_classes = num_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(
            images.size(0),
            self.num_classes,
            device=images.device,
        )
        logits[:, self.predicted_class] = 1.0
        return logits


class FakeClient:
    def __init__(
        self,
        client_id: int,
        is_malicious: bool = False,
        attack_active: bool = False,
        num_poisoned_samples: int = 0,
    ) -> None:
        self.client_id = client_id
        self.is_malicious = is_malicious
        self.attack_active = attack_active
        self.num_poisoned_samples = num_poisoned_samples

    def local_train(self, global_model: nn.Module, round_index: int) -> dict:
        return {
            "client_id": self.client_id,
            "state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in global_model.state_dict().items()
            },
            "num_samples": 4,
            "train_loss": 1.0,
            "is_malicious": self.is_malicious,
            "attack_active": self.attack_active,
            "num_poisoned_samples": self.num_poisoned_samples,
        }


def make_test_loader() -> DataLoader:
    images = torch.zeros(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 1, 1], dtype=torch.long)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=4, shuffle=False)


def make_server(attack_config: AttackConfig) -> Server:
    return Server(
        global_model=ConstantClassModel(predicted_class=0),
        clients=[
            FakeClient(
                client_id=0,
                is_malicious=True,
                attack_active=attack_config.enabled,
                num_poisoned_samples=2 if attack_config.enabled else 0,
            )
        ],
        test_loader=make_test_loader(),
        num_rounds=1,
        clients_per_round=1,
        device=torch.device("cpu"),
        seed=2026,
        attack_config=attack_config,
    )


def test_server_does_not_compute_asr_when_attack_disabled() -> None:
    attack_config = AttackConfig(enabled=False, target_label=0)
    result = make_server(attack_config).train_one_round(round_index=1)

    assert result["attack_success_rate"] is None
    assert result["num_attack_samples"] is None
    assert result["num_successful_attacks"] is None


def test_server_returns_asr_when_attack_enabled() -> None:
    attack_config = AttackConfig(enabled=True, target_label=0)
    result = make_server(attack_config).train_one_round(round_index=1)

    assert result["attack_success_rate"] == pytest.approx(100.0)
    assert result["num_attack_samples"] == 3
    assert result["num_successful_attacks"] == 3


def test_server_client_metrics_keep_attack_fields() -> None:
    attack_config = AttackConfig(enabled=True, target_label=0)
    result = make_server(attack_config).train_one_round(round_index=1)

    assert result["client_metrics"] == [
        {
            "client_id": 0,
            "num_samples": 4,
            "train_loss": 1.0,
            "is_malicious": True,
            "attack_active": True,
            "num_poisoned_samples": 2,
        }
    ]

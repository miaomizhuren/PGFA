import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from attacks.config import AttackConfig
from evaluation.evaluator import evaluate_backdoor_asr
from federated.malicious_client import MaliciousClient


class ConstantClassModel(nn.Module):
    def __init__(self, predicted_class: int, num_classes: int = 4) -> None:
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


def make_loader(labels: torch.Tensor) -> DataLoader:
    images = torch.zeros(labels.size(0), 3, 8, 8)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=labels.size(0), shuffle=False)


def test_evaluate_backdoor_asr_excludes_target_class_samples() -> None:
    labels = torch.tensor([0, 1, 2, 0, 3], dtype=torch.long)
    loader = make_loader(labels)
    model = ConstantClassModel(predicted_class=0)

    result = evaluate_backdoor_asr(
        model,
        loader,
        torch.device("cpu"),
        target_label=0,
    )

    assert result["num_attack_samples"] == 3


def test_evaluate_backdoor_asr_is_100_when_all_attacks_succeed() -> None:
    labels = torch.tensor([1, 2, 3], dtype=torch.long)
    loader = make_loader(labels)
    model = ConstantClassModel(predicted_class=0)

    result = evaluate_backdoor_asr(
        model,
        loader,
        torch.device("cpu"),
        target_label=0,
    )

    assert result["attack_success_rate"] == pytest.approx(100.0)
    assert result["num_successful_attacks"] == 3


def test_evaluate_backdoor_asr_is_0_when_no_attacks_succeed() -> None:
    labels = torch.tensor([1, 2, 3], dtype=torch.long)
    loader = make_loader(labels)
    model = ConstantClassModel(predicted_class=1)

    result = evaluate_backdoor_asr(
        model,
        loader,
        torch.device("cpu"),
        target_label=0,
    )

    assert result["attack_success_rate"] == pytest.approx(0.0)
    assert result["num_successful_attacks"] == 0


def test_evaluate_backdoor_asr_does_not_modify_original_inputs() -> None:
    images = torch.zeros(4, 3, 8, 8)
    original_images = images.clone()
    labels = torch.tensor([1, 2, 3, 1], dtype=torch.long)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = ConstantClassModel(predicted_class=0)

    evaluate_backdoor_asr(
        model,
        loader,
        torch.device("cpu"),
        target_label=0,
    )

    assert torch.equal(images, original_images)


def test_evaluate_backdoor_asr_raises_without_non_target_samples() -> None:
    labels = torch.tensor([0, 0, 0], dtype=torch.long)
    loader = make_loader(labels)
    model = ConstantClassModel(predicted_class=0)

    with pytest.raises(ValueError):
        evaluate_backdoor_asr(
            model,
            loader,
            torch.device("cpu"),
            target_label=0,
        )


def test_adversarial_asr_is_unavailable_without_shared_delta() -> None:
    MaliciousClient.reset_shared_delta()
    labels = torch.tensor([1, 2, 3], dtype=torch.long)
    result = evaluate_backdoor_asr(
        ConstantClassModel(predicted_class=0),
        make_loader(labels),
        torch.device("cpu"),
        target_label=0,
        trigger_type="adversarial",
        attack_config=AttackConfig(enabled=True, target_label=0),
    )

    assert result == {
        "attack_success_rate": None,
        "num_attack_samples": None,
        "num_successful_attacks": None,
    }
    assert MaliciousClient.shared_adversarial_delta is None

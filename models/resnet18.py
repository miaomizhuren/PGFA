"""ResNet-18 model builder for CIFAR-10 classification."""

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet


class CIFARResNet18(ResNet):
    """CIFAR-10 ResNet-18 with explicit penultimate-feature extraction.

    Inheriting torchvision's ``ResNet`` retains conventional parameter names,
    so checkpoints made by the previous builder remain state-dict compatible.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()

    def forward_features(
        self,
        x: torch.Tensor,
        feature_layer: str = "penultimate",
    ) -> torch.Tensor:
        """Return globally pooled features from a configured ResNet stage."""
        if feature_layer not in ("layer3", "layer4", "avgpool", "penultimate"):
            raise ValueError(
                "feature_layer must be 'layer3', 'layer4', 'avgpool', or 'penultimate'"
            )
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if feature_layer == "layer3":
            return torch.flatten(self.avgpool(x), 1)
        x = self.layer4(x)
        if feature_layer == "layer4":
            return torch.flatten(self.avgpool(x), 1)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits, preserving the previous builder behaviour."""
        return self.fc(self.forward_features(x))


def build_resnet18(num_classes: int = 10) -> CIFARResNet18:
    """Build a ResNet-18 adapted for CIFAR-sized images."""
    if type(num_classes) is not int:
        raise TypeError("num_classes must be an integer")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than 1")
    return CIFARResNet18(num_classes=num_classes)

#!/usr/bin/env python3
"""Lightweight structural compatibility check for a CIFAR-10 ResNet18 checkpoint."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.resnet18 import build_resnet18


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint is not a state_dict mapping")

    model = build_resnet18(num_classes=10)
    model.load_state_dict(state_dict, strict=True)
    if tuple(model.fc.weight.shape) != (10, 512):
        raise ValueError(f"unexpected fc.weight shape: {tuple(model.fc.weight.shape)}")

    print(f"PASS: {checkpoint}")
    print(f"state_dict_keys={len(state_dict)}")
    print(f"fc.weight_shape={tuple(model.fc.weight.shape)}")


if __name__ == "__main__":
    main()

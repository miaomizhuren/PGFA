"""Benign federated-learning client with local SGD training."""

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


class Client:
    """A benign client that trains a copied global model on local data."""

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
    ) -> None:
        """Initialize a client and its local training data loader."""
        if type(client_id) is not int:
            raise TypeError("client_id must be an integer")
        if client_id < 0:
            raise ValueError("client_id must be non-negative")
        if not isinstance(indices, list):
            raise TypeError("indices must be a list")
        if not indices:
            raise ValueError("indices must not be empty")
        if any(type(index) is not int for index in indices):
            raise TypeError("indices must contain only integers")
        if type(batch_size) is not int:
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if type(local_epochs) is not int:
            raise TypeError("local_epochs must be an integer")
        if local_epochs <= 0:
            raise ValueError("local_epochs must be positive")
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
            raise TypeError("learning_rate must be a number")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if isinstance(momentum, bool) or not isinstance(momentum, (int, float)):
            raise TypeError("momentum must be a number")
        if momentum < 0:
            raise ValueError("momentum must be non-negative")
        if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)):
            raise TypeError("weight_decay must be a number")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if type(num_workers) is not int:
            raise TypeError("num_workers must be an integer")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")

        self.client_id = client_id
        self.indices = indices
        self.local_dataset = Subset(dataset, indices)
        self.train_loader = DataLoader(
            self.local_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.device = device

    def local_train(
        self,
        global_model: nn.Module,
        round_index: int | None = None,
        neurotoxin_masks=None,
        effective_local_epochs: int | None = None,
    ) -> dict:
        """Train a local copy of the global model and return its metrics."""
        if round_index is not None:
            if type(round_index) is not int:
                raise TypeError("round_index must be an integer or None")
            if round_index <= 0:
                raise ValueError("round_index must be positive")
        if effective_local_epochs is None:
            effective_local_epochs = self.local_epochs
        if type(effective_local_epochs) is not int or effective_local_epochs <= 0:
            raise ValueError("effective_local_epochs must be a positive integer or None")

        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        total_loss = 0.0
        total_samples = 0
        for _ in range(effective_local_epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = local_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in local_model.state_dict().items()
        }
        return {
            "client_id": self.client_id,
            "state_dict": state_dict,
            "num_samples": len(self.local_dataset),
            "train_loss": total_loss / total_samples,
        }

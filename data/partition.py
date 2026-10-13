"""IID sample-index partitioning utilities."""

import numpy as np


def partition_iid(
    num_samples: int,
    num_clients: int,
    seed: int,
) -> dict[int, list[int]]:
    """Randomly split all sample indices into balanced IID client partitions."""
    if type(num_samples) is not int:
        raise TypeError("num_samples must be an integer")
    if type(num_clients) is not int:
        raise TypeError("num_clients must be an integer")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")

    if num_samples <= 0:
        raise ValueError("num_samples must be greater than 0")
    if num_clients <= 0:
        raise ValueError("num_clients must be greater than 0")
    if num_clients > num_samples:
        raise ValueError("num_clients cannot be greater than num_samples")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples) # 打乱
    client_indices = np.array_split(indices, num_clients)

    return {
        client_id: [int(index) for index in split]
        for client_id, split in enumerate(client_indices)
    }


def partition_dirichlet(
    labels,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples_per_client: int = 1,
    max_retries: int = 100,
) -> dict[int, list[int]]:
    """Deterministically create a label-skew Dirichlet partition without loss."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    if num_clients <= 0 or alpha <= 0 or min_samples_per_client <= 0 or max_retries <= 0:
        raise ValueError("invalid Dirichlet partition arguments")
    if num_clients * min_samples_per_client > labels.size:
        raise ValueError("min_samples_per_client is infeasible")
    rng = np.random.default_rng(seed)
    for _ in range(max_retries):
        partitions = [[] for _ in range(num_clients)]
        for class_id in np.unique(labels):
            class_indices = np.flatnonzero(labels == class_id).copy()
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            boundaries = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
            for client_id, split in enumerate(np.split(class_indices, boundaries)):
                partitions[client_id].extend(int(index) for index in split)
        if min(map(len, partitions)) >= min_samples_per_client:
            for partition in partitions:
                rng.shuffle(partition)
            return {client_id: partition for client_id, partition in enumerate(partitions)}
    raise RuntimeError(f"Dirichlet partition failed min_samples_per_client after {max_retries} retries")


def partition_dirichlet_balanced(
    labels,
    num_clients: int,
    alpha: float,
    seed: int,
    samples_per_client: int,
    min_classes_per_client: int = 1,
    max_retries: int = 100,
) -> dict[int, list[int]]:
    """Create an exact-size, label-skew Dirichlet partition without loss.

    Each client first receives a class preference vector sampled from a
    Dirichlet distribution.  We draw a fixed-size class histogram for every
    client and reconcile its column totals to the available class inventories.
    Consequently every client has exactly ``samples_per_client`` samples while
    every training index is assigned exactly once.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    if num_clients <= 0 or alpha <= 0 or samples_per_client <= 0:
        raise ValueError("num_clients, alpha, and samples_per_client must be positive")
    if min_classes_per_client <= 0 or min_classes_per_client > len(np.unique(labels)):
        raise ValueError("min_classes_per_client must be between 1 and the number of classes")
    if max_retries <= 0:
        raise ValueError("max_retries must be positive")
    if num_clients * samples_per_client != labels.size:
        raise ValueError(
            "balanced Dirichlet requires num_clients * samples_per_client to equal the dataset size"
        )

    class_values, inverse = np.unique(labels, return_inverse=True)
    class_capacities = np.bincount(inverse, minlength=len(class_values)).astype(int)
    rng = np.random.default_rng(seed)

    for _ in range(max_retries):
        preferences = rng.dirichlet(np.full(len(class_values), alpha), size=num_clients)
        counts = np.vstack([
            rng.multinomial(samples_per_client, preference)
            for preference in preferences
        ]).astype(int)

        # Move one sample at a time from over-subscribed global classes to
        # under-subscribed classes, keeping every client row sum unchanged.
        totals = counts.sum(axis=0)
        while not np.array_equal(totals, class_capacities):
            source = int(np.flatnonzero(totals > class_capacities)[0])
            destination = int(np.flatnonzero(totals < class_capacities)[0])
            # Prefer retaining support in the source class and moving samples
            # to clients whose sampled preference favors the destination.
            candidates = np.flatnonzero(counts[:, source] > 1)
            if candidates.size == 0:
                candidates = np.flatnonzero(counts[:, source] > 0)
            if candidates.size == 0:  # Defensive; impossible when totals[source] is positive.
                break
            ratios = preferences[candidates, destination] / (preferences[candidates, source] + 1e-12)
            client_id = int(candidates[np.argmax(ratios)])
            counts[client_id, source] -= 1
            counts[client_id, destination] += 1
            totals[source] -= 1
            totals[destination] += 1

        if not np.array_equal(totals, class_capacities):
            continue
        if np.any(counts.sum(axis=1) != samples_per_client):
            continue
        if np.any((counts > 0).sum(axis=1) < min_classes_per_client):
            continue

        class_indices = []
        for class_id in range(len(class_values)):
            indices = np.flatnonzero(inverse == class_id).copy()
            rng.shuffle(indices)
            class_indices.append(indices)
        offsets = np.zeros(len(class_values), dtype=int)
        partitions: dict[int, list[int]] = {}
        for client_id in range(num_clients):
            client_indices: list[int] = []
            for class_id, count in enumerate(counts[client_id]):
                start = offsets[class_id]
                end = start + int(count)
                client_indices.extend(int(index) for index in class_indices[class_id][start:end])
                offsets[class_id] = end
            rng.shuffle(client_indices)
            partitions[client_id] = client_indices
        if np.array_equal(offsets, class_capacities):
            return partitions
    raise RuntimeError(
        f"balanced Dirichlet partition failed min_classes_per_client after {max_retries} retries"
    )

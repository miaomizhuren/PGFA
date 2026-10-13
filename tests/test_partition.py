import pytest

from data.partition import partition_iid


def test_partition_iid_covers_all_samples() -> None:
    partitions = partition_iid(num_samples=100, num_clients=10, seed=2026)

    all_indices = [index for indices in partitions.values() for index in indices]

    assert len(partitions) == 10
    assert len(all_indices) == 100
    assert sorted(all_indices) == list(range(100))


def test_partition_iid_is_balanced() -> None:
    partitions = partition_iid(num_samples=103, num_clients=10, seed=2026)

    client_sizes = [len(indices) for indices in partitions.values()]

    assert max(client_sizes) - min(client_sizes) <= 1


def test_partition_iid_is_reproducible() -> None:
    first_partition = partition_iid(num_samples=100, num_clients=10, seed=2026)
    second_partition = partition_iid(num_samples=100, num_clients=10, seed=2026)

    assert first_partition == second_partition


def test_partition_iid_changes_with_seed() -> None:
    partition_2026 = partition_iid(num_samples=100, num_clients=10, seed=2026)
    partition_2027 = partition_iid(num_samples=100, num_clients=10, seed=2027)

    assert partition_2026 != partition_2027


def test_partition_iid_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        partition_iid(num_samples=0, num_clients=1, seed=2026)
    with pytest.raises(ValueError):
        partition_iid(num_samples=1, num_clients=0, seed=2026)
    with pytest.raises(ValueError):
        partition_iid(num_samples=1, num_clients=2, seed=2026)
    with pytest.raises(ValueError):
        partition_iid(num_samples=1, num_clients=1, seed=-1)
    with pytest.raises(TypeError):
        partition_iid(num_samples=1.0, num_clients=1, seed=2026)
    with pytest.raises(TypeError):
        partition_iid(num_samples=1, num_clients=True, seed=2026)

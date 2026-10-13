import pytest
import torch

from federated.aggregation import fedavg


def test_fedavg_equal_sample_counts() -> None:
    client_1 = {
        "w": torch.tensor([2.0]),
        "n": torch.tensor(1, dtype=torch.long),
    }
    client_2 = {
        "w": torch.tensor([4.0]),
        "n": torch.tensor(2, dtype=torch.long),
    }

    result = fedavg([client_1, client_2], [100, 100])

    torch.testing.assert_close(result["w"], torch.tensor([3.0]))
    assert torch.equal(result["n"], torch.tensor(1, dtype=torch.long))


def test_fedavg_weighted_by_sample_counts() -> None:
    client_1 = {"w": torch.tensor([2.0])}
    client_2 = {"w": torch.tensor([4.0])}

    result = fedavg([client_1, client_2], [100, 300])

    torch.testing.assert_close(result["w"], torch.tensor([3.5]))


def test_fedavg_multidimensional_tensor() -> None:
    client_1 = {
        "w": torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
    }
    client_2 = {
        "w": torch.tensor(
            [
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
    }

    result = fedavg([client_1, client_2], [100, 100])

    expected = torch.tensor(
        [
            [2.0, 3.0],
            [4.0, 5.0],
        ]
    )
    torch.testing.assert_close(result["w"], expected)


def test_fedavg_does_not_modify_inputs() -> None:
    client_1 = {"w": torch.tensor([2.0])}
    client_2 = {"w": torch.tensor([4.0])}
    client_1_original = client_1["w"].clone()
    client_2_original = client_2["w"].clone()

    result = fedavg([client_1, client_2], [100, 300])

    assert torch.equal(client_1["w"], client_1_original)
    assert torch.equal(client_2["w"], client_2_original)
    assert result["w"] is not client_1["w"]


def test_fedavg_rejects_empty_client_states() -> None:
    with pytest.raises(ValueError):
        fedavg([], [])


def test_fedavg_rejects_mismatched_list_lengths() -> None:
    with pytest.raises(ValueError):
        fedavg([{"w": torch.tensor([1.0])}], [100, 200])


def test_fedavg_rejects_zero_sample_count() -> None:
    with pytest.raises(ValueError):
        fedavg([{"w": torch.tensor([1.0])}], [0])


def test_fedavg_rejects_mismatched_parameter_keys() -> None:
    with pytest.raises(ValueError):
        fedavg(
            [{"w": torch.tensor([1.0])}, {"b": torch.tensor([1.0])}],
            [100, 100],
        )


def test_fedavg_rejects_mismatched_parameter_shapes() -> None:
    with pytest.raises(ValueError):
        fedavg(
            [{"w": torch.tensor([1.0])}, {"w": torch.tensor([1.0, 2.0])}],
            [100, 100],
        )

"""Data structures for federated experiment metrics."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ClientMetric:
    """Training metrics for one client in one federated round."""

    round_index: int
    client_id: int
    num_samples: int
    train_loss: float
    is_malicious: bool = False
    attack_active: bool = False
    protocol_warning: str | None = None
    num_poisoned_samples: int = 0
    ce_loss_avg: float | None = None
    proto_train_loss_avg: float | None = None
    proto_train_batches: int = 0
    adaptation_loss_avg: float | None = None
    projection_loss_avg: float | None = None
    supcon_valid_anchors: int = 0
    supcon_target_valid_anchors: int = 0

    def to_dict(self) -> dict:
        """Convert this metric to a plain dictionary."""
        return asdict(self)


@dataclass
class RoundMetric:
    """Global metrics collected after one federated round."""

    round_index: int
    selected_clients: list[int]
    average_train_loss: float
    test_loss: float
    clean_accuracy: float
    selected_malicious_clients: list[int] | None = None
    num_selected_malicious: int = 0
    attack_active: bool = False
    aggregation_weighting: str = "sample_weighted"
    bn_buffer_policy: str = "aggregate"
    protocol_warning: str | None = None
    attack_success_rate: float | None = None
    num_attack_samples: int | None = None
    num_successful_attacks: int | None = None
    psnr: float | None = None
    ssim: float | None = None
    linf: float | None = None
    recorded_at: str = ""

    def __post_init__(self) -> None:
        """Fill the record timestamp when one is not provided."""
        if not self.recorded_at:
            self.recorded_at = _utc_now_iso()

    def to_dict(self) -> dict:
        """Convert this metric to a plain dictionary."""
        return asdict(self)

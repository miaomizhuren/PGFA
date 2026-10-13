"""Recorder interface for federated experiment metrics."""

from pathlib import Path

from experiment_tracking.schema import ClientMetric, RoundMetric
from experiment_tracking.storage import MetricsStorage


class ExperimentRecorder:
    """Convert training results to metrics and persist them through storage."""

    def __init__(self, database_path: str | Path) -> None:
        """Initialize the recorder with a SQLite-backed metric store."""
        self.storage = MetricsStorage(database_path)

    def record_round(self, round_result: dict) -> None:
        """Convert and save one global federated round result."""
        metric = RoundMetric(
            round_index=round_result["round"],
            selected_clients=round_result["selected_clients"],
            selected_malicious_clients=round_result.get("selected_malicious_clients", []),
            num_selected_malicious=round_result.get("num_selected_malicious", 0),
            attack_active=round_result.get("attack_active", False),
            aggregation_weighting=round_result.get("aggregation_weighting", "sample_weighted"),
            bn_buffer_policy=round_result.get("bn_buffer_policy", "aggregate"),
            protocol_warning=round_result.get("protocol_warning"),
            average_train_loss=round_result["average_train_loss"],
            test_loss=round_result["test_loss"],
            clean_accuracy=round_result["clean_accuracy"],
            attack_success_rate=round_result.get("attack_success_rate"),
            num_attack_samples=round_result.get("num_attack_samples"),
            num_successful_attacks=round_result.get("num_successful_attacks"),
            psnr=round_result.get("psnr"),
            ssim=round_result.get("ssim"),
            linf=round_result.get("linf"),
        )
        self.storage.save_round_metric(metric)

    def record_clients(
        self,
        round_index: int,
        client_results: list[dict],
    ) -> None:
        """Convert and save client metrics for one federated round."""
        for client_result in client_results:
            metric = ClientMetric(
                round_index=round_index,
                client_id=client_result["client_id"],
                num_samples=client_result["num_samples"],
                train_loss=client_result["train_loss"],
                is_malicious=client_result.get("is_malicious", False),
                attack_active=client_result.get("attack_active", False),
                num_poisoned_samples=client_result.get("num_poisoned_samples", 0),
                ce_loss_avg=client_result.get("ce_loss_avg"),
                proto_train_loss_avg=client_result.get("proto_train_loss_avg"),
                proto_train_batches=client_result.get("proto_train_batches", 0),
                adaptation_loss_avg=client_result.get("adaptation_loss_avg"),
                projection_loss_avg=client_result.get("projection_loss_avg"),
                supcon_valid_anchors=client_result.get("supcon_valid_anchors", 0),
                supcon_target_valid_anchors=client_result.get("supcon_target_valid_anchors", 0),
            )
            self.storage.save_client_metric(metric)

    def load_round_metrics(self) -> list[dict]:
        """Load all recorded global round metrics."""
        return self.storage.load_round_metrics()

    def load_client_metrics(self) -> list[dict]:
        """Load all recorded client metrics."""
        return self.storage.load_client_metrics()

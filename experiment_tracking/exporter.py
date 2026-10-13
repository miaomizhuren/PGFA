"""Command-line CSV exporter for experiment metrics."""

import argparse
import csv
import json
from pathlib import Path

from experiment_tracking.storage import MetricsStorage


ROUND_FIELDS = [
    "round_index",
    "selected_clients",
    "selected_malicious_clients",
    "num_selected_malicious",
    "attack_active",
    "aggregation_weighting",
    "bn_buffer_policy",
    "average_train_loss",
    "test_loss",
    "clean_accuracy",
    "attack_success_rate",
    "num_attack_samples",
    "num_successful_attacks",
    "psnr",
    "ssim",
    "linf",
    "recorded_at",
]

CLIENT_FIELDS = [
    "round_index",
    "client_id",
    "num_samples",
    "train_loss",
    "is_malicious",
    "attack_active",
    "num_poisoned_samples",
    "ce_loss_avg",
    "proto_train_loss_avg",
    "proto_train_batches",
]


def _write_csv(path: Path, records: list[dict], fieldnames: list[str]) -> None:
    """Write dictionaries to a CSV file using the given field order."""
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in fieldnames}
            if isinstance(row.get("selected_clients"), list):
                row["selected_clients"] = json.dumps(row["selected_clients"])
            writer.writerow(row)


def export_metrics(database_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    """Export round and client metrics from SQLite to CSV files."""
    database_path = Path(database_path)
    if not database_path.exists():
        raise FileNotFoundError(f"metrics database not found: {database_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    storage = MetricsStorage(database_path)
    round_metrics_path = output_dir / "round_metrics.csv"
    client_metrics_path = output_dir / "client_metrics.csv"

    _write_csv(round_metrics_path, storage.load_round_metrics(), ROUND_FIELDS)
    _write_csv(client_metrics_path, storage.load_client_metrics(), CLIENT_FIELDS)
    return round_metrics_path, client_metrics_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export experiment metrics to CSV.")
    parser.add_argument("--db", default="outputs/metrics.db", help="Path to metrics.db")
    parser.add_argument("--out", default="outputs/exports", help="Output directory")
    return parser.parse_args()


def main() -> None:
    """Run the CSV exporter from the command line."""
    args = parse_args()
    round_path, client_path = export_metrics(args.db, args.out)
    print(f"Exported round metrics: {round_path}")
    print(f"Exported client metrics: {client_path}")


if __name__ == "__main__":
    main()

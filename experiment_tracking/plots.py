"""Command-line plotting utilities for experiment metrics."""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_tracking.storage import MetricsStorage


def _save_current_figure(output_dir: Path, filename_stem: str) -> list[Path]:
    """Save the current matplotlib figure as PNG and SVG."""
    paths = [
        output_dir / f"{filename_stem}.png",
        output_dir / f"{filename_stem}.svg",
    ]
    for path in paths:
        plt.savefig(path, bbox_inches="tight")
    plt.close()
    return paths


def _plot_clean_accuracy(round_metrics: list[dict], output_dir: Path) -> list[Path]:
    """Plot clean accuracy by federated round."""
    rounds = [record["round_index"] for record in round_metrics]
    clean_accuracy = [record["clean_accuracy"] for record in round_metrics]

    plt.figure()
    plt.plot(rounds, clean_accuracy, marker="o", label="Clean Accuracy")
    plt.title("Clean Accuracy vs Round")
    plt.xlabel("Round")
    plt.ylabel("Clean Accuracy (%)")
    plt.legend()
    return _save_current_figure(output_dir, "clean_accuracy_curve")


def _plot_asr(round_metrics: list[dict], output_dir: Path) -> list[Path]:
    """Plot attack success rate by federated round."""
    asr_records = [
        record
        for record in round_metrics
        if record.get("attack_success_rate") is not None
    ]
    if not asr_records:
        print("No ASR records found. Skipping ASR curve.")
        return []

    rounds = [record["round_index"] for record in asr_records]
    attack_success_rates = [
        record["attack_success_rate"]
        for record in asr_records
    ]

    plt.figure()
    plt.plot(rounds, attack_success_rates, marker="o", label="ASR")
    plt.title("Attack Success Rate vs Round")
    plt.xlabel("Round")
    plt.ylabel("Attack Success Rate (%)")
    plt.legend()
    return _save_current_figure(output_dir, "asr_curve")


def _plot_losses(round_metrics: list[dict], output_dir: Path) -> list[Path]:
    """Plot average train loss and test loss by federated round."""
    rounds = [record["round_index"] for record in round_metrics]
    train_losses = [record["average_train_loss"] for record in round_metrics]
    test_losses = [record["test_loss"] for record in round_metrics]

    plt.figure()
    plt.plot(rounds, train_losses, marker="o", label="Train Loss")
    plt.plot(rounds, test_losses, marker="o", label="Test Loss")
    plt.title("Loss vs Round")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    return _save_current_figure(output_dir, "loss_curve")


def _plot_optional_round_metric(
    round_metrics: list[dict],
    output_dir: Path,
    metric_name: str,
    title: str,
    ylabel: str,
    filename_stem: str,
) -> list[Path]:
    """Plot one optional metric, skipping missing or null-only records."""
    records = [record for record in round_metrics if record.get(metric_name) is not None]
    if not records:
        print(f"No {metric_name} records found. Skipping {title}.")
        return []

    plt.figure()
    plt.plot(
        [record["round_index"] for record in records],
        [record[metric_name] for record in records],
        marker="o",
        label=ylabel,
    )
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.legend()
    return _save_current_figure(output_dir, filename_stem)


def _plot_poisoned_samples(
    client_metrics: list[dict],
    output_dir: Path,
) -> list[Path]:
    """Plot total poisoned samples by federated round."""
    if not client_metrics or all(
        "num_poisoned_samples" not in record
        for record in client_metrics
    ):
        print("No poisoned sample records found. Skipping poisoned samples curve.")
        return []

    poisoned_by_round: dict[int, int] = defaultdict(int)
    for record in client_metrics:
        poisoned_by_round[record["round_index"]] += record.get(
            "num_poisoned_samples",
            0,
        )

    rounds = sorted(poisoned_by_round)
    poisoned_samples = [poisoned_by_round[round_index] for round_index in rounds]

    plt.figure()
    plt.plot(rounds, poisoned_samples, marker="o", label="Poisoned Samples")
    plt.title("Poisoned Samples vs Round")
    plt.xlabel("Round")
    plt.ylabel("Poisoned Samples")
    plt.legend()
    return _save_current_figure(output_dir, "poisoned_samples_curve")


def generate_plots(database_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Generate publication-ready metric curves from a SQLite database."""
    database_path = Path(database_path)
    if not database_path.exists():
        raise FileNotFoundError(f"metrics database not found: {database_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    storage = MetricsStorage(database_path)
    round_metrics = storage.load_round_metrics()
    client_metrics = storage.load_client_metrics()
    if not round_metrics:
        print("No round metrics found. Skipping plots.")
        return []

    generated_paths = []
    generated_paths.extend(_plot_clean_accuracy(round_metrics, output_dir))
    generated_paths.extend(_plot_asr(round_metrics, output_dir))
    generated_paths.extend(_plot_losses(round_metrics, output_dir))
    generated_paths.extend(
        _plot_optional_round_metric(
            round_metrics, output_dir, "psnr", "PSNR vs Round", "PSNR (dB)", "psnr_curve"
        )
    )
    generated_paths.extend(
        _plot_optional_round_metric(
            round_metrics, output_dir, "ssim", "SSIM vs Round", "SSIM", "ssim_curve"
        )
    )
    generated_paths.extend(
        _plot_optional_round_metric(
            round_metrics, output_dir, "linf", "L_inf vs Round", "L_inf", "linf_curve"
        )
    )
    generated_paths.extend(_plot_poisoned_samples(client_metrics, output_dir))
    return generated_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate experiment metric plots.")
    parser.add_argument("--db", default="outputs/metrics.db", help="Path to metrics.db")
    parser.add_argument("--out", default="outputs/figures", help="Output directory")
    return parser.parse_args()


def main() -> None:
    """Run the plot generator from the command line."""
    args = parse_args()
    generated_paths = generate_plots(args.db, args.out)
    for path in generated_paths:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()

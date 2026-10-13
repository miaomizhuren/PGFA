#!/usr/bin/env python3
"""Plot completed client-scale diagnostics without hard-coding result values."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/summaries/client_scale"
COMMON_ROUNDS = tuple(range(100, 201, 10))
LATE_ROUNDS = (100, 120, 150, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400)
E1_CONFIG = "configs/client_scale/client_scale_100c10_proto_uap_layer4_proposed.yaml"
E3_CONFIG = "configs/client_scale/client_scale_100c10_e3_proto_uap_layer4_proposed.yaml"
E5_CONFIG = "configs/client_scale/client_scale_100c10_e5_proto_uap_layer4_proposed.yaml"
E5_LATE_CONFIG = "configs/client_scale/client_scale_100c10_e5_r400_late_proto_uap_layer4_proposed.yaml"
E3_LATE_CONFIG = "configs/client_scale/client_scale_100c10_e3_r400_late_proto_uap_layer4_proposed.yaml"
MAINEXP_METHOD = "Proto-UAP Layer4 Proposed"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_row(rows: list[dict[str, str]], config_path: str) -> dict[str, str] | None:
    return next((row for row in rows if row.get("config_path") == config_path), None)


def complete(row: dict[str, str] | None) -> bool:
    return row is not None and row.get("status", "").lower() == "complete"


def points(row: dict[str, str], metric: str, rounds: tuple[int, ...]) -> tuple[list[int], list[float]]:
    result: list[tuple[int, float]] = []
    for round_index in rounds:
        value = row.get(f"{metric}@{round_index}", "").strip()
        if value:
            try:
                result.append((round_index, float(value)))
            except ValueError:
                print(f"Skipping invalid {metric}@{round_index}: {value!r}")
    return [round_index for round_index, _ in result], [value for _, value in result]


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_curves(metric: str, ylabel: str, curves: list[tuple[str, dict[str, str]]], rounds: tuple[int, ...], stop_round: int, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, row in curves:
        x_values, y_values = points(row, metric, rounds)
        if x_values:
            ax.plot(x_values, y_values, marker="o", label=label)
    ax.axvline(stop_round, color="black", linestyle="--", linewidth=1)
    ax.annotate("Attack stops", xy=(stop_round, 1), xycoords=("data", "axes fraction"),
                xytext=(5, -6), textcoords="offset points", va="top", fontsize=8)
    ax.set(xlabel="Round", ylabel=ylabel, xlim=(min(rounds), max(rounds)))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, stem)


def main() -> None:
    summary_path = OUT / "client_scale_summary.csv"
    mainexp_path = ROOT / "experiments/summaries/mainexp/mainexp_summary.csv"
    if not summary_path.exists() or not mainexp_path.exists():
        print("Required client-scale or mainexp summary is missing; run the summary scripts first.")
        return
    rows, mainexp_rows = read_csv(summary_path), read_csv(mainexp_path)
    mainexp = next((row for row in mainexp_rows if row.get("method") == MAINEXP_METHOD), None)
    common: list[tuple[str, dict[str, str]]] = []
    if mainexp is not None:
        common.append(("20 clients / 10 per round / e1", mainexp))
    for label, path in (
        ("100 clients / 10 per round / e1", E1_CONFIG),
        ("100 clients / 10 per round / e3", E3_CONFIG),
        ("100 clients / 10 per round / e5", E5_CONFIG),
    ):
        row = find_row(rows, path)
        if complete(row):
            common.append((label, row))
    if common:
        plot_curves("ASR", "ASR (%)", common, COMMON_ROUNDS, 100, "client_scale_asr_curve_common")
        plot_curves("CleanACC", "Clean ACC (%)", common, COMMON_ROUNDS, 100, "client_scale_clean_acc_curve")

    late_curves: list[tuple[str, dict[str, str]]] = []
    for label, path in (
        ("100 clients / 10 per round / e5 r400 late", E5_LATE_CONFIG),
        ("100 clients / 10 per round / e3 r400 late", E3_LATE_CONFIG),
    ):
        late = find_row(rows, path)
        if complete(late):
            late_curves.append((label, late))
        else:
            status = late.get("status", "missing") if late else "missing"
            print(f"{label} missing or incomplete (status: {status}); curve omitted.")
    if not late_curves:
        print("No completed r400 late result is available; generated only available common curves.")
        return
    plot_curves("ASR", "ASR (%)", late_curves, LATE_ROUNDS, 220, "client_scale_r400_late_asr_curve")
    plot_curves("CleanACC", "Clean ACC (%)", late_curves, LATE_ROUNDS, 220, "client_scale_r400_late_clean_acc_curve")


if __name__ == "__main__":
    main()

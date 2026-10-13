#!/usr/bin/env python3
"""Create final paper figures from existing CSV/SQLite experiment artifacts only.

No training is started and no source result file is modified.  Outputs are written
under outputs/final_paper_figures/ as PNG, PDF, and generation_summary.md.
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_paper_figures"
PLOT_TABLES = OUT / "plot_tables"
SUMMARY = ROOT / "outputs" / "final_paper_summary_20260918"

COLORS = {
    "CE-only UAP": "#708090",
    "DBA": "#C58B6A",
    "Neurotoxin": "#8C9A74",
    "A3FL": "#8D83A9",
    "PGFA (Ours)": "#2F6F8F",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (SUMMARY / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def save(fig: plt.Figure, stem: str) -> list[Path]:
    paths = [OUT / f"{stem}.png", OUT / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def write_plot_table(name: str, rows: list[dict[str, object]], fields: list[str]) -> Path:
    """Write derived plotting data only; source result artifacts remain untouched."""
    PLOT_TABLES.mkdir(parents=True, exist_ok=True)
    path = PLOT_TABLES / name
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def load_rounds(db: Path, start: int, end: int) -> list[tuple[int, float, float]]:
    query = """SELECT round_index, clean_accuracy, attack_success_rate
               FROM round_metrics WHERE round_index BETWEEN ? AND ?
               ORDER BY round_index"""
    with sqlite3.connect(db) as conn:
        data = conn.execute(query, (start, end)).fetchall()
    if len(data) != end - start + 1:
        raise ValueError(f"Incomplete rounds {start}-{end} in {db}: {len(data)} rows")
    return [(int(r), float(c), float(a)) for r, c, a in data]


def figure3() -> list[Path]:
    main = ROOT / "outputs" / "main_experiment_clean3000_force"
    sources: dict[str, list[tuple[Path, int, int]]] = {
        "CE-only UAP": [(main / "ce_uap_clean3000_attack3001_3200_observe3201_3700" / "metrics_ce_uap_clean3000_attack3001_3200_observe3201_3700.db", 3200, 3700)],
        # DBA's original run stopped at 3590; its recorded continuation supplies 3501-3700.
        "DBA": [(main / "dba_clean3000_attack3001_3200_observe3201_3700" / "metrics_dba_clean3000_attack3001_3200_observe3201_3700.db", 3200, 3500),
                (ROOT / "outputs" / "parallel_13_resume_unfinished" / "resume_dba_from3500_to3700" / "metrics_resume_dba_from3500_to3700.db", 3501, 3700)],
        "Neurotoxin": [(main / "neurotoxin_clean3000_attack3001_3200_observe3201_3700_rerun1" / "metrics_neurotoxin_clean3000_attack3001_3200_observe3201_3700_rerun1.db", 3200, 3700)],
        "A3FL": [(main / "a3fl_clean3000_attack3001_3200_observe3201_3700" / "metrics_a3fl_clean3000_attack3001_3200_observe3201_3700.db", 3200, 3700)],
        "PGFA (Ours)": [(ROOT / "outputs" / "pbu_force_clean_stage_sensitivity" / "clean3000_attack3001_3200_observe3201_3700" / "metrics_pbu_force_clean3000_attack3001_3200_observe3201_3700.db", 3200, 3700)],
    }
    table_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), sharex=True)
    for method, chunks in sources.items():
        points = [point for db, start, end in chunks for point in load_rounds(db, start, end)]
        x = [round_index - 3200 for round_index, _, _ in points]
        clean = [value for _, value, _ in points]
        asr = [value for _, _, value in points]
        table_rows.extend({"method": method, "relative_post_round": round_index - 3200,
                           "global_round": round_index, "clean_accuracy": clean_value,
                           "attack_success_rate": asr_value}
                          for round_index, clean_value, asr_value in points)
        width = 2.4 if method == "PGFA (Ours)" else 1.45
        zorder = 5 if method == "PGFA (Ours)" else 2
        axes[0].plot(x, asr, label=method, color=COLORS[method], lw=width, zorder=zorder)
        axes[1].plot(x, clean, label=method, color=COLORS[method], lw=width, zorder=zorder)
    axes[0].set_title("(a) Attack success rate", fontsize=11)
    axes[1].set_title("(b) Clean accuracy", fontsize=11)
    for ax in axes:
        style_axes(ax); ax.set_xlim(0, 500); ax.set_xlabel("Post-attack relative round")
    axes[0].set_ylabel("ASR (%)"); axes[1].set_ylabel("Clean Accuracy (%)")
    axes[0].legend(ncol=2, frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    write_plot_table("figure3_cifar10_trajectory.csv", table_rows,
                     ["method", "relative_post_round", "global_round", "clean_accuracy", "attack_success_rate"])
    return save(fig, "figure3_cifar10_post_attack_performance")


def figure4() -> list[Path]:
    rows = read_csv("06_ablation_results.csv")
    labels = ["CE-only UAP", "Prototype in\nUAP generation\nonly", "Prototype in\nmalicious training\nonly", "Full PGFA"]
    metrics = [("Post_AUC_1_500", "(a) PA-AUC"), ("ASR_post_500", "(b) ASR@+500 (%)"), ("ASR_ge_80_duration", "(c) N80 (rounds)")]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.05))
    for ax, (key, title) in zip(axes, metrics):
        values = [f(row, key) for row in rows]
        ax.bar(range(4), values, color=[COLORS["CE-only UAP"], "#A7B7BC", "#6F9AA8", COLORS["PGFA (Ours)"]], width=0.68)
        ax.set_title(title, fontsize=11); ax.set_xticks(range(4), labels, fontsize=7.5)
        style_axes(ax)
    fig.tight_layout()
    write_plot_table("figure4_ablation_plot_data.csv",
                     [{"ablation": labels[i], "pa_auc": f(row, "Post_AUC_1_500"),
                       "asr_at_500": f(row, "ASR_post_500"), "n80": f(row, "ASR_ge_80_duration")}
                      for i, row in enumerate(rows)],
                     ["ablation", "pa_auc", "asr_at_500", "n80"])
    return save(fig, "figure4_ablation")


def figure5() -> list[Path]:
    rows = read_csv("07_mechanism_feature_persistence.csv")
    metric_specs = [("Triggered_to_target_cosine", "(a) Target cosine similarity"), ("Feature_drift_distance", "(b) Feature drift"), ("Target_vs_source_margin", "(c) Target–source margin")]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.45))
    for ax, (key, title) in zip(axes, metric_specs):
        for original, label in [("CE-only UAP", "CE-only UAP"), ("Ours / PBU", "PGFA (Ours)")]:
            selected = sorted((row for row in rows if row["Method"] == original), key=lambda row: int(row["Post_round"]))
            ax.plot([int(row["Post_round"]) for row in selected], [f(row, key) for row in selected], marker="o", ms=4, lw=2.4 if label == "PGFA (Ours)" else 1.5, color=COLORS[label], label=label)
        style_axes(ax); ax.set_title(title, fontsize=11); ax.set_xticks([0, 100, 200, 300, 500], ["Attack end", "+100", "+200", "+300", "+500"], fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    write_plot_table("figure5_mechanism_plot_data.csv",
                     [{"method": "PGFA (Ours)" if row["Method"] == "Ours / PBU" else row["Method"],
                       "post_round": row["Post_round"], "target_cosine_similarity": row["Triggered_to_target_cosine"],
                       "feature_drift": row["Feature_drift_distance"], "target_source_margin": row["Target_vs_source_margin"]}
                      for row in rows],
                     ["method", "post_round", "target_cosine_similarity", "feature_drift", "target_source_margin"])
    return save(fig, "figure5_mechanism")


def figure6() -> list[Path]:
    rows = [r for r in read_csv("03_generalization_gtsrb_svhn.csv") if r["Dataset"] in {"CIFAR-10", "GTSRB", "SVHN"}]
    datasets = ["CIFAR-10", "GTSRB", "SVHN"]; x = np.arange(len(datasets)); width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    for ax, (key, title) in zip(axes, [("Post_AUC_1_500", "(a) PA-AUC across datasets"), ("ASR_post_500", "(b) ASR@+500 across datasets")]):
        for offset, original, label in [(-width / 2, "CE-only UAP", "CE-only UAP"), (width / 2, "Ours / PBU", "PGFA (Ours)")]:
            values = [f(next(r for r in rows if r["Dataset"] == dataset and r["Method"] == original), key) for dataset in datasets]
            ax.bar(x + offset, values, width, label=label, color=COLORS[label])
        style_axes(ax); ax.set_title(title, fontsize=11); ax.set_xticks(x, datasets)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    write_plot_table("figure6_generalization_plot_data.csv",
                     [{"dataset": row["Dataset"], "method": "PGFA (Ours)" if row["Method"] == "Ours / PBU" else row["Method"],
                       "pa_auc": row["Post_AUC_1_500"], "asr_at_500": row["ASR_post_500"]} for row in rows],
                     ["dataset", "method", "pa_auc", "asr_at_500"])
    return save(fig, "figure6_generalization")


def appendix_a1() -> list[Path]:
    rows = sorted(read_csv("08_sensitivity_results.csv"), key=lambda r: int(r["Attack_duration"]))
    x = np.arange(len(rows)); labels = [row["Attack_duration"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for ax, (key, title) in zip(axes, [("Post_AUC_1_500", "(a) PA-AUC"), ("ASR_post_500", "(b) ASR@+500 (%)")]):
        ax.plot(x, [f(row, key) for row in rows], marker="o", color=COLORS["PGFA (Ours)"], lw=2.2)
        style_axes(ax); ax.set_title(title, fontsize=11); ax.set_xticks(x, labels); ax.set_xlabel("Attack duration (rounds)")
    fig.tight_layout()
    write_plot_table("appendix_a1_duration_plot_data.csv",
                     [{"attack_duration": row["Attack_duration"], "pa_auc": row["Post_AUC_1_500"],
                       "asr_at_500": row["ASR_post_500"]} for row in rows],
                     ["attack_duration", "pa_auc", "asr_at_500"])
    return save(fig, "appendix_figure_a1_duration_sensitivity")


def appendix_a2() -> list[Path]:
    rows = read_csv("05_defense_results.csv"); labels = ["None", "Norm\nClipping", "RLR", "CRFL-style", "Multi-Krum"]
    x = np.arange(len(rows)); fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.45))
    for ax, (key, title) in zip(axes, [("Post_AUC_1_500", "(a) PA-AUC"), ("ASR_post_500", "(b) ASR@+500 (%)")]):
        ax.bar(x, [f(row, key) for row in rows], color=COLORS["PGFA (Ours)"], alpha=0.88)
        style_axes(ax); ax.set_title(title, fontsize=11); ax.set_xticks(x, labels, fontsize=8)
    fig.tight_layout()
    write_plot_table("appendix_a2_defense_plot_data.csv",
                     [{"defense": row["Defense"], "pa_auc": row["Post_AUC_1_500"],
                       "asr_at_500": row["ASR_post_500"]} for row in rows],
                     ["defense", "pa_auc", "asr_at_500"])
    return save(fig, "appendix_figure_a2_defense")


def appendix_a3() -> list[Path]:
    raw = read_csv("02_seed_stability_cifar10.csv"); summary = read_csv("02_seed_stability_cifar10_mean_std.csv")
    metric_map = [("Post_AUC_1_500", "PA-AUC"), ("ASR_post_500", "ASR@+500 (%)"), ("ASR_ge_80_duration", "N80 (rounds)")]
    fig, ax = plt.subplots(figsize=(6.1, 3.7)); x = np.arange(3)
    for idx, (raw_key, label) in enumerate(metric_map):
        stats = next(row for row in summary if row["Metric"] == raw_key)
        values = np.array([f(row, raw_key) for row in raw])
        jitter = np.linspace(-0.07, 0.07, len(values))
        ax.scatter(np.full(len(values), idx) + jitter, values, color=COLORS["PGFA (Ours)"], alpha=0.65, s=24, zorder=3)
        ax.errorbar(idx, f(stats, "Mean"), yerr=f(stats, "Std"), color="#1D4E63", capsize=5, lw=2.0, marker="o", ms=5, zorder=4)
    style_axes(ax); ax.set_xticks(x, [label for _, label in metric_map]); ax.set_title("Appendix Figure A3: Five-seed stability of PGFA", fontsize=11)
    ax.set_ylabel("Metric value (native units)")
    fig.tight_layout()
    table_rows = []
    for raw_key, label in metric_map:
        stats = next(row for row in summary if row["Metric"] == raw_key)
        for row in raw:
            table_rows.append({"metric": label, "seed": row["Seed"], "value": row[raw_key],
                               "mean": stats["Mean"], "std": stats["Std"]})
    write_plot_table("appendix_a3_seed_plot_data.csv", table_rows, ["metric", "seed", "value", "mean", "std"])
    return save(fig, "appendix_figure_a3_seed_stability")


def write_summary(created: dict[str, list[Path]]) -> None:
    source_files = ["06_ablation_results.csv", "07_mechanism_feature_persistence.csv", "03_generalization_gtsrb_svhn.csv", "08_sensitivity_results.csv", "05_defense_results.csv", "02_seed_stability_cifar10.csv", "02_seed_stability_cifar10_mean_std.csv"]
    lines = ["# Paper-figure generation summary", "", "All figures were generated by `scripts/plot_paper_figures.py` using existing result artifacts only.", "", "## Integrity and protocol checks", "", "- **No training was started.**", "- **No original CSV or SQLite result was modified.** Only new files under `outputs/final_paper_figures/` were written.", "- **CIFAR-100 was excluded** from Figure 6; its plot table contains only CIFAR-10, GTSRB, and SVHN.", "- **DBA uses the required composite trajectory:** original rounds 3200–3500 and saved continuation rounds 3501–3700, for 501 rows.", "", "## Source files", ""]
    lines += [f"- `{(SUMMARY / name).relative_to(ROOT)}`" for name in source_files]
    lines += ["", "## Figure 3 SQLite sources and trajectory rows", "", "All queries use `round_metrics(round_index, attack_success_rate, clean_accuracy)`.", "", "- CE-only UAP — 501 rows: `outputs/main_experiment_clean3000_force/ce_uap_clean3000_attack3001_3200_observe3201_3700/metrics_ce_uap_clean3000_attack3001_3200_observe3201_3700.db`", "- DBA — 501 rows (composite): `outputs/main_experiment_clean3000_force/dba_clean3000_attack3001_3200_observe3201_3700/metrics_dba_clean3000_attack3001_3200_observe3201_3700.db` (3200–3500) + `outputs/parallel_13_resume_unfinished/resume_dba_from3500_to3700/metrics_resume_dba_from3500_to3700.db` (3501–3700)", "- Neurotoxin — 501 rows: `outputs/main_experiment_clean3000_force/neurotoxin_clean3000_attack3001_3200_observe3201_3700_rerun1/metrics_neurotoxin_clean3000_attack3001_3200_observe3201_3700_rerun1.db`", "- A3FL — 501 rows: `outputs/main_experiment_clean3000_force/a3fl_clean3000_attack3001_3200_observe3201_3700/metrics_a3fl_clean3000_attack3001_3200_observe3201_3700.db`", "- PGFA (Ours; stored as PBU) — 501 rows: `outputs/pbu_force_clean_stage_sensitivity/clean3000_attack3001_3200_observe3201_3700/metrics_pbu_force_clean3000_attack3001_3200_observe3201_3700.db`", "", "## Generated files", ""]
    for figure, paths in created.items():
        lines.append(f"- **{figure}: success** — " + ", ".join(f"`{path.relative_to(ROOT)}`" for path in paths))
    lines += ["", "## Derived plotting tables", ""]
    lines += [f"- `outputs/final_paper_figures/plot_tables/{name}`" for name in ["figure3_cifar10_trajectory.csv", "figure4_ablation_plot_data.csv", "figure5_mechanism_plot_data.csv", "figure6_generalization_plot_data.csv", "appendix_a1_duration_plot_data.csv", "appendix_a2_defense_plot_data.csv", "appendix_a3_seed_plot_data.csv"]]
    (OUT / "generation_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "normal", "figure.facecolor": "white"})
    created = {
        "Figure 3": figure3(), "Figure 4": figure4(), "Figure 5": figure5(), "Figure 6": figure6(),
        "Appendix Figure A1": appendix_a1(), "Appendix Figure A2": appendix_a2(), "Appendix Figure A3": appendix_a3(),
    }
    write_summary(created)


if __name__ == "__main__":
    main()

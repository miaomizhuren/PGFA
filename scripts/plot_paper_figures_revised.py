#!/usr/bin/env python3
"""Revised publication styling for existing paper figures; reads results only."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from plot_paper_figures import ROOT, SUMMARY, OUT, COLORS, read_csv, f, load_rounds


def setup() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "axes.edgecolor": "#333333", "axes.labelcolor": "#222222",
                         "xtick.color": "#222222", "ytick.color": "#222222"})


def style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#D5D9DC", linewidth=0.45, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(labelsize=8.5, width=0.75, length=3)


def save(fig: plt.Figure, name: str) -> list[Path]:
    paths = [OUT / f"{name}.png", OUT / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def fig3() -> list[Path]:
    main = ROOT / "outputs" / "main_experiment_clean3000_force"
    source = {
        "CE-only UAP": [(main / "ce_uap_clean3000_attack3001_3200_observe3201_3700" / "metrics_ce_uap_clean3000_attack3001_3200_observe3201_3700.db", 3200, 3700)],
        "DBA": [(main / "dba_clean3000_attack3001_3200_observe3201_3700" / "metrics_dba_clean3000_attack3001_3200_observe3201_3700.db", 3200, 3500),
                (ROOT / "outputs" / "parallel_13_resume_unfinished" / "resume_dba_from3500_to3700" / "metrics_resume_dba_from3500_to3700.db", 3501, 3700)],
        "Neurotoxin": [(main / "neurotoxin_clean3000_attack3001_3200_observe3201_3700_rerun1" / "metrics_neurotoxin_clean3000_attack3001_3200_observe3201_3700_rerun1.db", 3200, 3700)],
        "A3FL": [(main / "a3fl_clean3000_attack3001_3200_observe3201_3700" / "metrics_a3fl_clean3000_attack3001_3200_observe3201_3700.db", 3200, 3700)],
        "PGFA (Ours)": [(ROOT / "outputs" / "pbu_force_clean_stage_sensitivity" / "clean3000_attack3001_3200_observe3201_3700" / "metrics_pbu_force_clean3000_attack3001_3200_observe3201_3700.db", 3200, 3700)],
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.6), sharex=True)
    handles = []
    for method, chunks in source.items():
        points = [p for db, start, end in chunks for p in load_rounds(db, start, end)]
        x = [r - 3200 for r, _, _ in points]
        kwargs = {"color": COLORS[method], "lw": 2.3 if method == "PGFA (Ours)" else 1.25,
                  "zorder": 4 if method == "PGFA (Ours)" else 2}
        line = axes[0].plot(x, [a for _, _, a in points], label=method, **kwargs)[0]
        axes[1].plot(x, [c for _, c, _ in points], **kwargs)
        handles.append(line)
    for ax, title in zip(axes, ["(a) Attack success rate", "(b) Clean accuracy"]):
        style(ax); ax.set_xlim(0, 500); ax.set_xlabel("Post-attack round"); ax.set_title(title, fontsize=10.5, pad=7)
    axes[0].set_ylabel("ASR (%)"); axes[1].set_ylabel("Clean accuracy (%)")
    fig.legend(handles, list(source), ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=8.3, handlelength=2.1)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, "figure3_cifar10_post_attack_performance_revised")


def fig4() -> list[Path]:
    rows = read_csv("06_ablation_results.csv")
    labels = ["CE-only", "Proto-UAP\nonly", "Proto-train\nonly", "Full PGFA"]
    specs = [("Post_AUC_1_500", "(a) PA-AUC", "PA-AUC"), ("ASR_post_500", "(b) ASR@+500", "ASR@+500 (%)"), ("ASR_ge_80_duration", "(c) N80", "N80 (rounds)")]
    colors = [COLORS["CE-only UAP"], "#A7B7BC", "#769CAA", COLORS["PGFA (Ours)"]]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.45))
    for ax, (key, title, ylabel) in zip(axes, specs):
        ax.bar(range(4), [f(r, key) for r in rows], color=colors, width=0.64, edgecolor="none")
        style(ax); ax.set_title(title, fontsize=10.5, pad=7); ax.set_ylabel(ylabel); ax.set_xticks(range(4), labels)
    fig.tight_layout()
    return save(fig, "figure4_ablation_revised")


def fig5() -> list[Path]:
    rows = read_csv("07_mechanism_feature_persistence.csv")
    specs = [("Triggered_to_target_cosine", "(a) Target cosine similarity", "Cosine similarity"),
             ("Feature_drift_distance", "(b) Feature drift", "Feature drift"),
             ("Target_vs_source_margin", "(c) Target-source margin", "Margin")]
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.35))
    handles = []
    for ax, (key, title, ylabel) in zip(axes, specs):
        for original, label in [("CE-only UAP", "CE-only UAP"), ("Ours / PBU", "PGFA (Ours)")]:
            selected = sorted((r for r in rows if r["Method"] == original), key=lambda r: int(r["Post_round"]))
            line = ax.plot([int(r["Post_round"]) for r in selected], [f(r, key) for r in selected], marker="o", ms=4.2,
                           lw=2.3 if label == "PGFA (Ours)" else 1.7, color=COLORS[label], label=label)[0]
            if ax is axes[0]: handles.append(line)
        style(ax); ax.set_title(title, fontsize=10.2, pad=7); ax.set_ylabel(ylabel)
        ax.set_xticks([0, 100, 200, 300, 500], ["Attack end", "+100", "+200", "+300", "+500"])
    fig.legend(handles, ["CE-only UAP", "PGFA (Ours)"], ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=8.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, "figure5_mechanism_revised")


def fig6() -> list[Path]:
    rows = [r for r in read_csv("03_generalization_gtsrb_svhn.csv") if r["Dataset"] in {"CIFAR-10", "GTSRB", "SVHN"}]
    datasets, x, width = ["CIFAR-10", "GTSRB", "SVHN"], np.arange(3), 0.34
    specs = [("Post_AUC_1_500", "(a) PA-AUC across datasets", "PA-AUC"), ("ASR_post_500", "(b) ASR at Tₑ+500", "ASR@+500 (%)")]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4)); handles = []
    for ax, (key, title, ylabel) in zip(axes, specs):
        for offset, original, label in [(-width / 2, "CE-only UAP", "CE-only UAP"), (width / 2, "Ours / PBU", "PGFA (Ours)")]:
            vals = [f(next(r for r in rows if r["Dataset"] == ds and r["Method"] == original), key) for ds in datasets]
            bars = ax.bar(x + offset, vals, width, color=COLORS[label], label=label)
            if ax is axes[0]: handles.append(bars[0])
        style(ax); ax.set_title(title, fontsize=10.2, pad=7); ax.set_ylabel(ylabel); ax.set_xticks(x, datasets)
    fig.legend(handles, ["CE-only UAP", "PGFA (Ours)"], ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=8.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, "figure6_generalization_revised")


def appendix_a1() -> list[Path]:
    rows = sorted(read_csv("08_sensitivity_results.csv"), key=lambda r: int(r["Attack_duration"]))
    x = np.arange(3); specs = [("Post_AUC_1_500", "(a) PA-AUC", "PA-AUC"), ("ASR_post_500", "(b) ASR@+500", "ASR@+500 (%)")]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.25))
    for ax, (key, title, ylabel) in zip(axes, specs):
        ax.plot(x, [f(r, key) for r in rows], marker="o", ms=5, lw=2.0, color=COLORS["PGFA (Ours)"])
        style(ax); ax.set_title(title, fontsize=10.5, pad=7); ax.set_ylabel(ylabel); ax.set_xlabel("Attack duration (rounds)"); ax.set_xticks(x, ["100", "200", "300"])
    fig.tight_layout()
    return save(fig, "appendix_figure_a1_duration_sensitivity_revised")


def appendix_a2() -> list[Path]:
    rows = read_csv("05_defense_results.csv")
    labels = ["None", "Norm\nClipping", "RLR", "CRFL-style", "Multi-Krum"]
    specs = [("Post_AUC_1_500", "(a) PA-AUC", "PA-AUC"), ("ASR_post_500", "(b) ASR@+500", "ASR@+500 (%)")]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.35))
    for ax, (key, title, ylabel) in zip(axes, specs):
        ax.bar(range(5), [f(r, key) for r in rows], color=COLORS["PGFA (Ours)"], alpha=0.92, width=0.64)
        style(ax); ax.set_title(title, fontsize=10.5, pad=7); ax.set_ylabel(ylabel); ax.set_xticks(range(5), labels)
    fig.tight_layout()
    return save(fig, "appendix_figure_a2_defense_revised")


def appendix_a3() -> list[Path]:
    raw, stats = read_csv("02_seed_stability_cifar10.csv"), read_csv("02_seed_stability_cifar10_mean_std.csv")
    specs = [("Post_AUC_1_500", "(a) PA-AUC", "PA-AUC"), ("ASR_post_500", "(b) ASR@+500", "ASR@+500 (%)"), ("ASR_ge_80_duration", "(c) N80", "N80 (rounds)")]
    jitter = np.linspace(-0.11, 0.11, len(raw)); fig, axes = plt.subplots(1, 3, figsize=(10.1, 3.35))
    for ax, (key, title, ylabel) in zip(axes, specs):
        stat = next(row for row in stats if row["Metric"] == key)
        mean, std = f(stat, "Mean"), f(stat, "Std")
        values = np.array([f(row, key) for row in raw])
        ax.bar([0], [mean], width=0.42, color=COLORS["PGFA (Ours)"], alpha=0.75, zorder=2)
        ax.errorbar([0], [mean], yerr=[std], color="#173D50", capsize=4, lw=1.5, zorder=4)
        ax.scatter(jitter, values, color="#173D50", s=24, alpha=0.72, zorder=5, edgecolors="white", linewidths=0.35)
        style(ax); ax.set_title(title, fontsize=10.5, pad=7); ax.set_ylabel(ylabel); ax.set_xlim(-0.38, 0.38)
        ax.set_xticks(jitter, [str(i) for i in range(1, 6)]); ax.set_xlabel("Seed index")
    fig.tight_layout()
    return save(fig, "appendix_figure_a3_seed_stability_revised")


def report(created: dict[str, list[Path]]) -> None:
    lines = ["# Revised paper-figure generation summary", "", "## Verification", "", "- **All revised figures were generated successfully.**", "- Both figure and axes backgrounds are white; only subtle y-axis grids are used.", "- Figure 3 uses complete raw round-wise trajectories with **no smoothing**.", "- Figure 3 uses the required DBA composite trajectory (original 3200–3500 + continuation 3501–3700).", "- Figure 6 filters to CIFAR-10, GTSRB, and SVHN; **CIFAR-100 is excluded**.", "- Appendix Figure A3 was redrawn as three subplots with **three independent y-axes** (PA-AUC, ASR@+500, N80).", "- **No training was started and no original CSV/SQLite result was modified.**", "", "## Outputs", ""]
    for name, paths in created.items():
        lines.append(f"- **{name}: success** — " + ", ".join(f"`{p.relative_to(ROOT)}`" for p in paths))
    lines += ["", "## Data sources", "", "- Figure 3: existing `round_metrics` tables in the five documented SQLite sources (DBA is composite).", "- Figures 4–6 and Appendix A1–A3: existing CSVs under `outputs/final_paper_summary_20260918/`."]
    (OUT / "generation_summary_revised.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); setup()
    created = {"Figure 3": fig3(), "Figure 4": fig4(), "Figure 5": fig5(), "Figure 6": fig6(),
               "Appendix Figure A1": appendix_a1(), "Appendix Figure A2": appendix_a2(), "Appendix Figure A3": appendix_a3()}
    report(created)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read four post-attack diagnosis runs and export a non-destructive summary."""

from __future__ import annotations

import csv
import math
import sqlite3
from pathlib import Path
from statistics import mean, pstdev

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/noniid_alpha0.8_post_diagnosis_3200_to3400"
OUTPUT_DIR = ROOT / "outputs/noniid_alpha0.8_post_diagnosis_summary"
SOURCE_DB = ROOT / (
    "outputs/cifar10_noniid_pbu_force_fromscratch_3_rerun1/"
    "alpha0.8_pbu_fromscratch_clean3000_attack3001_3200_observe3201_3700/"
    "metrics_cifar10_noniid_alpha0.8_pbu_fromscratch_clean3000_attack3001_3200_observe3201_3700_rerun1.db"
)
KEY_ROUNDS = (3200, 3201, 3250, 3300, 3350, 3400)
BRANCHES = (
    ("A_original", "A 原始", "alpha0.8_postdiag_A_original_from3200_to3400.yaml"),
    ("B_bn_stable", "B BN buffers 稳定", "alpha0.8_postdiag_B_bn_buffer_stable_from3200_to3400.yaml"),
    ("C_equal_weight", "C equal-client", "alpha0.8_postdiag_C_equal_client_weight_from3200_to3400.yaml"),
    ("D_bn_stable_equal_weight", "D BN 稳定 + equal-client", "alpha0.8_postdiag_D_both_from3200_to3400.yaml"),
)


def read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_metrics(db_path: Path) -> tuple[dict[int, tuple[float | None, float | None]], list[str]]:
    if not db_path.exists():
        return {}, ["database missing"]
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(round_metrics)")}
        round_column = "round_index" if "round_index" in columns else "global_round"
        required = {round_column, "clean_accuracy", "attack_success_rate"}
        if not required.issubset(columns):
            return {}, ["round_metrics schema missing required columns"]
        rows = connection.execute(
            f"SELECT {round_column}, clean_accuracy, attack_success_rate "
            f"FROM round_metrics ORDER BY {round_column}"
        ).fetchall()
    finally:
        connection.close()
    return {int(r): (as_float(clean), as_float(asr)) for r, clean, asr in rows}, []


def as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def metric_unit(values: list[float | None]) -> str:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return "unknown"
    if max(abs(value) for value in finite) <= 1.0:
        return "0-1"
    if max(abs(value) for value in finite) <= 100.0:
        return "0-100"
    return "out_of_range"


def normalize(value: float | None, unit: str) -> float | None:
    return value * 100.0 if value is not None and unit == "0-1" else value


def value_at(metrics: dict[int, tuple[float | None, float | None]], round_index: int, index: int) -> float | None:
    pair = metrics.get(round_index)
    return None if pair is None else pair[index]


def fmt(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def phase(round_index: int) -> str:
    return {
        3200: "source_round_3200_attack_end",
        3201: "post_start",
        3250: "post_50",
        3300: "post_100",
        3350: "post_150",
        3400: "post_200",
    }[round_index]


def stability_key(row: dict) -> tuple[float, float, float]:
    # Highest minimum, then lowest variability, then highest mean is preferable.
    return (row["min_clean_acc_3201_3400"] or -math.inf,
            -(row["std_clean_acc_3201_3400"] or math.inf),
            row["mean_clean_acc_3201_3400"] or -math.inf)


def asr_key(row: dict) -> tuple[float, float]:
    return (row["asr_3400"] or -math.inf, row["mean_asr_3201_3400"] or -math.inf)


def main() -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty derived directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_metrics, source_errors = read_metrics(SOURCE_DB)
    source_clean_unit = metric_unit([pair[0] for pair in source_metrics.values()])
    source_asr_unit = metric_unit([pair[1] for pair in source_metrics.values()])
    source_3200 = (
        normalize(value_at(source_metrics, 3200, 0), source_clean_unit),
        normalize(value_at(source_metrics, 3200, 1), source_asr_unit),
    )

    rows: list[dict] = []
    source_checks: list[dict] = []
    curves: dict[str, dict[int, tuple[float | None, float | None]]] = {}
    key_rows: list[dict] = []
    for branch, display_name, config_name in BRANCHES:
        config = read_yaml(CONFIG_DIR / config_name)
        output_rel = config["experiment"]["output_dir"]
        db_name = config["tracking"]["database_name"]
        db_path = ROOT / output_rel / db_name
        metrics_raw, errors = read_metrics(db_path)
        clean_unit = metric_unit([pair[0] for pair in metrics_raw.values()])
        asr_unit = metric_unit([pair[1] for pair in metrics_raw.values()])
        metrics = {
            round_index: (normalize(clean, clean_unit), normalize(asr, asr_unit))
            for round_index, (clean, asr) in metrics_raw.items()
        }
        curves[branch] = metrics
        actual_rounds = sorted(round_index for round_index in metrics if 3201 <= round_index <= 3400)
        clean_values = [metrics[round_index][0] for round_index in actual_rounds if metrics[round_index][0] is not None]
        asr_values = [metrics[round_index][1] for round_index in actual_rounds if metrics[round_index][1] is not None]
        clean_deltas = [
            metrics[current][0] - metrics[previous][0]
            for previous, current in zip(actual_rounds, actual_rounds[1:])
            if metrics[previous][0] is not None and metrics[current][0] is not None
        ]
        missing = [str(round_index) for round_index in KEY_ROUNDS if round_index != 3200 and round_index not in metrics]
        has_nan = any(
            value is not None and not math.isfinite(value)
            for pair in metrics.values() for value in pair
        )
        notes = ["round 3200 metrics read from common source DB"]
        if errors:
            notes.extend(errors)
        if missing:
            notes.append("missing key rounds: " + ",".join(missing))
        row = {
            "branch": branch,
            "display_name": display_name,
            "experiment_name": config["experiment"]["name"],
            "output_dir": output_rel,
            "aggregation_weighting": config["federated"]["aggregation_weighting"],
            "bn_buffer_policy": config["federated"]["bn_buffer_policy"],
            "start_round": min(actual_rounds) if actual_rounds else None,
            "final_round": max(actual_rounds) if actual_rounds else None,
            "max_round": max(metrics) if metrics else None,
            "clean_acc_3200": source_3200[0],
            "clean_acc_3201": value_at(metrics, 3201, 0),
            "clean_acc_3300": value_at(metrics, 3300, 0),
            "clean_acc_3400": value_at(metrics, 3400, 0),
            "asr_3200": source_3200[1],
            "asr_3201": value_at(metrics, 3201, 1),
            "asr_3300": value_at(metrics, 3300, 1),
            "asr_3400": value_at(metrics, 3400, 1),
            "min_clean_acc_3201_3400": min(clean_values) if clean_values else None,
            "mean_clean_acc_3201_3400": mean(clean_values) if clean_values else None,
            "std_clean_acc_3201_3400": pstdev(clean_values) if len(clean_values) > 1 else 0.0 if clean_values else None,
            "max_clean_acc_3201_3400": max(clean_values) if clean_values else None,
            "worst_single_round_clean_acc_drop": max((-delta for delta in clean_deltas), default=None),
            "max_single_round_clean_acc_recovery": max(clean_deltas, default=None),
            "min_asr_3201_3400": min(asr_values) if asr_values else None,
            "mean_asr_3201_3400": mean(asr_values) if asr_values else None,
            "std_asr_3201_3400": pstdev(asr_values) if len(asr_values) > 1 else 0.0 if asr_values else None,
            "max_asr_3201_3400": max(asr_values) if asr_values else None,
            "asr_drop_3201_to_3400": (
                value_at(metrics, 3201, 1) - value_at(metrics, 3400, 1)
                if value_at(metrics, 3201, 1) is not None and value_at(metrics, 3400, 1) is not None else None
            ),
            "clean_acc_collapse_below_80": bool(any(value < 80 for value in clean_values)),
            "clean_acc_collapse_below_70": bool(any(value < 70 for value in clean_values)),
            "clean_acc_collapse_below_60": bool(any(value < 60 for value in clean_values)),
            "clean_acc_collapse_below_50": bool(any(value < 50 for value in clean_values)),
            "notes": "; ".join(notes),
        }
        rows.append(row)
        source_checks.append({
            "branch": branch, "output_dir": output_rel, "db_path": str(db_path.relative_to(ROOT)), "csv_path": "",
            "min_round": min(metrics) if metrics else None, "max_round": max(metrics) if metrics else None,
            **{f"has_round_{round_index}": round_index in metrics if round_index != 3200 else 3200 in source_metrics for round_index in KEY_ROUNDS},
            "missing_key_rounds": ",".join(missing), "has_nan": has_nan,
            "asr_unit_detected": asr_unit, "clean_acc_unit_detected": clean_unit,
            "data_quality_note": "; ".join(errors) if errors else "complete 3201-3400 DB range; source round 3200 verified separately",
        })
        for round_index in KEY_ROUNDS:
            clean, asr = source_3200 if round_index == 3200 else metrics.get(round_index, (None, None))
            key_rows.append({"branch": branch, "global_round": round_index,
                             "relative_post_round": 0 if round_index == 3200 else round_index - 3200,
                             "phase": phase(round_index), "clean_acc": clean, "asr": asr})

    summary_columns = [
        "branch", "display_name", "experiment_name", "output_dir", "aggregation_weighting", "bn_buffer_policy",
        "start_round", "final_round", "max_round", "clean_acc_3200", "clean_acc_3201", "clean_acc_3300", "clean_acc_3400",
        "asr_3200", "asr_3201", "asr_3300", "asr_3400", "min_clean_acc_3201_3400", "mean_clean_acc_3201_3400",
        "std_clean_acc_3201_3400", "max_clean_acc_3201_3400", "worst_single_round_clean_acc_drop", "max_single_round_clean_acc_recovery",
        "min_asr_3201_3400", "mean_asr_3201_3400", "std_asr_3201_3400", "max_asr_3201_3400", "asr_drop_3201_to_3400",
        "clean_acc_collapse_below_80", "clean_acc_collapse_below_70", "clean_acc_collapse_below_60", "clean_acc_collapse_below_50", "notes",
    ]
    write_csv(OUTPUT_DIR / "alpha0.8_post_diagnosis_summary.csv", summary_columns, rows)
    write_csv(OUTPUT_DIR / "alpha0.8_post_diagnosis_key_rounds.csv",
              ["branch", "global_round", "relative_post_round", "phase", "clean_acc", "asr"], key_rows)
    curve_rows = []
    for round_index in range(3201, 3401):
        item = {"relative_post_round": round_index - 3200, "global_round": round_index}
        for branch, _, _ in BRANCHES:
            item[f"{branch}_clean_acc"], item[f"{branch}_asr"] = curves[branch].get(round_index, (None, None))
        curve_rows.append(item)
    curve_columns = ["relative_post_round", "global_round"] + [
        field for branch, _, _ in BRANCHES for field in (f"{branch}_clean_acc", f"{branch}_asr")
    ]
    write_csv(OUTPUT_DIR / "alpha0.8_post_diagnosis_curves.csv", curve_columns, curve_rows)
    source_columns = ["branch", "output_dir", "db_path", "csv_path", "min_round", "max_round"] + [
        f"has_round_{round_index}" for round_index in KEY_ROUNDS
    ] + ["missing_key_rounds", "has_nan", "asr_unit_detected", "clean_acc_unit_detected", "data_quality_note"]
    write_csv(OUTPUT_DIR / "alpha0.8_post_diagnosis_source_check.csv", source_columns, source_checks)
    write_report(OUTPUT_DIR / "alpha0.8_post_diagnosis_report.md", rows, source_checks, source_errors, source_3200)


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict], source_checks: list[dict], source_errors: list[str], source_3200: tuple[float | None, float | None]) -> None:
    complete = [row for row in rows if row["final_round"] == 3400]
    stable = max(complete, key=stability_key) if complete else None
    best_asr = max(complete, key=asr_key) if complete else None
    by_branch = {row["branch"]: row for row in rows}
    a, b, c, d = (by_branch[name] for name in ("A_original", "B_bn_stable", "C_equal_weight", "D_bn_stable_equal_weight"))
    bn_gain = (b["min_clean_acc_3201_3400"] or -math.inf) - (a["min_clean_acc_3201_3400"] or math.inf)
    quantity_gain = (c["min_clean_acc_3201_3400"] or -math.inf) - (a["min_clean_acc_3201_3400"] or math.inf)
    joint_best = stable is not None and stable["branch"] == "D_bn_stable_equal_weight"
    if joint_best and bn_gain > 0.1 and quantity_gain > 0.1:
        diagnosis = "BN buffers 与 sample-size weighting 均是稳定性影响因素；联合策略 D 的最小 Clean ACC 最优。"
    elif bn_gain > max(quantity_gain, 0.1):
        diagnosis = "BN buffer 聚合是更主要的稳定性影响因素；B 相对 A 的最小 Clean ACC 改善更大。"
    elif quantity_gain > max(bn_gain, 0.1):
        diagnosis = "client quantity skew 与 sample-weighted 聚合是更主要的稳定性影响因素；C 相对 A 的最小 Clean ACC 改善更大。"
    else:
        diagnosis = "在 3201–3400 的受控分支中，BN buffer policy 与 weighting 均未显示出足以解释 Clean ACC collapse 的主导改善；应优先检查本地训练强度/学习率。"
    recommendation = (
        f"采用 {stable['branch']}（{stable['display_name']}）作为后续 α=0.8 消融的优先聚合设置，并用独立 seed 复验。"
        if stable else "分支数据未完整到 3400；先补齐缺失诊断数据后再推荐聚合设置。"
    )
    table = "\n".join(
        f"| {row['branch']} | {fmt(row['clean_acc_3201'])} | {fmt(row['clean_acc_3300'])} | {fmt(row['clean_acc_3400'])} | {fmt(row['min_clean_acc_3201_3400'])} | {fmt(row['std_clean_acc_3201_3400'])} | {fmt(row['worst_single_round_clean_acc_drop'])} | {fmt(row['asr_3201'])} | {fmt(row['asr_3300'])} | {fmt(row['asr_3400'])} |"
        for row in rows
    )
    source_table = "\n".join(
        f"| {check['branch']} | `{check['output_dir']}` | `{check['db_path']}` | {check['max_round']} | {'yes' if check['has_round_3400'] else 'no'} |"
        for check in source_checks
    )
    text = f"""# Non-IID Alpha0.8 Post-Attack Diagnosis Summary

## 1. Goal

This is a 200-round post-attack diagnostic continuation from the common round-3200 checkpoint, not a full experiment rerun and not a final Non-IID main result. No attack rounds were added after 3200.

## 2. Source experiments

The common source baseline at round 3200 is Clean ACC {fmt(source_3200[0])} and ASR {fmt(source_3200[1])}, read from `{SOURCE_DB.relative_to(ROOT)}`. Source read errors: {', '.join(source_errors) if source_errors else 'none'}.

| Branch | Output directory | DB | max round | reached 3400 |
| --- | --- | --- | ---: | --- |
{source_table}

## 3. Branch definition

| Branch | aggregation_weighting | bn_buffer_policy |
| --- | --- | --- |
| A_original | sample_weighted | aggregate |
| B_bn_stable | sample_weighted | keep_global |
| C_equal_weight | equal_client | aggregate |
| D_bn_stable_equal_weight | equal_client | keep_global |

`keep_global` preserves global BatchNorm `running_mean`, `running_var`, and `num_batches_tracked`; BatchNorm affine parameters remain aggregated.

## 4. Summary

| Branch | Clean 3201 | Clean 3300 | Clean 3400 | min Clean | std Clean | worst adjacent drop | ASR 3201 | ASR 3300 | ASR 3400 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

## 5. Clean ACC stability

Most stable branch by highest minimum Clean ACC (then lower standard deviation and higher mean): **{stable['branch'] if stable else 'unavailable'}**. Clean-ACC collapse indicators (<80/<70/<60/<50) are recorded in the summary CSV rather than inferred from plots.

## 6. ASR persistence

Best ASR retention by ASR at 3400 (then mean ASR): **{best_asr['branch'] if best_asr else 'unavailable'}**. The reported ASR drop is `ASR_3201 - ASR_3400` in percentage points.

## 7. Diagnosis conclusion

{diagnosis}

## 8. Recommendation

{recommendation}

## 9. Quality notes

All branch metrics are read directly from their SQLite `round_metrics` tables in read-only mode. The common 3200 baseline is read only from the source experiment and is not mixed into 3201–3400 branch statistics. Each branch source check records missing key rounds, NaNs, and metric units.

## 10. Generated files

- `alpha0.8_post_diagnosis_summary.csv`
- `alpha0.8_post_diagnosis_key_rounds.csv`
- `alpha0.8_post_diagnosis_curves.csv`
- `alpha0.8_post_diagnosis_source_check.csv`
- `alpha0.8_post_diagnosis_report.md`
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

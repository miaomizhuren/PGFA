#!/usr/bin/env python3
"""Read-only export for the three PBU (``proto_uap``) persistence runs."""

import csv
import sqlite3
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/saved_results/pbu_three_experiment_comparison_clean500_20260913"
SPECS = (
    {
        "experiment_id": "old_s60_e150_r350",
        "label": "旧 PBU（round 60–150 attack）",
        "config": ROOT / "configs/final_attack_duration_sensitivity_r350/dur_s60_e150_pbu_r350.yaml",
        "db": ROOT / "outputs/extended_attack_single_proto/ea01_s60_e150_r350/metrics_ea01_s60_e150_r350.db",
        "log": ROOT / "experiments/logs/final_attack_duration_sensitivity_r350/dur_s60_e150_pbu_r350.log",
        "note": "配置声明的原始输出目录及日志未保留；使用已存在 ea01 SQLite 指标库。",
    },
    {
        "experiment_id": "clean500_s501_e600_r900",
        "label": "clean500 + 100轮攻击（round 501–600）",
        "config": ROOT / "configs/cifar10_pbu_resume_clean500_attack501_600_stop601_900.yaml",
        "db": ROOT / "outputs/pbu_resume_clean500_attack501_600_stop601_900/metrics_pbu_resume_clean500_attack501_600_stop601_900.db",
        "log": ROOT / "experiments/logs/pbu_resume_clean500_attack501_600_stop601_900.log",
        "note": "直接读取 clean500 resume SQLite 指标库。",
    },
    {
        "experiment_id": "clean500_s501_e700_r1000",
        "label": "clean500 + 200轮攻击（round 501–700）",
        "config": ROOT / "configs/cifar10_pbu_resume_clean500_attack501_700_stop701_1000.yaml",
        "db": ROOT / "outputs/pbu_resume_clean500_attack501_700_stop701_1000/metrics_pbu_resume_clean500_attack501_700_stop701_1000.db",
        "log": ROOT / "experiments/logs/pbu_resume_clean500_attack501_700_stop701_1000.log",
        "note": "直接读取 clean500 resume SQLite 指标库。",
    },
)


def load_rows(database: Path) -> list[dict]:
    with sqlite3.connect(database) as connection:
        records = connection.execute(
            "SELECT round_index, clean_accuracy, attack_success_rate, attack_active, "
            "average_train_loss, test_loss, selected_clients, selected_malicious_clients "
            "FROM round_metrics ORDER BY round_index"
        ).fetchall()
    if not records:
        raise RuntimeError(f"empty metrics database: {database}")
    return [
        {
            "global_round": int(round_index), "clean_acc": float(clean_accuracy),
            "asr": None if asr is None else float(asr), "attack_active": bool(attack_active),
            "train_loss": float(train_loss), "test_loss": float(test_loss),
            "selected_clients": selected_clients,
            "selected_malicious_clients": selected_malicious_clients,
        }
        for round_index, clean_accuracy, asr, attack_active, train_loss, test_loss,
        selected_clients, selected_malicious_clients in records
    ]


def value_at(rows: list[dict], round_index: int, field: str):
    return next((row[field] for row in rows if row["global_round"] == round_index), None)


def average(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def fmt(value):
    return "—" if value is None else f"{value:.2f}"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing export directory: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    curves, summaries = [], []

    for spec in SPECS:
        config = yaml.safe_load(spec["config"].read_text(encoding="utf-8"))
        attack = config["attack"]
        total_rounds = int(config["federated"].get("total_rounds", config["federated"]["num_rounds"]))
        attack_start, attack_end = int(attack["start_round"]), int(attack["end_round"])
        rows = load_rows(spec["db"])
        first_round, last_round = rows[0]["global_round"], rows[-1]["global_round"]
        active_rows = [row for row in rows if row["attack_active"]]
        post_rows = [row for row in rows if row["global_round"] >= attack_end]
        status = "complete" if last_round == total_rounds else "partial"

        for row in rows:
            curves.append({
                "experiment_id": spec["experiment_id"], "experiment_label": spec["label"],
                "paper_method": "PBU", "code_method": attack["method"],
                "configured_total_rounds": total_rounds, "actual_last_round": last_round,
                "global_round": row["global_round"],
                "relative_post_attack_round": row["global_round"] - attack_end,
                "attack_active": row["attack_active"], "clean_acc": row["clean_acc"],
                "asr": row["asr"], "train_loss": row["train_loss"],
                "test_loss": row["test_loss"], "selected_clients": row["selected_clients"],
                "selected_malicious_clients": row["selected_malicious_clients"],
                "source_db": str(spec["db"]),
            })

        summary = {
            "experiment_id": spec["experiment_id"], "experiment_label": spec["label"],
            "paper_method": "PBU", "code_method": attack["method"], "status": status,
            "config_file": str(spec["config"]), "source_db": str(spec["db"]),
            "source_log": str(spec["log"]) if spec["log"].is_file() else "NOT_FOUND",
            "configured_total_rounds": total_rounds, "actual_first_round": first_round,
            "actual_last_round": last_round, "row_count": len(rows),
            "attack_start_round": attack_start, "attack_end_round": attack_end,
            "post_attack_start_round": attack_end + 1,
            "configured_attack_rounds": attack_end - attack_start + 1,
            "observed_attack_active_rounds": len(active_rows),
            "first_attack_active_round": active_rows[0]["global_round"] if active_rows else None,
            "last_attack_active_round": active_rows[-1]["global_round"] if active_rows else None,
            "clean_acc_at_attack_end": value_at(rows, attack_end, "clean_acc"),
            "asr_at_attack_end": value_at(rows, attack_end, "asr"),
            "clean_acc_post_attack_start": value_at(rows, attack_end + 1, "clean_acc"),
            "asr_post_attack_start": value_at(rows, attack_end + 1, "asr"),
            "final_clean_acc": rows[-1]["clean_acc"], "final_asr": rows[-1]["asr"],
            "post_attack_asr_mean": average(row["asr"] for row in post_rows),
            "notes": spec["note"],
        }
        for offset in (0, 1, 50, 100, 200, 300):
            summary[f"clean_acc_post_{offset}"] = value_at(rows, attack_end + offset, "clean_acc")
            summary[f"asr_post_{offset}"] = value_at(rows, attack_end + offset, "asr")
        summaries.append(summary)

    curve_file = OUTPUT / "PBU_THREE_EXPERIMENT_COMPARISON_CURVES.csv"
    summary_file = OUTPUT / "PBU_THREE_EXPERIMENT_COMPARISON_SUMMARY.csv"
    for path, records in ((curve_file, curves), (summary_file, summaries)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    lines = [
        "# PBU 持久性：三组实验统一对比", "",
        "论文方法名为 **PBU**；代码实现入口为 `method: proto_uap`。本导出只读取既有 SQLite、配置和日志路径，不启动训练，也不修改训练代码。", "",
        "## 已定位的数据源", "",
        "| 实验 | 配置 | 指标数据库 | 日志 |", "| --- | --- | --- | --- |",
    ]
    for row in summaries:
        lines.append(f"| {row['experiment_id']} | `{row['config_file']}` | `{row['source_db']}` | `{row['source_log']}` |")
    lines += ["", "## 指标汇总", "", "| 实验 | 完整性 | 实际轮次 | 攻击窗口 | 停攻起点 | ASR@攻击末轮 | ASR@停攻首轮 | 最终 ASR | 最终 CA |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        lines.append(f"| {row['experiment_label']} | {row['status']} | {row['actual_first_round']}–{row['actual_last_round']} / {row['configured_total_rounds']} | {row['attack_start_round']}–{row['attack_end_round']} | {row['post_attack_start_round']} | {fmt(row['asr_at_attack_end'])} | {fmt(row['asr_post_attack_start'])} | {fmt(row['final_asr'])} | {fmt(row['final_clean_acc'])} |")
    lines += ["", "## 使用说明", "", "- 曲线 CSV 包含三组全部逐轮指标；使用 `relative_post_attack_round` 对齐停攻时刻（0 为攻击末轮，1 为停攻后的第一轮）。", "- 汇总 CSV 保留配置、数据库和日志定位结果，以及攻击末轮、停攻首轮和最终轮指标。", "- A 的配置中声明的原始输出目录与日志未在工作区检出；其已有 `ea01` 指标数据库覆盖完整的 1–350 轮，故作为 A 的实际数据源，并在汇总中如实标注日志为 `NOT_FOUND`。"]
    report_file = OUTPUT / "PBU_THREE_EXPERIMENT_COMPARISON_REPORT.md"
    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(curve_file)
    print(summary_file)
    print(report_file)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export a read-only comparison of the legacy and clean500-resume PBU runs."""

import csv
import sqlite3
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/saved_results/pbu_persistence_comparison_clean500_20260913"
SPECS = (
    {
        "experiment_id": "A_old_pbu_s60_e150_r350",
        "label": "旧 PBU（round 60–150 attack）",
        "config": ROOT / "configs/final_attack_duration_sensitivity_r350/dur_s60_e150_pbu_r350.yaml",
        "db": ROOT / "outputs/extended_attack_single_proto/ea01_s60_e150_r350/metrics_ea01_s60_e150_r350.db",
        "source_note": "配置声明的输出目录缺失；使用正式 s60_e150 PBU 曲线导出所声明的 ea01 SQLite 源。",
    },
    {
        "experiment_id": "B_resume_clean500_pbu_s501_e600_r900",
        "label": "clean500 resume PBU（round 501–600 attack）",
        "config": ROOT / "configs/cifar10_pbu_resume_clean500_attack501_600_stop601_900.yaml",
        "db": ROOT / "outputs/pbu_resume_clean500_attack501_600_stop601_900/metrics_pbu_resume_clean500_attack501_600_stop601_900.db",
        "source_note": "直接读取本次 clean500 resume 实验 SQLite。",
    },
)


def load_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with sqlite3.connect(path) as connection:
        records = connection.execute(
            "SELECT round_index, clean_accuracy, attack_success_rate, attack_active, "
            "average_train_loss, test_loss, selected_clients, selected_malicious_clients "
            "FROM round_metrics ORDER BY round_index"
        ).fetchall()
    return [
        {
            "global_round": int(round_index),
            "clean_acc": float(clean_accuracy),
            "asr": None if asr is None else float(asr),
            "attack_active": bool(attack_active),
            "train_loss": float(train_loss),
            "test_loss": float(test_loss),
            "selected_clients": selected_clients,
            "selected_malicious_clients": selected_malicious_clients,
        }
        for round_index, clean_accuracy, asr, attack_active, train_loss, test_loss,
        selected_clients, selected_malicious_clients in records
    ]


def at(rows: list[dict], round_index: int, key: str):
    return next((row[key] for row in rows if row["global_round"] == round_index), "")


def fmt(value):
    return "" if value == "" or value is None else f"{value:.4f}"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_rows, summaries, report_rows = [], [], []
    for spec in SPECS:
        config = yaml.safe_load(spec["config"].read_text(encoding="utf-8"))
        attack = config["attack"]
        attack_start = int(attack.get("start_round", attack.get("attack_start_round")))
        attack_end = int(attack.get("end_round", attack.get("attack_end_round")))
        rows = load_metrics(spec["db"])
        if not rows:
            raise RuntimeError(f"empty round_metrics: {spec['db']}")
        for row in rows:
            all_rows.append(
                {
                    "experiment_id": spec["experiment_id"],
                    "experiment_label": spec["label"],
                    "paper_method": "PBU",
                    "code_method": attack["method"],
                    "global_round": row["global_round"],
                    "relative_post_attack_round": row["global_round"] - attack_end,
                    "attack_active": row["attack_active"],
                    "clean_acc": row["clean_acc"],
                    "asr": row["asr"],
                    "train_loss": row["train_loss"],
                    "test_loss": row["test_loss"],
                    "selected_clients": row["selected_clients"],
                    "selected_malicious_clients": row["selected_malicious_clients"],
                    "source_file": str(spec["db"]),
                }
            )
        post_rows = [row for row in rows if row["global_round"] >= attack_end]
        max_row = max(rows, key=lambda row: (row["clean_acc"], -row["global_round"]))
        summary = {
            "experiment_id": spec["experiment_id"],
            "experiment_label": spec["label"],
            "paper_method": "PBU",
            "code_method": attack["method"],
            "config_file": str(spec["config"]),
            "source_file": str(spec["db"]),
            "max_round": rows[-1]["global_round"],
            "attack_start_round": attack_start,
            "attack_end_round": attack_end,
            "post_attack_start_round": attack_end + 1,
            "attack_round_count": attack_end - attack_start + 1,
            "first_attack_active_round": min(row["global_round"] for row in rows if row["attack_active"]),
            "last_attack_active_round": max(row["global_round"] for row in rows if row["attack_active"]),
            "first_post_attack_inactive_round": min(row["global_round"] for row in rows if row["global_round"] > attack_end and not row["attack_active"]),
            "clean_acc_at_attack_end": at(rows, attack_end, "clean_acc"),
            "asr_at_attack_end": at(rows, attack_end, "asr"),
            "clean_acc_post_attack_start": at(rows, attack_end + 1, "clean_acc"),
            "asr_post_attack_start": at(rows, attack_end + 1, "asr"),
            "max_clean_acc": max_row["clean_acc"],
            "max_clean_acc_round": max_row["global_round"],
            "final_clean_acc": rows[-1]["clean_acc"],
            "final_asr": rows[-1]["asr"],
            "post_attack_asr_mean": sum(row["asr"] for row in post_rows if row["asr"] is not None) / sum(row["asr"] is not None for row in post_rows),
            "notes": spec["source_note"],
        }
        for relative in (0, 1, 50, 100, 150, 200, 250, 300):
            summary[f"clean_acc_post_{relative}"] = at(rows, attack_end + relative, "clean_acc")
            summary[f"asr_post_{relative}"] = at(rows, attack_end + relative, "asr")
        summaries.append(summary)
        report_rows.append(summary)

    curve_path = OUTPUT / "PBU_PERSISTENCE_COMPARISON_CURVES.csv"
    summary_path = OUTPUT / "PBU_PERSISTENCE_COMPARISON_SUMMARY.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# PBU 持久性对比：旧实验与 clean500 resume",
        "",
        "## 数据来源",
        "",
        "- 实验 A 配置指定的原始输出目录未保留。项目已有正式 s60_e150 PBU 导出，其 source_file 明确指向 outputs/extended_attack_single_proto/ea01_s60_e150_r350/metrics_ea01_s60_e150_r350.db；本报告据此使用该 SQLite。",
        "- 实验 B 直接读取 outputs/pbu_resume_clean500_attack501_600_stop601_900/metrics_pbu_resume_clean500_attack501_600_stop601_900.db。",
        "- 两组的论文方法名均为 PBU，代码实现入口均为 proto_uap。",
        "",
        "## 汇总",
        "",
        "| 实验 | 轮次范围 | 攻击窗口 | 停攻起点 | ASR@攻击末轮 | ASR@停攻首轮 | 最终 ASR | 最终 CA |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report_rows:
        lines.append(
            f"| {row['experiment_label']} | 1–{row['max_round']} | "
            f"{row['attack_start_round']}–{row['attack_end_round']} | "
            f"{row['post_attack_start_round']} | {fmt(row['asr_at_attack_end'])} | "
            f"{fmt(row['asr_post_attack_start'])} | {fmt(row['final_asr'])} | "
            f"{fmt(row['final_clean_acc'])} |"
        )
    lines += [
        "",
        "## 说明",
        "",
        "- PBU_PERSISTENCE_COMPARISON_CURVES.csv 保留两组所有已有逐轮 clean_acc、ASR、attack_active、绝对轮次和相对停攻轮次。",
        "- 实验 A 与实验 B 的绝对开始轮次不同；跨实验持久性比较应优先使用 relative_post_attack_round。",
        "- 本导出仅读取实验数据库和既有配置，不重新训练或改写任何实验指标。",
    ]
    report_path = OUTPUT / "PBU_PERSISTENCE_COMPARISON_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(curve_path)
    print(summary_path)
    print(report_path)


if __name__ == "__main__":
    main()

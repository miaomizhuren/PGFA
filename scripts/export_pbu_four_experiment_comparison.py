#!/usr/bin/env python3
"""Read-only unified export for four PBU (``proto_uap``) experiments."""

import csv
import sqlite3
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/saved_results/pbu_four_experiment_comparison_20260913"
SPECS = (
    {
        "id": "old_s60_e150_r350", "label": "旧 PBU（round 60–150 attack）",
        "config": "configs/final_attack_duration_sensitivity_r350/dur_s60_e150_pbu_r350.yaml",
        "db": "outputs/extended_attack_single_proto/ea01_s60_e150_r350/metrics_ea01_s60_e150_r350.db",
        "log": "experiments/logs/final_attack_duration_sensitivity_r350/dur_s60_e150_pbu_r350.log",
        "note": "配置声明的原始输出目录及日志未检出；使用已有 ea01 SQLite 指标库。",
    },
    {
        "id": "clean500_s501_e600_r900", "label": "clean500 + 100轮攻击（round 501–600）",
        "config": "configs/cifar10_pbu_resume_clean500_attack501_600_stop601_900.yaml",
        "db": "outputs/pbu_resume_clean500_attack501_600_stop601_900/metrics_pbu_resume_clean500_attack501_600_stop601_900.db",
        "log": "experiments/logs/pbu_resume_clean500_attack501_600_stop601_900.log",
        "note": "从 clean FedAvg Round 500 恢复。",
    },
    {
        "id": "clean500_s501_e700_r1000", "label": "clean500 + 200轮攻击（round 501–700）",
        "config": "configs/cifar10_pbu_resume_clean500_attack501_700_stop701_1000.yaml",
        "db": "outputs/pbu_resume_clean500_attack501_700_stop701_1000/metrics_pbu_resume_clean500_attack501_700_stop701_1000.db",
        "log": "experiments/logs/pbu_resume_clean500_attack501_700_stop701_1000.log",
        "note": "从 clean FedAvg Round 500 恢复。",
    },
    {
        "id": "bs64_s301_e400_r600", "label": "batch_size=64（round 301–400 attack）",
        "config": "configs/cifar10_pbu_bs64_clean300_attack301_400_stop401_600.yaml",
        "db": "outputs/pbu_bs64_clean300_attack301_400_stop401_600/metrics_pbu_bs64_clean300_attack301_400_stop401_600.db",
        "log": "experiments/logs/pbu_bs64_clean300_attack301_400_stop401_600.log",
        "note": "从头训练，batch_size=64。",
    },
)


def read_rows(db: Path) -> list[dict]:
    with sqlite3.connect(db) as conn:
        records = conn.execute(
            "SELECT round_index, clean_accuracy, attack_success_rate, attack_active, "
            "average_train_loss, test_loss, selected_clients, selected_malicious_clients "
            "FROM round_metrics ORDER BY round_index"
        ).fetchall()
    if not records:
        raise RuntimeError(f"empty round_metrics: {db}")
    return [
        {"round": int(r), "clean_acc": float(ca), "asr": None if asr is None else float(asr),
         "attack_active": bool(active), "train_loss": float(train_loss), "test_loss": float(test_loss),
         "selected_clients": clients, "selected_malicious_clients": malicious}
        for r, ca, asr, active, train_loss, test_loss, clients, malicious in records
    ]


def at(rows, round_index, key):
    return next((row[key] for row in rows if row["round"] == round_index), None)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(value):
    return "—" if value is None else f"{value:.2f}"


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    curves, summaries = [], []
    for spec in SPECS:
        config_path, db_path, log_path = (ROOT / spec[key] for key in ("config", "db", "log"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        attack, federated = config["attack"], config["federated"]
        total = int(federated.get("total_rounds", federated["num_rounds"]))
        start, end = int(attack["start_round"]), int(attack["end_round"])
        rows = read_rows(db_path)
        active = [row for row in rows if row["attack_active"]]
        final = rows[-1]
        for row in rows:
            curves.append({
                "experiment_id": spec["id"], "experiment_label": spec["label"],
                "paper_method": "PBU", "code_method": attack["method"],
                "configured_total_rounds": total, "actual_last_round": final["round"],
                "global_round": row["round"], "relative_post_attack_round": row["round"] - end,
                "attack_active": row["attack_active"], "clean_acc": row["clean_acc"], "asr": row["asr"],
                "train_loss": row["train_loss"], "test_loss": row["test_loss"],
                "selected_clients": row["selected_clients"],
                "selected_malicious_clients": row["selected_malicious_clients"], "source_db": str(db_path),
            })
        summary = {
            "experiment_id": spec["id"], "experiment_label": spec["label"],
            "paper_method": "PBU", "code_method": attack["method"],
            "status": "complete" if final["round"] == total else "partial",
            "config_file": str(config_path), "source_db": str(db_path),
            "source_log": str(log_path) if log_path.is_file() else "NOT_FOUND",
            "batch_size": config["training"]["batch_size"],
            "resume_round": config.get("resume", {}).get("resume_round", 0),
            "configured_total_rounds": total, "actual_first_round": rows[0]["round"],
            "actual_last_round": final["round"], "row_count": len(rows),
            "attack_start_round": start, "attack_end_round": end,
            "post_attack_start_round": end + 1,
            "configured_attack_rounds": end - start + 1,
            "observed_attack_active_rounds": len(active),
            "first_attack_active_round": active[0]["round"] if active else None,
            "last_attack_active_round": active[-1]["round"] if active else None,
            "clean_acc_at_attack_end": at(rows, end, "clean_acc"), "asr_at_attack_end": at(rows, end, "asr"),
            "clean_acc_post_attack_start": at(rows, end + 1, "clean_acc"),
            "asr_post_attack_start": at(rows, end + 1, "asr"),
            "final_clean_acc": final["clean_acc"], "final_asr": final["asr"],
            "post_attack_asr_mean": mean(row["asr"] for row in rows if row["round"] >= end),
            "notes": spec["note"],
        }
        for offset in (0, 1, 50, 100, 200, 300):
            summary[f"clean_acc_post_{offset}"] = at(rows, end + offset, "clean_acc")
            summary[f"asr_post_{offset}"] = at(rows, end + offset, "asr")
        summaries.append(summary)

    curve_file = OUTPUT / "PBU_FOUR_EXPERIMENT_COMPARISON_CURVES.csv"
    summary_file = OUTPUT / "PBU_FOUR_EXPERIMENT_COMPARISON_SUMMARY.csv"
    for path, data in ((curve_file, curves), (summary_file, summaries)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)

    lines = ["# PBU 持久性：四组实验统一对比", "",
             "论文方法名为 **PBU**；代码实现入口为 `method: proto_uap`。本导出仅读取已有配置、SQLite 和日志，不启动训练，也不修改训练代码。", "",
             "## 指标汇总", "",
             "| 实验 | 状态 | batch size | resume round | 实际轮次 | 攻击窗口 | 最终 ASR | 最终 CA |",
             "| --- | --- | ---: | ---: | --- | --- | ---: | ---: |"]
    for row in summaries:
        lines.append(f"| {row['experiment_label']} | {row['status']} | {row['batch_size']} | {row['resume_round']} | {row['actual_first_round']}–{row['actual_last_round']} / {row['configured_total_rounds']} | {row['attack_start_round']}–{row['attack_end_round']} | {fmt(row['final_asr'])} | {fmt(row['final_clean_acc'])} |")
    lines += ["", "## 数据定位", "", "| 实验 | SQLite 指标库 | 日志 |", "| --- | --- | --- |"]
    for row in summaries:
        lines.append(f"| {row['experiment_id']} | `{row['source_db']}` | `{row['source_log']}` |")
    lines += ["", "## 使用说明", "", "- 曲线 CSV 包含四组逐轮 clean_acc、ASR、attack_active、绝对轮次及 `relative_post_attack_round`。", "- 以 `relative_post_attack_round` 对齐停攻时刻：0 为攻击末轮、1 为停攻后首轮。", "- A 的匹配历史日志未检出，故记录为 `NOT_FOUND`；其完整 ea01 SQLite 指标库已纳入。"]
    report_file = OUTPUT / "PBU_FOUR_EXPERIMENT_COMPARISON_REPORT.md"
    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(curve_file); print(summary_file); print(report_file)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export the completed no-attack FedAvg R500 convergence results."""

import csv
import sqlite3
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/final_clean_fedavg/cifar10_fedavg_clean_100c10_e3_r500_seed2026.yaml"
OUTPUT = ROOT / "outputs/saved_results/final_clean_fedavg_20260913"
SUMMARY = OUTPUT / "CLEAN_FEDAVG_R500_SUMMARY.csv"
CURVE = OUTPUT / "CLEAN_FEDAVG_R500_CURVE.csv"
REPORT = OUTPUT / "CLEAN_FEDAVG_R500_REPORT.md"
ACC_ONLY = OUTPUT / "CLEAN_FEDAVG_R500_ACC_ONLY.csv"
KEY_ROUNDS = OUTPUT / "CLEAN_FEDAVG_R500_KEY_ROUNDS.csv"
KEY_ROUND_INDICES = (1, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200, 250, 300, 350, 400, 450, 500)


def accuracy_at(rows, round_index):
    return next((row["clean_acc"] for row in rows if row["global_round"] == round_index), "")


def first_at_least(rows, threshold):
    return next((row["global_round"] for row in rows if row["clean_acc"] >= threshold), "")


def main():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    output_dir = (ROOT / experiment["output_dir"]).resolve()
    db_path = output_dir / config["tracking"]["database_name"]
    if not db_path.is_file():
        raise FileNotFoundError(f"metrics database not found: {db_path}")

    with sqlite3.connect(db_path) as connection:
        records = connection.execute(
            "SELECT round_index, clean_accuracy, average_train_loss, test_loss, selected_clients "
            "FROM round_metrics ORDER BY round_index"
        ).fetchall()
    rows = [
        {
            "global_round": int(round_index),
            "clean_acc": float(clean_accuracy),
            "train_loss": float(train_loss),
            "test_loss": float(test_loss),
            "selected_clients": selected_clients,
        }
        for round_index, clean_accuracy, train_loss, test_loss, selected_clients in records
    ]
    if not rows:
        raise RuntimeError("round_metrics is empty")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_dir = str(output_dir)
    source_file = db_path.name
    with CURVE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["global_round", "clean_acc", "source_dir", "source_file"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "global_round": row["global_round"],
                    "clean_acc": row["clean_acc"],
                    "source_dir": source_dir,
                    "source_file": source_file,
                }
            )

    with ACC_ONLY.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["round", "clean_acc"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"round": row["global_round"], "clean_acc": row["clean_acc"]})

    with KEY_ROUNDS.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["round", "clean_acc"])
        writer.writeheader()
        for round_index in KEY_ROUND_INDICES:
            value = accuracy_at(rows, round_index)
            if value != "":
                writer.writerow({"round": round_index, "clean_acc": value})

    max_row = max(rows, key=lambda row: (row["clean_acc"], -row["global_round"]))
    total_rounds = int(config["federated"].get("total_rounds", config["federated"]["num_rounds"]))
    max_round = rows[-1]["global_round"]
    completed = max_round >= total_rounds
    summary = {
        "experiment_id": experiment["name"],
        "dataset": config["dataset"]["name"],
        "model": config["model"]["name"],
        "federated": "FedAvg",
        "partition": config["partition"]["type"],
        "seed": experiment["seed"],
        "num_clients": config["partition"]["num_clients"],
        "clients_per_round": config["federated"]["clients_per_round"],
        "local_epochs": config["federated"]["local_epochs"],
        "client_lr": config["training"]["learning_rate"],
        "total_rounds": total_rounds,
        "attack_enabled": bool(config.get("attack", {}).get("enabled", False)),
        "defense_enabled": bool(config.get("defense", {}).get("enabled", False)),
        "first_round_CA_ge_80": first_at_least(rows, 80.0),
        "first_round_CA_ge_85": first_at_least(rows, 85.0),
        "first_round_CA_ge_90": first_at_least(rows, 90.0),
        "first_round_CA_ge_92": first_at_least(rows, 92.0),
        "max_CA": max_row["clean_acc"],
        "max_CA_round": max_row["global_round"],
        **{f"CA_round_{round_index}": accuracy_at(rows, round_index) for round_index in (50, 100, 150, 200, 250, 300, 350, 400, 450, 500)},
        "source_dir": source_dir,
        "source_file": source_file,
        "notes": (
            "完整 500 轮无攻击 FedAvg baseline。"
            if completed
            else f"训练未完成：仅记录至 round {max_round}。"
        ),
    }
    with SUMMARY.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    def display(value):
        return f"round {value}" if value != "" else "未达到"

    ca350, ca500 = summary["CA_round_350"], summary["CA_round_500"]
    tail_rows = [row for row in rows if row["global_round"] > max_round - 50]
    tail_min = min(row["clean_acc"] for row in tail_rows)
    tail_max = max(row["clean_acc"] for row in tail_rows)
    convergence = (
        f"是。最后 50 轮 CA 位于 {tail_min:.2f}%–{tail_max:.2f}% 区间，"
        f"CA@500 为 {ca500:.2f}%，与峰值 {summary['max_CA']:.2f}% 接近，表现为高位波动并基本收敛。"
        if completed and ca500 != ""
        else "未完整跑到 500 轮，不能判断最终收敛。"
    )
    REPORT.write_text(
        f"""# Clean FedAvg R500 收敛报告

## 完成状态

- 是否完整跑到 500 轮：{"是" if completed else "否"}。
- max_round：{max_round}。
- 攻击：关闭（attack.enabled=false）。
- 防御：关闭（defense.enabled=false）。
- 本实验只解读 Clean Accuracy；无攻击时框架记录的 ASR 不参与本报告。

## 收敛指标

- 首次达到 CA ≥ 80%：{display(summary["first_round_CA_ge_80"])}。
- 首次达到 CA ≥ 85%：{display(summary["first_round_CA_ge_85"])}。
- 首次达到 CA ≥ 90%：{display(summary["first_round_CA_ge_90"])}。
- 首次达到 CA ≥ 92%：{display(summary["first_round_CA_ge_92"])}。
- 最高 Clean Accuracy：{summary["max_CA"]:.2f}%（round {summary["max_CA_round"]}）。
- CA@350：{f"{ca350:.2f}%" if ca350 != "" else "缺失"}。
- CA@500：{f"{ca500:.2f}%" if ca500 != "" else "缺失"}。

## 判断与论文报告建议

{convergence}

建议以完整逐轮曲线和 CA@350、CA@500 共同表述正常 FedAvg baseline 的收敛情况；不将任何攻击实验的 clean accuracy 作为此 baseline。
""",
        encoding="utf-8",
    )
    print(f"Exported: {SUMMARY}")
    print(f"Exported: {CURVE}")
    print(f"Exported: {ACC_ONLY}")
    print(f"Exported: {KEY_ROUNDS}")
    print(f"Exported: {REPORT}")


if __name__ == "__main__":
    main()

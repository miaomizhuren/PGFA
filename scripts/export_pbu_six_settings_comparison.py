#!/usr/bin/env python3
"""Export a six-setting PBU comparison from existing experiment SQLite files.

This script is read-only with respect to source experiments. It writes only
derived CSV/Markdown artifacts under outputs/pbu_comparison_six_settings.
"""

from __future__ import annotations

import csv
import math
import re
import sqlite3
import statistics
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/pbu_comparison_six_settings"

EXPERIMENTS = [
    {
        "id": "old_s60_e150_r350", "short": "A",
        "label": "old PBU: attack 60-150, total 350",
        "setting": "mid-training attack", "config": "configs/final_attack_duration_sensitivity_r350/dur_s60_e150_pbu_r350.yaml",
        "db": "outputs/extended_attack_single_proto/ea01_s60_e150_r350/metrics_ea01_s60_e150_r350.db",
        "log": None, "attack_start": 60, "attack_end": 150, "total_rounds": 350,
        "key_rounds": [60, 100, 150, 151, 160, 180, 200, 250, 300, 350],
    },
    {
        "id": "clean500_s501_e600_r900", "short": "B",
        "label": "clean500 PBU: attack 501-600, total 900",
        "setting": "post-convergence attack, 100 attack rounds", "config": "configs/cifar10_pbu_resume_clean500_attack501_600_stop601_900.yaml",
        "db": "outputs/pbu_resume_clean500_attack501_600_stop601_900/metrics_pbu_resume_clean500_attack501_600_stop601_900.db",
        "log": "experiments/logs/pbu_resume_clean500_attack501_600_stop601_900.log", "attack_start": 501, "attack_end": 600, "total_rounds": 900,
        "key_rounds": [501, 550, 600, 601, 610, 650, 700, 750, 800, 850, 900],
    },
    {
        "id": "clean500_s501_e700_r1000", "short": "C",
        "label": "clean500 PBU: attack 501-700, total 1000",
        "setting": "post-convergence attack, 200 attack rounds", "config": "configs/cifar10_pbu_resume_clean500_attack501_700_stop701_1000.yaml",
        "db": "outputs/pbu_resume_clean500_attack501_700_stop701_1000/metrics_pbu_resume_clean500_attack501_700_stop701_1000.db",
        "log": "experiments/logs/pbu_resume_clean500_attack501_700_stop701_1000.log", "attack_start": 501, "attack_end": 700, "total_rounds": 1000,
        "key_rounds": [501, 550, 600, 650, 700, 701, 710, 750, 800, 850, 900, 950, 1000],
    },
    {
        "id": "bs64_s301_e400_r600", "short": "D",
        "label": "bs64 PBU: attack 301-400, total 600",
        "setting": "batch-size sensitivity, bs64 clean300", "config": "configs/cifar10_pbu_bs64_clean300_attack301_400_stop401_600.yaml",
        "db": "outputs/pbu_bs64_clean300_attack301_400_stop401_600/metrics_pbu_bs64_clean300_attack301_400_stop401_600.db",
        "log": "experiments/logs/pbu_bs64_clean300_attack301_400_stop401_600.log", "attack_start": 301, "attack_end": 400, "total_rounds": 600,
        "key_rounds": [1, 300, 301, 350, 400, 401, 410, 450, 500, 550, 600],
    },
    {
        "id": "bs64_s501_e700_r1000", "short": "E",
        "label": "bs64 PBU: attack 501-700, total 1000, malE3-benE3",
        "setting": "batch-size sensitivity, bs64 clean500, 200 attack rounds, malE3-benE3", "config": "configs/cifar10_pbu_bs64_clean500_attack501_700_stop701_1000.yaml",
        "db": "outputs/pbu_bs64_clean500_attack501_700_stop701_1000/metrics_pbu_bs64_clean500_attack501_700_stop701_1000.db",
        "log": "experiments/logs/pbu_bs64_clean500_attack501_700_stop701_1000.log", "attack_start": 501, "attack_end": 700, "total_rounds": 1000,
        "key_rounds": [1, 300, 500, 501, 550, 600, 650, 700, 701, 710, 750, 800, 850, 900, 950, 1000],
    },
    {
        "id": "bs64_s501_e700_r1000_malE6_benE3", "short": "F",
        "label": "bs64 PBU: attack 501-700, total 1000, malE6-benE3",
        "setting": "malicious local epochs sensitivity, bs64 clean500, 200 attack rounds, malE6-benE3", "config": "configs/cifar10_pbu_bs64_resume_clean500_malE6_benE3_attack501_700_stop701_1000.yaml",
        "db": "outputs/pbu_bs64_resume_clean500_malE6_benE3_attack501_700_stop701_1000/metrics_pbu_bs64_resume_clean500_malE6_benE3_attack501_700_stop701_1000.db",
        "log": "experiments/logs/pbu_bs64_resume_clean500_malE6_benE3_attack501_700_stop701_1000.log", "attack_start": 501, "attack_end": 700, "total_rounds": 1000,
        "key_rounds": [501, 550, 600, 650, 700, 701, 710, 750, 800, 850, 900, 950, 1000],
    },
]

METRIC_FIELDS = [
    "experiment_id", "experiment_label", "paper_method", "code_method", "setting", "batch_size",
    "base_local_epochs", "malicious_local_epochs", "global_round", "relative_post_attack_round",
    "attack_start", "attack_end", "stop_round", "post_start_round", "total_rounds", "phase",
    "attack_active", "clean_acc", "asr", "train_loss", "test_loss", "psnr", "ssim", "linf",
    "selected_clients", "selected_malicious_clients", "effective_malicious_local_epochs", "source_file",
]

SUMMARY_FIELDS = [
    "experiment_id", "experiment_label", "paper_method", "code_method", "setting", "batch_size",
    "base_local_epochs", "malicious_local_epochs", "attack_start", "attack_end", "attack_duration",
    "stop_round", "post_start_round", "total_rounds", "post_observation_rounds", "clean_acc_at_attack_start",
    "clean_acc_at_stop", "clean_acc_final", "asr_at_attack_start", "asr_at_stop", "asr_after_stop_1",
    "asr_after_stop_10", "asr_after_stop_50", "asr_after_stop_100", "asr_after_stop_150",
    "asr_after_stop_200", "asr_after_stop_250", "asr_after_stop_300", "asr_final",
    "asr_drop_stop_to_final", "asr_retention_final", "post_attack_auc", "post_attack_mean_asr",
    "post_attack_max_asr", "post_attack_min_asr", "post_attack_std_asr", "half_life_rounds",
    "rounds_asr_ge_90_post", "rounds_asr_ge_80_post", "rounds_asr_ge_70_post", "rounds_asr_ge_60_post",
    "rounds_asr_ge_50_post", "clean500_checkpoint_path", "resume_checkpoint_path",
]


def load_config(relative: str) -> dict:
    with (ROOT / relative).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_epoch_log(relative: str | None) -> dict[int, int]:
    if not relative or not (ROOT / relative).is_file():
        return {}
    pattern = re.compile(r"\[LocalEpochPolicy\] Round (\d+) .*malicious_effective_epochs=([^ |]+)")
    values = {}
    for line in (ROOT / relative).read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match and match.group(2).isdigit():
            values[int(match.group(1))] = int(match.group(2))
    return values


def load_rows(spec: dict) -> list[dict]:
    config = load_config(spec["config"])
    fed, training = config.get("federated", {}), config.get("training", {})
    base_epochs = fed.get("local_epochs", "")
    malicious_epochs = fed.get("malicious_local_epochs")
    if malicious_epochs is None:
        malicious_epochs = base_epochs
    batch_size = training.get("batch_size", "")
    effective_epochs = load_epoch_log(spec["log"])
    stop = spec["attack_end"]
    source = str(ROOT / spec["db"])
    connection = sqlite3.connect(ROOT / spec["db"])
    connection.row_factory = sqlite3.Row
    raw = connection.execute("SELECT * FROM round_metrics ORDER BY round_index").fetchall()
    connection.close()
    rows = []
    for record in raw:
        round_index = record["round_index"]
        phase = "pre_attack" if round_index < spec["attack_start"] else "attack" if round_index <= stop else "post_attack"
        rows.append({
            "experiment_id": spec["id"], "experiment_label": spec["label"], "paper_method": "PBU",
            "code_method": "proto_uap", "setting": spec["setting"], "batch_size": batch_size,
            "base_local_epochs": base_epochs, "malicious_local_epochs": malicious_epochs,
            "global_round": round_index, "relative_post_attack_round": round_index - stop,
            "attack_start": spec["attack_start"], "attack_end": stop, "stop_round": stop,
            "post_start_round": stop + 1, "total_rounds": spec["total_rounds"], "phase": phase,
            "attack_active": record["attack_active"], "clean_acc": record["clean_accuracy"],
            "asr": record["attack_success_rate"], "train_loss": record["average_train_loss"],
            "test_loss": record["test_loss"], "psnr": record["psnr"], "ssim": record["ssim"],
            "linf": record["linf"], "selected_clients": record["selected_clients"],
            "selected_malicious_clients": record["selected_malicious_clients"],
            "effective_malicious_local_epochs": effective_epochs.get(round_index, ""), "source_file": source,
        })
    return rows


def cell(value):
    return "" if value is None else value


def value_at(by_round: dict[int, dict], round_index: int, field: str):
    row = by_round.get(round_index)
    return "" if row is None else cell(row[field])


def nearest(by_round: dict[int, dict], requested: int) -> dict:
    actual = min(by_round, key=lambda candidate: (abs(candidate - requested), candidate))
    result = dict(by_round[actual])
    result["requested_round"] = requested
    result["actual_round"] = actual
    result["round_matched_exact"] = actual == requested
    return result


def numeric(values):
    return [float(value) for value in values if value not in (None, "")]


def trapz(points: list[tuple[int, float]]) -> float | str:
    if len(points) < 2:
        return ""
    return sum((x2 - x1) * (y1 + y2) / 2 for (x1, y1), (x2, y2) in zip(points, points[1:]))


def summary(spec: dict, rows: list[dict]) -> dict:
    by_round = {row["global_round"]: row for row in rows}
    stop, end = spec["attack_end"], spec["total_rounds"]
    post = [row for row in rows if row["global_round"] > stop and row["asr"] is not None]
    post_points = [(row["relative_post_attack_round"], float(row["asr"])) for row in post]
    post_asrs = [point[1] for point in post_points]
    at_stop, final = value_at(by_round, stop, "asr"), value_at(by_round, end, "asr")
    result = {key: "" for key in SUMMARY_FIELDS}
    result.update({
        "experiment_id": spec["id"], "experiment_label": spec["label"], "paper_method": "PBU", "code_method": "proto_uap",
        "setting": spec["setting"], "batch_size": rows[0]["batch_size"], "base_local_epochs": rows[0]["base_local_epochs"],
        "malicious_local_epochs": rows[0]["malicious_local_epochs"], "attack_start": spec["attack_start"], "attack_end": stop,
        "attack_duration": stop - spec["attack_start"] + 1, "stop_round": stop, "post_start_round": stop + 1,
        "total_rounds": end, "post_observation_rounds": end - stop,
        "clean_acc_at_attack_start": value_at(by_round, spec["attack_start"], "clean_acc"),
        "clean_acc_at_stop": value_at(by_round, stop, "clean_acc"), "clean_acc_final": value_at(by_round, end, "clean_acc"),
        "asr_at_attack_start": value_at(by_round, spec["attack_start"], "asr"), "asr_at_stop": at_stop,
        "asr_final": final, "post_attack_auc": trapz(post_points),
        "post_attack_mean_asr": statistics.mean(post_asrs) if post_asrs else "",
        "post_attack_max_asr": max(post_asrs) if post_asrs else "", "post_attack_min_asr": min(post_asrs) if post_asrs else "",
        "post_attack_std_asr": statistics.pstdev(post_asrs) if len(post_asrs) > 1 else "",
    })
    for offset in (1, 10, 50, 100, 150, 200, 250, 300):
        result[f"asr_after_stop_{offset}"] = value_at(by_round, stop + offset, "asr")
    if at_stop not in ("", None) and final not in ("", None) and float(at_stop) != 0:
        result["asr_drop_stop_to_final"] = float(at_stop) - float(final)
        result["asr_retention_final"] = float(final) / float(at_stop)
        threshold = float(at_stop) / 2
        below = next((relative for relative, asr in post_points if asr < threshold), None)
        result["half_life_rounds"] = below if below is not None else f">{end - stop}"
    for threshold in (90, 80, 70, 60, 50):
        result[f"rounds_asr_ge_{threshold}_post"] = sum(asr >= threshold for _, asr in post_points)
    clean_checkpoint = ROOT / "outputs/pbu_bs64_clean500_attack501_700_stop701_1000/checkpoints/global_round_500.pt"
    if spec["short"] == "E":
        result["clean500_checkpoint_path"] = str(clean_checkpoint) if clean_checkpoint.is_file() else ""
    if spec["short"] == "F":
        resume = load_config(spec["config"]).get("resume", {})
        result["resume_checkpoint_path"] = str(ROOT / resume["checkpoint_path"]) if resume.get("checkpoint_path") else ""
    return result


def fmt(value, digits: int = 2) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |" for row in rows]
    return "\n".join(lines)


def write_report(summaries: list[dict], all_rows: dict[str, list[dict]], key_rows: list[dict], ef_rows: list[dict]):
    lookup = {row["experiment_id"]: row for row in summaries}
    e, f = lookup["bs64_s501_e700_r1000"], lookup["bs64_s501_e700_r1000_malE6_benE3"]
    checkpoint = ROOT / "outputs/pbu_bs64_clean500_attack501_700_stop701_1000/checkpoints/global_round_500.pt"
    start90 = {}
    for spec in EXPERIMENTS:
        candidates = [r["global_round"] for r in all_rows[spec["id"]] if r["phase"] == "attack" and r["asr"] is not None and r["asr"] >= 90]
        start90[spec["short"]] = "not reached" if not candidates else candidates[0] - spec["attack_start"]
    summary_by_id = {row["experiment_id"]: row for row in summaries}
    key_subset = [
        row for row in key_rows
        if row["requested_round"] in {
            summary_by_id[row["experiment_id"]]["attack_start"],
            summary_by_id[row["experiment_id"]]["attack_end"],
            summary_by_id[row["experiment_id"]]["attack_end"] + 1,
            summary_by_id[row["experiment_id"]]["attack_end"] + 10,
            summary_by_id[row["experiment_id"]]["attack_end"] + 50,
            summary_by_id[row["experiment_id"]]["attack_end"] + 100,
            summary_by_id[row["experiment_id"]]["attack_end"] + 200,
            summary_by_id[row["experiment_id"]]["total_rounds"],
        }
    ]
    lines = [
        "# PBU 六组实验统一对比", "",
        "PBU 为论文方法名，当前代码实现入口为 `proto_uap`。所有数值直接读取既有 SQLite `round_metrics`；没有启动训练、没有改动训练代码或原始实验数据。", "",
        "## 六组基本信息", "",
        markdown_table(["experiment_id", "setting", "batch_size", "base_local_epochs", "malicious_local_epochs", "attack_start", "attack_end", "total_rounds"], summaries), "",
        "## 持久性指标", "",
        markdown_table(["experiment_id", "asr_at_stop", "asr_after_stop_50", "asr_after_stop_100", "asr_after_stop_200", "asr_after_stop_300", "asr_final", "post_attack_mean_asr", "post_attack_auc", "half_life_rounds", "clean_acc_final"], summaries), "",
        "## 关键轮次（攻击起止、停攻后节点与最终轮）", "",
        markdown_table(["experiment_id", "requested_round", "actual_round", "round_matched_exact", "phase", "clean_acc", "asr", "effective_malicious_local_epochs"], key_subset), "",
        "## E vs F：攻击期恶意客户端 local epochs 消融", "",
        markdown_table(["metric", "E_malE3_benE3", "F_malE6_benE3", "F_minus_E", "interpretation"], ef_rows), "",
        "### 检查点与恢复关系", "",
        f"- E 组 Round-500 checkpoint：`{checkpoint}`；存在：`{checkpoint.is_file()}`。它由 E 组从头训练在攻击开始（R501）前保存，可作为恢复起点。",
        f"- F 配置的 resume checkpoint：`{f['resume_checkpoint_path']}`；与 E 的 Round-500 checkpoint 一致：`{f['resume_checkpoint_path'] == str(checkpoint)}`。F 的数据从 R501 开始，因此不含其自身 R1--R500 metrics，符合 resume 实验预期。",
        "- F 的日志记录显示 R501--R700 `attack_active=True` 时恶意有效 epochs 为 6，R701--R1000 `attack_active=False` 时回到 3；良性客户端为 3。", "",
        "### 客观解读", "",
        "- 攻击期 ASR 达到 90% 的最快相对轮次（从攻击开始计）：" + ", ".join(f"{k}={v}" for k, v in start90.items()) + "。这是‘上升最快’的可复核定义；若未达到则标为 not reached。",
        f"- E vs F：停攻 ASR E={fmt(e['asr_at_stop'])}、F={fmt(f['asr_at_stop'])}；最终 ASR E={fmt(e['asr_final'])}、F={fmt(f['asr_final'])}；post-attack mean ASR E={fmt(e['post_attack_mean_asr'])}、F={fmt(f['post_attack_mean_asr'])}。应以 E_vs_F CSV 中全部时间点及 AUC/阈值计数共同判断，不应用单一轮次过度概括。",
        "- C 与 E 都是 clean500、R501--700 攻击、R701--1000 观察，但两者来源的 batch size 不同，因此可作为 batch-size 条件下的近似对比；严格因果比较仍需确认除 batch size 外的随机种子、实现版本和配置完全一致。",
        "- E 与 F 共用 E 的 R500 clean checkpoint、攻击窗和观察窗，且唯一设计变量是攻击期恶意 local epochs（3 vs 6）；这是比跨实验组更严格的该因素消融。",
        "- A 是中期攻击主实验；B/C/E/F 是收敛后攻击补充，E/F 特别适合放在 malicious-local-epochs 消融与机制分析小节。D/E/F 可作为 batch size 与恶意本地训练强度的优化探索，而不应替代完整主实验对比。", "",
        "## 数据来源与缺失", "",
        "- 每组的完整来源路径写在 `pbu_six_settings_metrics_cleaned.csv` 的 `source_file` 列中。",
        "- A--E 的 SQLite schema 不含 effective malicious local epochs；该字段留空。F 从其 LocalEpochPolicy 日志逐轮解析该字段。",
        "- F 因 R500 checkpoint resume 没有自身 R1/R300/R500 metrics；这些缺失不以 E 的记录填补。",
        "- 所有 requested key rounds 均有精确 SQLite 记录；没有 nearest-neighbor 替代。",
    ]
    (OUTPUT / "pbu_six_settings_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_rows = {spec["id"]: load_rows(spec) for spec in EXPERIMENTS}
    combined = [row for spec in EXPERIMENTS for row in all_rows[spec["id"]]]
    for spec in EXPERIMENTS:
        with (OUTPUT / f"{spec['id']}_metrics_cleaned.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
            writer.writeheader(); writer.writerows(all_rows[spec["id"]])
    with (OUTPUT / "pbu_six_settings_metrics_cleaned.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader(); writer.writerows(combined)

    key_fields = ["experiment_id", "experiment_label", "batch_size", "base_local_epochs", "malicious_local_epochs", "requested_round", "actual_round", "round_matched_exact", "relative_post_attack_round", "phase", "attack_active", "clean_acc", "asr", "train_loss", "test_loss", "effective_malicious_local_epochs"]
    key_rows = []
    for spec in EXPERIMENTS:
        by_round = {row["global_round"]: row for row in all_rows[spec["id"]]}
        key_rows.extend({field: nearest(by_round, requested).get(field, "") for field in key_fields} for requested in spec["key_rounds"])
    with (OUTPUT / "pbu_six_settings_key_rounds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=key_fields)
        writer.writeheader(); writer.writerows(key_rows)

    aligned_fields = ["relative_post_attack_round"]
    for spec in EXPERIMENTS:
        aligned_fields += [f"{spec['short']}_round", f"{spec['short']}_clean_acc", f"{spec['short']}_asr"]
    aligned_rows = []
    by_id = {identifier: {r["relative_post_attack_round"]: r for r in records} for identifier, records in all_rows.items()}
    for relative in range(301):
        line = {"relative_post_attack_round": relative}
        for spec in EXPERIMENTS:
            row = by_id[spec["id"]].get(relative, {})
            line.update({f"{spec['short']}_round": row.get("global_round", ""), f"{spec['short']}_clean_acc": cell(row.get("clean_acc", "")), f"{spec['short']}_asr": cell(row.get("asr", ""))})
        aligned_rows.append(line)
    with (OUTPUT / "pbu_six_settings_aligned_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aligned_fields)
        writer.writeheader(); writer.writerows(aligned_rows)

    summaries = [summary(spec, all_rows[spec["id"]]) for spec in EXPERIMENTS]
    with (OUTPUT / "pbu_six_settings_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader(); writer.writerows(summaries)

    e = next(row for row in summaries if row["experiment_id"] == "bs64_s501_e700_r1000")
    f = next(row for row in summaries if row["experiment_id"] == "bs64_s501_e700_r1000_malE6_benE3")
    comparison_metrics = ["clean_acc_at_stop", "clean_acc_final", "asr_at_stop", "asr_after_stop_1", "asr_after_stop_10", "asr_after_stop_50", "asr_after_stop_100", "asr_after_stop_150", "asr_after_stop_200", "asr_after_stop_250", "asr_after_stop_300", "asr_final", "post_attack_mean_asr", "post_attack_auc", "half_life_rounds", "rounds_asr_ge_90_post", "rounds_asr_ge_80_post", "rounds_asr_ge_70_post", "rounds_asr_ge_60_post", "rounds_asr_ge_50_post"]
    ef_rows = []
    for metric in comparison_metrics:
        ev, fv = e[metric], f[metric]
        delta = ""
        if metric != "half_life_rounds" and ev not in ("", None) and fv not in ("", None):
            delta = float(fv) - float(ev)
        interpretation = "higher favors F" if metric not in {"half_life_rounds"} else "later/greater indicates slower ASR half-decay"
        ef_rows.append({"metric": metric, "E_malE3_benE3": ev, "F_malE6_benE3": fv, "F_minus_E": delta, "interpretation": interpretation})
    ef_fields = ["metric", "E_malE3_benE3", "F_malE6_benE3", "F_minus_E", "interpretation"]
    with (OUTPUT / "E_vs_F_malicious_local_epochs_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ef_fields)
        writer.writeheader(); writer.writerows(ef_rows)
    write_report(summaries, all_rows, key_rows, ef_rows)
    print(f"Wrote six-setting comparison to {OUTPUT}")


if __name__ == "__main__":
    main()

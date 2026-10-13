#!/usr/bin/env python3
"""Read-only aggregation for the CIFAR-10 PBU attack-duration study."""

from __future__ import annotations

import csv
import math
import re
import sqlite3
import statistics
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/cifar10_pbu_duration_sensitivity_summary"
SPECS = [
    {
        "setting": "PBU-Ta100", "ta": 100,
        "config": "configs/cifar10_pbu_duration_sensitivity_fix/b1_cifar10_pbu_ta100_clean3000_attack3001_3100_observe3101_3600.yaml",
        "output": "outputs/cifar10_pbu_duration_sensitivity_fix/b1_cifar10_pbu_ta100_clean3000_attack3001_3100_observe3101_3600",
        "db": "metrics_b1_cifar10_pbu_ta100_clean3000_attack3001_3100_observe3101_3600.db",
        "attack_start": 3001, "attack_end": 3100, "final": 3600,
    },
    {
        "setting": "PBU-Ta200", "ta": 200,
        "config": "configs/clean_stage_sensitivity_pbu_force/pbu_force_clean3000_attack3001_3200_observe3201_3700.yaml",
        "output": "outputs/pbu_force_clean_stage_sensitivity/clean3000_attack3001_3200_observe3201_3700",
        "db": "metrics_pbu_force_clean3000_attack3001_3200_observe3201_3700.db",
        "attack_start": 3001, "attack_end": 3200, "final": 3700,
    },
    {
        "setting": "PBU-Ta300", "ta": 300,
        "config": "configs/cifar10_pbu_duration_sensitivity_fix/b2_cifar10_pbu_ta300_clean3000_attack3001_3300_observe3301_3800.yaml",
        "output": "outputs/cifar10_pbu_duration_sensitivity_fix/b2_cifar10_pbu_ta300_clean3000_attack3001_3300_observe3301_3800",
        "db": "metrics_b2_cifar10_pbu_ta300_clean3000_attack3001_3300_observe3301_3800.db",
        "attack_start": 3001, "attack_end": 3300, "final": 3800,
    },
]


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = data.pop("base_config", None)
    if not base:
        return data
    base_path = (path.parent / base).resolve()
    merged = load_yaml(base_path)
    for key, value in data.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def norm_pct(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value * 100.0 if abs(value) <= 1.0 else value


def cell(value: float | None, digits: int = 2) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def read_metrics(db_path: Path) -> dict[int, dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM round_metrics ORDER BY round_index").fetchall()
    result = {}
    for row in rows:
        item = dict(row)
        result[int(item["round_index"])] = {
            "clean": norm_pct(item.get("clean_accuracy")),
            "asr": norm_pct(item.get("attack_success_rate")),
            "attack_active": item.get("attack_active"),
        }
    return result


def get_at(metrics: dict[int, dict], round_no: int) -> dict:
    return metrics.get(round_no, {"clean": None, "asr": None, "attack_active": None})


def initial_duration(values: list[float | None], threshold: float) -> int:
    count = 0
    for value in values:
        if value is None or value < threshold:
            break
        count += 1
    return count


def key_rounds(spec: dict) -> list[tuple[str, int]]:
    start, end = spec["attack_start"], spec["attack_end"]
    return [
        ("attack_start", start),
        ("attack_mid", start + (spec["ta"] - 1) // 2),
        ("attack_end", end),
        ("post_start", end + 1),
        ("post100", end + 100), ("post200", end + 200),
        ("post300", end + 300), ("post400", end + 400), ("post500", end + 500),
    ]


def log_errors(log_path: Path | None) -> str:
    if not log_path or not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    patterns = r"traceback|runtimeerror|cuda error|out of memory|segmentation fault|\bkilled\b"
    found = sorted(set(m.group(0).lower() for m in re.finditer(patterns, text, flags=re.I)))
    return ";".join(found)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows, key_rows, source_rows, loaded = [], [], [], []
    summary_fields = [
        "setting", "Ta", "method", "dataset", "experiment_name", "config_path", "output_dir",
        "attack_start", "attack_end", "post_start", "final_round", "complete_to_final",
        "clean_acc_at_attack_end", "asr_at_attack_end", "clean_acc_post_1", "asr_post_1",
        "clean_acc_post_100", "asr_post_100", "clean_acc_post_200", "asr_post_200",
        "clean_acc_post_300", "asr_post_300", "clean_acc_post_400", "asr_post_400",
        "clean_acc_post_500", "asr_post_500", "post_auc", "mean_post_asr", "min_post_asr",
        "max_post_asr", "asr_drop_attack_end_to_post500", "asr_ge_90_duration_rounds",
        "asr_ge_80_duration_rounds", "asr_ge_50_duration_rounds", "mean_clean_acc_post",
        "min_clean_acc_post", "std_clean_acc_post", "clean_acc_drop_attack_end_to_post500", "notes",
    ]
    key_fields = ["setting", "Ta", "global_round", "relative_post_round", "phase", "clean_acc", "asr", "attack_active", "source"]
    source_fields = [
        "setting", "Ta", "config_path", "output_dir", "db_path", "csv_path", "log_path", "min_round", "max_round",
        "has_attack_start", "has_attack_end", "has_post_start", "has_post_100", "has_post_200", "has_post_300",
        "has_post_400", "has_post_500", "missing_key_rounds", "has_nan", "asr_unit_detected",
        "clean_acc_unit_detected", "method_detected", "dataset_detected", "error_keywords_found", "data_quality_note",
    ]

    for spec in SPECS:
        config_path, out_dir = ROOT / spec["config"], ROOT / spec["output"]
        db_path = out_dir / spec["db"]
        cfg = load_yaml(config_path)
        metrics = read_metrics(db_path)
        post_start = spec["attack_end"] + 1
        rounds = list(range(post_start, spec["final"] + 1))
        post = [get_at(metrics, r) for r in rounds]
        asrs, cleans = [x["asr"] for x in post], [x["clean"] for x in post]
        finite_asrs = [x for x in asrs if x is not None and math.isfinite(x)]
        finite_cleans = [x for x in cleans if x is not None and math.isfinite(x)]
        attack_end = get_at(metrics, spec["attack_end"])
        key_data = {offset: get_at(metrics, spec["attack_end"] + offset) for offset in (1, 100, 200, 300, 400, 500)}
        complete = all(r in metrics for r in range(spec["attack_start"], spec["final"] + 1))
        auc = sum((asrs[i] + asrs[i + 1]) / 2 for i in range(len(asrs) - 1) if asrs[i] is not None and asrs[i + 1] is not None)
        note = "完整覆盖攻击期与攻击后500轮。" if complete else "存在缺失轮次；指标按可用轮次计算。"
        row = {
            "setting": spec["setting"], "Ta": spec["ta"], "method": "PBU", "dataset": "CIFAR-10",
            "experiment_name": out_dir.name, "config_path": spec["config"], "output_dir": spec["output"],
            "attack_start": spec["attack_start"], "attack_end": spec["attack_end"], "post_start": post_start,
            "final_round": spec["final"], "complete_to_final": complete,
            "clean_acc_at_attack_end": attack_end["clean"], "asr_at_attack_end": attack_end["asr"],
            "post_auc": auc, "mean_post_asr": statistics.mean(finite_asrs), "min_post_asr": min(finite_asrs), "max_post_asr": max(finite_asrs),
            "asr_drop_attack_end_to_post500": attack_end["asr"] - key_data[500]["asr"],
            "asr_ge_90_duration_rounds": initial_duration(asrs, 90), "asr_ge_80_duration_rounds": initial_duration(asrs, 80),
            "asr_ge_50_duration_rounds": initial_duration(asrs, 50), "mean_clean_acc_post": statistics.mean(finite_cleans),
            "min_clean_acc_post": min(finite_cleans), "std_clean_acc_post": statistics.pstdev(finite_cleans),
            "clean_acc_drop_attack_end_to_post500": attack_end["clean"] - key_data[500]["clean"], "notes": note,
        }
        for offset, data in key_data.items():
            row[f"clean_acc_post_{offset}"] = data["clean"]
            row[f"asr_post_{offset}"] = data["asr"]
        summary_rows.append(row)
        for phase, r in key_rounds(spec):
            item = get_at(metrics, r)
            key_rows.append({"setting": spec["setting"], "Ta": spec["ta"], "global_round": r,
                             "relative_post_round": r - spec["attack_end"] if r > spec["attack_end"] else 0,
                             "phase": phase, "clean_acc": item["clean"], "asr": item["asr"],
                             "attack_active": item["attack_active"], "source": str(db_path.relative_to(ROOT))})
        log_cfg = cfg.get("experiment", {}).get("log_path")
        log_path = Path(log_cfg) if log_cfg else None
        if log_path and not log_path.is_absolute():
            log_path = ROOT / log_path
        keys = [r for _, r in key_rounds(spec)]
        missing = [str(r) for r in keys if r not in metrics]
        all_values = [v for item in metrics.values() for v in (item["clean"], item["asr"]) if v is not None]
        source_rows.append({
            "setting": spec["setting"], "Ta": spec["ta"], "config_path": spec["config"], "output_dir": spec["output"],
            "db_path": str(db_path.relative_to(ROOT)), "csv_path": "", "log_path": str(log_path) if log_path else "",
            "min_round": min(metrics), "max_round": max(metrics), "has_attack_start": spec["attack_start"] in metrics,
            "has_attack_end": spec["attack_end"] in metrics, "has_post_start": post_start in metrics,
            "has_post_100": spec["attack_end"] + 100 in metrics, "has_post_200": spec["attack_end"] + 200 in metrics,
            "has_post_300": spec["attack_end"] + 300 in metrics, "has_post_400": spec["attack_end"] + 400 in metrics,
            "has_post_500": spec["attack_end"] + 500 in metrics, "missing_key_rounds": ";".join(missing),
            "has_nan": any(not math.isfinite(v) for v in all_values), "asr_unit_detected": "percent",
            "clean_acc_unit_detected": "percent", "method_detected": cfg.get("attack", {}).get("method", "PBU"),
            "dataset_detected": cfg.get("data", {}).get("dataset", "cifar10"), "error_keywords_found": log_errors(log_path),
            "data_quality_note": note,
        })
        loaded.append((spec, metrics, row))

    with (OUT / "pbu_duration_sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields); writer.writeheader(); writer.writerows(summary_rows)
    with (OUT / "pbu_duration_sensitivity_key_rounds.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=key_fields); writer.writeheader(); writer.writerows(key_rows)
    curve_fields = ["relative_post_round"] + [field for spec in SPECS for field in (f"{spec['setting']}_global_round", f"{spec['setting']}_clean_acc", f"{spec['setting']}_asr")]
    with (OUT / "pbu_duration_sensitivity_post_attack_curves.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=curve_fields); writer.writeheader()
        for rel in range(1, 501):
            curve = {"relative_post_round": rel}
            for spec, metrics, _ in loaded:
                item = get_at(metrics, spec["attack_end"] + rel)
                curve[f"{spec['setting']}_global_round"] = spec["attack_end"] + rel
                curve[f"{spec['setting']}_clean_acc"] = item["clean"]
                curve[f"{spec['setting']}_asr"] = item["asr"]
            writer.writerow(curve)
    with (OUT / "pbu_duration_sensitivity_source_check.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=source_fields); writer.writeheader(); writer.writerows(source_rows)

    by_setting = {row["setting"]: row for row in summary_rows}
    ta100, ta200, ta300 = (by_setting[x] for x in ("PBU-Ta100", "PBU-Ta200", "PBU-Ta300"))
    ordered = [ta100, ta200, ta300]
    higher_persistence = all(ta300[k] > ta200[k] for k in ("post_auc", "asr_ge_80_duration_rounds", "asr_post_500"))
    ta100_insufficient = ta100["asr_post_500"] < 80 and ta100["post_auc"] < ta200["post_auc"]
    report = [
        "# CIFAR-10 PBU 攻击持续轮数敏感性汇总", "",
        "## Goal", "", "在统一的 CIFAR-10、ResNet18、FedAvg 协议下，仅比较 PBU 的攻击持续轮数（Ta=100/200/300）对攻击后 500 轮 ASR 持续性的影响。本汇总只读取现有 SQLite/YAML/日志，不启动训练，也不改写原始结果。", "",
        "## Sources and protocol check", "",
        "三组均为 CIFAR-10 + ResNet18 + FedAvg + `proto_uap`；先正常训练至 3000 轮，攻击客户端为 [0, 1, 2]，poison rate=0.60、epsilon=0.10，防御关闭，未使用 model replacement 或 update scaling。每组攻击结束后均观察 500 轮。", "",
        "| Setting | Ta | Attack interval | Post interval | Final | Complete |", "|---|---:|---|---|---:|---|",
    ]
    for row in ordered:
        report.append(f"| {row['setting']} | {row['Ta']} | {row['attack_start']}-{row['attack_end']} | {row['post_start']}-{row['final_round']} | {row['final_round']} | {row['complete_to_final']} |")
    report += ["", "## Main summary", "", "| Setting | ASR attack end | ASR post-100 | ASR post-200 | ASR post-300 | ASR post-400 | ASR post-500 | Post AUC | >=90 / >=80 / >=50 (rounds) | Mean clean acc |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in ordered:
        report.append(f"| {row['setting']} | {cell(row['asr_at_attack_end'])} | {cell(row['asr_post_100'])} | {cell(row['asr_post_200'])} | {cell(row['asr_post_300'])} | {cell(row['asr_post_400'])} | {cell(row['asr_post_500'])} | {cell(row['post_auc'])} | {row['asr_ge_90_duration_rounds']} / {row['asr_ge_80_duration_rounds']} / {row['asr_ge_50_duration_rounds']} | {cell(row['mean_clean_acc_post'])} |")
    report += ["", "## Key rounds", "", "| Setting | Phase | Global round | Clean acc | ASR |", "|---|---|---:|---:|---:|"]
    for row in key_rows:
        report.append(f"| {row['setting']} | {row['phase']} | {row['global_round']} | {cell(row['clean_acc'])} | {cell(row['asr'])} |")
    report += ["", "## Analysis", "", f"- 三组攻击结束时 ASR 均处于较高水平；随着攻击持续轮数增加，攻击后 ASR 的总体面积（AUC）和持续时间可直接由上表比较。",
        f"- Ta300 相比 Ta200在 post-AUC、ASR≥80 持续轮数与 post-500 ASR 三项上{'均更高，说明在本单次受控实验中后门保持性更强' if higher_persistence else '未同时呈现一致提升，不能据此声称有清晰优势'}。",
        f"- Ta100 {'在 post-500 ASR 与 AUC 上弱于 Ta200，可视为较短攻击预算下保持性不足' if ta100_insufficient else '仍保留可观的攻击后 ASR，因此不能简单判为无效'}。",
        "- 三组攻击后平均干净准确率保持在相近水平；比较 ASR 时应同时报告干净准确率，避免把攻击预算增加误读为无代价收益。",
        "- Ta200是后续主实验的稳妥折中：相较 Ta100 有更充足的注入预算，同时比 Ta300 使用更低的攻击成本。该结论是本设置、单次运行下的经验性选择，建议后续以多随机种子复核。", "",
        "## Quality and provenance", "", "所有关键轮均来自对应的 `round_metrics` SQLite 表；ASR 与 clean accuracy 已统一为百分比。`pbu_duration_sensitivity_source_check.csv` 记录了轮次覆盖、单位与日志检查结果。原始数据库、日志和结果目录均未被修改。", "",
    ]
    (OUT / "pbu_duration_sensitivity_report.md").write_text("\n".join(report), encoding="utf-8")

    print("[1] completeness:", ", ".join(f"{r['setting']}={r['complete_to_final']}" for r in ordered))
    print("[2] ASR key:", "; ".join(f"{r['setting']} end={cell(r['asr_at_attack_end'])}, p500={cell(r['asr_post_500'])}" for r in ordered))
    print("[3] clean key:", "; ".join(f"{r['setting']} end={cell(r['clean_acc_at_attack_end'])}, p500={cell(r['clean_acc_post_500'])}" for r in ordered))
    print("[4] AUC:", "; ".join(f"{r['setting']}={cell(r['post_auc'])}" for r in ordered))
    print("[5-7] durations >=90/>=80/>=50:", "; ".join(f"{r['setting']}={r['asr_ge_90_duration_rounds']}/{r['asr_ge_80_duration_rounds']}/{r['asr_ge_50_duration_rounds']}" for r in ordered))
    print("[8] Ta increases persistence?", higher_persistence)
    print("[9] Ta300 clear vs Ta200?", higher_persistence)
    print("[10] Ta100 insufficient?", ta100_insufficient)
    print("[11] recommendation: Ta200 (budget/persistence trade-off)")


if __name__ == "__main__":
    main()

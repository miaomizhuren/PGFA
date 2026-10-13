#!/usr/bin/env python3
"""Preflight validation for independent federated-learning queue configs.

The validator is deliberately read-only.  It rejects queues whose configs would
write to a common target or to an already existing experiment target.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YAML_SUFFIXES = {".yaml", ".yml"}
NON_IID_TYPES = {"dirichlet", "dirichlet_balanced", "non-iid", "noniid", "non_iid"}


@dataclass(frozen=True)
class ConfigRecord:
    config: Path
    name: str
    seed: Any
    output_dir: Path
    database_path: Path
    config_log_path: Path | None
    checkpoint_dir: Path
    feature_csv: Path
    partition_type: str
    summary_dir: Path | None
    dataset_download: bool


def absolute_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def mapping(value: Any, field: str, config: Path) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{config}: {field} must be a mapping")
    return value


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_with_base(config: Path) -> dict[str, Any]:
    try:
        with config.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{config}: cannot parse YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{config}: top-level YAML value must be a mapping")
    base_value = raw.pop("base_config", None)
    if base_value is None:
        return raw
    if not isinstance(base_value, str) or not base_value.strip():
        raise ValueError(f"{config}: base_config must be a non-empty string")
    base_path = Path(base_value)
    if not base_path.is_absolute():
        base_path = config.parent / base_path
    if not base_path.is_file():
        raise ValueError(f"{config}: base_config not found: {base_path}")
    return merge_config(load_yaml_with_base(base_path.resolve()), raw)


def load_record(config: Path) -> ConfigRecord:
    raw = load_yaml_with_base(config)

    experiment = mapping(raw.get("experiment"), "experiment", config)
    tracking = mapping(raw.get("tracking"), "tracking", config)
    partition = mapping(raw.get("partition"), "partition", config)
    dataset = mapping(raw.get("dataset"), "dataset", config)

    output_value = experiment.get("output_dir")
    if not isinstance(output_value, str) or not output_value.strip():
        raise ValueError(f"{config}: experiment.output_dir is required")
    output_dir = absolute_path(output_value)

    database_name = tracking.get("database_name", "metrics.db")
    if not isinstance(database_name, str) or not database_name.strip():
        raise ValueError(f"{config}: tracking.database_name must be a non-empty string")
    database_path = absolute_path(output_dir / database_name)

    log_value = experiment.get("log_path")
    if log_value is not None and (not isinstance(log_value, str) or not log_value.strip()):
        raise ValueError(f"{config}: experiment.log_path must be a non-empty string when set")
    config_log_path = absolute_path(log_value) if log_value else None

    partition_type = str(partition.get("type", "iid")).strip().lower()
    summary_value = partition.get("summary_dir")
    if summary_value is not None and (
        not isinstance(summary_value, str) or not summary_value.strip()
    ):
        raise ValueError(f"{config}: partition.summary_dir must be a non-empty string when set")
    summary_dir = absolute_path(summary_value) if summary_value else None
    if partition_type in NON_IID_TYPES and summary_dir is None:
        raise ValueError(f"{config}: non-IID partition requires partition.summary_dir")

    return ConfigRecord(
        config=config.resolve(strict=False),
        name=str(experiment.get("name", config.stem)),
        seed=experiment.get("seed", raw.get("seed")),
        output_dir=output_dir,
        database_path=database_path,
        config_log_path=config_log_path,
        checkpoint_dir=output_dir / "checkpoints",
        feature_csv=output_dir / "feature_margins.csv",
        partition_type=partition_type,
        summary_dir=summary_dir,
        dataset_download=dataset.get("download") is True,
    )


def config_files(config_dir: Path) -> list[Path]:
    if not config_dir.is_dir():
        raise ValueError(f"config directory does not exist or is not a directory: {config_dir}")
    return sorted(
        (path.resolve(strict=False) for path in config_dir.iterdir()
         if path.is_file() and path.suffix.lower() in YAML_SUFFIXES),
        key=lambda path: str(path),
    )


def collect_inputs(config_dir: str | None, paths: Iterable[str]) -> list[Path]:
    result: list[Path] = []
    if config_dir:
        result.extend(config_files(absolute_path(config_dir)))
    for raw in paths:
        path = absolute_path(raw)
        if not path.is_file():
            raise ValueError(f"config file does not exist or is not a file: {path}")
        if path.suffix.lower() not in YAML_SUFFIXES:
            raise ValueError(f"config file must end in .yaml or .yml: {path}")
        result.append(path)
    result = sorted(set(result), key=lambda path: str(path))
    if not result:
        raise ValueError("no .yaml or .yml config files were provided")
    return result


def report_duplicate(errors: list[str], label: str, pairs: dict[Path, list[ConfigRecord]]) -> None:
    for target, records in sorted(pairs.items(), key=lambda item: str(item[0])):
        if len(records) > 1:
            configs = " ; ".join(str(record.config) for record in records)
            errors.append(f"DUPLICATE {label}: {target}\n  configs: {configs}")


def validate(records: list[ConfigRecord]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    targets: dict[str, defaultdict[Path, list[ConfigRecord]]] = {
        "output_dir": defaultdict(list),
        "SQLite DB": defaultdict(list),
        "experiment.log_path": defaultdict(list),
        "checkpoint directory": defaultdict(list),
        "feature CSV": defaultdict(list),
        "non-IID partition.summary_dir": defaultdict(list),
    }
    for record in records:
        targets["output_dir"][record.output_dir].append(record)
        targets["SQLite DB"][record.database_path].append(record)
        if record.config_log_path:
            targets["experiment.log_path"][record.config_log_path].append(record)
        targets["checkpoint directory"][record.checkpoint_dir].append(record)
        targets["feature CSV"][record.feature_csv].append(record)
        if record.partition_type in NON_IID_TYPES and record.summary_dir:
            targets["non-IID partition.summary_dir"][record.summary_dir].append(record)
        if record.dataset_download:
            warnings.append(
                f"dataset.download=true: {record.config} (download/cache contention is possible)"
            )

        for label, target in (
            ("existing output_dir", record.output_dir),
            ("existing SQLite DB", record.database_path),
        ):
            if target.exists():
                errors.append(f"EXISTING {label}: {target}\n  config: {record.config}")
        if record.config_log_path and record.config_log_path.exists():
            errors.append(
                f"EXISTING experiment.log_path: {record.config_log_path}\n  config: {record.config}"
            )

    for label, pair_map in targets.items():
        report_duplicate(errors, label, pair_map)
    return errors, warnings


def print_human(records: list[ConfigRecord], errors: list[str], warnings: list[str]) -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config count: {len(records)}")
    for record in records:
        print(f"CONFIG: {record.config}")
        print(f"  output_dir: {record.output_dir}")
        print(f"  sqlite_db: {record.database_path}")
        print(f"  config_log: {record.config_log_path or '<not configured>'}")
        print(f"  checkpoint_dir: {record.checkpoint_dir}")
        print(f"  feature_csv: {record.feature_csv}")
        print(f"  partition.type: {record.partition_type}")
        print(f"  partition.summary_dir: {record.summary_dir or '<not configured>'}")
        print(f"  seed: {record.seed!r}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        help="Directory whose immediate .yaml/.yml files are validated in sorted order.",
    )
    parser.add_argument("configs", nargs="*", help="Individual YAML config paths.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config_dir and not args.configs:
        print("ERROR: provide --config-dir and/or one or more YAML config paths", file=sys.stderr)
        return 2
    try:
        records = [load_record(path) for path in collect_inputs(args.config_dir, args.configs)]
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    errors, warnings = validate(records)
    print_human(records, errors, warnings)
    if errors:
        print(f"Validation failed: {len(errors)} error(s). Queue will not start.", file=sys.stderr)
        return 1
    print("Validation passed: all write targets are unique and absent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

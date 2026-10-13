import csv
import sqlite3

import pytest

from experiment_tracking.exporter import export_metrics
from experiment_tracking.recorder import ExperimentRecorder


def test_recorder_record_round_writes_attack_success_rate(tmp_path) -> None:
    recorder = ExperimentRecorder(tmp_path / "metrics.db")

    recorder.record_round(
        {
            "round": 1,
            "selected_clients": [0, 1],
            "average_train_loss": 1.25,
            "test_loss": 1.05,
            "clean_accuracy": 55.5,
            "attack_success_rate": 80.0,
            "num_attack_samples": 90,
            "num_successful_attacks": 72,
            "psnr": 38.34,
            "ssim": 0.992,
            "linf": 0.05,
        }
    )

    records = recorder.load_round_metrics()
    assert records[0]["attack_success_rate"] == pytest.approx(80.0)
    assert records[0]["num_attack_samples"] == 90
    assert records[0]["num_successful_attacks"] == 72
    assert records[0]["psnr"] == pytest.approx(38.34)
    assert records[0]["ssim"] == pytest.approx(0.992)
    assert records[0]["linf"] == pytest.approx(0.05)


def test_recorder_record_clients_writes_attack_fields(tmp_path) -> None:
    recorder = ExperimentRecorder(tmp_path / "metrics.db")

    recorder.record_clients(
        round_index=1,
        client_results=[
            {
                "client_id": 0,
                "num_samples": 100,
                "train_loss": 1.1,
                "is_malicious": True,
                "attack_active": True,
                "num_poisoned_samples": 20,
            }
        ],
    )

    records = recorder.load_client_metrics()
    assert records[0]["is_malicious"] is True
    assert records[0]["attack_active"] is True
    assert records[0]["num_poisoned_samples"] == 20


def test_recorder_initialization_adds_missing_columns_to_old_database(tmp_path) -> None:
    database_path = tmp_path / "old_metrics.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE round_metrics (
                round_index INTEGER PRIMARY KEY,
                selected_clients TEXT NOT NULL,
                average_train_loss REAL NOT NULL,
                test_loss REAL NOT NULL,
                clean_accuracy REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE client_metrics (
                round_index INTEGER NOT NULL,
                client_id INTEGER NOT NULL,
                num_samples INTEGER NOT NULL,
                train_loss REAL NOT NULL,
                PRIMARY KEY (round_index, client_id)
            )
            """
        )

    ExperimentRecorder(database_path)

    with sqlite3.connect(database_path) as connection:
        round_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(round_metrics)")
        }
        client_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(client_metrics)")
        }

    assert "attack_success_rate" in round_columns
    assert "num_attack_samples" in round_columns
    assert "num_successful_attacks" in round_columns
    assert "psnr" in round_columns
    assert "ssim" in round_columns
    assert "linf" in round_columns
    assert "recorded_at" in round_columns
    assert "is_malicious" in client_columns
    assert "attack_active" in client_columns
    assert "num_poisoned_samples" in client_columns


def test_exporter_writes_round_and_client_csv_files(tmp_path) -> None:
    database_path = tmp_path / "metrics.db"
    output_dir = tmp_path / "exports"
    recorder = ExperimentRecorder(database_path)
    recorder.record_round(
        {
            "round": 1,
            "selected_clients": [0],
            "average_train_loss": 1.0,
            "test_loss": 0.9,
            "clean_accuracy": 60.0,
            "attack_success_rate": 50.0,
            "num_attack_samples": 10,
            "num_successful_attacks": 5,
        }
    )
    recorder.record_clients(
        1,
        [
            {
                "client_id": 0,
                "num_samples": 10,
                "train_loss": 1.0,
                "is_malicious": True,
                "attack_active": True,
                "num_poisoned_samples": 2,
            }
        ],
    )

    round_csv, client_csv = export_metrics(database_path, output_dir)

    assert round_csv.exists()
    assert client_csv.exists()
    with round_csv.open("r", newline="", encoding="utf-8") as csv_file:
        row = next(csv.DictReader(csv_file))
    assert row["attack_success_rate"] == "50.0"
    with client_csv.open("r", newline="", encoding="utf-8") as csv_file:
        row = next(csv.DictReader(csv_file))
    assert row["num_poisoned_samples"] == "2"


def test_plots_generate_asr_curve_when_asr_data_exists(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from experiment_tracking.plots import generate_plots

    database_path = tmp_path / "metrics.db"
    output_dir = tmp_path / "figures"
    recorder = ExperimentRecorder(database_path)
    recorder.record_round(
        {
            "round": 1,
            "selected_clients": [0],
            "average_train_loss": 1.0,
            "test_loss": 0.9,
            "clean_accuracy": 60.0,
            "attack_success_rate": 50.0,
            "num_attack_samples": 10,
            "num_successful_attacks": 5,
        }
    )
    recorder.record_clients(
        1,
        [
            {
                "client_id": 0,
                "num_samples": 10,
                "train_loss": 1.0,
                "num_poisoned_samples": 2,
            }
        ],
    )

    generate_plots(database_path, output_dir)

    assert (output_dir / "asr_curve.png").exists()
    assert (output_dir / "asr_curve.svg").exists()


def test_plots_skip_asr_curve_when_asr_is_null(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from experiment_tracking.plots import generate_plots

    database_path = tmp_path / "metrics.db"
    output_dir = tmp_path / "figures"
    recorder = ExperimentRecorder(database_path)
    recorder.record_round(
        {
            "round": 1,
            "selected_clients": [0],
            "average_train_loss": 1.0,
            "test_loss": 0.9,
            "clean_accuracy": 60.0,
        }
    )

    generate_plots(database_path, output_dir)

    assert not (output_dir / "asr_curve.png").exists()
    assert not (output_dir / "asr_curve.svg").exists()

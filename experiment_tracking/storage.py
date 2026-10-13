"""SQLite storage for federated experiment metrics."""

import json
import sqlite3
from pathlib import Path

from experiment_tracking.schema import ClientMetric, RoundMetric


class MetricsStorage:
    """Persist and retrieve round-level and client-level metrics in SQLite."""

    def __init__(self, database_path: str | Path) -> None:
        """Create storage backed by the given SQLite database path."""
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = path
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Create metric tables and add missing columns for older databases."""
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS round_metrics (
                    round_index INTEGER PRIMARY KEY,
                    selected_clients TEXT NOT NULL,
                    selected_malicious_clients TEXT,
                    num_selected_malicious INTEGER NOT NULL DEFAULT 0,
                    attack_active INTEGER NOT NULL DEFAULT 0,
                    aggregation_weighting TEXT NOT NULL DEFAULT 'sample_weighted',
                    bn_buffer_policy TEXT NOT NULL DEFAULT 'aggregate',
                    protocol_warning TEXT,
                    average_train_loss REAL NOT NULL,
                    test_loss REAL NOT NULL,
                    clean_accuracy REAL NOT NULL,
                    attack_success_rate REAL,
                    num_attack_samples INTEGER,
                    num_successful_attacks INTEGER,
                    psnr REAL,
                    ssim REAL,
                    linf REAL,
                    recorded_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS client_metrics (
                    round_index INTEGER NOT NULL,
                    client_id INTEGER NOT NULL,
                    num_samples INTEGER NOT NULL,
                    train_loss REAL NOT NULL,
                    is_malicious INTEGER NOT NULL DEFAULT 0,
                    attack_active INTEGER NOT NULL DEFAULT 0,
                    num_poisoned_samples INTEGER NOT NULL DEFAULT 0,
                    ce_loss_avg REAL,
                    proto_train_loss_avg REAL,
                    proto_train_batches INTEGER NOT NULL DEFAULT 0,
                    adaptation_loss_avg REAL,
                    projection_loss_avg REAL,
                    supcon_valid_anchors INTEGER NOT NULL DEFAULT 0,
                    supcon_target_valid_anchors INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (round_index, client_id)
                )
                """
            )
            self._add_missing_columns(
                connection,
                "round_metrics",
                {
                    "selected_malicious_clients": "TEXT",
                    "num_selected_malicious": "INTEGER NOT NULL DEFAULT 0",
                    "attack_active": "INTEGER NOT NULL DEFAULT 0",
                    "aggregation_weighting": "TEXT NOT NULL DEFAULT 'sample_weighted'",
                    "bn_buffer_policy": "TEXT NOT NULL DEFAULT 'aggregate'",
                    "protocol_warning": "TEXT",
                    "attack_success_rate": "REAL",
                    "num_attack_samples": "INTEGER",
                    "num_successful_attacks": "INTEGER",
                    "psnr": "REAL",
                    "ssim": "REAL",
                    "linf": "REAL",
                    "recorded_at": "TEXT",
                },
            )
            self._add_missing_columns(
                connection,
                "client_metrics",
                {
                    "is_malicious": "INTEGER NOT NULL DEFAULT 0",
                    "attack_active": "INTEGER NOT NULL DEFAULT 0",
                    "num_poisoned_samples": "INTEGER NOT NULL DEFAULT 0",
                    "ce_loss_avg": "REAL",
                    "proto_train_loss_avg": "REAL",
                    "proto_train_batches": "INTEGER NOT NULL DEFAULT 0",
                    "adaptation_loss_avg": "REAL",
                    "projection_loss_avg": "REAL",
                    "supcon_valid_anchors": "INTEGER NOT NULL DEFAULT 0",
                    "supcon_target_valid_anchors": "INTEGER NOT NULL DEFAULT 0",
                },
            )

    def _add_missing_columns(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        columns: dict[str, str],
    ) -> None:
        """Add missing columns to an existing SQLite table."""
        existing_columns = {
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for column_name, column_definition in columns.items():
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE {table_name} "
                    f"ADD COLUMN {column_name} {column_definition}"
                )

    def save_round_metric(self, metric: RoundMetric) -> None:
        """Insert or replace one round-level metric record."""
        selected_clients = json.dumps(metric.selected_clients)
        selected_malicious_clients = json.dumps(metric.selected_malicious_clients or [])
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO round_metrics (
                    round_index,
                    selected_clients,
                    selected_malicious_clients,
                    num_selected_malicious,
                    attack_active,
                    aggregation_weighting,
                    bn_buffer_policy,
                    protocol_warning,
                    average_train_loss,
                    test_loss,
                    clean_accuracy,
                    attack_success_rate,
                    num_attack_samples,
                    num_successful_attacks,
                    psnr,
                    ssim,
                    linf,
                    recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metric.round_index,
                    selected_clients,
                    selected_malicious_clients,
                    metric.num_selected_malicious,
                    int(metric.attack_active),
                    metric.aggregation_weighting,
                    metric.bn_buffer_policy,
                    metric.protocol_warning,
                    metric.average_train_loss,
                    metric.test_loss,
                    metric.clean_accuracy,
                    metric.attack_success_rate,
                    metric.num_attack_samples,
                    metric.num_successful_attacks,
                    metric.psnr,
                    metric.ssim,
                    metric.linf,
                    metric.recorded_at,
                ),
            )

    def save_client_metric(self, metric: ClientMetric) -> None:
        """Insert or replace one client-level metric record."""
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO client_metrics (
                    round_index, client_id, num_samples, train_loss,
                    is_malicious, attack_active, num_poisoned_samples,
                    ce_loss_avg, proto_train_loss_avg, proto_train_batches,
                    adaptation_loss_avg, projection_loss_avg,
                    supcon_valid_anchors, supcon_target_valid_anchors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metric.round_index, metric.client_id, metric.num_samples, metric.train_loss,
                    int(metric.is_malicious), int(metric.attack_active), metric.num_poisoned_samples,
                    metric.ce_loss_avg, metric.proto_train_loss_avg, metric.proto_train_batches,
                    metric.adaptation_loss_avg, metric.projection_loss_avg,
                    metric.supcon_valid_anchors, metric.supcon_target_valid_anchors,
                ),
            )

    def load_round_metrics(self) -> list[dict]:
        """Load all round-level metrics ordered by round index."""
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT
                    round_index,
                    selected_clients,
                    selected_malicious_clients,
                    num_selected_malicious,
                    attack_active,
                    aggregation_weighting,
                    bn_buffer_policy,
                    protocol_warning,
                    average_train_loss,
                    test_loss,
                    clean_accuracy,
                    attack_success_rate,
                    num_attack_samples,
                    num_successful_attacks,
                    psnr,
                    ssim,
                    linf,
                    recorded_at
                FROM round_metrics
                ORDER BY round_index ASC
                """
            ).fetchall()

        return [
            {
                "round_index": row[0], "selected_clients": [int(x) for x in json.loads(row[1])],
                "selected_malicious_clients": [int(x) for x in json.loads(row[2] or "[]")],
                "num_selected_malicious": row[3], "attack_active": bool(row[4]),
                "aggregation_weighting": row[5], "bn_buffer_policy": row[6], "protocol_warning": row[7],
                "average_train_loss": row[8], "test_loss": row[9], "clean_accuracy": row[10],
                "attack_success_rate": row[11], "num_attack_samples": row[12],
                "num_successful_attacks": row[13], "psnr": row[14], "ssim": row[15],
                "linf": row[16], "recorded_at": row[17],
            }
            for row in rows
        ]

    def load_client_metrics(self) -> list[dict]:
        """Load all client-level metrics ordered by round and client."""
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT round_index, client_id, num_samples, train_loss,
                       is_malicious, attack_active, num_poisoned_samples,
                       ce_loss_avg, proto_train_loss_avg, proto_train_batches,
                       adaptation_loss_avg, projection_loss_avg,
                       supcon_valid_anchors, supcon_target_valid_anchors
                FROM client_metrics
                ORDER BY round_index ASC, client_id ASC
                """
            ).fetchall()

        return [
            {
                "round_index": row[0],
                "client_id": row[1],
                "num_samples": row[2],
                "train_loss": row[3],
                "is_malicious": bool(row[4]),
                "attack_active": bool(row[5]),
                "num_poisoned_samples": row[6],
                "ce_loss_avg": row[7],
                "proto_train_loss_avg": row[8],
                "proto_train_batches": row[9],
                "adaptation_loss_avg": row[10],
                "projection_loss_avg": row[11],
                "supcon_valid_anchors": row[12],
                "supcon_target_valid_anchors": row[13],
            }
            for row in rows
        ]

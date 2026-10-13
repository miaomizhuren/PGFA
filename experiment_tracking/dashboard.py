"""Streamlit dashboard for monitoring federated-learning experiments."""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Federated Learning Dashboard",
    layout="wide",
)

st.title("Federated Learning Experiment Dashboard")

project_root = Path(__file__).resolve().parent.parent
default_output_dir = project_root / "outputs"


def resolve_project_path(path_text: str) -> Path:
    """Resolve a dashboard path relative to the project root."""
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def find_database_files(output_dir: Path) -> list[Path]:
    """Find SQLite database files sorted by most recent modification time."""
    if not output_dir.exists() or not output_dir.is_dir():
        return []
    return sorted(
        output_dir.glob("*.db"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def table_exists(database_path: Path, table_name: str) -> bool:
    """Return whether a SQLite database contains a table."""
    connection_uri = f"file:{database_path}?mode=ro"
    with sqlite3.connect(connection_uri, uri=True) as connection:
        row = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (table_name,),
        ).fetchone()
    return row is not None


def load_table(database_path: Path, table_name: str) -> pd.DataFrame:
    """Load a SQLite table into a pandas DataFrame without writing to the database."""
    connection_uri = f"file:{database_path}?mode=ro"
    with sqlite3.connect(connection_uri, uri=True) as connection:
        return pd.read_sql_query(
            f"SELECT * FROM {table_name}",
            connection,
        )


def render_optional_metric(column, label: str, value, precision: int, suffix: str = "") -> None:
    """Render a metric card, using N/A when a database value is unavailable."""
    if value is not None and pd.notna(value):
        column.metric(label, f"{value:.{precision}f}{suffix}")
    else:
        column.metric(label, "N/A")


def render_optional_curve(round_metrics: pd.DataFrame, metric_name: str, title: str) -> None:
    """Render one optional round-metric curve without breaking on old databases."""
    st.subheader(title)
    if metric_name not in round_metrics.columns:
        st.info(f"Current database has no {metric_name} column.")
    elif not round_metrics[metric_name].notna().any():
        st.info(f"Current database has no {metric_name} records.")
    else:
        st.line_chart(round_metrics.set_index("round_index")[[metric_name]])


output_dir = resolve_project_path(
    st.sidebar.text_input(
        "DB search directory",
        value="outputs",
    )
)
st.sidebar.caption(f"Resolved search directory: {output_dir}")

database_files = find_database_files(output_dir)
use_manual_database_path = st.sidebar.checkbox("Use manual database path", value=False)
manual_database_path = resolve_project_path(
    st.sidebar.text_input(
        "Manual database path",
        value=str(default_output_dir / "metrics.db"),
    )
)

if st.sidebar.button("Refresh database list"):
    st.rerun()

if use_manual_database_path:
    database_path = manual_database_path
elif database_files:
    selected_database = st.sidebar.selectbox(
        "Experiment database",
        options=database_files,
        index=0,
        format_func=lambda path: path.name,
    )
    database_path = selected_database
else:
    database_path = None


@st.fragment(run_every="5s")
def render_dashboard() -> None:
    """Reload and display the latest experiment metrics."""
    if database_path is None:
        st.warning("No .db files found in outputs directory.")
        return

    st.caption(f"Current database: {database_path}")

    if not database_path.exists():
        st.warning("Metrics database not found. Start training first.")
        return

    try:
        has_round_metrics = table_exists(database_path, "round_metrics")
        has_client_metrics = table_exists(database_path, "client_metrics")
    except sqlite3.DatabaseError as error:
        st.error(f"Unable to read selected database: {error}")
        return

    if not has_round_metrics:
        st.warning("Selected database does not contain round_metrics table.")
        return

    round_metrics = load_table(database_path, "round_metrics").sort_values(
        "round_index"
    )

    if round_metrics.empty:
        st.info("No round metrics have been recorded yet.")
        return

    latest = round_metrics.iloc[-1]
    (
        current_round,
        clean_accuracy,
        attack_success_rate,
        test_loss,
        psnr,
        ssim,
        linf,
    ) = st.columns(7)
    current_round.metric("Current Round", int(latest["round_index"]))
    clean_accuracy.metric("Clean Accuracy", f"{latest['clean_accuracy']:.2f}%")
    render_optional_metric(
        attack_success_rate,
        "Attack Success Rate",
        latest.get("attack_success_rate"),
        2,
        "%",
    )
    test_loss.metric("Test Loss", f"{latest['test_loss']:.4f}")
    render_optional_metric(psnr, "PSNR", latest.get("psnr"), 2, " dB")
    render_optional_metric(ssim, "SSIM", latest.get("ssim"), 3)
    render_optional_metric(linf, "L_inf", latest.get("linf"), 3)

    st.subheader("Clean Accuracy vs Round")
    st.line_chart(round_metrics.set_index("round_index")[["clean_accuracy"]])

    render_optional_curve(round_metrics, "attack_success_rate", "ASR vs Round")

    st.subheader("Train Loss / Test Loss vs Round")
    st.line_chart(
        round_metrics.set_index("round_index")[
            ["average_train_loss", "test_loss"]
        ]
    )

    render_optional_curve(round_metrics, "psnr", "PSNR vs Round")
    render_optional_curve(round_metrics, "ssim", "SSIM vs Round")
    render_optional_curve(round_metrics, "linf", "L_inf vs Round")

    st.subheader("Round Metrics")
    st.dataframe(round_metrics, width="stretch")

    if not has_client_metrics:
        st.info("Selected database does not contain client_metrics table.")
        return

    client_metrics = load_table(database_path, "client_metrics").sort_values(
        ["round_index", "client_id"]
    )
    if client_metrics.empty:
        st.info("No client metrics have been recorded yet.")
        return

    st.subheader("Client Metrics")
    st.dataframe(client_metrics, width="stretch")

    st.subheader("Poisoned Samples vs Round")
    if "num_poisoned_samples" in client_metrics.columns:
        poisoned_samples = (
            client_metrics.groupby("round_index", as_index=True)[
                "num_poisoned_samples"
            ]
            .sum()
            .to_frame()
        )
        st.line_chart(poisoned_samples)
    else:
        st.info("No poisoned sample records found.")


render_dashboard()

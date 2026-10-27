
"""Compare GARCH and LSTM volatility forecasting results."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

GARCH_METRICS = RESULTS_DIR / "garch_metrics.csv"
LSTM_METRICS = RESULTS_DIR / "lstm_metrics.csv"
GARCH_FORECASTS = RESULTS_DIR / "garch_forecasts.csv"
LSTM_FORECASTS = RESULTS_DIR / "lstm_forecasts.csv"

COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Check that the required outputs exist.
    required_files = [
        GARCH_METRICS,
        LSTM_METRICS,
        GARCH_FORECASTS,
        LSTM_FORECASTS,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # Combine evaluation metrics from both models.
    garch_metrics = pd.read_csv(GARCH_METRICS)
    lstm_metrics = pd.read_csv(LSTM_METRICS)

    comparison = pd.concat(
        [garch_metrics, lstm_metrics],
        ignore_index=True,
    )

    comparison.to_csv(COMPARISON_PATH, index=False)

    print("=" * 60)
    print("GARCH VS LSTM COMPARISON")
    print("=" * 60)
    print(comparison.to_string(index=False))
    print(f"\nSaved comparison: {COMPARISON_PATH}")

    # --------------------------------------------------
    # GRAPH 1: METRIC COMPARISON
    # --------------------------------------------------

    metrics_to_plot = ["QLIKE", "MSE", "RMSE", "MAE"]

    for split_name in ["validation", "test"]:
        subset = comparison[
            comparison["split"] == split_name
        ]

        if subset.empty:
            continue

        ax = subset.set_index("model")[
            metrics_to_plot
        ].plot(
            kind="bar",
            figsize=(10, 6),
            subplots=True,
            layout=(2, 2),
            legend=False,
            rot=0,
            title=[
                f"{metric} — {split_name.title()}"
                for metric in metrics_to_plot
            ],
        )

        plt.tight_layout()

        figure_path = (
            FIGURES_DIR
            / f"model_metrics_{split_name}.png"
        )

        plt.savefig(figure_path, dpi=150)
        plt.close()

        print(f"Saved graph: {figure_path}")

    # --------------------------------------------------
    # GRAPH 2: TEST ACTUAL VS FORECAST VARIANCE
    # --------------------------------------------------

    garch = pd.read_csv(
        GARCH_FORECASTS,
        parse_dates=["date"],
    )

    lstm = pd.read_csv(
        LSTM_FORECASTS,
        parse_dates=["date"],
    )

    garch_test = garch[
        garch["split"] == "test"
    ][
        ["date", "actual_variance", "forecast_variance"]
    ].rename(
        columns={
            "actual_variance": "actual_garch",
            "forecast_variance": "garch_forecast",
        }
    )

    lstm_test = lstm[
        lstm["split"] == "test"
    ][
        ["date", "actual_variance", "forecast_variance"]
    ].rename(
        columns={
            "actual_variance": "actual_lstm",
            "forecast_variance": "lstm_forecast",
        }
    )

    merged = pd.merge(
        garch_test,
        lstm_test,
        on="date",
        how="inner",
    ).sort_values("date")

    if merged.empty:
        raise ValueError(
            "No matching test dates were found between GARCH and LSTM."
        )

    # Both models should use the same realized-variance proxy.
    if not np.allclose(
        merged["actual_garch"],
        merged["actual_lstm"],
        rtol=1e-8,
        atol=1e-8,
    ):
        raise ValueError(
            "GARCH and LSTM actual variance values do not match."
        )

    plt.figure(figsize=(13, 6))

    plt.plot(
        merged["date"],
        merged["actual_garch"],
        label="Actual squared return",
        alpha=0.65,
        linewidth=1,
    )

    plt.plot(
        merged["date"],
        merged["garch_forecast"],
        label="GARCH forecast variance",
        linewidth=1.2,
    )

    plt.plot(
        merged["date"],
        merged["lstm_forecast"],
        label="LSTM forecast variance",
        linewidth=1.2,
    )

    plt.title("Test Period: Actual vs Forecast Variance")
    plt.xlabel("Date")
    plt.ylabel("Daily variance (squared percentage returns)")
    plt.legend()
    plt.tight_layout()

    forecast_figure = (
        FIGURES_DIR / "test_actual_vs_forecast_variance.png"
    )

    plt.savefig(forecast_figure, dpi=150)
    plt.close()

    print(f"Saved graph: {forecast_figure}")
    print(f"Matching test dates: {len(merged)}")

    # --------------------------------------------------
    # GRAPH 3: LSTM TRAINING AND VALIDATION LOSS
    # --------------------------------------------------

    history_path = RESULTS_DIR / "lstm_training_history.csv"

    if history_path.exists():
        history = pd.read_csv(history_path)

        plt.figure(figsize=(10, 6))

        for seed, group in history.groupby("seed"):
            plt.plot(
                group["epoch"],
                group["training_loss"],
                label=f"Seed {seed} training",
            )

            plt.plot(
                group["epoch"],
                group["validation_loss"],
                linestyle="--",
                label=f"Seed {seed} validation",
            )

        plt.title("LSTM Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Mean squared error on log variance")
        plt.legend()
        plt.tight_layout()

        loss_figure = (
            FIGURES_DIR / "lstm_training_validation_loss.png"
        )

        plt.savefig(loss_figure, dpi=150)
        plt.close()

        print(f"Saved graph: {loss_figure}")

    print("\nComparison completed successfully.")


if __name__ == "__main__":
    main()

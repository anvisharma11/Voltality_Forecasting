
"""Rolling one-step-ahead GARCH(1,1) volatility forecasts."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed.csv"
RESULTS_DIR = BASE_DIR / "results"
FORECAST_PATH = RESULTS_DIR / "garch_forecasts.csv"
SUMMARY_PATH = RESULTS_DIR / "garch_metrics.csv"

# Project settings
REFIT_EVERY = 20
MIN_TRAINING_OBSERVATIONS = 500
VARIANCE_FLOOR = 1e-6


def fit_garch(returns):
    """Fit a GARCH(1,1) model with Student's t innovations."""
    model = arch_model(
        returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted_model = model.fit(disp="off", show_warning=False)

    return fitted_model


def forecast_next_variance(fitted_model):
    """Forecast the variance of the next daily return."""
    forecast = fitted_model.forecast(horizon=1, reindex=False)
    variance = float(forecast.variance.iloc[-1, 0])
    return max(variance, VARIANCE_FLOOR)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {DATA_PATH}. "
            "Run python data.py first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load processed data. The CSV may store dates in its first column.
    data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    data.index.name = "date"
    data = data.sort_index()

    required_columns = {"ret", "split"}
    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns in processed.csv: {sorted(missing)}. "
            f"Columns found: {list(data.columns)}"
        )

    data["ret"] = pd.to_numeric(data["ret"], errors="coerce")
    data = data.dropna(subset=["ret", "split"])

    if data.index.has_duplicates:
        raise ValueError("Duplicate dates found in processed data.")

    # The first fit uses training returns only.
    train_data = data[data["split"] == "train"]
    forecast_data = data[
        data["split"].isin(["validation", "test"])
    ]

    if len(train_data) < MIN_TRAINING_OBSERVATIONS:
        raise ValueError(
            "Not enough training observations to fit the GARCH model."
        )

    if forecast_data.empty:
        raise ValueError("No validation or test observations were found.")

    history = train_data["ret"].astype(float).tolist()
    fitted_model = fit_garch(np.asarray(history))

    print("Initial GARCH(1,1) model fitted.")
    print(f"Training observations: {len(history)}")
    print("Error distribution: Student's t")
    print(f"Forecast observations: {len(forecast_data)}")
    print(f"Parameter refit frequency: every {REFIT_EVERY} forecasts")
    print()

    records = []
    forecasts_since_refit = 0

    for date, row in forecast_data.iterrows():
        # Forecast before adding the current day's actual return.
        # Therefore, today's return cannot influence today's forecast.
        predicted_variance = forecast_next_variance(fitted_model)
        actual_return = float(row["ret"])

        records.append(
            {
                "date": date,
                "split": row["split"],
                "actual_return": actual_return,
                # Squared daily return is a noisy daily variance proxy.
                "actual_variance": actual_return ** 2,
                "forecast_variance": predicted_variance,
                "forecast_volatility": np.sqrt(predicted_variance),
            }
        )

        # The actual return becomes available after the forecast.
        history.append(actual_return)
        forecasts_since_refit += 1

        # Refit periodically using only returns observed so far.
        if forecasts_since_refit >= REFIT_EVERY:
            fitted_model = fit_garch(np.asarray(history))
            forecasts_since_refit = 0

    forecasts = pd.DataFrame(records)
    forecasts.to_csv(FORECAST_PATH, index=False)

    # Calculate metrics separately for validation and test.
    # Metrics are computed against daily squared returns because this
    # model produces one-step-ahead daily variance forecasts.
    metric_rows = []

    for split_name in ["validation", "test"]:
        subset = forecasts[forecasts["split"] == split_name]

        if subset.empty:
            continue

        actual = np.maximum(
            subset["actual_variance"].to_numpy(dtype=float),
            VARIANCE_FLOOR,
        )
        predicted = np.maximum(
            subset["forecast_variance"].to_numpy(dtype=float),
            VARIANCE_FLOOR,
        )

        qlike = np.mean(np.log(predicted) + actual / predicted)
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predicted))

        metric_rows.append(
            {
                "model": "GARCH(1,1)-Student-t",
                "split": split_name,
                "observations": len(subset),
                "QLIKE": float(qlike),
                "MSE": float(mse),
                "RMSE": float(rmse),
                "MAE": float(mae),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(SUMMARY_PATH, index=False)

    print("GARCH FORECASTING COMPLETED")
    print(f"Forecast file: {FORECAST_PATH}")
    print(f"Metrics file:  {SUMMARY_PATH}")
    print()
    print("Forecast counts:")
    print(forecasts.groupby("split").size().to_string())
    print()
    print("Evaluation metrics:")
    print(metrics.to_string(index=False))

    # Basic output checks
    assert forecasts["forecast_variance"].notna().all()
    assert np.isfinite(forecasts["forecast_variance"]).all()
    assert (forecasts["forecast_variance"] > 0).all()
    assert set(forecasts["split"].unique()).issubset(
        {"validation", "test"}
    )
    assert {"validation", "test"}.issubset(
        set(forecasts["split"].unique())
    )

    print("\nAll GARCH output checks passed.")


if __name__ == "__main__":
    main()

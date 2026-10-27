
"""Evaluation metrics for volatility forecasting.

Actual and forecast inputs are variances, not standard deviations.
All metrics use the same observations for a fair model comparison.
"""

import numpy as np


# Small positive floor to prevent division by zero and log(0).
QLIKE_FLOOR = 1e-6


def _prepare_arrays(actual_variance, forecast_variance, floor=QLIKE_FLOOR):
    """Validate and align actual and forecast variance arrays."""
    actual = np.asarray(actual_variance, dtype=float).reshape(-1)
    forecast = np.asarray(forecast_variance, dtype=float).reshape(-1)

    if actual.size == 0 or forecast.size == 0:
        raise ValueError("Actual and forecast arrays must not be empty.")

    if actual.size != forecast.size:
        raise ValueError(
            "Actual and forecast arrays must have the same length."
        )

    if not np.all(np.isfinite(actual)):
        raise ValueError("Actual variance contains missing or infinite values.")

    if not np.all(np.isfinite(forecast)):
        raise ValueError("Forecast variance contains missing or infinite values.")

    if np.any(actual < 0):
        raise ValueError("Actual variance cannot be negative.")

    if floor <= 0:
        raise ValueError("The variance floor must be positive.")

    # Variance forecasts must be strictly positive for QLIKE.
    forecast = np.maximum(forecast, floor)

    return actual, forecast


def qlike(actual_variance, forecast_variance, floor=QLIKE_FLOOR):
    """Calculate mean QLIKE loss. Lower is better."""
    actual, forecast = _prepare_arrays(
        actual_variance, forecast_variance, floor
    )

    actual = np.maximum(actual, floor)

    losses = np.log(forecast) + actual / forecast
    return float(np.mean(losses))


def mse(actual_variance, forecast_variance):
    """Calculate mean squared error. Lower is better."""
    actual, forecast = _prepare_arrays(actual_variance, forecast_variance)
    return float(np.mean((actual - forecast) ** 2))


def rmse(actual_variance, forecast_variance):
    """Calculate root mean squared error. Lower is better."""
    return float(np.sqrt(mse(actual_variance, forecast_variance)))


def mae(actual_variance, forecast_variance):
    """Calculate mean absolute error. Lower is better."""
    actual, forecast = _prepare_arrays(actual_variance, forecast_variance)
    return float(np.mean(np.abs(actual - forecast)))


def evaluate_forecasts(actual_variance, forecast_variance):
    """Return all project evaluation metrics in one dictionary."""
    actual, forecast = _prepare_arrays(
        actual_variance, forecast_variance
    )

    return {
        "observations": int(actual.size),
        "QLIKE": qlike(actual, forecast),
        "MSE": mse(actual, forecast),
        "RMSE": rmse(actual, forecast),
        "MAE": mae(actual, forecast),
    }


if __name__ == "__main__":
    # Small known example to check that the metrics work correctly.
    actual = np.array([1.0, 2.0, 3.0])
    forecast = np.array([1.1, 1.9, 3.2])

    scores = evaluate_forecasts(actual, forecast)

    print("METRICS MODULE TEST")
    for metric, value in scores.items():
        print(f"{metric}: {value}")

    # Sanity checks using known expected values.
    assert scores["observations"] == 3
    assert np.isclose(scores["MSE"], 0.02)
    assert np.isclose(scores["RMSE"], np.sqrt(0.02))
    assert np.isclose(scores["MAE"], 0.13333333333333333)
    assert np.isfinite(scores["QLIKE"])

    print("\nAll metric tests passed.")

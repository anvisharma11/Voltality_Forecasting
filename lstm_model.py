
"""LSTM model for one-day-ahead S&P 500 variance forecasting."""

from pathlib import Path
import os
import random
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed.csv"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = RESULTS_DIR / "models"

FORECAST_PATH = RESULTS_DIR / "lstm_forecasts.csv"
METRICS_PATH = RESULTS_DIR / "lstm_metrics.csv"
TRAINING_HISTORY_PATH = RESULTS_DIR / "lstm_training_history.csv"

LOOKBACK = 30
LSTM_UNITS = 64
DROPOUT = 0.2
MAX_EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10

# Train three versions with different random seeds.
SEEDS = [42, 43, 44]

VARIANCE_FLOOR = 1e-6
FEATURES = ["ret", "hl_range", "vol_chg", "rv5"]


# --------------------------------------------------
# 2. REPRODUCIBILITY
# --------------------------------------------------

def set_seeds(seed):
    """Set random seeds for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# --------------------------------------------------
# 3. BUILD THE LSTM NETWORK
# --------------------------------------------------

def build_model(input_shape):
    """Create an LSTM neural network for log variance."""
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(LSTM_UNITS),
            Dropout(DROPOUT),
            Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mse",
    )

    return model


# --------------------------------------------------
# 4. CREATE CHRONOLOGICAL SEQUENCES
# --------------------------------------------------

def create_sequences(data, scaled_features):
    """
    Use the previous 30 trading days to predict the
    current day's squared return.

    The current day's features are never included in
    the input sequence for its own prediction.
    """
    returns = data["ret"].to_numpy(dtype=float)
    dates = data.index
    splits = data["split"].to_numpy()

    X = []
    y = []
    target_dates = []
    target_splits = []

    for position in range(LOOKBACK, len(data)):
        split_name = splits[position]

        if split_name not in ("train", "validation", "test"):
            continue

        # Features from the previous 30 trading days.
        sequence = scaled_features[
            position - LOOKBACK:position
        ]

        # Daily squared return is the realized variance proxy.
        actual_variance = returns[position] ** 2

        # Log transformation makes the target less skewed.
        log_variance = np.log(
            actual_variance + VARIANCE_FLOOR
        )

        X.append(sequence)
        y.append(log_variance)
        target_dates.append(dates[position])
        target_splits.append(split_name)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        pd.DatetimeIndex(target_dates),
        np.asarray(target_splits),
    )


# --------------------------------------------------
# 5. EVALUATION METRICS
# --------------------------------------------------

def calculate_metrics(actual_variance, forecast_variance):
    """Calculate QLIKE, MSE, RMSE and MAE."""
    actual = np.maximum(
        np.asarray(actual_variance, dtype=float),
        VARIANCE_FLOOR,
    )

    forecast = np.maximum(
        np.asarray(forecast_variance, dtype=float),
        VARIANCE_FLOOR,
    )

    qlike = np.mean(
        np.log(forecast) + actual / forecast
    )

    mse = np.mean((actual - forecast) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - forecast))

    return {
        "observations": len(actual),
        "QLIKE": float(qlike),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
    }


# --------------------------------------------------
# 6. MAIN PROGRAM
# --------------------------------------------------

def main():
    print("=" * 60)
    print("LSTM VOLATILITY FORECASTING")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found: {DATA_PATH}\n"
            "Run python data.py first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load processed data.
    data = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True,
    )

    data.index.name = "date"
    data = data.sort_index()

    required_columns = FEATURES + ["split"]
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in processed.csv: {missing_columns}"
        )

    for column in FEATURES:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=FEATURES + ["split"]
    )

    if data.index.has_duplicates:
        raise ValueError(
            "Duplicate dates found in processed.csv."
        )

    # Fit the scaler using training rows only.
    training_rows = data[data["split"] == "train"]

    if len(training_rows) <= LOOKBACK:
        raise ValueError(
            "Insufficient training observations."
        )

    scaler = StandardScaler()
    scaler.fit(training_rows[FEATURES])

    # Transform features using training-set statistics.
    scaled_features = scaler.transform(data[FEATURES])

    # Build sequences.
    X, y, dates, splits = create_sequences(
        data,
        scaled_features,
    )

    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"

    if not train_mask.any():
        raise ValueError("No training sequences were created.")

    if not validation_mask.any():
        raise ValueError("No validation sequences were created.")

    if not test_mask.any():
        raise ValueError("No test sequences were created.")

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_validation = X[validation_mask]
    y_validation = y[validation_mask]

    print(f"Features: {FEATURES}")
    print(f"Lookback: {LOOKBACK} trading days")
    print(f"Training sequences: {len(X_train)}")
    print(f"Validation sequences: {len(X_validation)}")
    print(f"Test sequences: {int(test_mask.sum())}")
    print(f"Random seeds: {SEEDS}")
    print()

    # --------------------------------------------------
    # TRAIN THREE MODELS AND AVERAGE THEIR FORECASTS
    # --------------------------------------------------

    evaluation_mask = validation_mask | test_mask

    X_evaluation = X[evaluation_mask]
    evaluation_dates = dates[evaluation_mask]
    evaluation_splits = splits[evaluation_mask]

    all_predictions = []
    training_history_records = []

    for seed in SEEDS:
        print("-" * 60)
        print(f"Training LSTM with random seed {seed}")
        print("-" * 60)

        set_seeds(seed)

        model = build_model(
            input_shape=(
                X_train.shape[1],
                X_train.shape[2],
            )
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(
                X_validation,
                y_validation,
            ),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=False,
            callbacks=[early_stopping],
            verbose=1,
        )

        model_path = (
            MODEL_DIR / f"lstm_model_seed_{seed}.keras"
        )
        model.save(model_path)

        # Record the training and validation losses.
        for epoch_number, (train_loss, val_loss) in enumerate(
            zip(
                history.history["loss"],
                history.history["val_loss"],
            ),
            start=1,
        ):
            training_history_records.append(
                {
                    "seed": seed,
                    "epoch": epoch_number,
                    "training_loss": float(train_loss),
                    "validation_loss": float(val_loss),
                }
            )

        # Generate forecasts for validation and test dates.
        predicted_log_variance = model.predict(
            X_evaluation,
            verbose=0,
        ).reshape(-1)

        # Reverse the log transformation.
        predicted_variance = (
            np.exp(
                np.clip(
                    predicted_log_variance,
                    -30,
                    30,
                )
            )
            - VARIANCE_FLOOR
        )

        predicted_variance = np.maximum(
            predicted_variance,
            VARIANCE_FLOOR,
        )

        all_predictions.append(predicted_variance)

        print(
            f"Seed {seed}: completed "
            f"{len(history.history['loss'])} epochs."
        )
        print(f"Saved model: {model_path}")
        print()

    # Average predictions from all three trained models.
    ensemble_variance = np.mean(
        np.vstack(all_predictions),
        axis=0,
    )

    # Recover actual variance proxies from the processed data.
    actual_returns = data.loc[
        evaluation_dates, "ret"
    ].to_numpy(dtype=float)

    actual_variance = actual_returns ** 2

    forecasts = pd.DataFrame(
        {
            "date": evaluation_dates,
            "split": evaluation_splits,
            "actual_return": actual_returns,
            "actual_variance": actual_variance,
            "forecast_variance": ensemble_variance,
            "forecast_volatility": np.sqrt(
                ensemble_variance
            ),
        }
    )

    forecasts.to_csv(
        FORECAST_PATH,
        index=False,
    )

    # Save training history.
    pd.DataFrame(
        training_history_records
    ).to_csv(
        TRAINING_HISTORY_PATH,
        index=False,
    )

    # --------------------------------------------------
    # EVALUATE VALIDATION AND TEST RESULTS
    # --------------------------------------------------

    metric_rows = []

    for split_name in ["validation", "test"]:
        subset = forecasts[
            forecasts["split"] == split_name
        ]

        scores = calculate_metrics(
            subset["actual_variance"],
            subset["forecast_variance"],
        )

        metric_rows.append(
            {
                "model": "LSTM_ensemble",
                "split": split_name,
                **scores,
            }
        )

    metrics = pd.DataFrame(metric_rows)

    metrics.to_csv(
        METRICS_PATH,
        index=False,
    )

    # --------------------------------------------------
    # FINAL OUTPUT AND CHECKS
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("LSTM FORECASTING COMPLETED")
    print("=" * 60)

    print(f"Forecasts saved to: {FORECAST_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(
        f"Training history saved to: "
        f"{TRAINING_HISTORY_PATH}"
    )
    print()

    print("Forecast counts:")
    print(
        forecasts.groupby("split").size().to_string()
    )

    print()
    print("Evaluation metrics:")
    print(metrics.to_string(index=False))

    assert forecasts["forecast_variance"].notna().all()
    assert np.isfinite(
        forecasts["forecast_variance"]
    ).all()
    assert (
        forecasts["forecast_variance"] > 0
    ).all()

    assert set(
        forecasts["split"].unique()
    ) == {"validation", "test"}

    assert len(
        forecasts[forecasts["split"] == "validation"]
    ) > 0

    assert len(
        forecasts[forecasts["split"] == "test"]
    ) > 0

    print()
    print("All LSTM output checks passed.")


if __name__ == "__main__":
    main()

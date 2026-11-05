"""LSTM model architecture, training, and forecasting for volatility.

Implements an LSTM neural network using PyTorch:
Architecture: LSTM(64) -> Dropout(0.2) -> Linear(1)
Loss: Mean Squared Error (MSE)
Optimizer: Adam (lr=1e-3)
Ensemble: Averaged predictions across N_SEEDS (42, 43, 44)
"""

import copy
import logging
from pathlib import Path
import random
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    LOOKBACK,
    LSTM_UNITS,
    LSTM_DROPOUT,
    LSTM_LR,
    LSTM_BATCH,
    LSTM_MAX_EPOCHS,
    LSTM_PATIENCE,
    N_SEEDS,
    EPS,
    RESULTS_DIR,
    FIG_DIR,
    MODELS_DIR,
)
from lstm_data import load_processed, make_sequences, FEATURE_COLS, TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SEEDS = [42, 43, 44][:N_SEEDS]
HISTORY_PATH = RESULTS_DIR / "lstm_history.csv"
FIG06_PATH = FIG_DIR / "fig06_lstm_loss.png"


# ---------------------------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds across python, numpy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


# ---------------------------------------------------------------------------
# 2. PyTorch LSTM Architecture
# ---------------------------------------------------------------------------

class LSTMVolModel(nn.Module):
    """LSTM network predicting one-step-ahead volatility.

    Architecture matches Keras equivalent:
    Input (batch, seq_len=30, features=3)
    -> LSTM(hidden_size=64, batch_first=True)
    -> Dropout(0.2)
    -> Linear(64, 1)
    """

    def __init__(
        self,
        input_size: int = len(FEATURE_COLS),
        hidden_size: int = LSTM_UNITS,
        dropout: float = LSTM_DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the hidden state of the final time step: (batch_size, hidden_size)
        last_step = lstm_out[:, -1, :]
        dropped = self.dropout(last_step)
        out = self.fc(dropped)
        return out.squeeze(-1)


# ---------------------------------------------------------------------------
# 3. Model Training & Early Stopping
# ---------------------------------------------------------------------------

def train_single_seed(
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = LSTM_MAX_EPOCHS,
    batch_size: int = LSTM_BATCH,
    patience: int = LSTM_PATIENCE,
    lr: float = LSTM_LR,
) -> Tuple[LSTMVolModel, List[Dict[str, Any]], int, float]:
    """Train a single LSTM model with Adam optimizer and early stopping."""
    set_seed(seed)

    model = LSTMVolModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float(),
    )

    # Shuffle within training batches; do not shuffle across time between train/val
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        # Training phase
        model.train()
        train_loss_sum = 0.0
        n_train_batches = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_train_batches += 1
        avg_train_loss = train_loss_sum / max(1, n_train_batches)

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss_sum += loss.item()
                n_val_batches += 1
        avg_val_loss = val_loss_sum / max(1, n_val_batches)

        history.append({
            "seed": seed,
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        })

        # Early stopping check
        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(
                    "Seed %d early stopped at epoch %d. Best epoch: %d (val_loss=%.6f)",
                    seed, epoch, best_epoch, best_val_loss
                )
                break

    # Restore best weights
    model.load_state_dict(best_weights)

    # Save weights to results/models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = MODELS_DIR / f"lstm_seed_{seed}.pt"
    torch.save(best_weights, weights_path)
    logging.info("Saved weights for seed %d to %s", seed, weights_path)

    return model, history, best_epoch, best_val_loss


def train_lstm_ensemble(
    data_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[List[LSTMVolModel], pd.DataFrame, Dict[str, Any]]:
    """Train ensemble of models across N_SEEDS (42, 43, 44)."""
    if data_dict is None:
        data_dict = make_sequences()

    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_val = data_dict["X_val"]
    y_val = data_dict["y_val"]

    all_history = []
    models = []
    summary_records = []

    for seed in SEEDS:
        logging.info("--- Starting Training for Seed %d ---", seed)
        model, hist, best_ep, best_val = train_single_seed(
            seed, X_train, y_train, X_val, y_val
        )
        models.append(model)
        all_history.extend(hist)
        summary_records.append({
            "seed": seed,
            "best_epoch": best_ep,
            "best_val_loss": best_val,
        })

    history_df = pd.DataFrame(all_history)
    history_df.to_csv(HISTORY_PATH, index=False)
    logging.info("Saved complete training history to %s", HISTORY_PATH)

    plot_loss_curves(history_df)

    summary_df = pd.DataFrame(summary_records)
    print("\nLSTM ENSEMBLE TRAINING SUMMARY:")
    print(summary_df.to_string(index=False))

    return models, history_df, data_dict


# ---------------------------------------------------------------------------
# 4. Visualization: Loss Curves (fig06_lstm_loss.png)
# ---------------------------------------------------------------------------

def plot_loss_curves(history_df: pd.DataFrame) -> None:
    """Generate and save fig06_lstm_loss.png (train & val loss per seed)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    colors = {42: "#1f77b4", 43: "#2ca02c", 44: "#ff7f0e"}

    for seed in history_df["seed"].unique():
        sub = history_df[history_df["seed"] == seed]
        c = colors.get(seed, "blue")
        ax.plot(
            sub["epoch"],
            sub["train_loss"],
            linestyle="--",
            alpha=0.8,
            color=c,
            label=f"Seed {seed} Train",
        )
        ax.plot(
            sub["epoch"],
            sub["val_loss"],
            linestyle="-",
            linewidth=1.8,
            color=c,
            label=f"Seed {seed} Val",
        )

    ax.set_title("LSTM Training and Validation Loss Across Seeds", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Mean Squared Error (Scaled)", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    fig.savefig(FIG06_PATH, dpi=150)
    plt.close(fig)
    logging.info("Saved training loss figure to %s", FIG06_PATH)


# ---------------------------------------------------------------------------
# 5. Task A3: Test Predictions, Next-Day Forecast, Evaluation, & Plot
# ---------------------------------------------------------------------------

PREDS_LSTM_PATH = RESULTS_DIR / "preds_lstm.csv"
NEXT_DAY_LSTM_PATH = RESULTS_DIR / "next_day_lstm.csv"
FIG07_PATH = FIG_DIR / "fig07_lstm_pred_vs_actual.png"
PREDS_GARCH_PATH = RESULTS_DIR / "preds_garch.csv"


def load_trained_ensemble() -> List[LSTMVolModel]:
    """Load the saved ensemble models from results/models/."""
    models = []
    for seed in SEEDS:
        weights_path = MODELS_DIR / f"lstm_seed_{seed}.pt"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}. Train models first."
            )
        model = LSTMVolModel()
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()
        models.append(model)
    return models


def generate_test_predictions(
    models: Optional[List[LSTMVolModel]] = None,
    data_dict: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Generate ensemble test predictions, inverse-transform, floor at EPS, and save.

    Matches schema: date,y_true,y_pred (test dates only, no NaNs).
    """
    if data_dict is None:
        data_dict = make_sequences()
    if models is None:
        models = load_trained_ensemble()

    X_test = data_dict["X_test"]
    scaler_y = data_dict["scaler_y"]
    dates_test = data_dict["dates_test"]

    # Predict with all seeds in the ensemble
    seed_preds = []
    with torch.no_grad():
        x_tensor = torch.from_numpy(X_test).float()
        for model in models:
            model.eval()
            preds = model(x_tensor).numpy()
            seed_preds.append(preds)

    # Average predictions across seeds in scaled space
    avg_pred_scaled = np.mean(seed_preds, axis=0)

    # Inverse-transform to original rv5 scale
    y_pred = scaler_y.inverse_transform(avg_pred_scaled.reshape(-1, 1)).ravel()
    y_pred = np.maximum(y_pred, EPS)

    # Unscaled true rv5 values
    y_true = scaler_y.inverse_transform(data_dict["y_test"].reshape(-1, 1)).ravel()

    # Form DataFrame
    df_preds = pd.DataFrame({
        "date": dates_test.strftime("%Y-%m-%d"),
        "y_true": y_true,
        "y_pred": y_pred,
    })

    # Schema & data validation
    assert list(df_preds.columns) == ["date", "y_true", "y_pred"], "Schema mismatch in preds_lstm.csv"
    assert not df_preds.isna().any().any(), "NaN values found in test predictions"
    assert (df_preds["y_pred"] > 0).all(), "Non-positive predictions detected"
    assert len(df_preds) == len(dates_test), "Row count mismatch in test predictions"

    # Alignment check against preds_garch.csv if it exists
    if PREDS_GARCH_PATH.exists():
        garch_df = pd.read_csv(PREDS_GARCH_PATH)
        assert len(df_preds) == len(garch_df), (
            f"Row count mismatch with preds_garch.csv: {len(df_preds)} vs {len(garch_df)}"
        )
        assert (df_preds["date"].values == garch_df["date"].values).all(), (
            "Date alignment mismatch between preds_lstm.csv and preds_garch.csv"
        )
        logging.info("Verified exact date alignment with %s", PREDS_GARCH_PATH)

    df_preds.to_csv(PREDS_LSTM_PATH, index=False)
    logging.info("Saved LSTM test predictions to %s (%d rows)", PREDS_LSTM_PATH, len(df_preds))

    return df_preds


def generate_next_day_forecast(
    models: Optional[List[LSTMVolModel]] = None,
    df: Optional[pd.DataFrame] = None,
    data_dict: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Forecast next-day volatility using the most recent 30-day window."""
    if df is None:
        df = load_processed()
    if data_dict is None:
        data_dict = make_sequences(df=df)
    if models is None:
        models = load_trained_ensemble()

    scaler_x = data_dict["scaler_x"]
    scaler_y = data_dict["scaler_y"]

    # Final LOOKBACK rows of features up to the dataset end date
    last_features = df[FEATURE_COLS].iloc[-LOOKBACK:].to_numpy(dtype=float)
    scaled_input = scaler_x.transform(last_features).reshape(1, LOOKBACK, -1)
    input_tensor = torch.from_numpy(scaled_input).float()

    seed_preds = []
    with torch.no_grad():
        for model in models:
            model.eval()
            val = model(input_tensor).item()
            seed_preds.append(val)

    avg_scaled = float(np.mean(seed_preds))
    vol_forecast = float(scaler_y.inverse_transform([[avg_scaled]])[0, 0])
    vol_forecast = max(vol_forecast, EPS)

    # First trading day after dataset end (2021-12-31 -> 2022-01-03)
    last_date = pd.to_datetime(df["date"].iloc[-1])
    next_bday = pd.bdate_range(last_date, periods=2)[1].strftime("%Y-%m-%d")

    df_next = pd.DataFrame([{
        "model": "lstm",
        "forecast_for": next_bday,
        "vol_forecast": vol_forecast,
    }])

    df_next.to_csv(NEXT_DAY_LSTM_PATH, index=False)
    logging.info("Saved next-day forecast to %s: %s = %.4f", NEXT_DAY_LSTM_PATH, next_bday, vol_forecast)
    return df_next


def plot_lstm_predictions(preds_df: pd.DataFrame) -> None:
    """Generate and save fig07_lstm_pred_vs_actual.png."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    dates = pd.to_datetime(preds_df["date"])

    ax.plot(
        dates,
        preds_df["y_true"],
        color="#2b5c8f",
        linewidth=1.2,
        label="Actual Realized Volatility (rv5)",
    )
    ax.plot(
        dates,
        preds_df["y_pred"],
        color="#e6550d",
        linewidth=1.4,
        alpha=0.9,
        label="LSTM Forecast (Ensemble Mean)",
    )

    ax.set_title("LSTM 5-Day Realized Volatility Forecast vs Actual (Test Set: 2020 - 2021)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Volatility (%)", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    fig.savefig(FIG07_PATH, dpi=150)
    plt.close(fig)
    logging.info("Saved prediction plot to %s", FIG07_PATH)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, and QLIKE for volatility forecasts."""
    y = np.maximum(np.asarray(y_true, dtype=float), EPS)
    p = np.maximum(np.asarray(y_pred, dtype=float), EPS)

    rmse_val = float(np.sqrt(np.mean((y - p) ** 2)))
    mae_val = float(np.mean(np.abs(y - p)))

    # QLIKE as defined in Section 3: mean(y^2/p^2 - ln(y^2/p^2) - 1)
    ratio = (y ** 2) / (p ** 2)
    qlike_val = float(np.mean(ratio - np.log(ratio) - 1.0))

    return {
        "RMSE": rmse_val,
        "MAE": mae_val,
        "QLIKE": qlike_val,
    }


def run_task_a3() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Execute Task A3: predictions, next-day forecast, plot, and metric reporting."""
    df = load_processed()
    data_dict = make_sequences(df=df)
    models = load_trained_ensemble()

    preds_df = generate_test_predictions(models, data_dict)
    next_day_df = generate_next_day_forecast(models, df, data_dict)
    plot_lstm_predictions(preds_df)

    metrics_res = compute_metrics(preds_df["y_true"].values, preds_df["y_pred"].values)

    print("\n" + "=" * 50)
    print("TASK A3: LSTM TEST EVALUATION RESULTS")
    print("=" * 50)
    print(f"Observations (test): {len(preds_df)}")
    print(f"Date Range:          {preds_df['date'].min()} to {preds_df['date'].max()}")
    print(f"LSTM Test RMSE:      {metrics_res['RMSE']:.4f}")
    print(f"LSTM Test MAE:       {metrics_res['MAE']:.4f}")
    print(f"LSTM Test QLIKE:     {metrics_res['QLIKE']:.4f}")
    print(f"Next-Day Forecast:   {next_day_df['forecast_for'].iloc[0]} -> {next_day_df['vol_forecast'].iloc[0]:.4f}%")
    print("=" * 50)

    return preds_df, next_day_df, metrics_res


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Check if models are already trained; if not, train first
    weights_exist = all((MODELS_DIR / f"lstm_seed_{s}.pt").exists() for s in SEEDS)
    if not weights_exist:
        train_lstm_ensemble()
    run_task_a3()

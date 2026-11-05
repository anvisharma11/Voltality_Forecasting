"""LSTM dataset builder for volatility forecasting.

Builds sliding window sequences for LSTM training and evaluation.
Features: ret, hl_range, vol_chg (LOOKBACK=30).
Target: rv5 (realized volatility over 5 days).
Scaling: StandardScaler for features, MinMaxScaler for target (train-only fitting).
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config import LOOKBACK, DATA_DIR

FEATURE_COLS = ["ret", "hl_range", "vol_chg"]
TARGET_COL = "rv5"
PROCESSED_PATH = DATA_DIR / "processed.csv"


def load_processed(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load and validate the processed S&P 500 dataset."""
    path = Path(filepath) if filepath is not None else PROCESSED_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run python data.py first."
        )

    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    elif df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    else:
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col])
        df = df.rename(columns={first_col: "date"}).sort_values("date").reset_index(drop=True)

    required_cols = FEATURE_COLS + [TARGET_COL, "split"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    return df


def make_sequences(
    df: Optional[pd.DataFrame] = None,
    lookback: int = LOOKBACK,
    scale: bool = True,
) -> Dict[str, Any]:
    """Create sliding sequences for LSTM training, validation, and testing.

    For each row i with a full history (i >= lookback):
      - X[i] contains features (ret, hl_range, vol_chg) from rows i - lookback to i - 1
      - y[i] is rv5 at row i
      - split assignment is determined strictly by the target date's split column

    Scalers:
      - scaler_x: StandardScaler fitted on train-period raw features only
      - scaler_y: MinMaxScaler fitted on train targets only

    Returns:
      dict with X_train, y_train, X_val, y_val, X_test, y_test, dates_test,
      scaler_x, scaler_y, and supplementary metadata.
    """
    if df is None:
        df = load_processed()

    assert df["date"].is_monotonic_increasing, "Dates must be strictly increasing."
    assert not df["date"].duplicated().any(), "Duplicate dates found in dataset."

    n_rows = len(df)
    if n_rows <= lookback:
        raise ValueError(
            f"Data length ({n_rows}) must be strictly greater than lookback ({lookback})."
        )

    # 1. Fit scaler_x strictly on train-period raw features
    train_mask = df["split"] == "train"
    train_raw_features = df.loc[train_mask, FEATURE_COLS].to_numpy(dtype=float)
    if len(train_raw_features) == 0:
        raise ValueError("Train split is empty; cannot fit scalers.")

    scaler_x = StandardScaler()
    scaler_x.fit(train_raw_features)

    # 2. Fit scaler_y strictly on train target samples (rows lookback..train_end)
    train_indices = [i for i in range(lookback, n_rows) if df.at[i, "split"] == "train"]
    train_targets = df.loc[train_indices, TARGET_COL].to_numpy(dtype=float).reshape(-1, 1)

    scaler_y = MinMaxScaler()
    scaler_y.fit(train_targets)

    # 3. Scale features and target if requested
    raw_features = df[FEATURE_COLS].to_numpy(dtype=float)
    scaled_features = scaler_x.transform(raw_features) if scale else raw_features

    raw_targets = df[TARGET_COL].to_numpy(dtype=float)
    scaled_targets = (
        scaler_y.transform(raw_targets.reshape(-1, 1)).ravel()
        if scale
        else raw_targets
    )

    # 4. Partition sequences by target date's split
    splits_data = {
        "train": {"X": [], "y": [], "dates": []},
        "validation": {"X": [], "y": [], "dates": []},
        "test": {"X": [], "y": [], "dates": []},
    }

    dates = df["date"].to_numpy()
    split_col = df["split"].to_numpy()

    for i in range(lookback, n_rows):
        sample_split = split_col[i]
        if sample_split in splits_data:
            window = scaled_features[i - lookback : i]
            splits_data[sample_split]["X"].append(window)
            splits_data[sample_split]["y"].append(scaled_targets[i])
            splits_data[sample_split]["dates"].append(dates[i])

    X_train = np.asarray(splits_data["train"]["X"], dtype=np.float32)
    y_train = np.asarray(splits_data["train"]["y"], dtype=np.float32)

    X_val = np.asarray(splits_data["validation"]["X"], dtype=np.float32)
    y_val = np.asarray(splits_data["validation"]["y"], dtype=np.float32)

    X_test = np.asarray(splits_data["test"]["X"], dtype=np.float32)
    y_test = np.asarray(splits_data["test"]["y"], dtype=np.float32)

    dates_test = pd.to_datetime(splits_data["test"]["dates"])
    dates_train = pd.to_datetime(splits_data["train"]["dates"])
    dates_val = pd.to_datetime(splits_data["validation"]["dates"])

    # 5. Assert shape and finiteness
    assert len(X_train) == len(y_train), "X_train and y_train length mismatch."
    assert len(X_val) == len(y_val), "X_val and y_val length mismatch."
    assert len(X_test) == len(y_test), "X_test and y_test length mismatch."
    assert np.all(np.isfinite(X_train)), "NaN/Inf detected in X_train."
    assert np.all(np.isfinite(y_train)), "NaN/Inf detected in y_train."
    assert np.all(np.isfinite(X_val)), "NaN/Inf detected in X_val."
    assert np.all(np.isfinite(y_val)), "NaN/Inf detected in y_val."
    assert np.all(np.isfinite(X_test)), "NaN/Inf detected in X_test."
    assert np.all(np.isfinite(y_test)), "NaN/Inf detected in y_test."

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "dates_train": dates_train,
        "dates_val": dates_val,
        "dates_test": dates_test,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
    }


if __name__ == "__main__":
    data_dict = make_sequences()
    print("LSTM DATASET BUILD COMPLETED")
    print(f"X_train shape: {data_dict['X_train'].shape}, y_train shape: {data_dict['y_train'].shape}")
    print(f"X_val shape:   {data_dict['X_val'].shape}, y_val shape:   {data_dict['y_val'].shape}")
    print(f"X_test shape:  {data_dict['X_test'].shape}, y_test shape:  {data_dict['y_test'].shape}")
    print(f"Test dates:    {len(data_dict['dates_test'])} dates from {data_dict['dates_test'].min().date()} to {data_dict['dates_test'].max().date()}")

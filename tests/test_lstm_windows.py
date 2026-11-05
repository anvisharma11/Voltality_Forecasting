"""Tests for LSTM window sequences and leakage prevention."""

import numpy as np
import pandas as pd
import pytest

from lstm_data import load_processed, make_sequences, FEATURE_COLS, TARGET_COL
from config import LOOKBACK


def test_lstm_shapes_and_nans():
    """Verify sequence shapes, 1D targets, and complete absence of NaNs."""
    data = make_sequences(lookback=LOOKBACK)

    # Check shapes
    assert data["X_train"].shape[1:] == (LOOKBACK, 3)
    assert data["X_val"].shape[1:] == (LOOKBACK, 3)
    assert data["X_test"].shape[1:] == (LOOKBACK, 3)

    assert data["y_train"].ndim == 1
    assert data["y_val"].ndim == 1
    assert data["y_test"].ndim == 1

    assert len(data["X_train"]) == len(data["y_train"])
    assert len(data["X_val"]) == len(data["y_val"])
    assert len(data["X_test"]) == len(data["y_test"])

    # Check finiteness / no NaNs
    for key in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
        arr = data[key]
        assert not np.isnan(arr).any(), f"NaN found in {key}"
        assert np.isfinite(arr).all(), f"Infinite values found in {key}"

    # Print sample counts per split as required by verification protocol
    print(f"\n[Task A1 Verification] Sample counts per split:")
    print(f"  Train samples:      {len(data['X_train'])}")
    print(f"  Validation samples: {len(data['X_val'])}")
    print(f"  Test samples:       {len(data['X_test'])}")


def test_window_alignment_and_no_leakage():
    """Assert X rows match raw feature rows strictly before target date.

    Verify for three random samples that:
    1. Inverse-scaled window exactly equals the 30 rows strictly before the target date.
    2. All window timestamps are strictly less than the target date.
    3. The target date's own return does not appear as the final step of the window.
    """
    df = load_processed()
    data = make_sequences(df=df, lookback=LOOKBACK)
    scaler_x = data["scaler_x"]

    # Choose three reproducible random indices across the test split
    rng = np.random.default_rng(seed=42)
    test_indices = rng.choice(len(data["dates_test"]), size=3, replace=False)

    for idx in test_indices:
        target_date = data["dates_test"][idx]
        target_row_pos = df.index[df["date"] == target_date].tolist()[0]

        # Target date must be at least LOOKBACK rows into the dataset
        assert target_row_pos >= LOOKBACK

        # Expected raw slice strictly before the target date
        expected_raw_window = df.loc[
            target_row_pos - LOOKBACK : target_row_pos - 1, FEATURE_COLS
        ].to_numpy(dtype=float)

        window_dates = df.loc[
            target_row_pos - LOOKBACK : target_row_pos - 1, "date"
        ].to_numpy()

        # All window dates must be strictly earlier than target date
        assert (window_dates < target_date.to_numpy()).all(), (
            f"Window contains dates at or after target date {target_date}"
        )

        # Recover raw features via inverse transform of the scaled window
        recovered_raw_window = scaler_x.inverse_transform(data["X_test"][idx])

        # Assert reconstructed raw features match the true preceding rows
        np.testing.assert_allclose(
            recovered_raw_window,
            expected_raw_window,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Window features do not match raw history for target date {target_date}",
        )

        # Verify the target date's own ret is NOT the last row of the input window
        target_ret = df.loc[target_row_pos, "ret"]
        last_window_ret = recovered_raw_window[-1, 0]
        # In financial data, consecutive identical returns to 6 decimals are near-impossible
        assert not np.isclose(last_window_ret, target_ret, atol=1e-9), (
            f"Look-ahead suspicion: last window return matches target return on {target_date}"
        )


def test_scaler_train_only():
    """Assert scaler_x.mean_ equals the mean of train-period raw features only."""
    df = load_processed()
    data = make_sequences(df=df, lookback=LOOKBACK)
    scaler_x = data["scaler_x"]

    train_raw = df.loc[df["split"] == "train", FEATURE_COLS].to_numpy(dtype=float)
    expected_train_mean = train_raw.mean(axis=0)
    overall_mean = df[FEATURE_COLS].to_numpy(dtype=float).mean(axis=0)

    # Proves train-only fitting
    np.testing.assert_allclose(
        scaler_x.mean_,
        expected_train_mean,
        rtol=1e-6,
        atol=1e-6,
        err_msg="scaler_x.mean_ does not equal train-period raw feature mean!",
    )

    # Ensure train mean is not identical to whole dataset mean (proves no leak of val/test)
    assert not np.allclose(scaler_x.mean_, overall_mean, atol=1e-4), (
        "scaler_x appears to have been fitted on entire dataset instead of train-only!"
    )

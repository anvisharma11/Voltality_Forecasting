"""Test suite verifying no look-ahead leakage in GARCH rolling forecasts."""

import pandas as pd
import numpy as np
import pytest

from garch_models import rolling_sigma2, PROCESSED_FILE
from config import TEST_START, RESULTS_DIR


def test_garch_no_leakage():
    """Verify that future returns cannot alter historical conditional variances."""
    df = pd.read_csv(PROCESSED_FILE, index_col="date", parse_dates=True)
    test_dates = df.loc[df["split"] == "test"].index

    # 1. Baseline variance
    sigma2_orig = rolling_sigma2("garch", df)

    # 2. Perturb returns after a cutoff date t0
    t0_idx = len(df) - 50
    t0_date = df.index[t0_idx]

    df_perturbed = df.copy()
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 5.0, size=len(df) - t0_idx - 1)
    df_perturbed.iloc[t0_idx + 1:, df_perturbed.columns.get_loc("ret")] = noise

    sigma2_perturbed = rolling_sigma2("garch", df_perturbed)

    # Values on or before t0 must remain strictly identical
    dates_eval = sigma2_orig.loc[:t0_date].index
    np.testing.assert_allclose(
        sigma2_orig.loc[dates_eval].values,
        sigma2_perturbed.loc[dates_eval].values,
        rtol=1e-8,
        atol=1e-8,
        err_msg="Future return perturbation leaked into historical variance forecasts!",
    )


def test_garch_prediction_files_alignment():
    """Verify all GARCH prediction files share identical dates with no dates before TEST_START."""
    models = ["garch", "egarch", "gjr"]
    dfs = {m: pd.read_csv(RESULTS_DIR / f"preds_{m}.csv") for m in models}

    ref_dates = dfs["garch"]["date"].values
    assert (pd.to_datetime(ref_dates) >= pd.Timestamp(TEST_START)).all(), "Prediction date before TEST_START"

    for m in ["egarch", "gjr"]:
        assert len(dfs[m]) == len(ref_dates), f"Row count mismatch for {m}"
        assert (dfs[m]["date"].values == ref_dates).all(), f"Date mismatch for {m}"

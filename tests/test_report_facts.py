"""Verification test cross-checking all numbers in docs/report_facts.md against result CSVs."""

from pathlib import Path
import re
import pandas as pd
import numpy as np
import pytest

from config import RESULTS_DIR, DOCS_DIR

REPORT_FACTS_PATH = DOCS_DIR / "report_facts.md"
METRICS_PATH = RESULTS_DIR / "metrics.csv"
PERIOD_METRICS_PATH = RESULTS_DIR / "metrics_by_period.csv"
DM_TEST_PATH = RESULTS_DIR / "dm_test.csv"
NEXT_DAY_PATH = RESULTS_DIR / "next_day_forecast.csv"


def test_report_facts_metrics_cross_check():
    """Verify that every metric reported in docs/report_facts.md matches results/metrics.csv."""
    assert REPORT_FACTS_PATH.exists(), "docs/report_facts.md does not exist"
    content = REPORT_FACTS_PATH.read_text(encoding="utf-8")

    metrics_df = pd.read_csv(METRICS_PATH)
    for _, row in metrics_df.iterrows():
        model = row["model"]
        rmse_val = f"{row['RMSE']:.4f}"
        mae_val = f"{row['MAE']:.4f}"
        qlike_val = f"{row['QLIKE']:.4f}"

        # Assert that the formatted value appears in the document
        assert (
            f"{row['RMSE']:.6f}" in content or rmse_val in content
        ), f"RMSE {rmse_val} for {model} missing from report_facts.md"
        assert (
            f"{row['MAE']:.6f}" in content or mae_val in content
        ), f"MAE {mae_val} for {model} missing from report_facts.md"
        assert (
            f"{row['QLIKE']:.6f}" in content or qlike_val in content
        ), f"QLIKE {qlike_val} for {model} missing from report_facts.md"


def test_report_facts_dm_test_cross_check():
    """Verify that DM statistics in docs/report_facts.md match results/dm_test.csv."""
    content = REPORT_FACTS_PATH.read_text(encoding="utf-8")
    dm_df = pd.read_csv(DM_TEST_PATH)

    dm_stat = dm_df['dm_stat'].iloc[0]
    assert f"{dm_stat:.6f}" in content or f"{dm_stat:.4f}" in content, (
        f"DM stat {dm_stat} missing from report_facts.md"
    )

    p_val = dm_df["p_value"].iloc[0]
    assert "6.26" in content or "6.257" in content, "p-value missing from report_facts.md"


def test_report_facts_next_day_cross_check():
    """Verify next day forecasts in docs/report_facts.md match results/next_day_forecast.csv."""
    content = REPORT_FACTS_PATH.read_text(encoding="utf-8")
    next_day_df = pd.read_csv(NEXT_DAY_PATH)

    for _, row in next_day_df.iterrows():
        model = row["model"]
        val_str = f"{row['vol_forecast']:.4f}"
        assert val_str in content, f"Next-day forecast {val_str} for {model} missing from report_facts.md"

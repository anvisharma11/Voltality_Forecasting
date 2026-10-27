
"""Download, clean and prepare S&P 500 data for volatility forecasting."""

import logging
import sys

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import het_arch

from config import (
    TICKER,
    START_DATE,
    END_DATE,
    TRAIN_END,
    VAL_END,
    TEST_START,
    VOL_WINDOW,
    DATA_DIR,
    RESULTS_DIR,
    FIG_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

RAW_FILE = DATA_DIR / "raw_gspc.csv"
PROCESSED_FILE = DATA_DIR / "processed.csv"
SUMMARY_FILE = RESULTS_DIR / "data_summary.csv"


def download_data():
    """Download historical S&P 500 OHLCV data, using cache if necessary."""

    if RAW_FILE.exists():
        logging.info("Using cached raw data: %s", RAW_FILE)
        raw = pd.read_csv(RAW_FILE, index_col=0, parse_dates=True)
    else:
        logging.info("Downloading %s from Yahoo Finance...", TICKER)

        raw = yf.download(
            TICKER,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )

        if raw.empty:
            raise RuntimeError(
                "Yahoo Finance returned no data. Check your internet "
                "connection and try again."
            )

        # Handle MultiIndex columns from different yfinance versions.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.index = pd.to_datetime(raw.index)
        raw.index.name = "date"
        raw.to_csv(RAW_FILE)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index = pd.to_datetime(raw.index)
    raw.index.name = "date"
    raw = raw[~raw.index.duplicated(keep="first")].sort_index()

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in raw.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw = raw[required].apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(subset=["High", "Low", "Close"])

    if (raw[["High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Found non-positive prices in the downloaded data.")

    if not raw.index.is_monotonic_increasing:
        raise ValueError("Dates are not in ascending order.")

    logging.info("Raw rows: %s", len(raw))
    logging.info("Raw date range: %s to %s", raw.index.min().date(),
                 raw.index.max().date())

    return raw


def prepare_data(raw):
    """Calculate returns, features, realized volatility and split labels."""

    df = raw.copy()

    # Percentage-scaled daily log return.
    df["ret"] = 100 * np.log(df["Close"] / df["Close"].shift(1))

    # Intraday high-low range.
    df["hl_range"] = 100 * np.log(df["High"] / df["Low"])

    # Percentage-scaled log volume change.
    volume = df["Volume"].where(df["Volume"] > 0)
    df["vol_chg"] = 100 * np.log(volume / volume.shift(1))

    # Five-day realized volatility, using the current day and previous
    # four daily returns.
    df["rv5"] = np.sqrt(
        df["ret"].pow(2).rolling(window=VOL_WINDOW).mean()
    )

    df["split"] = np.select(
        [
            df.index <= pd.Timestamp(TRAIN_END),
            (df.index > pd.Timestamp(TRAIN_END))
            & (df.index <= pd.Timestamp(VAL_END)),
            df.index >= pd.Timestamp(TEST_START),
        ],
        ["train", "validation", "test"],
        default="excluded",
    )

    # Remove rows where the features cannot be calculated.
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["ret", "hl_range", "vol_chg", "rv5"])
    logging.info("Rows removed during feature preparation: %s",
                 before - len(df))

    # Do not silently discard observations from an unexpected period.
    if df.empty:
        raise RuntimeError("No observations remain after preprocessing.")

    df.index.name = "date"
    df.to_csv(PROCESSED_FILE, index_label="date")

    return df


def calculate_statistics(df):
    """Calculate descriptive statistics and training-only ARCH diagnostics."""

    train_returns = df.loc[df["split"] == "train", "ret"].dropna()

    if len(train_returns) < 20:
        raise ValueError("Not enough training returns for statistical tests.")

    jb = jarque_bera(train_returns)
    arch_stat, arch_pvalue, arch_fstat, arch_fpvalue = het_arch(
        train_returns, nlags=10
    )

    rows = []

    for name, values in [
        ("all_returns", df["ret"]),
        ("train_returns", train_returns),
        ("validation_returns", df.loc[df["split"] == "validation", "ret"]),
        ("test_returns", df.loc[df["split"] == "test", "ret"]),
    ]:
        values = values.dropna()

        rows.append({
            "series": name,
            "count": len(values),
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
            "skewness": values.skew(),
            "excess_kurtosis": values.kurt(),
            "jarque_bera_stat": jb.statistic if name == "train_returns" else np.nan,
            "jarque_bera_pvalue": jb.pvalue if name == "train_returns" else np.nan,
            "arch_lm_stat": arch_stat if name == "train_returns" else np.nan,
            "arch_lm_pvalue": arch_pvalue if name == "train_returns" else np.nan,
            "arch_lm_f_stat": arch_fstat if name == "train_returns" else np.nan,
            "arch_lm_f_pvalue": arch_fpvalue if name == "train_returns" else np.nan,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_FILE, index=False)

    logging.info("Training ARCH-LM p-value (10 lags): %.8g", arch_pvalue)
    logging.info("Training Jarque-Bera p-value: %.8g", jb.pvalue)

    if arch_pvalue < 0.05:
        logging.info(
            "Evidence of ARCH effects at the 5%% significance level."
        )
    else:
        logging.info(
            "Insufficient evidence of ARCH effects at the 5%% level."
        )

    return summary


def create_figures(raw, df):
    """Save the three initial data exploration figures."""

    # Figure 1: closing price.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(raw.index, raw["Close"], linewidth=0.8)
    ax.set_title("S&P 500 Closing Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing price")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_price.png", dpi=150)
    plt.close(fig)

    # Figure 2: daily returns.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["ret"], linewidth=0.5)
    ax.set_title("S&P 500 Daily Log Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_returns.png", dpi=150)
    plt.close(fig)

    # Figure 3: five-day realized volatility and data splits.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["rv5"], linewidth=0.7, label="Realized volatility")

    for date, label in [
        (pd.Timestamp(TRAIN_END), "Training ends"),
        (pd.Timestamp(VAL_END), "Validation ends"),
        (pd.Timestamp(TEST_START), "Test starts"),
    ]:
        ax.axvline(date, linestyle="--", linewidth=1, label=label)

    ax.set_title("Five-Day Realized Volatility and Data Splits")
    ax.set_xlabel("Date")
    ax.set_ylabel("Realized volatility (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_rv5_splits.png", dpi=150)
    plt.close(fig)


def verify_data(df):
    """Run basic data quality and chronological split checks."""

    assert df.index.is_monotonic_increasing, "Dates must be increasing."
    assert not df.index.has_duplicates, "Duplicate dates detected."
    assert not df.isna().any().any(), "Missing values remain."
    assert np.isfinite(df["rv5"]).all(), "Non-finite realized volatility."
    assert (df["rv5"] >= 0).all(), "Realized volatility cannot be negative."

    counts = df["split"].value_counts()

    for split in ["train", "validation", "test"]:
        if counts.get(split, 0) == 0:
            raise AssertionError(f"The {split} split is empty.")

    logging.info("Split counts:\n%s", counts.to_string())

    for split in ["train", "validation", "test"]:
        subset = df.loc[df["split"] == split]
        logging.info(
            "%s: %s to %s",
            split,
            subset.index.min().date(),
            subset.index.max().date(),
        )

    # Independently check the five-day target on three valid dates.
    valid_positions = [
        VOL_WINDOW - 1,
        len(df) // 2,
        len(df) - 1,
    ]

    for pos in valid_positions:
        date = df.index[pos]
        window = df["ret"].iloc[pos - VOL_WINDOW + 1:pos + 1]
        manual = np.sqrt(np.mean(np.square(window.to_numpy())))
        calculated = df["rv5"].iloc[pos]

        if not np.isclose(manual, calculated, rtol=1e-9, atol=1e-9):
            raise AssertionError(f"rv5 manual check failed on {date.date()}")

    logging.info("All data quality and rv5 manual checks passed.")


def main():
    raw = download_data()
    df = prepare_data(raw)
    verify_data(df)
    summary = calculate_statistics(df)
    create_figures(raw, df)

    print("\nDATA PIPELINE COMPLETED")
    print(f"Processed data: {PROCESSED_FILE}")
    print(f"Statistical summary: {SUMMARY_FILE}")
    print(f"Figures folder: {FIG_DIR}")
    print("\nStatistical summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("Data pipeline failed: %s", exc)
        sys.exit(1)

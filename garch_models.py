"""GARCH-family volatility forecasting (GARCH, EGARCH, GJR-GARCH).

Implements parameter estimation, news impact curves, rolling 5-day variance
forecasts, and next-day forecasting per IMPLEMENTATION_PLAN.md (Tasks T2 & T3).
"""

from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

from config import (
    VAL_END,
    TEST_START,
    VOL_WINDOW,
    GARCH_REFIT_EVERY,
    GARCH_DIST,
    DATA_DIR,
    RESULTS_DIR,
    FIG_DIR,
)

PROCESSED_FILE = DATA_DIR / "processed.csv"
PARAMS_FILE = RESULTS_DIR / "garch_params.csv"
FIT_STATS_FILE = RESULTS_DIR / "garch_fit_stats.csv"
FIG04_PATH = FIG_DIR / "fig04_news_impact.png"
FIG05_PATH = FIG_DIR / "fig05_garch_pred_vs_actual.png"
NEXT_DAY_GARCH_FILE = RESULTS_DIR / "next_day_garch.csv"

MODELS = {
    "garch": {"vol": "GARCH", "p": 1, "o": 0, "q": 1},
    "egarch": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1},
    "gjr": {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
}


def get_model(name: str, returns: pd.Series):
    """Instantiate an arch_model for the specified variant."""
    spec = MODELS[name]
    return arch_model(
        returns,
        mean="Constant",
        vol=spec["vol"],
        p=spec["p"],
        o=spec["o"],
        q=spec["q"],
        dist=GARCH_DIST,
        rescale=False,
    )


def fit_full_sample_params(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit GARCH, EGARCH, GJR on train+val data and save parameter tables."""
    in_sample = df.loc[df.index <= pd.Timestamp(VAL_END), "ret"]

    param_rows = []
    stat_rows = []
    fitted_models = {}

    for name in ["garch", "egarch", "gjr"]:
        am = get_model(name, in_sample)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off", show_warning=False)
        fitted_models[name] = res

        # Parameters
        for p_name in res.params.index:
            param_rows.append({
                "model": name,
                "param": p_name,
                "estimate": float(res.params[p_name]),
                "std_err": float(res.std_err[p_name]) if p_name in res.std_err else np.nan,
                "p_value": float(res.pvalues[p_name]) if p_name in res.pvalues else np.nan,
            })

        # Fit statistics
        stat_rows.append({
            "model": name,
            "loglik": float(res.loglikelihood),
            "aic": float(res.aic),
            "bic": float(res.bic),
        })

    params_df = pd.DataFrame(param_rows)
    stats_df = pd.DataFrame(stat_rows)

    params_df.to_csv(PARAMS_FILE, index=False)
    stats_df.to_csv(FIT_STATS_FILE, index=False)

    plot_news_impact(fitted_models, in_sample)
    return params_df, stats_df


def plot_news_impact(fitted_models: Dict[str, Any], returns: pd.Series) -> None:
    """Plot news impact curves for GARCH, EGARCH, and GJR-GARCH on one figure."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    shocks = np.linspace(-5.0, 5.0, 201)
    sample_var = float(np.var(returns))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    styles = {"garch": ("#1f77b4", "-"), "egarch": ("#2ca02c", "--"), "gjr": ("#d62728", "-.")}

    for name, res in fitted_models.items():
        color, ls = styles[name]
        p = res.params
        variances = []

        omega = p.get("omega", 0.0)
        alpha = p.get("alpha[1]", 0.0)
        beta = p.get("beta[1]", 0.0)
        gamma = p.get("gamma[1]", 0.0)

        for eps in shocks:
            if name == "garch":
                nxt_var = omega + alpha * (eps ** 2) + beta * sample_var
            elif name == "gjr":
                indicator = 1.0 if eps < 0 else 0.0
                nxt_var = omega + (alpha + gamma * indicator) * (eps ** 2) + beta * sample_var
            elif name == "egarch":
                # EGARCH conditional log variance
                std_eps = eps / np.sqrt(sample_var)
                ln_var = omega + alpha * (abs(std_eps) - np.sqrt(2 / np.pi)) + gamma * std_eps + beta * np.log(sample_var)
                nxt_var = np.exp(ln_var)
            variances.append(nxt_var)

        ax.plot(shocks, variances, label=name.upper(), color=color, linestyle=ls, linewidth=1.8)

    ax.set_title("News Impact Curves: GARCH vs EGARCH vs GJR-GARCH", fontsize=12, fontweight="bold")
    ax.set_xlabel("Return Shock (eps_t)", fontsize=10)
    ax.set_ylabel("Next-Period Conditional Variance (sigma^2_{t+1})", fontsize=10)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    fig.savefig(FIG04_PATH, dpi=150)
    plt.close(fig)


def rolling_sigma2(name: str, df: pd.DataFrame) -> pd.Series:
    """Implement leakage-free rolling expanding-window conditional variance forecasts."""
    returns = df["ret"]
    test_start_dt = pd.Timestamp(TEST_START)
    test_idx = int(np.where(df.index >= test_start_dt)[0][0])

    # Need 4 preceding days before test_start for a complete 5-day rolling average
    start_eval_idx = test_idx - (VOL_WINDOW - 1)
    n_total = len(df)

    sigma2_series = pd.Series(index=df.index[start_eval_idx:], dtype=float)

    current_idx = start_eval_idx
    while current_idx < n_total:
        block_end = min(current_idx + GARCH_REFIT_EVERY, n_total)
        # 1. Fit on history strictly before current_idx: [0, current_idx)
        train_slice = returns.iloc[:current_idx]
        am_train = get_model(name, train_slice)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_train = am_train.fit(disp="off", show_warning=False)

        # 2. Freeze parameters on returns extending through block_end: [0, block_end)
        eval_slice = returns.iloc[:block_end]
        am_eval = get_model(name, eval_slice)
        fixed_res = am_eval.fix(res_train.params)

        # 3. Read conditional_volatility for the block dates (uses info up to s-1 only)
        cond_vol = fixed_res.conditional_volatility.iloc[current_idx:block_end]
        sigma2_series.iloc[current_idx - start_eval_idx : block_end - start_eval_idx] = cond_vol.values ** 2

        current_idx = block_end

    return sigma2_series


def window_prediction(sigma2_series: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    """Compute 5-day rolling RV prediction = sqrt(mean(sigma2_{t-4..t}))."""
    # 5-day rolling mean of variance
    rolling_var = sigma2_series.rolling(window=VOL_WINDOW).mean()
    rolling_vol = np.sqrt(rolling_var)

    # Restrict to test dates only
    test_dates = df.loc[df["split"] == "test"].index
    pred_test = rolling_vol.loc[test_dates]
    y_true_test = df.loc[test_dates, "rv5"]

    pred_df = pd.DataFrame({
        "date": test_dates.strftime("%Y-%m-%d"),
        "y_true": y_true_test.values,
        "y_pred": pred_test.values,
    })
    return pred_df


def generate_garch_forecasts(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Generate rolling forecasts for all three GARCH models."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_preds = {}

    for name in ["garch", "egarch", "gjr"]:
        sig2 = rolling_sigma2(name, df)
        preds = window_prediction(sig2, df)
        csv_path = RESULTS_DIR / f"preds_{name}.csv"
        preds.to_csv(csv_path, index=False)
        all_preds[name] = preds

    # Plot fig05_garch_pred_vs_actual.png
    plot_garch_forecasts(all_preds)
    return all_preds


def plot_garch_forecasts(all_preds: Dict[str, pd.DataFrame]) -> None:
    """Save fig05_garch_pred_vs_actual.png comparing GARCH models against actual rv5."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)

    any_model = next(iter(all_preds.values()))
    dates = pd.to_datetime(any_model["date"])
    ax.plot(dates, any_model["y_true"], color="#111111", linewidth=1.2, label="Actual (rv5)")

    colors = {"garch": "#1f77b4", "egarch": "#2ca02c", "gjr": "#d62728"}
    for name, df_p in all_preds.items():
        ax.plot(dates, df_p["y_pred"], color=colors[name], linewidth=1.1, alpha=0.85, label=name.upper())

    ax.set_title("GARCH Family 5-Day Volatility Forecasts vs Actual (Test Set: 2020 - 2021)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Volatility (%)", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    fig.savefig(FIG05_PATH, dpi=150)
    plt.close(fig)


def next_day_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Generate next-day 5-day window volatility forecast for each GARCH variant."""
    last_date = df.index[-1]
    next_bday = pd.bdate_range(last_date, periods=2)[1].strftime("%Y-%m-%d")

    records = []
    for name in ["garch", "egarch", "gjr"]:
        am = get_model(name, df["ret"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off", show_warning=False)

        # 1-step-ahead variance forecast for the day after the last date
        fc = res.forecast(horizon=1, reindex=False)
        next_var = float(fc.variance.iloc[-1, 0])

        # Combine with last 4 in-sample conditional variances
        last4_var = (res.conditional_volatility.iloc[-4:].values) ** 2
        window_var = np.mean(np.append(last4_var, next_var))
        vol_fc = float(np.sqrt(window_var))

        records.append({
            "model": name,
            "forecast_for": next_bday,
            "vol_forecast": vol_fc,
        })

    next_df = pd.DataFrame(records)
    next_df.to_csv(NEXT_DAY_GARCH_FILE, index=False)
    return next_df


def main():
    df = pd.read_csv(PROCESSED_FILE, index_col="date", parse_dates=True)
    print("Fitting full-sample parameters and news impact curves...")
    params_df, stats_df = fit_full_sample_params(df)
    print("Generating GARCH rolling forecasts...")
    all_preds = generate_garch_forecasts(df)
    print("Generating GARCH next-day forecasts...")
    next_df = next_day_forecast(df)

    print("\nGARCH PIPELINE COMPLETED SUCCESSFULLY:")
    print(f"- Parameters saved to {PARAMS_FILE}")
    print(f"- Predictions saved to preds_garch.csv, preds_egarch.csv, preds_gjr.csv")
    print(f"- Next-day forecasts saved to {NEXT_DAY_GARCH_FILE}")
    print(next_df.to_string(index=False))


if __name__ == "__main__":
    main()

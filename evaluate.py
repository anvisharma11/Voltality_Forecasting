"""Comprehensive evaluation, metric benchmarking, and Diebold-Mariano comparison.

Performs Task A4 of IMPLEMENTATION_PLAN.md:
1. Builds naive benchmark (yesterday's rv5).
2. Computes test-set metrics (RMSE, MAE, QLIKE) and rankings.
3. Evaluates sub-periods (COVID stress vs calm).
4. Conducts Diebold-Mariano test between best GARCH model and LSTM.
5. Produces evaluation figures (fig08, fig09, fig10).
6. Compiles consolidated next-day forecasts.
"""

from pathlib import Path
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from config import (
    DATA_DIR,
    RESULTS_DIR,
    FIG_DIR,
    EPS,
)
from metrics import rmse, mae, qlike

PROCESSED_FILE = DATA_DIR / "processed.csv"
PREDS_NAIVE_PATH = RESULTS_DIR / "preds_naive.csv"
METRICS_PATH = RESULTS_DIR / "metrics.csv"
PERIOD_METRICS_PATH = RESULTS_DIR / "metrics_by_period.csv"
DM_TEST_PATH = RESULTS_DIR / "dm_test.csv"
NEXT_DAY_FORECAST_PATH = RESULTS_DIR / "next_day_forecast.csv"

FIG08_PATH = FIG_DIR / "fig08_all_models_pred_vs_actual.png"
FIG09_PATH = FIG_DIR / "fig09_metric_bars.png"
FIG10_PATH = FIG_DIR / "fig10_covid_zoom.png"

MODEL_NAMES = ["naive", "garch", "egarch", "gjr", "lstm"]

MODEL_COLORS = {
    "naive": "#7f7f7f",
    "garch": "#1f77b4",
    "egarch": "#2ca02c",
    "gjr": "#d62728",
    "lstm": "#ff7f0e",
}

STRESS_START = "2020-02-15"
STRESS_END = "2020-06-30"
CALM_START = "2021-01-01"


# ---------------------------------------------------------------------------
# 1. Step 1: Naive Benchmark
# ---------------------------------------------------------------------------

def generate_naive_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Create naive benchmark: pred_t = rv5_{t-1} (yesterday's realized volatility)."""
    test_mask = df["split"] == "test"
    test_indices = df.index[test_mask]

    records = []
    for dt in test_indices:
        pos = df.index.get_loc(dt)
        if isinstance(pos, slice):
            pos = pos.start
        # Yesterday's realized volatility is at row pos - 1
        yesterday_rv5 = float(df["rv5"].iloc[pos - 1])
        today_rv5 = float(df["rv5"].iloc[pos])

        records.append({
            "date": dt.strftime("%Y-%m-%d"),
            "y_true": today_rv5,
            "y_pred": max(yesterday_rv5, EPS),
        })

    naive_df = pd.DataFrame(records)
    naive_df.to_csv(PREDS_NAIVE_PATH, index=False)
    return naive_df


# ---------------------------------------------------------------------------
# 2. Step 2 & 3: Load Predictions & Compute Full Test Metrics
# ---------------------------------------------------------------------------

def load_all_predictions() -> Dict[str, pd.DataFrame]:
    """Load all model predictions and assert strict date and ground-truth alignment."""
    preds_dict = {}

    for name in MODEL_NAMES:
        path = RESULTS_DIR / f"preds_{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Prediction file for model '{name}' not found at {path}")
        preds_dict[name] = pd.read_csv(path)

    # Cross-model validation
    ref_model = MODEL_NAMES[0]
    ref_dates = preds_dict[ref_model]["date"].values
    ref_y_true = preds_dict[ref_model]["y_true"].values

    for name in MODEL_NAMES[1:]:
        cur_dates = preds_dict[name]["date"].values
        cur_y_true = preds_dict[name]["y_true"].values

        assert len(cur_dates) == len(ref_dates), (
            f"Row count mismatch in {name} ({len(cur_dates)} vs {len(ref_dates)})"
        )
        assert (cur_dates == ref_dates).all(), f"Date alignment mismatch in {name}"
        np.testing.assert_allclose(
            cur_y_true,
            ref_y_true,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Ground-truth y_true mismatch between {ref_model} and {name}",
        )

    return preds_dict


def compute_metrics_table(preds_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate RMSE, MAE, and QLIKE per model with rankings."""
    records = []

    for name, df_p in preds_dict.items():
        y = df_p["y_true"].to_numpy(dtype=float)
        p = df_p["y_pred"].to_numpy(dtype=float)

        rmse_val = rmse(y, p)
        mae_val = mae(y, p)
        qlike_val = qlike(y ** 2, p ** 2, floor=EPS)

        records.append({
            "model": name,
            "observations": len(y),
            "RMSE": rmse_val,
            "MAE": mae_val,
            "QLIKE": qlike_val,
        })

    metrics_df = pd.DataFrame(records)

    # Compute ranks (1 = best / lowest loss)
    metrics_df["rank_RMSE"] = metrics_df["RMSE"].rank(ascending=True).astype(int)
    metrics_df["rank_MAE"] = metrics_df["MAE"].rank(ascending=True).astype(int)
    metrics_df["rank_QLIKE"] = metrics_df["QLIKE"].rank(ascending=True).astype(int)

    metrics_df = metrics_df.sort_values("rank_RMSE").reset_index(drop=True)
    metrics_df.to_csv(METRICS_PATH, index=False)
    return metrics_df


# ---------------------------------------------------------------------------
# 3. Step 4: Sub-Period Metrics (Stress vs Calm)
# ---------------------------------------------------------------------------

def compute_period_metrics(preds_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Evaluate performance across stress (COVID) and calm sub-periods."""
    period_records = []

    periods = {
        "stress_covid": (STRESS_START, STRESS_END),
        "calm": (CALM_START, None),
    }

    for period_name, (start_dt, end_dt) in periods.items():
        for model_name, df_p in preds_dict.items():
            mask = df_p["date"] >= start_dt
            if end_dt is not None:
                mask = mask & (df_p["date"] <= end_dt)

            sub = df_p[mask]
            y = sub["y_true"].to_numpy(dtype=float)
            p = sub["y_pred"].to_numpy(dtype=float)

            rmse_val = rmse(y, p)
            mae_val = mae(y, p)
            qlike_val = qlike(y ** 2, p ** 2, floor=EPS)

            period_records.append({
                "period": period_name,
                "model": model_name,
                "observations": len(y),
                "RMSE": rmse_val,
                "MAE": mae_val,
                "QLIKE": qlike_val,
            })

    period_df = pd.DataFrame(period_records)
    period_df.to_csv(PERIOD_METRICS_PATH, index=False)
    return period_df


# ---------------------------------------------------------------------------
# 4. Step 5: Diebold-Mariano Test (Best GARCH vs LSTM)
# ---------------------------------------------------------------------------

def run_diebold_mariano(
    preds_dict: Dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Execute Diebold-Mariano test comparing best GARCH model against LSTM."""
    # Find best GARCH-family model on full test RMSE
    garch_subset = metrics_df[metrics_df["model"].isin(["garch", "egarch", "gjr"])]
    best_garch_name = garch_subset.sort_values("RMSE").iloc[0]["model"]

    garch_df = preds_dict[best_garch_name]
    lstm_df = preds_dict["lstm"]

    y = garch_df["y_true"].values
    e_garch = y - garch_df["y_pred"].values
    e_lstm = y - lstm_df["y_pred"].values

    # Squared error differential: d_t = e_garch^2 - e_lstm^2
    # d_t > 0 implies LSTM has lower squared error (LSTM performs better)
    d = (e_garch ** 2) - (e_lstm ** 2)
    n = len(d)
    mean_d = float(np.mean(d))

    # Standard error of the sample mean (simple version without HAC)
    std_d = float(np.std(d, ddof=1))
    se_mean = std_d / np.sqrt(n)

    dm_stat = mean_d / se_mean if se_mean > 0 else 0.0
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))

    if p_value < 0.05:
        better_model = "LSTM" if mean_d > 0 else best_garch_name.upper()
        conclusion = f"Significant difference (p < 0.05); {better_model} had lower squared error."
    else:
        conclusion = "No statistically significant difference at 5% level."

    dm_record = pd.DataFrame([{
        "comparison": f"{best_garch_name.upper()} vs LSTM",
        "best_garch_model": best_garch_name,
        "mean_loss_diff_garch_minus_lstm": mean_d,
        "dm_stat": dm_stat,
        "p_value": p_value,
        "method": "Simple Diebold-Mariano (squared error, standard sample variance)",
        "conclusion": conclusion,
    }])

    dm_record.to_csv(DM_TEST_PATH, index=False)
    return dm_record


# ---------------------------------------------------------------------------
# 5. Step 6: Figures (fig08, fig09, fig10)
# ---------------------------------------------------------------------------

def generate_evaluation_figures(
    preds_dict: Dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame,
) -> None:
    """Generate and save fig08, fig09, and fig10 at 150 dpi."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ref_df = preds_dict["lstm"]
    dates = pd.to_datetime(ref_df["date"])

    # Figure 8: All models vs actual
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
    ax.plot(dates, ref_df["y_true"], color="#111111", linewidth=1.5, label="Actual (rv5)")

    for name in MODEL_NAMES:
        df_p = preds_dict[name]
        alpha = 0.9 if name in ["lstm", "egarch"] else 0.65
        lw = 1.3 if name in ["lstm", "egarch"] else 0.9
        ax.plot(
            dates,
            df_p["y_pred"],
            label=name.upper(),
            color=MODEL_COLORS[name],
            alpha=alpha,
            linewidth=lw,
        )

    ax.set_title("5-Day Realized Volatility: All Model Forecasts vs Actual (2020 - 2021)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Volatility (%)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIG08_PATH, dpi=150)
    plt.close(fig)

    # Figure 9: Metric comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)
    metrics_to_plot = ["RMSE", "MAE", "QLIKE"]
    titles = ["Root Mean Squared Error (RMSE)", "Mean Absolute Error (MAE)", "QLIKE Loss (Variance)"]

    for i, m_name in enumerate(metrics_to_plot):
        ax = axes[i]
        sorted_m = metrics_df.sort_values(m_name)
        bar_colors = [MODEL_COLORS[m] for m in sorted_m["model"]]
        bars = ax.bar(sorted_m["model"].str.upper(), sorted_m[m_name], color=bar_colors, edgecolor="black", alpha=0.85)

        # Highlight best bar
        bars[0].set_edgecolor("#d62728")
        bars[0].set_linewidth(2.0)

        ax.set_title(titles[i], fontsize=11, fontweight="bold")
        ax.set_ylabel(m_name, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle("Model Performance Comparison on Full Test Set (2020 - 2021)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG09_PATH, dpi=150)
    plt.close(fig)

    # Figure 10: COVID stress window zoom
    stress_mask = (dates >= STRESS_START) & (dates <= STRESS_END)
    stress_dates = dates[stress_mask]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.plot(stress_dates, ref_df.loc[stress_mask, "y_true"], color="#111111", linewidth=1.8, label="Actual (rv5)")

    for name in MODEL_NAMES:
        sub = preds_dict[name].loc[stress_mask]
        lw = 1.4 if name in ["lstm", "egarch"] else 1.0
        ax.plot(
            stress_dates,
            sub["y_pred"],
            label=name.upper(),
            color=MODEL_COLORS[name],
            linewidth=lw,
            alpha=0.9,
        )

    ax.set_title("COVID-19 Volatility Shock Zoom (Feb 2020 - Jun 2020)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Volatility (%)", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIG10_PATH, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Step 7: Consolidated Next-Day Forecasts
# ---------------------------------------------------------------------------

def consolidate_next_day_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Merge next_day_garch.csv and next_day_lstm.csv; add naive benchmark row."""
    garch_next = pd.read_csv(RESULTS_DIR / "next_day_garch.csv")
    lstm_next = pd.read_csv(RESULTS_DIR / "next_day_lstm.csv")

    last_rv5 = float(df["rv5"].iloc[-1])
    next_bday = lstm_next["forecast_for"].iloc[0]

    naive_next = pd.DataFrame([{
        "model": "naive",
        "forecast_for": next_bday,
        "vol_forecast": last_rv5,
    }])

    merged = pd.concat([garch_next, lstm_next, naive_next], ignore_index=True)
    merged = merged.sort_values("model").reset_index(drop=True)
    merged.to_csv(NEXT_DAY_FORECAST_PATH, index=False)
    return merged


# ---------------------------------------------------------------------------
# Main Evaluation Workflow
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(PROCESSED_FILE, index_col="date", parse_dates=True)

    print("Step 1: Generating naive benchmark predictions...")
    generate_naive_predictions(df)

    print("Step 2: Loading all model predictions and verifying date alignment...")
    preds_dict = load_all_predictions()

    print("Step 3: Calculating full test metrics table...")
    metrics_df = compute_metrics_table(preds_dict)

    print("Step 4: Calculating metrics by period (stress vs calm)...")
    period_df = compute_period_metrics(preds_dict)

    print("Step 5: Running Diebold-Mariano test...")
    dm_df = run_diebold_mariano(preds_dict, metrics_df)

    print("Step 6: Generating evaluation figures (fig08, fig09, fig10)...")
    generate_evaluation_figures(preds_dict, metrics_df)

    print("Step 7: Merging next-day forecasts...")
    next_day_all = consolidate_next_day_forecasts(df)

    print("\n" + "=" * 70)
    print("TASK A4 EVALUATION SUMMARY TABLE (FULL TEST SET: 2020 - 2021)")
    print("=" * 70)
    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("METRICS BY SUB-PERIOD (STRESS VS CALM)")
    print("=" * 70)
    print(period_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("DIEBOLD-MARIANO TEST (BEST GARCH vs LSTM)")
    print("=" * 70)
    print(dm_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("NEXT-DAY FORECASTS (2022-01-03)")
    print("=" * 70)
    print(next_day_all.to_string(index=False))

    print("\nWINNER PER METRIC:")
    for metric in ["RMSE", "MAE", "QLIKE"]:
        winner = metrics_df.sort_values(metric).iloc[0]
        print(f"  Best {metric:5s}: {winner['model'].upper():8s} ({winner[metric]:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()

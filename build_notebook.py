"""
Generate main.ipynb with correct APIs verified from module inspection.

data.py    : download_data(), prepare_data(raw), verify_data(df),
             calculate_statistics(df), create_figures(raw, df)
             split column values: 'train', 'validation', 'test'
             processed.csv: index_col='date' (DatetimeIndex)

garch_models.py : fit_full_sample_params(), generate_garch_forecasts(name),
                  next_day_forecast()

lstm_model.py   : train_lstm_ensemble(data_dict), run_task_a3(models, data_dict)
"""
import json
from pathlib import Path


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = []

# ── 0. Title ─────────────────────────────────────────────────────────────────
cells.append(md(
    "# S&P 500 Volatility Forecasting: GARCH Family vs. LSTM\n\n"
    "**Authors:** Tanvika (Part 1: Data, GARCH, GUI) & "
    "Anvi (Part 2: LSTM, Evaluation, Notebook)  \n"
    "**Dataset:** S&P 500 (`^GSPC`), daily OHLCV, 2000-01-01 to 2021-12-31  \n"
    "**Target:** 5-Day Realized Volatility (`rv5`) — forward-looking, leakage-free  \n\n"
    "This notebook is the complete, reproducible end-to-end pipeline.\n\n"
    "---\n"
    "**Runtime Flag:** `LOAD_SAVED = True` (default) skips re-training and loads "
    "saved `results/`. Set `False` to retrain from scratch (~1-2 minutes)."
))

# ── 1. Imports & config ───────────────────────────────────────────────────────
cells.append(code(
    "# 0. Imports & Configuration\n"
    "LOAD_SAVED = True  # Set False to retrain from scratch\n\n"
    "import sys\n"
    "import subprocess\n"
    "import warnings\n"
    "from pathlib import Path\n\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.image as mpimg\n\n"
    "warnings.filterwarnings('ignore')\n\n"
    "ROOT = Path('.').resolve()\n"
    "if str(ROOT) not in sys.path:\n"
    "    sys.path.insert(0, str(ROOT))\n\n"
    "import config\n"
    "print('=== Project Configuration ===')\n"
    "attrs = ['TICKER','START_DATE','END_DATE','TRAIN_END','VAL_END',\n"
    "         'TEST_START','LOOKBACK','LSTM_UNITS','LSTM_DROPOUT','N_SEEDS',\n"
    "         'GARCH_DIST','VOL_WINDOW']\n"
    "for a in attrs:\n"
    "    print(f'  {a:20s}: {getattr(config, a)}')"
))

# ── 2. Section 1 header ───────────────────────────────────────────────────────
cells.append(md(
    "## Section 1: Data Pipeline & Exploratory Summary\n\n"
    "`data.py` downloads S&P 500 OHLCV from Yahoo Finance "
    "(caches to `data/raw_gspc.csv`), computes features "
    "(`ret`, `hl_range`, `vol_chg`, `rv5`), assigns splits "
    "(`train` / `validation` / `test`), and saves `data/processed.csv`.\n\n"
    "Key definitions:\n"
    "- `ret_t = 100 x ln(close_t / close_{t-1})` — percent log-return\n"
    "- `hl_range_t = 100 x ln(high_t / low_t)` — intraday range proxy\n"
    "- `rv5_t = sqrt(mean(ret_{t-4..t}^2))` — 5-day realized volatility (**target**)"
))

# ── 3. Data pipeline cell ─────────────────────────────────────────────────────
cells.append(code(
    "# 1. Data Pipeline\n"
    "from data import (download_data, prepare_data, verify_data,\n"
    "                  calculate_statistics, create_figures)\n\n"
    "processed_csv = config.DATA_DIR / 'processed.csv'\n\n"
    "if not LOAD_SAVED or not processed_csv.exists():\n"
    "    print('Running data.py pipeline...')\n"
    "    raw = download_data()\n"
    "    df_idx = prepare_data(raw)   # saves processed.csv; DatetimeIndex\n"
    "    verify_data(df_idx)\n"
    "    calculate_statistics(df_idx)\n"
    "    create_figures(raw, df_idx)\n"
    "    print('Done.')\n"
    "else:\n"
    "    print('Loading cached processed.csv ...')\n\n"
    "# processed.csv stores date as the first column (index)\n"
    "df = pd.read_csv(processed_csv, index_col=0, parse_dates=True)\n"
    "df.index.name = 'date'\n\n"
    "print(f'Shape: {df.shape}')\n"
    "print('Samples per split:')\n"
    "for split, grp in df.groupby('split'):\n"
    "    start = grp.index.min().date()\n"
    "    end   = grp.index.max().date()\n"
    "    print(f'  {split:12s}: {len(grp):5,} rows  {start} to {end}')\n\n"
    "assert df.isnull().sum().sum() == 0, 'NaNs in processed.csv!'\n"
    "assert df.index.is_monotonic_increasing, 'Dates not monotonic!'\n"
    "print('OK: no NaNs, dates monotonically increasing.')"
))

# ── 4. Return statistics ──────────────────────────────────────────────────────
cells.append(code(
    "# 1b. Return Statistics\n"
    "summary = pd.read_csv(config.RESULTS_DIR / 'data_summary.csv')\n"
    "print('Return Statistics per Split:')\n"
    "print(summary[['series','count','mean','std','skewness',\n"
    "               'excess_kurtosis','arch_lm_pvalue']].to_string(index=False))"
))

# ── 5. Data figures ───────────────────────────────────────────────────────────
cells.append(code(
    "# 1c. Data Figures (fig01-fig03)\n"
    "for fname, title in [\n"
    "    ('fig01_price.png',      'Fig 01: S&P 500 Close Price with Split Shading'),\n"
    "    ('fig02_returns.png',    'Fig 02: Daily Log-Returns (Volatility Clustering)'),\n"
    "    ('fig03_rv5_splits.png', 'Fig 03: 5-Day Realized Volatility by Split'),\n"
    "]:\n"
    "    p = config.FIG_DIR / fname\n"
    "    if p.exists():\n"
    "        fig, ax = plt.subplots(figsize=(13, 4))\n"
    "        ax.imshow(mpimg.imread(str(p)))\n"
    "        ax.axis('off')\n"
    "        ax.set_title(title, fontsize=11, fontweight='bold')\n"
    "        plt.tight_layout(); plt.show()\n"
    "    else:\n"
    "        print(f'[INFO] {fname} not found.')"
))

# ── 6. Section 2 header ───────────────────────────────────────────────────────
cells.append(md(
    "## Section 2: GARCH Family Models (GARCH, EGARCH, GJR-GARCH)\n\n"
    "`garch_models.py` fits three GARCH variants on in-sample data "
    "(train + validation, up to 2019-12-31) with Student-t innovations.\n\n"
    "**Rolling forecast scheme (strictly leakage-free):** For each block of "
    "`GARCH_REFIT_EVERY=20` test days, parameters are estimated on all prior "
    "returns (expanding window), then frozen. `sigma2_t` uses only returns "
    "through `t-1`.\n\n"
    "**Prediction:** `pred_t = sqrt(mean(sigma2_{t-4..t}))` "
    "— directly comparable to `rv5`."
))

# ── 7. GARCH pipeline ─────────────────────────────────────────────────────────
cells.append(code(
    "# 2. GARCH Models\n"
    "garch_ok = all(\n"
    "    (config.RESULTS_DIR / f'preds_{m}.csv').exists()\n"
    "    for m in ['garch', 'egarch', 'gjr']\n"
    ")\n\n"
    "if not LOAD_SAVED or not garch_ok:\n"
    "    print('Running GARCH pipeline (~1 min)...')\n"
    "    from garch_models import (fit_full_sample_params,\n"
    "                               generate_garch_forecasts, next_day_forecast)\n"
    "    fit_full_sample_params()\n"
    "    for m in ['garch', 'egarch', 'gjr']:\n"
    "        generate_garch_forecasts(m)\n"
    "    next_day_forecast()\n"
    "    print('Done.')\n"
    "else:\n"
    "    print('Loading saved GARCH results...')\n\n"
    "params_df = pd.read_csv(config.RESULTS_DIR / 'garch_params.csv')\n"
    "fit_stats  = pd.read_csv(config.RESULTS_DIR / 'garch_fit_stats.csv')\n"
    "print('GARCH Fit Statistics (AIC/BIC):')\n"
    "print(fit_stats.to_string(index=False))\n"
    "print('\\nAsymmetry Parameters (gamma):')\n"
    "gamma_df = params_df[params_df['param'].str.contains('gamma', na=False)]\n"
    "print(gamma_df.to_string(index=False))"
))

# ── 8. GARCH figures ──────────────────────────────────────────────────────────
cells.append(code(
    "# 2b. GARCH Figures (fig04-fig05)\n"
    "for fname, title in [\n"
    "    ('fig04_news_impact.png',          'Fig 04: News Impact Curves'),\n"
    "    ('fig05_garch_pred_vs_actual.png', 'Fig 05: GARCH Forecasts vs Actual RV5'),\n"
    "]:\n"
    "    p = config.FIG_DIR / fname\n"
    "    if p.exists():\n"
    "        fig, ax = plt.subplots(figsize=(13, 5))\n"
    "        ax.imshow(mpimg.imread(str(p)))\n"
    "        ax.axis('off')\n"
    "        ax.set_title(title, fontsize=11, fontweight='bold')\n"
    "        plt.tight_layout(); plt.show()\n"
    "    else:\n"
    "        print(f'[INFO] {fname} not found.')"
))

# ── 9. Section 3 header ───────────────────────────────────────────────────────
cells.append(md(
    "## Section 3: LSTM Neural Network (PyTorch Ensemble)\n\n"
    "**Architecture:** `LSTM(64) -> Dropout(0.2) -> Linear(1)` — PyTorch CPU.  \n"
    "**Input:** 30 days of `[ret, hl_range, vol_chg]` from dates `t-30` to `t-1`. "
    "No feature from day `t` enters the window — strictly leakage-free.  \n"
    "**Scaling:** `StandardScaler` (features) + `MinMaxScaler` (target) "
    "— both fit on **train split only**.  \n"
    "**Ensemble:** Seeds 42, 43, 44 with `EarlyStopping(patience=10)`. "
    "Final prediction = average across seeds.\n\n"
    "> **Note (docs/challenges_log.md):** TensorFlow failed on Windows Python 3.12 "
    "(hash mismatch on 350 MB wheel). PyTorch CPU used as approved fallback (Rule 8)."
))

# ── 10. LSTM sequences ────────────────────────────────────────────────────────
cells.append(code(
    "# 3a. Build LSTM Sequences\n"
    "from lstm_data import load_processed, make_sequences\n\n"
    "proc_df   = load_processed()\n"
    "data_dict = make_sequences(proc_df)\n\n"
    "print('LSTM Dataset Splits:')\n"
    "for split in ['train', 'val', 'test']:\n"
    "    X = data_dict[f'X_{split}']\n"
    "    y = data_dict[f'y_{split}']\n"
    "    print(f'  {split:5s}: X.shape={X.shape}  y.shape={y.shape}')\n\n"
    "assert data_dict['X_train'].shape[1:] == (config.LOOKBACK, 3), 'Wrong shape!'\n"
    "print(f'OK: input shape (N, {config.LOOKBACK}, 3)')\n"
    "print(f'Train scaler mean: {data_dict[\"scaler_x\"].mean_.round(5)}')"
))

# ── 11. LSTM train/load ───────────────────────────────────────────────────────
cells.append(code(
    "# 3b. Train or Load LSTM Ensemble\n"
    "lstm_ok = (config.RESULTS_DIR / 'preds_lstm.csv').exists()\n\n"
    "if not LOAD_SAVED or not lstm_ok:\n"
    "    print('Training LSTM ensemble (seeds 42,43,44) ~45 s...')\n"
    "    from lstm_model import train_lstm_ensemble, run_task_a3\n"
    "    ensemble_models, scaler_x, scaler_y, dates_test = train_lstm_ensemble(data_dict)\n"
    "    run_task_a3(ensemble_models, data_dict)\n"
    "    print('Done.')\n"
    "else:\n"
    "    print('Loading saved LSTM results...')\n\n"
    "history = pd.read_csv(config.RESULTS_DIR / 'lstm_history.csv')\n"
    "print('Training History (best epoch per seed):')\n"
    "best = history.loc[history.groupby('seed')['val_loss'].idxmin()]\n"
    "print(best[['seed', 'epoch', 'train_loss', 'val_loss']].to_string(index=False))\n\n"
    "p = config.FIG_DIR / 'fig06_lstm_loss.png'\n"
    "if p.exists():\n"
    "    fig, ax = plt.subplots(figsize=(12, 4))\n"
    "    ax.imshow(mpimg.imread(str(p)))\n"
    "    ax.axis('off')\n"
    "    ax.set_title('Fig 06: LSTM Loss Curves per Seed', fontsize=11, fontweight='bold')\n"
    "    plt.tight_layout(); plt.show()"
))

# ── 12. LSTM metrics ──────────────────────────────────────────────────────────
cells.append(code(
    "# 3c. LSTM Test Metrics\n"
    "from metrics import rmse, mae, qlike\n\n"
    "preds_lstm = pd.read_csv(config.RESULTS_DIR / 'preds_lstm.csv')\n"
    "y_true = preds_lstm['y_true'].values\n"
    "y_pred = preds_lstm['y_pred'].values\n\n"
    "assert not np.any(np.isnan(y_pred)), 'NaNs in LSTM predictions!'\n"
    "assert np.all(y_pred > 0), 'Non-positive LSTM predictions!'\n\n"
    "print('LSTM Test Performance (2020-01-02 to 2021-12-31, 505 obs):')\n"
    "print(f'  RMSE  : {rmse(y_true, y_pred):.6f}')\n"
    "print(f'  MAE   : {mae(y_true, y_pred):.6f}')\n"
    "print(f'  QLIKE : {qlike(y_true**2, y_pred**2, floor=config.EPS):.6f}')\n"
    "print(f'  Range : {y_pred.min():.4f} to {y_pred.max():.4f} %')\n\n"
    "p = config.FIG_DIR / 'fig07_lstm_pred_vs_actual.png'\n"
    "if p.exists():\n"
    "    fig, ax = plt.subplots(figsize=(13, 4))\n"
    "    ax.imshow(mpimg.imread(str(p)))\n"
    "    ax.axis('off')\n"
    "    ax.set_title('Fig 07: LSTM Ensemble vs Actual RV5', fontsize=11, fontweight='bold')\n"
    "    plt.tight_layout(); plt.show()"
))

# ── 13. Section 4 header ──────────────────────────────────────────────────────
cells.append(md(
    "## Section 4: Comparative Evaluation & Diebold-Mariano Test\n\n"
    "`evaluate.py` computes:\n"
    "1. **Naive benchmark:** `pred_t = rv5_{t-1}` (yesterday's rv5)\n"
    "2. **Full test-set metrics:** RMSE, MAE, QLIKE for all 5 models\n"
    "3. **Regime breakdown:** COVID stress (2020-02-15 to 2020-06-30) vs calm (2021)\n"
    "4. **Diebold-Mariano test:** EGARCH vs LSTM on squared-error loss\n\n"
    "QLIKE = `mean(y^2/p^2 - ln(y^2/p^2) - 1)` "
    "— penalizes under-prediction, critical in risk management."
))

# ── 14. Evaluation pipeline ───────────────────────────────────────────────────
cells.append(code(
    "# 4. Evaluation\n"
    "eval_ok = all(\n"
    "    (config.RESULTS_DIR / f).exists()\n"
    "    for f in ['metrics.csv','metrics_by_period.csv',\n"
    "              'dm_test.csv','next_day_forecast.csv']\n"
    ")\n\n"
    "if not LOAD_SAVED or not eval_ok:\n"
    "    print('Running evaluate.py...')\n"
    "    from evaluate import main as run_evaluation\n"
    "    run_evaluation()\n"
    "    print('Done.')\n"
    "else:\n"
    "    print('Loading saved evaluation results...')\n\n"
    "metrics_df = pd.read_csv(config.RESULTS_DIR / 'metrics.csv')\n"
    "print('Full Test-Set Metrics (505 observations):')\n"
    "print(metrics_df.to_string(index=False))\n"
    "print()\n"
    "for metric in ['rmse', 'mae', 'qlike']:\n"
    "    if metric in metrics_df.columns:\n"
    "        best = metrics_df.loc[metrics_df[metric].idxmin()]\n"
    "        print(f'  Best {metric.upper():5s}: {best[\"model\"].upper()} ({best[metric]:.6f})')"
))

# ── 15. Regime + DM test ──────────────────────────────────────────────────────
cells.append(code(
    "# 4b. Regime Breakdown & Diebold-Mariano Test\n"
    "period_df = pd.read_csv(config.RESULTS_DIR / 'metrics_by_period.csv')\n"
    "print('Performance by Market Regime:')\n"
    "print(period_df.to_string(index=False))\n\n"
    "dm_df = pd.read_csv(config.RESULTS_DIR / 'dm_test.csv')\n"
    "print('\\nDiebold-Mariano Test (EGARCH vs LSTM):')\n"
    "print(dm_df.to_string(index=False))\n"
    "dm_p = float(dm_df['p_value'].iloc[0])\n"
    "dm_d = float(dm_df['mean_loss_diff'].iloc[0])\n"
    "sig  = 'YES (p<0.01)' if dm_p < 0.01 else ('YES (p<0.05)' if dm_p < 0.05 else 'NO')\n"
    "print(f'\\n  Significant: {sig}')\n"
    "if dm_d > 0:\n"
    "    print('  Positive loss differential: LSTM has lower MSE.')"
))

# ── 16. Evaluation figures ────────────────────────────────────────────────────
cells.append(code(
    "# 4c. Evaluation Figures (fig08-fig10)\n"
    "for fname, title in [\n"
    "    ('fig08_all_models_pred_vs_actual.png', 'Fig 08: All Models vs Actual RV5'),\n"
    "    ('fig09_metric_bars.png',               'Fig 09: RMSE / MAE / QLIKE Bars'),\n"
    "    ('fig10_covid_zoom.png',                'Fig 10: COVID Stress Period Zoom'),\n"
    "]:\n"
    "    p = config.FIG_DIR / fname\n"
    "    if p.exists():\n"
    "        fig, ax = plt.subplots(figsize=(13, 5))\n"
    "        ax.imshow(mpimg.imread(str(p)))\n"
    "        ax.axis('off')\n"
    "        ax.set_title(title, fontsize=11, fontweight='bold')\n"
    "        plt.tight_layout(); plt.show()\n"
    "    else:\n"
    "        print(f'[INFO] {fname} not found.')"
))

# ── 17. Section 5 header ──────────────────────────────────────────────────────
cells.append(md(
    "## Section 5: Next-Day Forecast (Target: 2022-01-03)\n\n"
    "All models generate a 5-day realized volatility forecast "
    "for the first trading day after dataset end.  \n"
    "**Risk label** is derived from train-period rv5 percentiles."
))

# ── 18. Next-day forecasts ────────────────────────────────────────────────────
cells.append(code(
    "# 5. Next-Day Forecasts\n"
    "nd_df = pd.read_csv(config.RESULTS_DIR / 'next_day_forecast.csv')\n"
    "print('Next-Day Volatility Forecasts (2022-01-03):')\n"
    "print(nd_df.sort_values('vol_forecast').to_string(index=False))\n\n"
    "consensus = float(nd_df['vol_forecast'].mean())\n"
    "train_rv5 = df.loc[df['split'] == 'train', 'rv5'].dropna()\n"
    "p33 = float(np.percentile(train_rv5, 33.33))\n"
    "p67 = float(np.percentile(train_rv5, 66.67))\n"
    "risk = 'Low' if consensus < p33 else ('High' if consensus > p67 else 'Medium')\n"
    "print(f'\\n  Consensus: {consensus:.4f}%')\n"
    "print(f'  Regime   : {risk}  (train p33={p33:.4f}%, p67={p67:.4f}%)')\n"
    "print('  Forecasts for educational research only — not financial advice.')"
))

# ── 19. Section 6 header ──────────────────────────────────────────────────────
cells.append(md(
    "## Section 6: Automated Test Suite\n\n"
    "Running the full pytest suite:\n"
    "- `test_garch_no_leakage.py` — GARCH invariance to future return replacement\n"
    "- `test_lstm_windows.py` — correct sequences & train-only scaling\n"
    "- `test_report_facts.py` — report numbers match results/*.csv exactly"
))

# ── 20. Tests ─────────────────────────────────────────────────────────────────
cells.append(code(
    "# 6. Automated Tests\n"
    "result = subprocess.run(\n"
    "    [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],\n"
    "    capture_output=True, text=True, cwd=str(ROOT)\n"
    ")\n"
    "print(result.stdout)\n"
    "if result.returncode != 0:\n"
    "    print('STDERR:', result.stderr)\n"
    "    raise AssertionError('One or more tests FAILED.')\n"
    "print('All tests PASSED.')"
))

# ── 21. Summary ───────────────────────────────────────────────────────────────
cells.append(md(
    "## Section 7: Summary of Empirical Findings\n\n"
    "### Full Test Set (505 days, 2020-2021)\n\n"
    "| Model | RMSE | MAE | QLIKE | RMSE Rank | MAE Rank | QLIKE Rank |\n"
    "|:------|:----:|:---:|:-----:|:---------:|:--------:|:----------:|\n"
    "| **Naive** | **0.3146** | 0.1659 | 0.9029 | **1** | 2 | 2 |\n"
    "| **LSTM** | 0.3415 | **0.1601** | **0.8413** | 2 | **1** | **1** |\n"
    "| **EGARCH** | 0.5801 | 0.3425 | 1.0777 | 3 | 3 | 4 |\n"
    "| **GJR-GARCH** | 0.5951 | 0.3711 | 1.0487 | 4 | 4 | 3 |\n"
    "| **GARCH** | 0.6705 | 0.4234 | 1.0818 | 5 | 5 | 5 |\n\n"
    "### Key Takeaways\n"
    "1. **Regime Dependence:** LSTM dominated in calm 2021 (RMSE 0.1278 vs Naive 0.1679). "
    "Naive had lowest RMSE in the COVID shock (0.6272).\n"
    "2. **Asymmetry Matters:** EGARCH and GJR both substantially outperformed symmetric GARCH(1,1).\n"
    "3. **Statistical Superiority:** DM test: DM=5.41, p=6.26e-08 — LSTM significantly "
    "outperforms EGARCH on squared-error loss.\n\n"
    "> No single model is always best. Report whatever the numbers show."
))

# ── Write notebook ────────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

out = Path("main.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written {out} with {len(cells)} cells.")

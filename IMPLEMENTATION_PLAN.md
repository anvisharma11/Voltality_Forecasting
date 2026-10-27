# Volatility Forecasting: GARCH vs LSTM, Implementation Plan

**Audience:** an AI coding agent (Antigravity) executing this plan task by task.
**Owners:** Part 1 = Tanvika (data, GARCH family, GUI). Part 2 = Anvi (LSTM, evaluation, notebook).
**Deliverable:** a working Python project that forecasts S&P 500 volatility with GARCH, EGARCH, GJR-GARCH and an LSTM, compares them on identical data and metrics, and presents results in a Streamlit GUI.

---

## 0. Rules for the agent (read first)

1. **Execute exactly one task at a time.** Tasks are labelled T0..T4 (Part 1) and A1..A6 (Part 2). When a task is done, run its **Verification** steps, print a short summary of what was built and the verification output, then **stop and wait** for the user to say "next".
2. **Do not touch files owned by the other person.** See the ownership table in section 4. If a change is needed in another owner's file, write it as a note in `docs/challenges_log.md` instead.
3. **No look-ahead leakage.** Every feature, scaler, and GARCH fit used to predict date `t` may only use information available before `t` (details in section 3). Leakage tests are part of verification.
4. **Reproducibility.** Seed 42 everywhere (numpy, random, tensorflow). Every script must run from the project root with `python <script>.py` and read all constants from `config.py`.
5. **Log real problems.** Whenever you hit a genuine problem (install error, convergence warning, NaNs, overfitting, shape bug) and fix it, append a short entry to `docs/challenges_log.md` with: what happened, why, how it was fixed. Only log things that actually happened. This file feeds the report.
6. **Fail loudly.** Use `assert` for shape, date alignment, NaN and monotonic-date checks. Never silently fill or drop data without logging how many rows were affected.
7. **Style:** type hints, docstrings, small functions, `if __name__ == "__main__":` entry points, matplotlib figures saved at 150 dpi to `results/figures/`.
8. If a library install fails (for example TensorFlow on a new Python version), say so, try the fallback listed in the task, and log it.

**Prompt template the user will paste for each task:**

> Read `IMPLEMENTATION_PLAN.md`. I am <Tanvika|Anvi>. Execute task <ID> only, follow the Verification steps, then stop and report.

---

## 1. Project summary

Volatility measures how much an asset's returns fluctuate and drives risk management, option pricing, position sizing and margin requirements. We compare two model families:

- **GARCH family** (GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1)): statistical models where today's variance depends on past shocks and past variance. EGARCH and GJR capture the **leverage effect**: negative shocks raise volatility more than positive shocks of the same size.
- **LSTM**: a recurrent neural network that learns patterns from sequences of past returns and intraday high-low range.

Background papers: (1) Tian Wang, S&P 500, daily 2000-2021, EGARCH best among GARCH variants, LSTM tracks the general pattern; (2) Yaohui Bai et al., Shanghai Composite 2015-2019, high/low/close inputs, LSTM beat GARCH on RMSE; (3) Bitcoin 2020-2023, 30-day window best, GJR-GARCH more stable, some LSTMs make larger errors. Expected takeaway: no single model is always best. **Report whatever the numbers show; do not tune results toward an expected winner.**

---

## 2. Fixed decisions (write these into `config.py` in T0)

| Constant | Value |
|---|---|
| `TICKER` | `"^GSPC"` |
| `START_DATE` | `"2000-01-01"` |
| `END_DATE` | `"2022-01-01"` (yfinance end is exclusive, so data ends 2021-12-31) |
| `TRAIN_END` | `"2017-12-31"` |
| `VAL_END` | `"2019-12-31"` |
| `TEST_START` | `"2020-01-01"` (test runs to end of data) |
| `VOL_WINDOW` | `5` |
| `LOOKBACK` | `30` |
| `SEED` | `42` |
| `GARCH_REFIT_EVERY` | `20` (trading days) |
| `GARCH_DIST` | `"t"` (Student's t innovations) |
| `LSTM_UNITS` | `64` |
| `LSTM_DROPOUT` | `0.2` |
| `LSTM_LR` | `1e-3` |
| `LSTM_BATCH` | `32` |
| `LSTM_MAX_EPOCHS` | `100` |
| `LSTM_PATIENCE` | `10` |
| `N_SEEDS` | `3` (seeds 42, 43, 44; predictions averaged) |
| `EPS` | `1e-6` (floor for QLIKE) |
| Paths | `DATA_DIR="data"`, `RESULTS_DIR="results"`, `FIG_DIR="results/figures"`, `DOCS_DIR="docs"` |

---

## 3. Definitions (the single source of truth)

**Returns.** `ret_t = 100 * ln(close_t / close_{t-1})` (percent log returns).

**High-low range.** `hl_range_t = 100 * ln(high_t / low_t)`.

**Target (realized volatility, 5 day).** `rv5_t = sqrt( mean( ret_{t-4..t}^2 ) )`. This is the label for date `t`. It is computed from returns on days `t-4` to `t` and is only the *target*, never a feature for day `t`.

**LSTM alignment.** To predict `rv5_t`, the input is the 30 rows of features from dates `t-30 .. t-1` (features: `ret`, `hl_range`, plus `vol_chg = 100*ln(volume_t/volume_{t-1})`). Nothing from day `t` is in the input.

**GARCH alignment.** For each day `s`, a GARCH-family model gives a one-step-ahead conditional variance `sigma2_s` using returns up to `s-1` only. The comparable prediction of `rv5_t` is
`pred_t = sqrt( mean( sigma2_{t-4..t} ) )`. Each `sigma2_s` is itself a forecast made before day `s`, so this is leakage-free and measures the same quantity as the target.

**GARCH rolling scheme.** Test dates are split into blocks of `GARCH_REFIT_EVERY` days. For each block starting at index `s`: fit the model on returns `[0, s)` (expanding window), then freeze the parameters (`am.fix(params)`) on the series that extends through the end of the block, and read `conditional_volatility` for the block dates. In `arch`, `conditional_volatility[s]` uses information up to `s-1` only. Compute `sigma2` from 4 days before `TEST_START` so the first test target has a full 5-day window.

**Splits (chronological, never shuffled).** train: date <= `TRAIN_END`; val: `TRAIN_END` < date <= `VAL_END`; test: date >= `TEST_START`. Test includes the COVID-19 volatility spike of 2020, which is a deliberate stress test.

**Metrics.** With `y` = actual `rv5` and `p` = predicted (both floored at `EPS`):
- RMSE = sqrt(mean((y-p)^2))
- MAE = mean(|y-p|)
- QLIKE = mean( y^2/p^2 - ln(y^2/p^2) - 1 ) (lower is better; penalizes under-prediction more heavily)

---

## 4. Shared contracts

### 4.1 Prediction file schema

Every model writes `results/preds_<model>.csv` with columns exactly `date,y_true,y_pred`, covering **test dates only**, sorted ascending, no NaNs, same dates in every file. Model names: `garch`, `egarch`, `gjr`, `lstm`, and `naive` (benchmark, created in A4).

### 4.2 Repo structure and ownership

```
volatility-forecasting/
├── IMPLEMENTATION_PLAN.md
├── README.md                  (Anvi, A6)
├── requirements.txt           (Tanvika, T0)
├── config.py                  (Tanvika, T0)
├── data.py                    (Tanvika, T1)
├── metrics.py                 (Tanvika, T1)
├── garch_models.py            (Tanvika, T2-T3)
├── app.py                     (Tanvika, T4)
├── lstm_data.py               (Anvi, A1)
├── lstm_model.py              (Anvi, A2-A3)
├── evaluate.py                (Anvi, A4)
├── main.ipynb                 (Anvi, A6)
├── tests/
│   ├── test_garch_no_leakage.py   (Tanvika)
│   └── test_lstm_windows.py       (Anvi)
├── data/        raw_gspc.csv, processed.csv
├── results/     preds_*.csv, metrics*.csv, next_day_*.csv, figures/
└── docs/        challenges_log.md, part1_facts.md, report_facts.md
```

### 4.3 Handoffs

| From | To | Files | Needed before |
|---|---|---|---|
| Tanvika | Anvi | `config.py`, `data.py`, `metrics.py`, `data/processed.csv` | A1 |
| Anvi | Tanvika | `results/metrics.csv`, `results/metrics_by_period.csv`, `results/preds_lstm.csv`, `results/next_day_forecast.csv` | final GUI check in T4 |

Anvi can start the LSTM skeleton on dummy data while waiting for T1. Tanvika builds the GUI to tolerate missing files, then verifies it once Anvi's outputs exist.

---

# PART 1: Tanvika (data, GARCH family, GUI)

## T0. Project setup

**Do:**
- Create the folder structure from 4.2 (empty folders get a `.gitkeep`).
- Create `config.py` with every constant from section 2.
- Create `requirements.txt`: `numpy`, `pandas`, `yfinance`, `arch`, `statsmodels`, `scikit-learn`, `scipy`, `matplotlib`, `tensorflow`, `streamlit`, `plotly`, `jupyter`, `pytest`. (If TensorFlow cannot install, replace with `torch` and tell the user; Anvi's LSTM then uses PyTorch with the same architecture.)
- Create a virtual environment and install requirements. Create `docs/challenges_log.md` with a header.

**Verification:** `python -c "import config, arch, yfinance, sklearn, statsmodels, streamlit"` runs without error; `pytest --version` works; print installed versions of `arch`, `tensorflow` (or `torch`), `pandas`.

## T1. Data pipeline and metrics

**Do in `data.py`:**
1. `download_data()`: pull `TICKER` OHLCV from `START_DATE` to `END_DATE` with `yfinance` using `auto_adjust=False`. Newer yfinance returns MultiIndex columns even for one ticker; flatten them. Cache to `data/raw_gspc.csv` and reuse the cache if the download fails.
2. `build_features(df)`: compute `ret`, `hl_range`, `vol_chg`, `rv5` as defined in section 3. Drop the initial NaN rows and log how many were dropped.
3. Add a `split` column (`train`/`val`/`test`) from the date boundaries.
4. Save `data/processed.csv` with columns: `date, open, high, low, close, volume, ret, hl_range, vol_chg, rv5, split`.
5. `data_summary()`: save `results/data_summary.csv` per split and overall: count, mean, std, skewness, excess kurtosis of `ret`; Jarque-Bera p-value; and an ARCH-LM test p-value (`statsmodels.stats.diagnostic.het_arch`) on train returns. This is the statistical evidence of volatility clustering.
6. Figures in `results/figures/`: `fig01_price.png` (close with train/val/test shading), `fig02_returns.png` (returns showing volatility clustering), `fig03_rv5_splits.png` (rv5 coloured by split).

**Do in `metrics.py`:** functions `rmse(y,p)`, `mae(y,p)`, `qlike(y,p,eps)` and `all_metrics(y,p)` returning a dict, exactly as defined in section 3.

**Verification:**
- `processed.csv` has no NaNs, strictly increasing dates, about 5,500 rows, and all three splits non-empty; print the row count and date range per split.
- Recompute `rv5` for three random dates by hand in the check script and assert equality.
- `rmse` of identical arrays is 0; `qlike(y, y)` is about 0.
- Print the ARCH-LM p-value and state in one sentence whether clustering is significant.

**Handoff:** tell the user to send `config.py`, `data.py`, `metrics.py`, `data/processed.csv` to Anvi.

## T2. GARCH family, fit and parameter table

**Do in `garch_models.py`** (using the `arch` library, returns in percent, `dist=GARCH_DIST`, constant mean):
- Model specs: `garch` = `arch_model(r, vol="GARCH", p=1, q=1)`; `egarch` = `arch_model(r, vol="EGARCH", p=1, o=1, q=1)`; `gjr` = `arch_model(r, vol="GARCH", p=1, o=1, q=1)`.
- `fit_full_sample_params()`: fit each model on train+val data (dates up to `VAL_END`). Save `results/garch_params.csv` with columns: `model, param, estimate, std_err, p_value`, plus a second file `results/garch_fit_stats.csv` with `model, loglik, aic, bic`.
- Extract and highlight the asymmetry parameter `gamma[1]` for EGARCH and GJR. In EGARCH a negative-shock effect shows up as a negative `gamma`; in GJR a positive `gamma` means negative shocks add extra variance. Write a plain-English interpretation line per model into `docs/part1_facts.md`.
- Figure `fig04_news_impact.png`: news impact curves (next-period variance vs shock `eps` from -5 to +5, lagged variance fixed at the sample average) for GARCH, EGARCH, GJR on one plot. The asymmetric curves should lean left.
- Log any convergence warnings in `docs/challenges_log.md` with the fix (for example, rescaling, changing `dist`, or starting values).

**Verification:** all three models fit with finite estimates; `garch_params.csv` and `garch_fit_stats.csv` exist; stationarity check (alpha+beta < 1 for GARCH, alpha+beta+gamma/2 < 1 for GJR) printed for each model; figure saved and visually shows asymmetry for GJR/EGARCH.

## T3. GARCH rolling forecasts (leakage-safe) and next-day forecast

**Do in `garch_models.py`:**
1. `rolling_sigma2(model_name)`: implement the rolling scheme in section 3 (expanding window, refit every `GARCH_REFIT_EVERY` days, freeze parameters with `fix`, read `conditional_volatility`). Return a Series of one-step-ahead variance `sigma2` indexed by date from 4 days before `TEST_START` through the end of the data.
2. `window_prediction(sigma2)`: `pred_t = sqrt(rolling mean of sigma2 over 5 days ending t)`, restricted to test dates.
3. Save `results/preds_garch.csv`, `results/preds_egarch.csv`, `results/preds_gjr.csv` in the schema from 4.1 (`y_true` = `rv5` from `processed.csv`).
4. `next_day_forecast()`: fit each model on all data, take `forecast(horizon=1).variance` for the day after the last date, combine with the last four in-sample `conditional_volatility**2` values, compute the 5-day-window prediction the same way, and save `results/next_day_garch.csv` with columns `model, forecast_for, vol_forecast`. The forecast date is the first trading day after the dataset end. State this in the file header comment.
5. Figure `fig05_garch_pred_vs_actual.png`: actual `rv5` vs the three GARCH predictions over the test period.
6. `tests/test_garch_no_leakage.py`: take one model, compute `sigma2` for a block, then replace all returns after date `t0` with random noise and recompute: assert `sigma2` for every date `<= t0` is unchanged (tolerance 1e-8). Also assert that no prediction date appears before `TEST_START` and that all three files share the same dates.

**Verification:** `pytest tests/test_garch_no_leakage.py` passes; three CSVs with equal row counts and no NaNs; print RMSE, MAE, QLIKE for each GARCH model using `metrics.all_metrics`; check the predictions are in a plausible range (same order of magnitude as `rv5`, around 0.3 to 5 in percent units); runtime under about 10 minutes (if slower, log it and increase the refit interval only with the user's approval).

## T4. Streamlit GUI

**Do in `app.py`** (reads only saved files from `results/` and `data/processed.csv`, so it starts instantly and cannot break during a demo; use `st.cache_data`):
- **Sidebar:** model multiselect (garch, egarch, gjr, lstm, naive), date-range slider limited to the test period, metric used to highlight the best model (RMSE, MAE, QLIKE).
- **Tab 1 "Forecast vs actual":** interactive plotly line chart of `y_true` and selected predictions for the chosen date range.
- **Tab 2 "Model comparison":** metrics table recomputed for the selected date range using `metrics.all_metrics`; best model highlighted per metric; bar chart of the chosen metric; a second table from `results/metrics_by_period.csv` (calm vs COVID stress).
- **Tab 3 "Asymmetry":** the gamma parameters from `garch_params.csv` with the plain-English interpretation, and `fig04_news_impact.png`.
- **Tab 4 "Next-day forecast":** model estimates from `results/next_day_forecast.csv`, the average, and a **risk label** (Low / Medium / High) based on where the forecast falls within the train-period distribution of `rv5` (below 33rd percentile = Low, 33rd to 67th = Medium, above = High). Add a footer: forecasts are for the first trading day after the dataset end; educational use, not financial advice.
- **Tab 5 "About":** project description, data source, the 3 papers, model descriptions.
- Missing files must show a friendly `st.info` message, not a traceback.

**Verification:**
- `streamlit run app.py` starts; every tab renders with GARCH files only (LSTM and next-day pieces show the info message).
- After Anvi's outputs arrive (A4 and A5), re-check that LSTM and naive appear in every tab with no code changes.
- Save a screenshot checklist to `docs/screenshots_todo.md`: one screenshot per tab and one per sidebar state for the report.

## T5. Part 1 handoff note

**Do:** write `docs/part1_facts.md` summarising, with real numbers read from the result files: data source and date range, row counts per split, return statistics, ARCH-LM result, GARCH parameter table highlights, AIC/BIC ranking, gamma interpretation, GARCH test-set metrics, runtime, and each entry of `docs/challenges_log.md` related to Part 1. No invented numbers: every number must be read programmatically from the saved files.

**Verification:** every figure and filename mentioned in the note exists.

---

# PART 2: Anvi (LSTM, evaluation, notebook)

*Prerequisite: the handoff files from Tanvika's T1 (`config.py`, `data.py`, `metrics.py`, `data/processed.csv`). Until they arrive, build against a dummy `processed.csv` with the same columns.*

## A1. LSTM dataset builder

**Do in `lstm_data.py`:**
- `load_processed()` reads `data/processed.csv`.
- `make_sequences()`: for each row `i` with a full history, `X[i]` = features `ret, hl_range, vol_chg` from rows `i-LOOKBACK .. i-1`, shape `(LOOKBACK, 3)`; `y[i]` = `rv5` at row `i`; keep the target date. Assign each sample to a split by its **target date**. (Test samples therefore use some validation-period rows in their input window, which is correct and not leakage.)
- Scaling: fit a `StandardScaler` for features and a `MinMaxScaler` for the target on **train samples only**; apply to val and test. Return the fitted scalers so predictions can be inverse-transformed.
- Return a dict with `X_train, y_train, X_val, y_val, X_test, y_test, dates_test, scaler_x, scaler_y`.

**Verification (`tests/test_lstm_windows.py`):**
- Shapes are `(n, 30, 3)` and `(n,)`; no NaNs.
- For three random samples, assert that `X` rows equal the raw feature rows at dates strictly before the target date, and that the target date's own `ret` does not appear in its window.
- `scaler_x.mean_` equals the mean of **train-period** raw features (proves train-only fitting).
- Print the sample count per split.

## A2. LSTM model and training

**Do in `lstm_model.py`:**
- Architecture (Keras): `LSTM(LSTM_UNITS) -> Dropout(LSTM_DROPOUT) -> Dense(1)`; Adam with `LSTM_LR`; MSE loss; batch `LSTM_BATCH`; up to `LSTM_MAX_EPOCHS`; `EarlyStopping(monitor="val_loss", patience=LSTM_PATIENCE, restore_best_weights=True)`. Do not shuffle across time between train and val (shuffling within the training batches is fine).
- Set seeds with `tf.keras.utils.set_random_seed(seed)`.
- Train `N_SEEDS` models (seeds 42, 43, 44) and keep the averaged prediction. Save each model's weights to `results/models/`.
- Save training history to `results/lstm_history.csv` and the figure `fig06_lstm_loss.png` (train and val loss per epoch, one line per seed).
- If TensorFlow is unavailable, implement the identical architecture in PyTorch and note it in `docs/challenges_log.md`.

**Verification:** training completes; val loss decreases at first and early stopping triggers before the epoch cap; print best epoch per seed and final val loss; if train loss is far below val loss, log it as overfitting in `docs/challenges_log.md` and (with the user's approval) try dropout 0.3 or fewer units; do not change the test set.

## A3. LSTM predictions, next-day forecast, optional lookback study

**Do in `lstm_model.py`:**
1. Predict on the test samples, inverse-transform with `scaler_y`, floor at `EPS`, and save `results/preds_lstm.csv` in the schema from 4.1 (`y_true` must be the unscaled `rv5`).
2. Assert the date column is identical to `results/preds_garch.csv` (equal length and equal values). If lengths differ, fix the alignment instead of trimming silently, and log it.
3. Next-day forecast: take the final 30 rows of features, scale with the train scalers, predict with all seeds, average, inverse-transform, and save `results/next_day_lstm.csv` with columns `model, forecast_for, vol_forecast`.
4. Figure `fig07_lstm_pred_vs_actual.png`.
5. **Optional (only if the user asks):** lookback study with 10, 30 and 60 days; save `results/lookback_study.csv` with test RMSE, MAE, QLIKE per lookback.

**Verification:** file has the right columns, no NaNs, the same row count as `preds_garch.csv`; print LSTM RMSE, MAE, QLIKE; predictions are positive and the same order of magnitude as `rv5`.

## A4. Evaluation and comparison

**Do in `evaluate.py`** (import functions from `metrics.py`, do not redefine them):
1. Create the **naive benchmark** `results/preds_naive.csv`: `pred_t = rv5_{t-1}` (yesterday's realized volatility), same schema. Any model should beat this to be worth using.
2. Load every `results/preds_*.csv`; assert identical date indices across files.
3. Compute RMSE, MAE, QLIKE per model over the full test set and save `results/metrics.csv` (rows = models, plus a `rank` column per metric).
4. `results/metrics_by_period.csv`: the same metrics for two sub-periods: **stress** (2020-02-15 to 2020-06-30) and **calm** (2021-01-01 to the end).
5. Diebold-Mariano style comparison between the best GARCH-family model and LSTM on squared error (a simple implementation using `scipy`; report the statistic and p-value in `results/dm_test.csv`). Note HAC corrections only if implemented; otherwise say it is a simple version.
6. Figures: `fig08_all_models_pred_vs_actual.png`, `fig09_metric_bars.png` (RMSE, MAE, QLIKE side by side), `fig10_covid_zoom.png` (actual vs models during the stress window).
7. Merge `results/next_day_garch.csv` and `results/next_day_lstm.csv` into `results/next_day_forecast.csv` (`model, forecast_for, vol_forecast`); add the naive row using the last `rv5`.
8. Print a plain table of the results and the winner per metric. **Do not write conclusions that the numbers do not support.**

**Verification:** `metrics.csv` has one row per model with finite values; the naive row is present; best model per metric is printed; all five figures exist; the dates in every preds file match.

**Handoff:** tell the user to send `results/` (metrics, preds, next-day, figures) to Tanvika for the GUI check.

## A5. Interpretation helper

**Do:** write `docs/report_facts.md` containing real values read from files: final metrics table (as markdown), winner per metric, stress vs calm comparison, DM test result, LSTM training details (best epochs, final losses), lookback study if it was run, and factual one-line observations such as "model X had the lowest RMSE overall but model Y had the lowest QLIKE" generated from the numbers. Include a section "Possible explanations to discuss in the report" that is clearly labelled as hypotheses (for example: LSTM lagging during sudden spikes, GARCH persistence, effect of the high-low feature), not as findings.

**Verification:** every number in the file matches `metrics.csv` and related files (write a script that cross-checks them).

## A6. End-to-end notebook and README

**Do:**
- `main.ipynb`: sections for Config, Data (call `data.py`), GARCH (call `garch_models.py`), LSTM, Evaluation, Figures. Include a markdown cell explaining what each step does. It must run top to bottom on a fresh kernel. If a step takes long, add a flag at the top to load saved results instead of retraining.
- `README.md`: project description, structure, install steps, exact commands in order (`python data.py`, `python garch_models.py`, `python lstm_model.py`, `python evaluate.py`, `streamlit run app.py`), expected outputs, and runtime.

**Verification:**
- Run `jupyter nbconvert --to notebook --execute main.ipynb` on a clean kernel with no errors.
- Run `pytest` and confirm every test passes.
- Confirm that following only the README from a fresh clone reproduces `results/metrics.csv`.

---

## 5. Final integration checklist (both people, after Part 1 and Part 2)

- [ ] `results/` contains `preds_garch.csv`, `preds_egarch.csv`, `preds_gjr.csv`, `preds_lstm.csv`, `preds_naive.csv` with identical dates.
- [ ] `metrics.csv`, `metrics_by_period.csv`, `dm_test.csv`, `next_day_forecast.csv` exist.
- [ ] Ten figures `fig01` to `fig10` exist in `results/figures/`.
- [ ] `pytest` passes; `main.ipynb` runs on a fresh kernel.
- [ ] `streamlit run app.py` shows all models in all tabs; screenshots captured per `docs/screenshots_todo.md`.
- [ ] `docs/part1_facts.md`, `docs/report_facts.md` and `docs/challenges_log.md` exist and contain only real, verified content.
- [ ] Report numbers match `metrics.csv` exactly.

## 6. Notes for the written report (not for the agent to invent)

The report should describe what was actually built and found: methodology per phase, design, parameters, results with the saved figures and GUI screenshots, real challenges from `docs/challenges_log.md`, limitations (one main asset, no extensive hyperparameter search, a single test window dominated by COVID, simple DM test) and future work (GARCH-LSTM hybrid, more assets such as Bitcoin, extra features).

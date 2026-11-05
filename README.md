# S&P 500 Volatility Forecasting: GARCH Family vs. LSTM

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A rigorous empirical comparison benchmarking classical econometric volatility models (**GARCH**, **EGARCH**, **GJR-GARCH**) against a deep recurrent neural network (**LSTM**) and a **naive 1-day benchmark** on identical historical S&P 500 index data (`^GSPC`, 2000–2021).

---

## 1. Project Overview

Volatility measures the dispersion of financial asset returns and is a vital input for risk management (Value-at-Risk), option pricing, dynamic hedging, and margin requirements. This repository investigates whether deep sequence models utilizing intraday high-low ranges and volume dynamics can systematically outperform traditional parametric autoregressive conditional heteroskedasticity models.

### Key Highlights
- **Identical Evaluation Base:** All models forecast 5-day realized volatility ($rv5_t$) evaluated over the identical 505 test trading days (2020–2021), covering the severe COVID-19 volatility shock.
- **Strict Leakage Prevention:** All scalers, parameters, and sequence windows strictly use information available prior to the forecast target date ($t-1$).
- **Ensemble Robustness:** The LSTM uses a 3-seed ensemble (seeds 42, 43, 44) with early stopping.
- **Interactive Streamlit GUI:** Web interface featuring interactive Plotly charts, dynamic date slicing, and asymmetry diagnostics.

---

## 2. Repository Structure

```
Voltality_Forecasting/
├── config.py                  # Fixed configuration constants, paths, and hyperparameters
├── data.py                    # Data retrieval, feature engineering, and ARCH-LM clustering tests
├── garch_models.py            # GARCH, EGARCH, and GJR-GARCH rolling forecasts and news impact curves
├── lstm_data.py               # 30-day sliding window sequence generator with train-only scaling
├── lstm_model.py              # PyTorch LSTM architecture, 3-seed ensemble training, and predictions
├── evaluate.py                # Naive benchmark, comparative metrics (RMSE/MAE/QLIKE), and DM test
├── metrics.py                 # Pure metric calculation functions (RMSE, MAE, QLIKE)
├── app.py                     # Streamlit web application GUI
├── main.ipynb                 # Executable end-to-end Jupyter Notebook
├── requirements.txt           # Project dependencies
├── tests/
│   ├── test_garch_no_leakage.py # Leakage and invariance tests for GARCH rolling forecasts
│   ├── test_lstm_windows.py     # Sequence shape, alignment, and train-only scaling assertions
│   └── test_report_facts.py     # Cross-check verification of numerical documentation
├── data/                      # raw_gspc.csv, processed.csv
├── results/                   # Model predictions (preds_*.csv), metrics.csv, next_day_*.csv
│   ├── figures/               # Figures fig01 through fig10 (150 dpi)
│   └── models/                # Saved PyTorch model weights (lstm_seed_*.pt)
└── docs/                      # challenges_log.md, part1_facts.md, report_facts.md
```

---

## 3. Installation & Setup

Clone the repository and install dependencies in your Python 3.12 environment:

```bash
git clone https://github.com/anvisharma11/Voltality_Forecasting.git
cd Voltality_Forecasting

# Install dependencies
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytest streamlit plotly nbconvert ipykernel
```

*(Note: Per project fallback decisions, PyTorch CPU is utilized for deep learning components to guarantee cross-platform reproducibility on Windows Python 3.12).*

---

## 4. Exact Execution Order

Execute the pipeline sequentially from the project root:

```bash
# Step 1: Download raw S&P 500 data and build features (~5 seconds)
python data.py

# Step 2: Fit GARCH models, news impact curves, and rolling forecasts (~10 seconds)
python garch_models.py

# Step 3: Train LSTM ensemble across seeds 42, 43, 44 and generate predictions (~45 seconds)
python lstm_model.py

# Step 4: Run comparative evaluation, period breakdown, and Diebold-Mariano test (~5 seconds)
python evaluate.py

# Step 5: Launch the Streamlit Interactive Dashboard
streamlit run app.py
```

### Total Pipeline Runtime
Approx. **1 minute to 1.5 minutes** from fresh clone to final figures and metrics.

---

## 5. Summary of Empirical Findings (Full Test Set: 2020–2021)

| Model | Observations | RMSE | MAE | QLIKE | Rank (RMSE) | Rank (MAE) | Rank (QLIKE) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive ($rv5_{t-1}$)** | 505 | **0.3146** | 0.1659 | 0.9029 | **1** | 2 | 2 |
| **LSTM (Ensemble)** | 505 | 0.3415 | **0.1601** | **0.8413** | 2 | **1** | **1** |
| **EGARCH(1,1)** | 505 | 0.5801 | 0.3425 | 1.0777 | 3 | 3 | 4 |
| **GJR-GARCH(1,1)** | 505 | 0.5951 | 0.3711 | 1.0487 | 4 | 4 | 3 |
| **GARCH(1,1)** | 505 | 0.6705 | 0.4234 | 1.0818 | 5 | 5 | 5 |

### Key Takeaways
1. **Regime Dependence:** In the tranquil 2021 period (252 trading days), the **LSTM outperformed all models** across every metric (RMSE `0.1278` vs Naive `0.1679` vs GJR `0.2811`). In the extreme March 2020 COVID shock, the **Naive benchmark** responded with the lowest RMSE (`0.6272`) due to immediate 1-day adjustment.
2. **Asymmetry Matters:** Both asymmetric models (EGARCH $\gamma_1 = -0.1646$, GJR $\gamma_1 = +0.2027$, $p < 10^{-16}$) substantially outperformed symmetric GARCH(1,1).
3. **Statistical Superiority:** The Diebold-Mariano test between the best GARCH model (`EGARCH`) and the `LSTM` yields $DM = 5.4113$, $p = 6.26 \times 10^{-8}$, confirming a statistically significant advantage for the LSTM on squared error.

---

## 6. Running Tests

Run the full automated test suite verifying leakage-free windows, GARCH invariance, and numerical report consistency:

```bash
python -m pytest -v
```

---

## 7. Next-Day Forecast (Target Date: 2022-01-03)
- **LSTM:** `0.4789%`
- **Naive:** `0.6446%`
- **GJR-GARCH:** `0.8740%`
- **EGARCH:** `0.9588%`
- **GARCH:** `1.0587%`

*Disclaimer: Forecasts are generated for educational and quantitative research purposes only and do not constitute financial advice.*

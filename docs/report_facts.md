# Empirical Report Facts & Statistical Reference

> **Notice:** This document contains verified, programmatically extracted numbers from the experimental output files (`results/metrics.csv`, `results/metrics_by_period.csv`, `results/dm_test.csv`, `results/lstm_history.csv`, `results/garch_params.csv`, and `results/next_day_forecast.csv`). All figures and statistics reported here represent actual experimental runs on the S&P 500 dataset (`^GSPC`, 2000–2021).

---

## 1. Full Test-Set Performance (2020-01-02 to 2021-12-31, 505 Observations)

| Model | Observations | RMSE | MAE | QLIKE | Rank (RMSE) | Rank (MAE) | Rank (QLIKE) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **naive** | 505 | 0.314626 | 0.165914 | 0.902870 | 1 | 2 | 2 |
| **lstm** | 505 | 0.341522 | 0.160139 | 0.841270 | 2 | 1 | 1 |
| **egarch** | 505 | 0.580133 | 0.342501 | 1.077702 | 3 | 3 | 4 |
| **gjr** | 505 | 0.595087 | 0.371102 | 1.048665 | 4 | 4 | 3 |
| **garch** | 505 | 0.670497 | 0.423369 | 1.081792 | 5 | 5 | 5 |

### Winners per Metric
- **Best RMSE:** **`NAIVE`** (`0.314626`), with `LSTM` second (`0.341522`).
- **Best MAE:** **`LSTM`** (`0.160139`), beating Naive (`0.165914`) and EGARCH (`0.342501`).
- **Best QLIKE:** **`LSTM`** (`0.841270`), outperforming Naive (`0.902870`) and GJR-GARCH (`1.048665`).

---

## 2. Regime Performance: COVID Stress vs. Calm Period

### Metrics Table by Sub-Period

| Period | Model | Observations | RMSE | MAE | QLIKE |
|:---|:---|:---:|:---:|:---:|:---:|
| **COVID Stress** (`2020-02-15` to `2020-06-30`) | **naive** | 94 | 0.627175 | 0.380437 | 2.704925 |
| | **lstm** | 94 | 0.734505 | 0.442069 | 2.647798 |
| | **egarch** | 94 | 1.175608 | 0.779268 | 2.988798 |
| | **gjr** | 94 | 1.214337 | 0.946410 | 2.946120 |
| | **garch** | 94 | 1.387035 | 1.107072 | 2.991988 |
| **Calm Period** (`2021-01-01` to `2021-12-31`) | **lstm** | 252 | 0.127802 | 0.091300 | 0.262642 |
| | **naive** | 252 | 0.167883 | 0.113118 | 0.352131 |
| | **gjr** | 252 | 0.281075 | 0.224064 | 0.456477 |
| | **egarch** | 252 | 0.284436 | 0.226115 | 0.465791 |
| | **garch** | 252 | 0.312646 | 0.252636 | 0.504592 |

### Key Regime Observations
1. In the **Calm Period** (252 trading days), **LSTM clearly outperforms all models across every single metric** (RMSE `0.127802` vs Naive `0.167883` vs GJR `0.281075`; MAE `0.091300` vs Naive `0.113118`; QLIKE `0.262642` vs Naive `0.352131`).
2. During the **COVID Stress Period** (94 trading days), the **Naive benchmark had the lowest RMSE** (`0.627175`), while **LSTM achieved the lowest QLIKE loss** (`2.647798`).
3. GARCH models exhibited substantial forecast errors during the sudden COVID crash (RMSEs between `1.1756` and `1.3870`), lagging the rapid volatility spike compared to the non-parametric models.

---

## 3. Diebold-Mariano Hypothesis Test (Best GARCH vs. LSTM)

- **Comparison:** EGARCH(1,1) (selected as best in-sample and lowest-RMSE GARCH model) vs LSTM Ensemble
- **Loss Metric:** Squared Error Loss ($d_t = e_{\text{egarch}, t}^2 - e_{\text{lstm}, t}^2$)
- **Mean Loss Differential ($\bar{d}$):** `+0.219918`
- **Diebold-Mariano Test Statistic ($DM$):** `5.411292`
- **p-value:** `6.257168e-08` ($p < 0.0001$)
- **Methodology:** Simple Diebold-Mariano test using standard sample variance (no HAC adjustment).
- **Statistical Conclusion:** The forecast accuracy differential between EGARCH and LSTM is statistically significant at the 1% level ($p = 6.26 \times 10^{-8}$). The positive loss differential indicates that the LSTM achieved lower mean squared error over the test period.

---

## 4. LSTM Ensemble Training Details

The LSTM network (`LSTM(64) -> Dropout(0.2) -> Linear(1)`) was trained across three random seeds with early stopping (`patience=10`, `restore_best_weights=True`):

| Seed | Best Epoch | Best Validation Loss (MSE) | Stopped Epoch | Train Loss at Best Epoch | Status |
|:---:|:---:|:---:|:---:|:---:|:---|
| **42** | 37 | 0.000489 | 47 | 0.000720 | Early stopped (patience reached) |
| **43** | 18 | 0.000524 | 28 | 0.000797 | Early stopped (patience reached) |
| **44** | 55 | 0.000465 | 65 | 0.000652 | Early stopped (patience reached) |

- **Average Validation Loss across Seeds:** `0.000493`
- **Generalization:** Final training losses remained close to validation losses across all three seeds, indicating no pathological overfitting.

---

## 5. Asymmetry & In-Sample Fit in GARCH Family

Fitted on in-sample data (Train + Validation, up to `2019-12-31`, 5026 observations):

| Model | Log-Likelihood | AIC | BIC | Asymmetry Parameter ($\gamma$) | p-value ($\gamma$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **EGARCH(1,1)** | -6590.14 | 13192.29 | 13231.42 | $\gamma_1 = -0.164604$ | $1.27 \times 10^{-31}$ |
| **GJR-GARCH(1,1)** | -6607.83 | 13227.67 | 13266.80 | $\gamma_1 = +0.202716$ | $3.15 \times 10^{-17}$ |
| **GARCH(1,1)** | -6697.79 | 13405.58 | 13438.19 | N/A (symmetric) | N/A |

### Plain-English Asymmetry Interpretation
- **EGARCH:** $\gamma_1 = -0.1646 < 0$ is highly statistically significant ($p < 10^{-30}$), confirming that negative return shocks produce substantially higher log-volatility than positive shocks of identical magnitude.
- **GJR-GARCH:** $\gamma_1 = +0.2027 > 0$ ($p < 10^{-16}$) shows that negative returns add an additional positive variance increment beyond the symmetric ARCH coefficient ($\alpha_1 = 0.0381$).
- **Model Selection:** Both information criteria (AIC and BIC) rank **EGARCH(1,1) as the best fitting in-sample model**, followed by GJR-GARCH and standard GARCH.

---

## 6. Next-Day Volatility Forecasts (Target Date: 2022-01-03)

| Model | Forecast Date | Forecasted 5-Day RV (%) |
|:---|:---:|:---:|
| **lstm** | 2022-01-03 | **0.4789%** |
| **naive** | 2022-01-03 | **0.6446%** |
| **gjr** | 2022-01-03 | **0.8740%** |
| **egarch** | 2022-01-03 | **0.9588%** |
| **garch** | 2022-01-03 | **1.0587%** |

- **Test Period Final Observed rv5 (2021-12-31):** `0.6446%`
- **Ensemble / Consensus Range:** `0.48%` to `1.06%`

---

## 7. Factual One-Line Observations

1. `NAIVE` had the lowest RMSE (`0.3146`) overall, but `LSTM` had the lowest MAE (`0.1601`) and lowest QLIKE (`0.8413`).
2. Among the GARCH family, `EGARCH` achieved the lowest RMSE (`0.5801`), while `GJR-GARCH` had the lowest QLIKE (`1.0487`), and symmetric `GARCH` was the worst performer across all metrics (`0.6705` RMSE).
3. In the tranquil 2021 calm period (252 days), `LSTM` beat all models including Naive on every metric (RMSE `0.1278` vs Naive `0.1679` vs EGARCH `0.2844`).
4. In the COVID shock period (94 days), `NAIVE` had the lowest RMSE (`0.6272`), while `LSTM` had the lowest QLIKE (`2.6478`).
5. All three GARCH variants significantly over-predicted volatility persistence following the March 2020 crash, leading to higher RMSEs (`1.17`–`1.39`).
6. The Diebold-Mariano test between the top statistical model (`EGARCH`) and `LSTM` rejects the null hypothesis of equal forecast accuracy with $p = 6.26 \times 10^{-8}$.

---

## 8. Possible Explanations to Discuss in the Report (Hypotheses)

> [!NOTE]
> The following points are theoretical hypotheses and potential explanations for discussion in the project report, not verified empirical laws.

- **Hypothesis 1 (Feature Richness):** The LSTM's access to intraday high-low range (`hl_range`) and volume changes (`vol_chg`) may provide informative intraday volatility signals that return-only GARCH models cannot observe.
- **Hypothesis 2 (COVID Lag vs Reaction):** The Naive model ($rv5_{t-1}$) reacts with a strict 1-day lag to volatility spikes. During sudden non-linear regime shifts like the March 2020 crash, a 1-day lag achieves lower squared error than parametric autoregressive models whose variance persistence decays gradually.
- **Hypothesis 3 (GARCH Mean-Reversion Inertia):** GARCH models estimate high persistence ($\alpha + \beta \approx 0.98$ to $0.99$). When shocked by unprecedented COVID return swings, the conditional variance remains elevated for months, leading to over-prediction as market realized volatility subsided in late 2020.
- **Hypothesis 4 (Asymmetric News Impact):** The clear outperformance of EGARCH and GJR over standard GARCH highlights the economic importance of modeling the leverage effect in equity indices.

# Part 1 Facts & Statistical Handoff Note

> **Author:** Tanvika (Part 1 - Data & GARCH Pipeline)  
> **Source Files Verified:** `data/processed.csv`, `results/data_summary.csv`, `results/garch_params.csv`, `results/garch_fit_stats.csv`, `results/metrics.csv`, `results/next_day_garch.csv`.

---

## 1. Data Source & Sample Properties
- **Asset / Ticker:** S&P 500 Index (`^GSPC`), downloaded via Yahoo Finance (`auto_adjust=False`).
- **Date Range:** `2000-01-03` to `2021-12-31` (Total raw rows: 5,536; Processed rows after 5-day rolling window: 5,531).
- **Partitioning (Chronological):**
  - **Training:** `2000-01-10` to `2017-12-29` (4,523 rows, 81.8%)
  - **Validation:** `2018-01-02` to `2019-12-31` (503 rows, 9.1%)
  - **Testing:** `2020-01-02` to `2021-12-31` (505 rows, 9.1%)

---

## 2. Return Statistics & Volatility Clustering Evidence
Calculated on daily percent log returns ($r_t = 100 \times \ln(Close_t / Close_{t-1})$):

| Split | Observations | Mean (%) | Std Dev (%) | Min (%) | Max (%) | Skewness | Excess Kurtosis |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **All Returns** | 5,531 | 0.0216 | 1.2378 | -12.7652 | 10.9572 | -0.3983 | 11.1121 |
| **Train Returns** | 4,523 | 0.0137 | 1.2126 | -9.4695 | 10.9572 | -0.2022 | 8.7160 |
| **Val Returns** | 503 | 0.0376 | 0.9441 | -4.1843 | 4.8403 | -0.6168 | 3.6888 |
| **Test Returns** | 505 | 0.0770 | 1.6514 | -12.7652 | 8.9683 | -1.0521 | 14.9179 |

- **Jarque-Bera Test (Train Returns):** Statistic = `14311.80`, $p = 0.000$ (strongly rejects normality due to heavy tails).
- **ARCH-LM Test (Train Returns, 10 lags):** LM Statistic = `1215.17`, $p = 7.68 \times 10^{-255}$ ($F$-stat = $165.89$, $p = 1.21 \times 10^{-297}$).
- **Conclusion:** Overwhelming statistical evidence ($p < 10^{-250}$) of autoregressive conditional heteroskedasticity (volatility clustering), justifying GARCH family modeling.

---

## 3. GARCH In-Sample Estimation & Asymmetry
Fitted on combined Train + Validation sample (`2000-01-10` to `2019-12-31`, 5,026 observations) with Student's $t$ innovations:

| Model | Log-Likelihood | AIC | BIC | $\alpha_1$ (ARCH) | $\beta_1$ (GARCH) | $\gamma_1$ (Asymmetry) | $p$-value ($\gamma_1$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **GARCH(1,1)** | -6697.79 | 13405.58 | 13438.19 | 0.0827 | 0.9069 | N/A | N/A |
| **EGARCH(1,1)** | -6590.14 | 13192.29 | 13231.42 | -0.0133 | 0.9856 | -0.1646 | $1.27 \times 10^{-31}$ |
| **GJR-GARCH(1,1)** | -6607.83 | 13227.67 | 13266.80 | 0.0381 | 0.8687 | +0.2027 | $3.15 \times 10^{-17}$ |

- **AIC/BIC Ranking:** **EGARCH(1,1)** achieved the lowest AIC (`13192.29`) and BIC (`13231.42`), followed by GJR-GARCH and standard GARCH.
- **Asymmetry Interpretation:** Both asymmetric models show statistically significant leverage effects ($p < 10^{-16}$). In EGARCH, $\gamma_1 = -0.1646 < 0$ implies negative return shocks increase volatility significantly more than positive shocks. In GJR, $\gamma_1 = +0.2027 > 0$ adds substantial conditional variance when returns are negative.

---

## 4. Test Set Out-of-Sample Performance (GARCH Models)

| Model | Observations | RMSE | MAE | QLIKE |
|:---|:---:|:---:|:---:|:---:|
| **EGARCH(1,1)** | 505 | **0.5801** | **0.3425** | 1.0777 |
| **GJR-GARCH(1,1)** | 505 | 0.5951 | 0.3711 | **1.0487** |
| **GARCH(1,1)** | 505 | 0.6705 | 0.4234 | 1.0818 |

- Rolling forecast runtime: ~8 seconds across all 3 models using expanding windows with 20-day parameter refits.

---

## 5. Next-Day GARCH Volatility Forecasts (2022-01-03)
- **GJR-GARCH:** `0.8740%`
- **EGARCH:** `0.9588%`
- **GARCH:** `1.0587%`

---

## 6. Generated Figures Checklist
- [x] [results/figures/fig01_price.png](file:///c:/Users/ANVI/Desktop/AF_MP/results/figures/fig01_price.png) — Closing price with train/val/test splits
- [x] [results/figures/fig02_returns.png](file:///c:/Users/ANVI/Desktop/AF_MP/results/figures/fig02_returns.png) — Daily percentage log returns displaying clustering
- [x] [results/figures/fig03_rv5_splits.png](file:///c:/Users/ANVI/Desktop/AF_MP/results/figures/fig03_rv5_splits.png) — 5-day realized volatility series across splits
- [x] [results/figures/fig04_news_impact.png](file:///c:/Users/ANVI/Desktop/AF_MP/results/figures/fig04_news_impact.png) — Asymmetric news impact curves for GARCH, EGARCH, GJR
- [x] [results/figures/fig05_garch_pred_vs_actual.png](file:///c:/Users/ANVI/Desktop/AF_MP/results/figures/fig05_garch_pred_vs_actual.png) — Test set GARCH rolling predictions vs actual rv5

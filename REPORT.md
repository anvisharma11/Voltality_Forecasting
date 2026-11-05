# Technical Project Report

The complete, publication-grade academic project report is located in:
👉 **[docs/project_report.md](docs/project_report.md)**

### Quick Reference: Empirical Summary
- **Asset / Period:** S&P 500 Index (`^GSPC`), 2000–2021 (505 out-of-sample test days: 2020–2021)
- **Target:** 5-Day Realized Volatility ($rv5_t$)
- **Best Overall MAE & QLIKE:** LSTM 3-Seed Ensemble (`MAE = 0.1601`, `QLIKE = 0.8413`)
- **Best Crisis RMSE (March 2020 Shock):** Naive Persistence Benchmark (`RMSE = 0.6272`)
- **Best Calm Regime Performer (2021):** LSTM Ensemble across all metrics (`RMSE = 0.1278` vs Naive `0.1679` vs GJR `0.2811`)
- **Statistical Significance:** Diebold-Mariano Test ($DM = 5.4113$, $p = 6.26 \times 10^{-8}$) confirming statistically superior accuracy of LSTM over best econometric model (EGARCH).

For full literature review (15 papers), mathematical formulations, Mermaid workflow architecture, regime analysis, and references, see [`docs/project_report.md`](docs/project_report.md).


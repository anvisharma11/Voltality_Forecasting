"""Streamlit Web GUI for S&P 500 Volatility Forecasting Comparison.

Interactive dashboard presenting forecasts, model comparison, asymmetry analysis,
and next-day forecasts per IMPLEMENTATION_PLAN.md (Task T4).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from config import (
    RESULTS_DIR,
    DATA_DIR,
    FIG_DIR,
    EPS,
)
from metrics import rmse, mae, qlike

st.set_page_config(
    page_title="Volatility Forecasting: GARCH vs LSTM",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROCESSED_FILE = DATA_DIR / "processed.csv"
METRICS_FILE = RESULTS_DIR / "metrics.csv"
PERIOD_METRICS_FILE = RESULTS_DIR / "metrics_by_period.csv"
PARAMS_FILE = RESULTS_DIR / "garch_params.csv"
NEXT_DAY_FILE = RESULTS_DIR / "next_day_forecast.csv"
FIG04_PATH = FIG_DIR / "fig04_news_impact.png"

ALL_MODELS = ["garch", "egarch", "gjr", "lstm", "naive"]
MODEL_COLORS = {
    "naive": "#7f7f7f",
    "garch": "#1f77b4",
    "egarch": "#2ca02c",
    "gjr": "#d62728",
    "lstm": "#ff7f0e",
}


@st.cache_data
def load_processed_data():
    if not PROCESSED_FILE.exists():
        return None
    df = pd.read_csv(PROCESSED_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_model_predictions():
    preds = {}
    for m in ALL_MODELS:
        path = RESULTS_DIR / f"preds_{m}.csv"
        if path.exists():
            df_p = pd.read_csv(path)
            df_p["date"] = pd.to_datetime(df_p["date"])
            preds[m] = df_p
    return preds


# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------

st.sidebar.title("Forecast Controls")

preds_dict = load_model_predictions()
processed_df = load_processed_data()

available_models = list(preds_dict.keys())

if not available_models:
    st.sidebar.warning("No prediction files found in results/. Run pipeline first.")
    selected_models = []
else:
    selected_models = st.sidebar.multiselect(
        "Select Models to Display",
        options=available_models,
        default=available_models,
    )

# Date Range Slider
if available_models:
    ref_df = preds_dict[available_models[0]]
    min_date = ref_df["date"].min().date()
    max_date = ref_df["date"].max().date()

    date_range = st.sidebar.slider(
        "Test Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
    )
else:
    date_range = None

chosen_metric = st.sidebar.selectbox(
    "Highlight Metric",
    options=["RMSE", "MAE", "QLIKE"],
    index=0,
)

st.title("S&P 500 Volatility Forecasting: GARCH Family vs. LSTM")
st.markdown("A comparative benchmark of econometric and deep learning volatility models on identical data.")

# ---------------------------------------------------------------------------
# Tabs Layout
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Forecast vs Actual",
    "Model Comparison",
    "Asymmetry",
    "Next-Day Forecast",
    "About",
])

# ---------------------------------------------------------------------------
# Tab 1: Forecast vs Actual
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Interactive Forecast vs. Actual Realized Volatility")
    if not selected_models:
        st.info("Select at least one model in the sidebar to view predictions.")
    else:
        start_d, end_d = date_range
        fig = go.Figure()

        # Plot Ground Truth
        ref_p = preds_dict[selected_models[0]]
        mask = (ref_p["date"].dt.date >= start_d) & (ref_p["date"].dt.date <= end_d)
        sub_ref = ref_p[mask]

        fig.add_trace(go.Scatter(
            x=sub_ref["date"],
            y=sub_ref["y_true"],
            mode="lines",
            name="Actual (rv5)",
            line=dict(color="#111111", width=2.5),
        ))

        # Plot Models
        for m in selected_models:
            df_m = preds_dict[m]
            sub_m = df_m[(df_m["date"].dt.date >= start_d) & (df_m["date"].dt.date <= end_d)]
            fig.add_trace(go.Scatter(
                x=sub_m["date"],
                y=sub_m["y_pred"],
                mode="lines",
                name=m.upper(),
                line=dict(color=MODEL_COLORS.get(m, "#333333"), width=1.5),
            ))

        fig.update_layout(
            title=f"5-Day Realized Volatility Forecasts ({start_d} to {end_d})",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Model Comparison
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Performance Benchmarks & Metrics")

    if not available_models:
        st.info("Results not yet available. Run evaluate.py first.")
    else:
        start_d, end_d = date_range
        recalc_rows = []

        for m in available_models:
            df_m = preds_dict[m]
            sub = df_m[(df_m["date"].dt.date >= start_d) & (df_m["date"].dt.date <= end_d)]
            if len(sub) > 0:
                y = sub["y_true"].values
                p = sub["y_pred"].values
                recalc_rows.append({
                    "Model": m.upper(),
                    "Observations": len(y),
                    "RMSE": rmse(y, p),
                    "MAE": mae(y, p),
                    "QLIKE": qlike(y ** 2, p ** 2, floor=EPS),
                })

        if recalc_rows:
            recalc_df = pd.DataFrame(recalc_rows)
            best_val = recalc_df[chosen_metric].min()
            best_model = recalc_df.loc[recalc_df[chosen_metric] == best_val, "Model"].iloc[0]

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Metrics for Selected Date Range ({start_d} to {end_d})**")
                # Highlight best per metric
                st.dataframe(
                    recalc_df.style.highlight_min(subset=["RMSE", "MAE", "QLIKE"], color="#d4edda"),
                    use_container_width=True,
                )
            with col2:
                st.metric(f"Best Model by {chosen_metric}", best_model, f"{best_val:.4f}")

            # Bar Chart of chosen metric
            bar_fig = go.Figure(go.Bar(
                x=recalc_df["Model"],
                y=recalc_df[chosen_metric],
                marker_color=[MODEL_COLORS.get(m.lower(), "#333") for m in recalc_df["Model"]],
            ))
            bar_fig.update_layout(
                title=f"{chosen_metric} Comparison for Chosen Date Range",
                xaxis_title="Model",
                yaxis_title=chosen_metric,
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        # Sub-period table
        if PERIOD_METRICS_FILE.exists():
            st.markdown("---")
            st.markdown("### Regime Breakdown: COVID Stress vs. Calm Period")
            p_df = pd.read_csv(PERIOD_METRICS_FILE)
            st.dataframe(p_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Asymmetry
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Asymmetric Volatility & The Leverage Effect")

    if not PARAMS_FILE.exists():
        st.info("GARCH parameter table results/garch_params.csv not found.")
    else:
        params_df = pd.read_csv(PARAMS_FILE)
        gamma_params = params_df[params_df["param"].str.contains("gamma", na=False)]

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("#### Estimated Asymmetry Parameters (gamma)")
            st.dataframe(gamma_params, use_container_width=True)

            st.markdown("""
            **Interpretation:**
            - **EGARCH:** $\\gamma_1 < 0$ (statistically significant) indicates that negative shocks generate higher log-variance than positive shocks of equal size.
            - **GJR-GARCH:** $\\gamma_1 > 0$ indicates that negative innovations add an extra variance component, confirming the equity leverage effect.
            """)

        with col_b:
            st.markdown("#### News Impact Curves")
            if FIG04_PATH.exists():
                st.image(str(FIG04_PATH), caption="News Impact Curves for GARCH Family", use_container_width=True)
            else:
                st.info("News impact plot fig04_news_impact.png not found.")

# ---------------------------------------------------------------------------
# Tab 4: Next-Day Forecast
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Next-Day Volatility Forecast")

    if not NEXT_DAY_FILE.exists():
        st.info("results/next_day_forecast.csv not found.")
    else:
        nd_df = pd.read_csv(NEXT_DAY_FILE)
        forecast_date = nd_df["forecast_for"].iloc[0]
        avg_forecast = float(nd_df["vol_forecast"].mean())

        # Determine Risk Label based on train rv5 percentiles
        risk_label = "Medium"
        if processed_df is not None:
            train_rv5 = processed_df.loc[processed_df["split"] == "train", "rv5"].dropna()
            p33 = float(np.percentile(train_rv5, 33.33))
            p67 = float(np.percentile(train_rv5, 66.67))

            if avg_forecast < p33:
                risk_label = "Low"
            elif avg_forecast > p67:
                risk_label = "High"
            else:
                risk_label = "Medium"

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Forecast Target Date", forecast_date)
        m_col2.metric("Consensus Forecast (rv5)", f"{avg_forecast:.4f}%")
        m_col3.metric("Market Volatility Regime", risk_label)

        st.markdown("#### Model-Specific Next-Day Estimates")
        st.dataframe(nd_df, use_container_width=True)

        st.caption(
            "Forecasts are for the first trading day after dataset end (2022-01-03). "
            "For educational and research evaluation only; not financial advice."
        )

# ---------------------------------------------------------------------------
# Tab 5: About
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("About the Volatility Forecasting Project")
    st.markdown("""
    ### Project Overview
    This project benchmarks classical econometric volatility models against a sequence-based deep learning architecture on the S&P 500 index (`^GSPC`):
    - **GARCH Family:** GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1) with Student's $t$ innovations.
    - **Deep Learning:** LSTM neural network with 30-day lookback sequence learning intraday high-low ranges, volume dynamics, and returns.
    - **Benchmark:** Naive 1-day lag ($rv5_{t-1}$).

    ### Background Literature
    1. **Tian Wang:** S&P 500 daily data (2000–2021). Finds EGARCH superior among GARCH variants, with LSTM capturing broad regime shifts.
    2. **Yaohui Bai et al.:** Shanghai Composite (2015–2019). Demonstrates that multi-feature inputs (high, low, close) enable LSTMs to outperform standard GARCH on RMSE.
    3. **Cryptocurrency Volatility Studies (2020–2023):** 30-day lookback windows prove optimal, with asymmetric GJR-GARCH offering high stability during regime breaks.
    """)

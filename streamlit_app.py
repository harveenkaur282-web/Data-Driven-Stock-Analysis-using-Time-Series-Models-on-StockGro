import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
FORECAST_DIR = ROOT / "results" / "forecasts"
DELIVERABLES_DIR = ROOT / "deliverables"
TASK4_DIR = DELIVERABLES_DIR / "task4_analysis"
TASK5_DIR = DELIVERABLES_DIR / "task5_portfolio"
PORTFOLIO_CSV = TASK5_DIR / "portfolio_allocation.csv"
CORRELATION_CSV = TASK4_DIR / "correlation_matrix.csv"
TREND_CSV = TASK4_DIR / "trend_metrics.csv"
VOLATILITY_CSV = TASK4_DIR / "volatility_metrics.csv"
IMAGE_DIR = DELIVERABLES_DIR / "images"

MODEL_DISPLAY = {
    "ARIMA": "ARIMA",
    "PROPHET": "Prophet",
    "LSTM": "LSTM",
    "GRU": "GRU",
    "TRANSFORMER": "Transformer",
}


def load_portfolio_allocation() -> pd.DataFrame:
    return pd.read_csv(PORTFOLIO_CSV)


def load_trend_metrics() -> pd.DataFrame:
    return pd.read_csv(TREND_CSV)


def load_volatility_metrics() -> pd.DataFrame:
    return pd.read_csv(VOLATILITY_CSV)


def load_correlation_matrix() -> pd.DataFrame:
    raw = pd.read_csv(CORRELATION_CSV, index_col=0)
    raw.index.name = "Ticker"
    return raw


def load_forecast_data() -> dict:
    forecasts = {}
    for file in sorted(FORECAST_DIR.glob("*.csv")):
        stem = file.stem
        if "_" not in stem:
            continue
        ticker, model_key = stem.rsplit("_", 1)
        model_name = MODEL_DISPLAY.get(model_key.upper(), model_key.upper())
        df = pd.read_csv(file, parse_dates=["Date"])
        df = df.sort_values("Date")
        pred_cols = [c for c in df.columns if c.endswith("_Pred") or c.lower().endswith("pred")]
        if not pred_cols:
            continue
        pred_col = pred_cols[0]
        df = df.rename(columns={pred_col: "Predicted"})
        df["Model"] = model_name
        if ticker not in forecasts:
            forecasts[ticker] = {}
        forecasts[ticker][model_name] = df
    return forecasts


def calculate_errors(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mse = np.mean((actual - predicted) ** 2)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1e-8))) * 100
    return rmse, mape


st.set_page_config(
    page_title="StockGro Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("StockGro Interactive Forecast + Portfolio Dashboard")
st.markdown(
    "This dashboard brings together the project outputs for forecast comparison, portfolio allocation, correlation diversification, and risk/trend analysis."
)

forecasts = load_forecast_data()
portfolio_df = load_portfolio_allocation()
trend_df = load_trend_metrics()
vol_df = load_volatility_metrics()
corr_df = load_correlation_matrix()

sector_map = portfolio_df.set_index("Ticker")["Sector"].to_dict()
trend_df["Sector"] = trend_df["Ticker"].map(sector_map)
vol_df["Sector"] = vol_df["Ticker"].map(sector_map)

stock_list = sorted(forecasts.keys())
selected_ticker = st.sidebar.selectbox("Select stock for forecast comparison", stock_list)
available_models = sorted(forecasts[selected_ticker].keys())
selected_models = st.sidebar.multiselect(
    "Select forecast model(s)", available_models, default=available_models
)

show_static_images = st.sidebar.checkbox("Show provided deliverable images", value=True)

st.header("1. Forecast Plots: Actual vs Predicted Prices")
if not selected_models:
    st.warning("Please select at least one forecast model from the sidebar.")
else:
    actual = forecasts[selected_ticker][selected_models[0]]["Actual"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecasts[selected_ticker][selected_models[0]]["Date"],
            y=forecasts[selected_ticker][selected_models[0]]["Actual"],
            name="Actual",
            mode="lines",
            line=dict(color="black", width=3),
        )
    )
    for model_name in selected_models:
        forecast_df = forecasts[selected_ticker][model_name]
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Predicted"],
                name=f"{model_name} Forecast",
                mode="lines",
                line=dict(dash="dash"),
            )
        )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Series",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    errors = []
    for model_name in selected_models:
        forecast_df = forecasts[selected_ticker][model_name]
        rmse, mape = calculate_errors(forecast_df["Actual"], forecast_df["Predicted"])
        errors.append({"Model": model_name, "RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2)})
    st.subheader("Model performance on historical test data")
    st.table(pd.DataFrame(errors))

    st.markdown(
        "**Interpretation:** The chart compares actual closing prices with predicted values from the selected models. Use this to validate which method tracked market behavior best before portfolio allocation."
    )

st.markdown("---")

st.header("2. Portfolio Allocation & Sector Exposure")
col1, col2 = st.columns([2, 1])
with col1:
    fig_weight = px.bar(
        portfolio_df,
        x="Name",
        y="Weight",
        color="Sector",
        title="Portfolio Weights by Stock",
        labels={"Weight": "Portfolio Weight", "Name": "Stock"},
    )
    fig_weight.update_layout(xaxis_tickangle=-45, template="plotly_white")
    st.plotly_chart(fig_weight, use_container_width=True)

    fig_amount = px.bar(
        portfolio_df.groupby("Sector", as_index=False)["Amount"].sum(),
        x="Sector",
        y="Amount",
        title="Capital Exposure by Sector",
        text_auto="$.2s",
    )
    fig_amount.update_layout(template="plotly_white")
    st.plotly_chart(fig_amount, use_container_width=True)

with col2:
    fig_pie = px.pie(
        portfolio_df,
        names="Name",
        values="Weight",
        title="Portfolio Allocation by Holding",
        hole=0.35,
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    top_holdings = portfolio_df.sort_values("Weight", ascending=False).head(5)
    st.subheader("Top 5 Holdings")
    st.table(top_holdings[["Ticker", "Name", "Sector", "Weight", "Amount"]].assign(Weight=lambda df: df["Weight"].round(4)))

if show_static_images:
    image_path = IMAGE_DIR / "task5_portfolio_allocation.png"
    if image_path.exists():
        st.image(str(image_path), caption="Provided portfolio allocation chart", use_column_width=True)

st.markdown("---")

st.header("3. Correlation Heatmap for Diversification")
fig_corr = px.imshow(
    corr_df,
    text_auto=True,
    aspect="auto",
    zmin=-1,
    zmax=1,
    color_continuous_scale=px.colors.diverging.RdBu,
    title="Stock Return Correlation Heatmap",
)
fig_corr.update_xaxes(side="bottom")
st.plotly_chart(fig_corr, use_container_width=True)
st.markdown(
    "This heatmap shows return correlations across the stock universe. Lower correlations suggest better diversification when constructing the portfolio."
)
if show_static_images:
    corr_image_path = IMAGE_DIR / "task4_correlation_matrix.png"
    if corr_image_path.exists():
        st.image(str(corr_image_path), caption="Provided correlation heatmap image", use_column_width=True)

st.markdown("---")

st.header("4. Trend & Volatility Analysis")
trend_combined = trend_df.merge(vol_df, on=["Ticker", "Sector"], how="left")
trend_combined = trend_combined.sort_values("Trend_Strength", ascending=False)

col3, col4 = st.columns(2)
with col3:
    fig_trend = px.bar(
        trend_combined,
        x="Ticker",
        y="Trend_Strength",
        color="Trend_Direction",
        title="Trend Strength by Stock",
        labels={"Trend_Strength": "Trend Strength", "Ticker": "Stock"},
    )
    fig_trend.update_layout(template="plotly_white")
    st.plotly_chart(fig_trend, use_container_width=True)

    fig_vol = px.bar(
        trend_combined,
        x="Ticker",
        y="Rolling_Vol_Ann",
        color="Sector",
        title="Annualized Rolling Volatility by Stock",
        labels={"Rolling_Vol_Ann": "Annualized Volatility (%)"},
    )
    fig_vol.update_layout(template="plotly_white")
    st.plotly_chart(fig_vol, use_container_width=True)

with col4:
    fig_scatter = px.scatter(
        trend_combined,
        x="Trend_Strength",
        y="Rolling_Vol_Ann",
        color="Sector",
        size="Sharpe_Approx",
        hover_name="Ticker",
        title="Trend Strength vs Volatility",
        labels={
            "Trend_Strength": "Trend Strength",
            "Rolling_Vol_Ann": "Annualized Volatility (%)",
            "Sharpe_Approx": "Approx. Sharpe",
        },
    )
    fig_scatter.update_layout(template="plotly_white")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Trend and volatility summary")
    st.dataframe(
        trend_combined[
            ["Ticker", "Sector", "Trend_Strength", "Trend_Direction", "Rolling_Vol_Ann", "GARCH_FcstVol", "Sharpe_Approx"]
        ].round(4),
        use_container_width=True,
    )

st.markdown(
    "These charts demonstrate why the portfolio mixes stocks with strong trend signals while keeping risk exposure balanced by volatility and correlation."
)

st.markdown("---")

st.header("5. Strategy Rationale")
st.markdown(
    "- Forecast plots compare actual vs predicted prices from each model, supporting model selection and risk control.\n"
    "- The portfolio allocation chart shows the chosen weights by stock and the resulting sector exposure.\n"
    "- The correlation heatmap is used to identify diversification benefits and reduce concentration risk.\n"
    "- Trend strength and volatility metrics help prioritize stocks with upward momentum while limiting exposure to higher-risk names."
)

st.markdown("### How to use this app")
st.markdown(
    "1. Select a stock ticker and the forecasting models to compare.\n"
    "2. Review portfolio weights and sector exposure.\n"
    "3. Examine the correlation heatmap to understand diversification.\n"
    "4. Use trend strength and volatility results to validate why each stock is included."
)

st.markdown("### Data sources included")
st.write(
    "`results/forecasts` for model forecasts, `deliverables/task5_portfolio/portfolio_allocation.csv` for allocation, "
    "`deliverables/task4_analysis/correlation_matrix.csv` for correlation, and `deliverables/task4_analysis/trend_metrics.csv` / `volatility_metrics.csv` for trend and volatility analysis."
)

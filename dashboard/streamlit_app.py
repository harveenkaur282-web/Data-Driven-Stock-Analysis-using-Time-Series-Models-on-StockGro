import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

st.set_page_config(
    page_title="TSA 2026 | Portfolio Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #00d4ff;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.portfolio.evaluation import get_current_performance, load_holdings
from config import STOCK_UNIVERSE, SECTOR_MAP

def load_allocation_data():
    path = ROOT / "deliverables" / "task5_portfolio" / "portfolio_allocation.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

def main():
    st.sidebar.title("💎 TSA 2026")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox("Navigation", [
        "🏠 Dashboard Overview",
        "📊 Model Forecasts",
        "💼 Portfolio Strategy",
        "🎯 Performance Tracking (Live)"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.info("Capstone Project: Time Series Stock Analysis 2026")

    if page == "🏠 Dashboard Overview":
        st.title("🚀 Portfolio Overview")
        st.markdown("Welcome to your automated stock analysis hub. Here we combine **Time Series AI** with **Strategic Capital Allocation**.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Strategy", "Multi-Model Ensemble")
        with col2:
            st.metric("Stocks Under Watch", f"{len(STOCK_UNIVERSE)}")
        with col3:
            st.metric("Total Capital", "₹10,00,000")

        alloc_df = load_allocation_data()
        if alloc_df is not None:
            st.subheader("Recommended Portfolio Structure")
            fig = px.pie(alloc_df, values='Amount', names='Ticker', hole=.4, 
                         color_discrete_sequence=px.colors.sequential.RdBu,
                         title="Capital Allocation by Ticker")
            st.plotly_chart(fig, use_container_width=True)

    elif page == "🎯 Performance Tracking (Live)":
        st.title("📈 Live Performance Tracking")
        st.markdown("> **Task 8**: Comparing Model Predictions vs. Actual Market Outcomes.")
        
        perf_df, invested, current = get_current_performance()
        
        if perf_df.empty:
            st.warning("No holdings found. Please ensure data/holdings.csv is populated.")
        else:
            pnl = current - invested
            pnl_pct = (pnl / invested) * 100

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Invested", f"₹{invested:,.2f}")
            m2.metric("Current Value", f"₹{current:,.2f}", f"{pnl:,.2f}")
            m3.metric("Total ROI (%)", f"{pnl_pct:.2f}%")
            
            st.markdown("---")
            st.subheader("Holdings Detail")
            st.dataframe(perf_df.style.format({
                "Invested": "₹{:.2f}",
                "CurrentValue": "₹{:.2f}",
                "PnL": "₹{:.2f}",
                "ROI%": "{:.2f}%"
            }).background_gradient(subset=["PnL", "ROI%"], cmap="RdYlGn", vmin=-5, vmax=5), use_container_width=True)

            st.markdown("---")
            st.subheader("🔍 Gap & Sentiment Analysis")
            st.markdown("Understanding the difference between the **Overnight Gap** and **Intraday Movement**.")
            
            gap_data = []
            for _, row in perf_df.iterrows():
                ticker = row["Ticker"]
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        open_p = hist["Open"].iloc[-1]
                        prev_close = stock.history(period="5d")["Close"].iloc[-2] # Simplified
                        gap = ((open_p - prev_close) / prev_close) * 100
                        intraday = ((row["CurrentPrice"] - open_p) / open_p) * 100
                        
                        gap_data.append({
                            "Ticker": ticker,
                            "Overnight Gap (%)": round(gap, 2),
                            "Intraday Move (%)": round(intraday, 2),
                            "Sentiment": "Bullish Recovery" if intraday > 0 and gap < 0 else "Strong Trend" if intraday > 0 and gap > 0 else "Fading"
                        })
                except:
                    continue
            
            if gap_data:
                gap_df = pd.DataFrame(gap_data)
                col_a, col_b = st.columns(2)
                with col_a:
                    fig_gap = px.bar(gap_df, x='Ticker', y='Overnight Gap (%)', title="Overnight Gaps (%)", color_discrete_sequence=['#ff4b4b'])
                    st.plotly_chart(fig_gap, use_container_width=True)
                with col_b:
                    fig_intra = px.bar(gap_df, x='Ticker', y='Intraday Move (%)', title="Intraday Performance (%)", color_discrete_sequence=['#00d4ff'])
                    st.plotly_chart(fig_intra, use_container_width=True)
                
                st.table(gap_df)

    elif page == "💼 Portfolio Strategy":
        st.title("💼 Allocation Strategy")
        alloc_df = load_allocation_data()
        if alloc_df is not None:
            st.subheader("Task 5 Strategy Breakdown")
            st.write("Our allocation combines **Forecast Returns (40%)**, **Volatility Inverse (20%)**, **Correlation (20%)**, and **Sector Momentum (20%)**.")
            st.dataframe(alloc_df, use_container_width=True)
            
            st.subheader("Sector Exposure")
            sector_dist = alloc_df.groupby('Sector')['Amount'].sum().reset_index()
            fig_sec = px.bar(sector_dist, x='Sector', y='Amount', color='Sector', title="Capital by Sector")
            st.plotly_chart(fig_sec, use_container_width=True)

    else:
        st.title("📊 Model Forecasts")
        st.info("Select a stock to view detailed forecasts from ARIMA, Prophet, and LSTM.")
        ticker = st.selectbox("Select Ticker", list(STOCK_UNIVERSE.keys()))
        
        st.image("https://via.placeholder.com/800x400.png?text=Interactive+Forecast+Chart+Coming+Soon", use_container_width=True)

if __name__ == "__main__":
    main()

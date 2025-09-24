# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from backtesting_engine import run_full_backtest, calculate_performance_metrics

# --- UI CONFIGURATION ---
st.set_page_config(page_title="AI Trading Dashboard", layout="wide")
st.title("🤖 AI-Powered Dynamic Trading Strategy Dashboard")

# --- SIDEBAR (CONTROLS) ---
st.sidebar.header("Simulation Controls")

start_button = st.sidebar.button("🚀 Run Backtest")
initial_capital = st.sidebar.number_input("Initial Capital (₹)", value=100000, step=10000)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

st.sidebar.header("Asset Universe")
default_stocks = "RELIANCE.NS, TCS.NS, HDFCBANK.NS, ICICIBANK.NS, INFY.NS, HINDUNILVR.NS, ITC.NS, BHARTIARTL.NS, LT.NS, BAJFINANCE.NS"
stock_tickers_input = st.sidebar.text_area("Enter Stock Tickers (comma-separated)", default_stocks, height=150)

st.sidebar.header("AI Risk Tuning")
risk_off_slider = st.sidebar.slider("Defense Threshold (% Drawdown to Trigger)", 1, 20, 8, help="Lower value means more sensitive.")
risk_on_slider = st.sidebar.slider("Attack Threshold (% Upward Move to Trigger)", 1, 20, 5, help="Lower value means more aggressive.")

# --- MAIN CONTENT AREA ---
if start_button:
    with st.spinner("Running simulation... This may take a moment."):
        stock_list = [ticker.strip().upper() for ticker in stock_tickers_input.split(',')]
        
        results_df, trade_log = run_full_backtest(
            capital=initial_capital,
            start_date=start_date,
            end_date=end_date,
            stock_list=stock_list,
            risk_off_pct=risk_off_slider,
            risk_on_pct=risk_on_slider
        )

    if results_df.empty:
        st.error("Could not fetch data or run backtest. Please check tickers and date range.")
    else:
        st.success("Simulation Complete!")

        # Calculate metrics for all strategies
        bh_metrics = calculate_performance_metrics(results_df['buy_and_hold_cumulative'], initial_capital)
        sh_metrics = calculate_performance_metrics(results_df['safe_hedge_cumulative'], initial_capital)
        ds_metrics = calculate_performance_metrics(results_df['dynamic_strat_cumulative'], initial_capital)

        # Create the tabs for displaying results
        tab1, tab2, tab3 = st.tabs(["📈 Dashboard Summary", "📉 Performance Chart", "📜 Trade Log"])

        with tab1:
            st.header("Strategy Performance Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("1. Buy and Hold (Nifty 50)")
                for key, val in bh_metrics.items(): st.metric(key, val)
            with col2:
                st.subheader("2. Safe Hedge Strategy")
                for key, val in sh_metrics.items(): st.metric(key, val)
            with col3:
                st.subheader("3. Dynamic Risk Dial")
                for key, val in ds_metrics.items(): st.metric(key, val)

        with tab2:
            st.header("Portfolio Growth Over Time")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(results_df.index, results_df['buy_and_hold_cumulative'] * initial_capital, label='1. Buy and Hold')
            ax.plot(results_df.index, results_df['safe_hedge_cumulative'] * initial_capital, label='2. Safe Hedge')
            ax.plot(results_df.index, results_df['dynamic_strat_cumulative'] * initial_capital, label='3. Dynamic Risk Dial', lw=2.5)
            ax.set_title('Strategy Comparison')
            ax.set_ylabel('Portfolio Value (₹)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        with tab3:
            st.header("Dynamic Strategy Trade Log")
            st.dataframe(trade_log)

else:
    st.info("Adjust the parameters on the left and click 'Run Backtest' to start.")
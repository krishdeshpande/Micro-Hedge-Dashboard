# backtesting_engine.py (Definitive Fix v7 - NumPy Core)

import yfinance as yf
import pandas as pd
import numpy as np

def get_data(tickers, start, end):
    """
    Downloads data for each ticker individually to be more robust against failures.
    """
    print(f"Downloading data for: {tickers}")
    all_data = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if not data.empty:
                all_data.append(data[['Close']].rename(columns={'Close': ticker}))
            else:
                print(f"Warning: No data returned for {ticker}. Skipping.")
        except Exception as e:
            print(f"Warning: Could not download {ticker}. Error: {e}. Skipping.")
    if not all_data: return pd.DataFrame()
    combined_df = pd.concat(all_data, axis=1).ffill()
    return combined_df

def calculate_performance_metrics(cumulative_returns, initial_capital):
    """Calculates key performance metrics."""
    if cumulative_returns.empty or pd.isna(cumulative_returns.iloc[-1]): return {}
    total_return = (cumulative_returns.iloc[-1] - 1) * 100
    daily_returns = cumulative_returns.pct_change().dropna()
    if daily_returns.std() == 0: return {"Total Return (%)": f"{total_return:.2f}"}
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    return {
        "Final Value (₹)": f"{initial_capital * cumulative_returns.iloc[-1]:,.2f}",
        "Total Return (%)": f"{total_return:.2f}",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown (%)": f"{max_drawdown:.2f}",
    }

def run_full_backtest(capital, start_date, end_date, stock_list, risk_off_pct, risk_on_pct):
    """The main backtesting engine for all three strategies."""
    benchmark_index = "^NSEI"
    neutral_etf = "NIFTYBEES.NS"
    safe_haven_etf = "GOLDBEES.NS"
    essential_tickers = [benchmark_index, neutral_etf, safe_haven_etf]
    all_tickers = list(set(essential_tickers + stock_list))
    
    price_data = get_data(all_tickers, start_date, end_date)
    price_data.dropna(axis='columns', how='all', inplace=True)

    for ticker in essential_tickers:
        if ticker not in price_data.columns:
            print(f"CRITICAL ERROR: Failed to fetch valid data for essential ticker: {ticker}.")
            return pd.DataFrame(), pd.DataFrame()

    valid_stock_list = [t for t in stock_list if t in price_data.columns]
    if len(valid_stock_list) < 2:
        print(f"WARNING: Not enough valid stocks survived data cleaning. Need at least 2.")
        return pd.DataFrame(), pd.DataFrame()

    returns = price_data.pct_change()

    # --- SIGNAL GENERATION ---
    rolling_max = price_data[benchmark_index].rolling(window=20).max()
    drawdown = (price_data[benchmark_index] / rolling_max) - 1
    rolling_min = price_data[benchmark_index].rolling(window=20).min()
    upward_move = (price_data[benchmark_index] / rolling_min) - 1
    
    signal_risk_off = (drawdown < (-risk_off_pct / 100)).shift(1).fillna(False)
    signal_risk_on = ((upward_move > (risk_on_pct / 100)) & (drawdown > (-risk_off_pct / 100))).shift(1).fillna(False)
    
    # --- STRATEGY SIMULATION (RE-ENGINEERED WITH NUMPY CORE) ---
    
    # Strategy 1: Buy and Hold
    returns['buy_and_hold'] = returns[benchmark_index]

    # Strategy 2: Safe Hedge (using pure numpy arrays to bypass pandas bugs)
    returns['safe_hedge'] = np.where(
        signal_risk_off.values,
        returns[safe_haven_etf].values,
        returns[neutral_etf].values
    )

    # Strategy 3: Dynamic Risk Dial
    momentum = price_data[valid_stock_list].pct_change(60).shift(1)
    top_stocks_series = momentum.apply(lambda row: row.nlargest(2).index.tolist(), axis=1)
    # This calculation MUST happen before dropping NaNs from the returns dataframe
    top_stock_returns = returns.apply(lambda row: row[top_stocks_series.loc[row.name]].mean(), axis=1)

    # Now drop NaNs from the main returns dataframe
    returns.dropna(inplace=True)
    
    # --- THIS IS THE CORRECTED SECTION ---
    # Re-align signals and top_stock_returns with the now smaller returns dataframe using .loc
    signal_risk_off = signal_risk_off.loc[returns.index]
    signal_risk_on = signal_risk_on.loc[returns.index]
    top_stock_returns = top_stock_returns.loc[returns.index]

    returns['dynamic_strat'] = np.select(
        [signal_risk_on.values, signal_risk_off.values],
        [top_stock_returns.values, returns[safe_haven_etf].values],
        default=returns[neutral_etf].values
    )
    
    # --- TRADE LOG GENERATION ---
    log_data = pd.DataFrame(index=returns.index)
    log_data['Mode'] = np.select([signal_risk_on, signal_risk_off], ["Attack", "Defense"], default="Neutral")
    log_data['Assets'] = log_data['Mode'].apply(lambda x: neutral_etf)
    log_data.loc[log_data['Mode'] == 'Defense', 'Assets'] = safe_haven_etf
    log_data.loc[log_data['Mode'] == 'Attack', 'Assets'] = top_stocks_series[returns.index]

    log_data['prev_mode'] = log_data['Mode'].shift(1)
    trade_log_df = log_data[log_data['Mode'] != log_data['prev_mode']].copy()
    trade_log_df.rename(columns={'Assets': 'New Assets'}, inplace=True)
    trade_log_df['Action'] = 'ENTER'
    trade_log_df = trade_log_df[['Mode', 'Action', 'New Assets']]
    trade_log_df.reset_index(inplace=True)
    trade_log_df.rename(columns={'index':'Date'}, inplace=True)

    # --- FINAL CUMULATIVE CALCULATION ---
    results = pd.DataFrame(index=returns.index)
    results['buy_and_hold_cumulative'] = (1 + returns['buy_and_hold']).cumprod()
    results['safe_hedge_cumulative'] = (1 + returns['safe_hedge']).cumprod()
    results['dynamic_strat_cumulative'] = (1 + returns['dynamic_strat']).cumprod()
    
    return results, trade_log_df


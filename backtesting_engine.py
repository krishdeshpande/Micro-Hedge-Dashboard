# backtesting_engine.py (Definitive Fix)

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
            
    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, axis=1)
    combined_df.ffill(inplace=True) 
    return combined_df

def calculate_performance_metrics(cumulative_returns, initial_capital):
    """Calculates key performance metrics."""
    if cumulative_returns.empty or pd.isna(cumulative_returns.iloc[-1]):
        return {}
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
    
    # --- DEFINITIVE FIX: REWRITTEN VALIDATION LOGIC ---
    # 1. Proactively drop any columns that are entirely empty.
    price_data.dropna(axis='columns', how='all', inplace=True)

    # 2. Now, simply check if the essential tickers still exist as columns. This avoids the ValueError.
    for ticker in essential_tickers:
        if ticker not in price_data.columns:
            print(f"CRITICAL ERROR: Failed to fetch valid data for essential ticker: {ticker}.")
            return pd.DataFrame(), pd.DataFrame()

    # 3. Re-create the valid stock list from the cleaned data.
    valid_stock_list = [t for t in stock_list if t in price_data.columns]
    if len(valid_stock_list) < 2:
        print(f"WARNING: Not enough valid stocks survived the data cleaning. Need at least 2.")
        return pd.DataFrame(), pd.DataFrame()

    returns = price_data.pct_change().dropna(how='all')

    rolling_max = price_data[benchmark_index].rolling(window=20).max()
    drawdown = (price_data[benchmark_index] / rolling_max) - 1
    rolling_min = price_data[benchmark_index].rolling(window=20).min()
    upward_move = (price_data[benchmark_index] / rolling_min) - 1

    returns['signal_risk_off'] = (drawdown < (-risk_off_pct / 100)).shift(1).fillna(False)
    returns['signal_risk_on'] = ((upward_move > (risk_on_pct / 100)) & (drawdown > (-risk_off_pct / 100))).shift(1).fillna(False)
    
    returns['buy_and_hold'] = returns[benchmark_index]
    
    # --- DEFINITIVE FIX v2: REPLACING np.where with PANDAS .loc ---
    returns['safe_hedge'] = returns[neutral_etf]
    hedge_days = returns['signal_risk_off'] == True
    returns.loc[hedge_days, 'safe_hedge'] = returns.loc[hedge_days, safe_haven_etf]

    momentum = price_data[valid_stock_list].pct_change(60).shift(1)
    top_stocks_series = momentum.apply(lambda row: row.nlargest(2).index.tolist(), axis=1)
    top_stock_returns = returns.apply(lambda row: row[top_stocks_series.loc[row.name]].mean(), axis=1)

    # --- DEFINITIVE FIX v3: REBUILDING DYNAMIC STRATEGY WITH .loc ---
    returns['dynamic_strat'] = returns[neutral_etf]
    risk_off_days = returns['signal_risk_off'] == True
    risk_on_days = returns['signal_risk_on'] == True
    returns.loc[risk_off_days, 'dynamic_strat'] = returns.loc[risk_off_days, safe_haven_etf]
    returns.loc[risk_on_days, 'dynamic_strat'] = top_stock_returns[risk_on_days]
    
    # --- TRADE LOG GENERATION ---
    mode_conditions = [returns['signal_risk_on'], returns['signal_risk_off']]
    mode_choices = ["Attack", "Defense"]
    log_data = pd.DataFrame(index=returns.index)
    log_data['Mode'] = np.select(mode_conditions, mode_choices, default="Neutral")
    log_data['Assets'] = log_data['Mode'].apply(lambda x: neutral_etf)
    log_data.loc[log_data['Mode'] == 'Defense', 'Assets'] = safe_haven_etf
    log_data.loc[log_data['Mode'] == 'Attack', 'Assets'] = top_stocks_series

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


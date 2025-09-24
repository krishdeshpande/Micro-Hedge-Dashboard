# backtesting_engine.py

import yfinance as yf
import pandas as pd
import numpy as np

def get_data(tickers, start, end):
    """Downloads and cleans historical price data."""
    print(f"Downloading data for: {tickers}")
    try:
        data = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']
        data.ffill(inplace=True) # Forward-fill missing values
        data.dropna(axis=1, how='any', inplace=True) # Drop any tickers with no data
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()

def calculate_performance_metrics(cumulative_returns, initial_capital):
    """Calculates key performance metrics."""
    if cumulative_returns.empty:
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
    
    # --- 1. DATA PREPARATION ---
    benchmark_index = "^NSEI"
    neutral_etf = "NIFTYBEES.NS"
    safe_haven_etf = "GOLDBEES.NS"
    all_tickers = list(set([benchmark_index, neutral_etf, safe_haven_etf] + stock_list))
    
    price_data = get_data(all_tickers, start_date, end_date)
    if price_data.empty or benchmark_index not in price_data.columns:
        return pd.DataFrame(), pd.DataFrame()

    returns = price_data.pct_change().dropna()

    # --- 2. SIGNAL GENERATION ---
    rolling_max = price_data[benchmark_index].rolling(window=20).max()
    drawdown = (price_data[benchmark_index] / rolling_max) - 1
    rolling_min = price_data[benchmark_index].rolling(window=20).min()
    upward_move = (price_data[benchmark_index] / rolling_min) - 1

    returns['signal_risk_off'] = (drawdown < (-risk_off_pct / 100)).shift(1).fillna(False)
    returns['signal_risk_on'] = ((upward_move > (risk_on_pct / 100)) & (drawdown > (-risk_off_pct / 100))).shift(1).fillna(False)
    
    # --- 3. STRATEGY SIMULATION ---
    # Strategy 1: Buy and Hold
    returns['buy_and_hold'] = returns[benchmark_index]

    # Strategy 2: Safe Hedge
    returns['safe_hedge'] = np.where(returns['signal_risk_off'], returns[safe_haven_etf], returns[neutral_etf])

    # Strategy 3: Dynamic Risk Dial
    # Find top momentum stocks for Risk-On days
    momentum = price_data[stock_list].pct_change(60).shift(1) # 3-month momentum, shifted
    top_stocks = momentum.apply(lambda row: row.nlargest(2).index.tolist(), axis=1)

    dynamic_returns = []
    trade_log = []
    current_mode = "Neutral"

    for i in range(len(returns)):
        row = returns.iloc[i]
        new_mode = "Neutral"
        daily_return = row[neutral_etf]
        assets = [neutral_etf]
        
        if row['signal_risk_on']:
            new_mode = "Attack"
            selected_stocks = top_stocks.iloc[i]
            daily_return = row[selected_stocks].mean()
            assets = selected_stocks
        elif row['signal_risk_off']:
            new_mode = "Defense"
            daily_return = row[safe_haven_etf]
            assets = [safe_haven_etf]
        
        dynamic_returns.append(daily_return)
        
        if new_mode != current_mode:
            trade_log.append({
                "Date": returns.index[i].date(),
                "Mode": new_mode,
                "Action": "ENTER",
                "Assets": ", ".join(assets)
            })
            current_mode = new_mode
            
    returns['dynamic_strat'] = dynamic_returns
    
    # --- 4. CALCULATE CUMULATIVE RETURNS ---
    results = pd.DataFrame(index=returns.index)
    results['buy_and_hold_cumulative'] = (1 + returns['buy_and_hold']).cumprod()
    results['safe_hedge_cumulative'] = (1 + returns['safe_hedge']).cumprod()
    results['dynamic_strat_cumulative'] = (1 + returns['dynamic_strat']).cumprod()
    
    trade_log_df = pd.DataFrame(trade_log)
    return results, trade_log_df
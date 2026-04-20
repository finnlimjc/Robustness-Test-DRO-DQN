from datetime import date
import numpy as np
import pandas as pd

ANNUAL_WINDOW = 252

def compute_wealth(log_returns:np.ndarray) -> np.ndarray:
    wealth = np.exp(np.concatenate([[0.0], np.cumsum(log_returns)])) # Start wealth = exp(0) = 1.0
    return wealth

def compute_drawdown(wealth:np.ndarray) -> np.ndarray:
    running_peak = np.maximum.accumulate(wealth)
    return (wealth - running_peak) / running_peak

def compute_mdd(wealth:np.ndarray) -> float:
    dd = compute_drawdown(wealth)
    return float(dd.min())

def compute_sharpe(log_returns:np.ndarray) -> float:
    mean_r = log_returns.mean()
    std_r = log_returns.std()
    annual_r = mean_r * ANNUAL_WINDOW
    annual_std = std_r * np.sqrt(ANNUAL_WINDOW)
    sharpe = float(annual_r / annual_std) if annual_std > 0 else np.nan
    return sharpe

def compute_sortino(log_returns:np.ndarray) -> float:
    mean_r = log_returns.mean()
    neg_r = log_returns[log_returns < 0]
    annual_r = mean_r * ANNUAL_WINDOW
    annual_downside_deviation = float(np.sqrt((neg_r ** 2).mean()) * np.sqrt(ANNUAL_WINDOW)) if len(neg_r) > 0 else 0.0
    sortino = float(annual_r / annual_downside_deviation) if annual_downside_deviation > 0 else np.nan
    return sortino

def compute_cagr(log_returns:np.ndarray) -> float:
    wealth = compute_wealth(log_returns) # Start wealth = 1.0 at time 0
    T = len(log_returns)
    final_value = float(wealth[-1])
    reciprocal_total_years = ANNUAL_WINDOW/T # 1/N where N is T/ANNUAL_WINDOW
    cagr = float(final_value**reciprocal_total_years - 1) if T > 0 else np.nan
    return cagr

def compute_calmar(log_returns:np.ndarray) -> float:
    cagr = compute_cagr(log_returns)
    wealth = compute_wealth(log_returns)
    mdd = compute_mdd(wealth)
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    return calmar

def compute_rolling_sharpe(log_returns:np.ndarray, window:int=252) -> np.ndarray:
    """Annualised rolling Sharpe (rf=0). NaN for first window-1 points."""
    s = pd.Series(log_returns)
    roll_mean = s.rolling(window).mean()
    annualized_mean = roll_mean * ANNUAL_WINDOW
    
    roll_std = s.rolling(window).std()
    annualized_std = roll_std * np.sqrt(ANNUAL_WINDOW)
    
    return (annualized_mean / annualized_std).values

def compute_rolling_sortino(log_returns:np.ndarray, window:int=252) -> np.ndarray:
    """Annualised rolling Sortino (rf=0). NaN for first window-1 points."""
    s = pd.Series(log_returns)
    roll_mean = s.rolling(window).mean()
    annualized_mean = roll_mean * ANNUAL_WINDOW
    
    neg_sq = s.clip(upper=0) ** 2
    rolling_avg = neg_sq.rolling(window).mean()
    roll_downside_dev = np.sqrt(rolling_avg)
    annualized_downside_dev = roll_downside_dev * np.sqrt(ANNUAL_WINDOW)
    
    return (annualized_mean / annualized_downside_dev).values

def compute_metrics(log_returns:np.ndarray) -> dict:
    """Compute summary performance metrics from log returns and wealth series."""
    wealth = compute_wealth(log_returns)
    cagr = compute_cagr(log_returns)
    sharpe = compute_sharpe(log_returns)
    sortino = compute_sortino(log_returns)
    
    dd = compute_drawdown(wealth)
    mdd = float(dd.min())
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    
    return {
        'Ending Portfolio Value': wealth[-1],
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'MDD': mdd,
        'Calmar Ratio': calmar
    }

def compute_top_drawdowns(wealth:np.ndarray, wealth_dates:pd.DatetimeIndex, n:int=5) -> pd.DataFrame:
    """Identify the top n largest drawdown episodes from the wealth series."""
    dd = compute_drawdown(wealth)
    in_dd = (dd < 0).astype(int)
    
    transitions = np.diff(in_dd, prepend=0)
    starts = np.where(transitions == 1)[0] #0 → 1 or start of drawdown
    recoveries = np.where(transitions == -1)[0] #1 → 0 or end of drawdown
    
    records = []
    for drawdown_episode, s in enumerate(starts):
        peak_idx = max(0, s - 1) #-1 due to prepend
        recovered = drawdown_episode < len(recoveries)
        end_idx = recoveries[drawdown_episode] if recovered else len(wealth) #Ongoing drawdown
        trough_idx = s + np.argmin(wealth[s:end_idx])
        
        records.append({
            'Start': wealth_dates[peak_idx].date(),
            'Trough': wealth_dates[trough_idx].date(),
            'Recovery Date': wealth_dates[recoveries[drawdown_episode]].date() if recovered else None,
            'MDD (%)': dd[trough_idx] * 100,
            'Duration (days)': trough_idx - peak_idx,
            'Recovery (days)': recoveries[drawdown_episode] - trough_idx if recovered else None,
        })
    
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No drawdown episodes found in the wealth series. Check if the input log returns are valid.")
    
    return df.sort_values('MDD (%)').head(n).reset_index(drop=True)

def filter_results(result:dict, view_start:date, view_end:date) -> dict:
    """Slice backtest result arrays to the view date range."""
    dates = result['dates']
    vs = pd.Timestamp(view_start)
    ve = pd.Timestamp(view_end)
    
    mask = (dates >= vs) & (dates <= ve)
    wealth_dates = pd.DatetimeIndex([dates[0] - pd.Timedelta(days=1)]).append(dates) #add an additional date for the initial wealth = 1.0 point
    wealth_mask = (wealth_dates >= vs) & (wealth_dates <= ve)
    
    return {
        'dates': dates[mask],
        'agent_log_returns': result['agent_log_returns'][mask],
        'benchmark_log_returns': result['benchmark_log_returns'][mask],
        'positions': result['positions'][mask],
        'agent_wealth': result['agent_wealth'][wealth_mask],
        'benchmark_wealth': result['benchmark_wealth'][wealth_mask],
        'wealth_dates': wealth_dates[wealth_mask],
    }

def format_metric(metric:str, value:float) -> str:
    """Format a metric value for display."""
    if value is None or np.isnan(value):
        return 'N/A'
    if metric == 'Ending Portfolio Value':
        return f'{value:.4f}'
    if metric in ('CAGR', 'MDD'):
        return f'{value * 100:.2f}%'
    if metric in ('Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'):
        return f'{value:.3f}'
    return str(value)

def metrics_df(agent_metrics:dict, bm_metrics:dict) -> pd.DataFrame:
    """Build a comparison DataFrame of agent vs benchmark metrics."""
    rows = []
    for k in agent_metrics:
        rows.append({
            'Metric': k,
            'Agent': format_metric(k, agent_metrics[k]),
            'Buy & Hold': format_metric(k, bm_metrics[k]),
        })
    return pd.DataFrame(rows).set_index('Metric')
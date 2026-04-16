import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ANNUAL_WINDOW=252

def _compute_wealth(log_returns:np.ndarray) -> np.ndarray:
    return np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))

def _compute_drawdown(wealth:np.ndarray) -> np.ndarray:
    running_peak = np.maximum.accumulate(wealth)
    return (wealth - running_peak) / running_peak

def _compute_rolling_sharpe(log_returns:np.ndarray, window:int=252) -> np.ndarray:
    """Annualised rolling Sharpe (rf=0). NaN for first window-1 points."""
    s = pd.Series(log_returns)
    
    roll_mean = s.rolling(window).mean()
    annualized_mean = roll_mean * ANNUAL_WINDOW
    roll_std = s.rolling(window).std()
    annualized_std = roll_std * np.sqrt(ANNUAL_WINDOW)
    
    return (annualized_mean / annualized_std).values

def _compute_rolling_sortino(log_returns:np.ndarray, window:int=252) -> np.ndarray:
    """Annualised rolling Sortino (rf=0). NaN for first window-1 points."""
    s = pd.Series(log_returns)
    
    roll_mean = s.rolling(window).mean()
    annualized_mean = roll_mean * ANNUAL_WINDOW

    neg_sq = s.clip(upper=0)**2
    rolling_avg = neg_sq.rolling(window).mean()
    roll_downside_dev = np.sqrt(rolling_avg)
    annualized_downside_dev = roll_downside_dev * np.sqrt(ANNUAL_WINDOW)
    
    return (annualized_mean / annualized_downside_dev).values

def _prepend_date(dates:pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Prepend one calendar day before the first date (anchor for starting wealth = 1.0)."""
    return pd.DatetimeIndex([dates[0] - pd.Timedelta(days=1)]).append(dates)

def plot_wealth(dates:pd.DatetimeIndex, agent_log_returns:np.ndarray, benchmark_log_returns:np.ndarray=None) -> plt.Figure:
    """Wealth index chart starting at 1.0 (agent vs optional benchmark)."""
    agent_wealth = _compute_wealth(agent_log_returns)
    wealth_dates = _prepend_date(dates)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wealth_dates, agent_wealth, label='Agent', linewidth=1.2)
    if benchmark_log_returns is not None:
        ax.plot(wealth_dates, _compute_wealth(benchmark_log_returns),
                label='Benchmark', linewidth=1.2, alpha=0.8)
    
    ax.set_title('Wealth Over Time')
    ax.set_ylabel('Wealth (start = 1.0)')
    ax.legend()
    plt.tight_layout()
    
    return fig

def plot_drawdown(dates:pd.DatetimeIndex, agent_log_returns:np.ndarray, benchmark_log_returns:np.ndarray=None) -> plt.Figure:
    """Drawdown from running peak."""
    agent_dd = _compute_drawdown(_compute_wealth(agent_log_returns))
    wealth_dates = _prepend_date(dates)

    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.fill_between(wealth_dates, agent_dd, 0, alpha=0.4, label='Agent')
    ax.plot(wealth_dates, agent_dd, linewidth=0.8)
    if benchmark_log_returns is not None:
        bm_dd = _compute_drawdown(_compute_wealth(benchmark_log_returns))
        ax.fill_between(wealth_dates, bm_dd, 0, alpha=0.3, label='Benchmark')
        ax.plot(wealth_dates, bm_dd, linewidth=0.8)
    
    ax.set_title('Drawdown Over Time')
    ax.set_ylabel('Drawdown')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))
    
    ax.legend()
    plt.tight_layout()
    return fig

def plot_monthly_returns(dates:pd.DatetimeIndex, agent_log_returns:np.ndarray) -> plt.Figure:
    """Violin plot of daily log returns grouped by calendar month."""
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    s = pd.Series(agent_log_returns, index=dates)
    df_plot = pd.DataFrame({'return': s.values, 'month': s.index.month})
    
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.boxplot(data=df_plot, x='month', y='return', ax=ax)
    
    ax.set_title('Daily Returns by Month')
    ax.set_ylabel('Log Return')
    ax.set_xlabel('')
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels)
    plt.tight_layout()
    
    return fig

def plot_rolling_sharpe(dates:pd.DatetimeIndex, agent_log_returns:np.ndarray, benchmark_log_returns:np.ndarray=None, window:int=252) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, _compute_rolling_sharpe(agent_log_returns, window),
            label='Agent', linewidth=1.0)
    if benchmark_log_returns is not None:
        ax.plot(dates, _compute_rolling_sharpe(benchmark_log_returns, window),
                label='Benchmark', linewidth=1.0, alpha=0.8)
    
    ax.set_title(f'Rolling {window}-Day Sharpe Ratio')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_rolling_sortino(dates:pd.DatetimeIndex, agent_log_returns:np.ndarray, benchmark_log_returns:np.ndarray=None, window:int=252) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, _compute_rolling_sortino(agent_log_returns, window),
            label='Agent', linewidth=1.0)
    if benchmark_log_returns is not None:
        ax.plot(dates, _compute_rolling_sortino(benchmark_log_returns, window),
                label='Benchmark', linewidth=1.0, alpha=0.8)
    
    ax.set_title(f'Rolling {window}-Day Sortino Ratio')
    ax.set_ylabel('Sortino Ratio')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_return_distribution(agent_log_returns:np.ndarray, benchmark_log_returns:np.ndarray=None, is_log_scale:bool=False) -> plt.Figure:
    """Histogram of daily log returns with skew and excess kurtosis annotation."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(agent_log_returns, bins=50, density=True, alpha=0.6, label='Agent')
    if benchmark_log_returns is not None:
        ax.hist(benchmark_log_returns, bins=50, density=True, alpha=0.5, label='Benchmark')
    
    if is_log_scale:
        ax.set_yscale('log')
    
    skew_val = stats.skew(agent_log_returns)
    kurt_val = stats.kurtosis(agent_log_returns)  # excess kurtosis
    textstr = f'Skew:     {skew_val:+.3f}\nKurtosis: {kurt_val:+.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.97, 0.95, textstr, transform=ax.transAxes, ha='right',
            va='top', bbox=props, fontsize=10, fontfamily='monospace')
    
    ax.set_title('Distribution of Daily Log Returns')
    ax.set_xlabel('Log Return')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_position(dates: pd.DatetimeIndex, positions:np.ndarray) -> plt.Figure:
    """Portfolio allocation weight over time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, positions, linewidth=0.7, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    
    ax.set_title('Portfolio Position Over Time')
    ax.set_ylabel('Allocation to Risky Asset')
    ax.set_ylim([-1.15, 1.15])
    plt.tight_layout()
    
    return fig
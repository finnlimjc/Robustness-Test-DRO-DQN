import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from .metrics import *

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

def _prepend_date(dates:pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Prepend one calendar day before the first date (anchor for starting wealth = 1.0)."""
    return pd.DatetimeIndex([dates[0] - pd.Timedelta(days=1)]).append(dates)

def plot_wealth(dates:pd.DatetimeIndex, agent_log_returns:np.ndarray, benchmark_log_returns:np.ndarray=None) -> plt.Figure:
    """Wealth index chart starting at 1.0 (agent vs optional benchmark)."""
    agent_wealth = compute_wealth(agent_log_returns)
    wealth_dates = _prepend_date(dates)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wealth_dates, agent_wealth, label='Agent', linewidth=1.2)
    if benchmark_log_returns is not None:
        ax.plot(wealth_dates, compute_wealth(benchmark_log_returns),
                label='Benchmark', linewidth=1.2, alpha=0.8)
    
    ax.set_title('Wealth Over Time')
    ax.set_ylabel('Wealth (start = 1.0)')
    ax.legend()
    plt.tight_layout()
    
    return fig

def plot_drawdown(dates:pd.DatetimeIndex, agent_log_returns:np.ndarray, benchmark_log_returns:np.ndarray=None) -> plt.Figure:
    """Drawdown from running peak."""
    agent_dd = compute_drawdown(compute_wealth(agent_log_returns))
    wealth_dates = _prepend_date(dates)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.fill_between(wealth_dates, agent_dd, 0, alpha=0.4, label='Agent')
    ax.plot(wealth_dates, agent_dd, linewidth=0.8)
    if benchmark_log_returns is not None:
        bm_dd = compute_drawdown(compute_wealth(benchmark_log_returns))
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
    ax.plot(dates, compute_rolling_sharpe(agent_log_returns, window),
            label='Agent', linewidth=1.0)
    if benchmark_log_returns is not None:
        ax.plot(dates, compute_rolling_sharpe(benchmark_log_returns, window),
                label='Benchmark', linewidth=1.0, alpha=0.8)
    
    ax.set_title(f'Rolling {window}-Day Sharpe Ratio')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_rolling_sortino(dates:pd.DatetimeIndex, agent_log_returns:np.ndarray, benchmark_log_returns:np.ndarray=None, window:int=252) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, compute_rolling_sortino(agent_log_returns, window),
            label='Agent', linewidth=1.0)
    if benchmark_log_returns is not None:
        ax.plot(dates, compute_rolling_sortino(benchmark_log_returns, window),
                label='Benchmark', linewidth=1.0, alpha=0.8)
    
    ax.set_title(f'Rolling {window}-Day Sortino Ratio')
    ax.set_ylabel('Sortino Ratio')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_return_distribution(agent_log_returns:np.ndarray, benchmark_log_returns:np.ndarray=None, is_log_scale:bool=False) -> plt.Figure:
    """Histogram of daily log returns with skew and excess kurtosis annotation."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(agent_log_returns, bins=50, density=True, alpha=0.6, label='Agent')
    if benchmark_log_returns is not None:
        ax.hist(benchmark_log_returns, bins=50, density=True, alpha=0.5, label='Benchmark')
    
    if is_log_scale:
        ax.set_yscale('log')
    
    skew_val = stats.skew(agent_log_returns)
    kurt_val = stats.kurtosis(agent_log_returns)
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

def plot_position(dates:pd.DatetimeIndex, positions:np.ndarray) -> plt.Figure:
    """Portfolio allocation weight over time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, positions, linewidth=0.7, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    
    ax.set_title('Portfolio Position Over Time')
    ax.set_ylabel('Allocation to Risky Asset')
    ax.set_ylim([-1.15, 1.15])
    plt.tight_layout()
    
    return fig

def plot_wealth_multi(wealth_dates:pd.DatetimeIndex, agent_mean:np.ndarray, agent_q25:np.ndarray, agent_q75:np.ndarray, bm_wealth:np.ndarray) -> plt.Figure:
    """Wealth chart for multiple runs showing mean and interquartile range."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(wealth_dates, agent_mean, label='Agent (mean)', linewidth=1.2)
    ax.fill_between(wealth_dates, agent_q25, agent_q75, alpha=0.3, label='Q25-Q75')
    ax.plot(wealth_dates, bm_wealth, label='Buy & Hold', linewidth=1.2, alpha=0.8)
    
    ax.set_title('Wealth Over Time — All Runs')
    ax.set_ylabel('Wealth (start = 1.0)')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_drawdown_multi(wealth_dates:pd.DatetimeIndex, mean_wealth:np.ndarray, bm_wealth:np.ndarray) -> plt.Figure:
    """Drawdown chart from pre-computed mean agent and benchmark wealth."""
    agent_dd = compute_drawdown(mean_wealth)
    bm_dd = compute_drawdown(bm_wealth)
    
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(wealth_dates, agent_dd, 0, alpha=0.4, label='Agent (mean)')
    ax.plot(wealth_dates, agent_dd, linewidth=0.8)
    ax.fill_between(wealth_dates, bm_dd, 0, alpha=0.3, label='Buy & Hold')
    ax.plot(wealth_dates, bm_dd, linewidth=0.8)
    
    ax.set_title('Drawdown Over Time (Mean Agent)')
    ax.set_ylabel('Drawdown')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))
    ax.legend()
    plt.tight_layout()
    return fig

def plot_metrics_bars(all_agent_metrics:list, bm_metrics:dict, nrows:int=2, ncols:int=3) -> plt.Figure:
    """Bar chart with one bar per run plus a Buy & Hold bar for each metric."""
    metric_names = list(bm_metrics.keys())
    n_runs = len(all_agent_metrics)
    labels = [f'Run {i + 1}' for i in range(n_runs)] + ['B&H']
    colors = ['steelblue'] * n_runs + ['darkorange']
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows*3.5))
    axes = axes.flatten()
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        vals = [m[metric] for m in all_agent_metrics] + [bm_metrics[metric]]
        ax.bar(labels, vals, color=colors)
        ax.set_title(metric, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.tick_params(axis='x', rotation=45)
        
    plt.suptitle('Metrics Comparison: All Runs vs Buy & Hold', fontsize=11)
    plt.tight_layout()
    return fig
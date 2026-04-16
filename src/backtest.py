import torch
import numpy as np
import pandas as pd

from src.agents.dqn import QFunc
from src.data.price_data import YahooFinance
from src.utils.config import Config

def load_ticker_data(ticker:str, start_date:str, end_date:str) -> pd.DataFrame:
    """
    Download and prepare price data for a ticker. Call this once and pass the
    result to run_backtest to avoid re-downloading for each model.
    
    Returns a DataFrame with a 'log_return' column and a date index, first date is dropped due to NaN log return.
    """
    yf = YahooFinance(symbol=ticker, start_date=start_date, end_date=end_date)
    df = yf.pipeline().dropna(subset=['log_return'])
    return df.iloc[1:]  # Drop first row with NaN log return

def load_qfunc_from_checkpoint(config_path:str, checkpoint_path:str) -> dict:
    """
    Load the Q-network and action values from a checkpoint for inference only.
    No full agent reconstruction required.
    
    Inputs:
    - config_path : path to the configuration file
    - checkpoint_path : path to the checkpoint file

    Outputs:
        A dictionary containing:
        - 'qfunc': QFunc in eval mode (on CPU)
        - 'action_values' : asset weight values
        - 'rf_rate' : risk-free rate from config
        - 'trans_cost' : transaction cost from config
    """
    c = Config()
    _, env_params, _, _, q_params, *_ = c.load_config(config_path)
    
    qfunc = QFunc(**q_params)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    qfunc.load_state_dict(checkpoint['q_network'])
    qfunc.eval()
    
    action_values = checkpoint['action_values']
    if isinstance(action_values, torch.Tensor):
        action_values = action_values.cpu().numpy()
    
    return {
        'qfunc': qfunc,
        'action_values': action_values,
        'rf_rate': env_params.get('rf_rate', 0.0),
        'trans_cost': env_params.get('trans_cost', 0.0),
    }

def run_backtest(qfunc, action_values, rf_rate:float, trans_cost:float, df:pd.DataFrame=None, ticker:str=None, start_date:str=None, end_date:str=None, hist_len:int=60) -> dict:
    """
    Run the agent greedily (no exploration) on real price data for any ticker.
    
    Pass a pre-loaded DataFrame via `df` (from load_ticker_data) to avoid re-downloading data when evaluating multiple models on the same ticker.
    If `df` is None, `ticker`, `start_date`, and `end_date` must be provided.
    
    Inputs:
    - qfunc : QFunc (nn.Module) in eval mode
    - action_values: array-like of shape (9,), weight values (e.g. linspace(-1,1,9))
    - rf_rate : annualised risk-free rate (continuous compounding)
    - trans_cost : transaction cost as fraction of absolute position change
    - df : pre-loaded DataFrame from load_ticker_data (preferred)
    - ticker : Yahoo Finance symbol — only used if df is None
    - start_date : only used if df is None
    - end_date : only used if df is None
    - hist_len : days of past returns in state (must match training, default 60)
    
    Outputs:
        A dictionary containing:
        - 'dates'                 : pd.DatetimeIndex of length T (trading days after warm-up)
        - 'agent_log_returns'     : agent log returns
        - 'benchmark_log_returns' : ticker log returns
        - 'positions'             : daily asset weight
        - 'agent_wealth'          : agent wealth over time, starting at 1.0
        - 'benchmark_wealth'      : benchmark wealth over time, starting at 1.0
    """
    if isinstance(action_values, torch.Tensor):
        action_values = action_values.cpu().numpy()
    
    if df is None:
        if ticker is None or start_date is None or end_date is None:
            raise ValueError("Provide either `df` or all of `ticker`, `start_date`, `end_date`.")
        df = load_ticker_data(ticker, start_date, end_date)
    
    # Calendar days between consecutive dates, expressed in years
    dt = df.index.to_series().diff().dt.days.div(365).values  # NaN at index 0
    
    log_return_col = df.columns.get_loc('log_return')
    
    log_wealth = 0.0
    position = 0.0
    agent_log_returns = []
    positions = []
    
    for i in range(hist_len + 1, len(df)):
        past_returns = df.iloc[i - hist_len:i, log_return_col].values
        state = np.concatenate([past_returns, [log_wealth], [position], [dt[i]]])
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        with torch.no_grad():
            act_idx = qfunc(state_tensor).argmax().item()
        asset_weight = float(action_values[act_idx])
        
        interest_return = (np.exp(rf_rate * dt[i]) - 1.0) * (1.0 - asset_weight)
        asset_return = asset_weight * (np.exp(df.iloc[i, log_return_col]) - 1.0)
        transaction_cost = trans_cost * abs(asset_weight - position)
        simple_return = max(1e-8, 1.0 + interest_return + asset_return - transaction_cost)
        log_ret = np.log(simple_return)
        
        log_wealth += log_ret
        position = asset_weight
        agent_log_returns.append(log_ret)
        positions.append(asset_weight)
    
    agent_log_returns = np.array(agent_log_returns, dtype=np.float64)
    positions = np.array(positions, dtype=np.float64)
    dates = df.index[hist_len + 1:]
    benchmark_log_returns = df.iloc[hist_len + 1:, log_return_col].values.astype(np.float64)
    
    agent_wealth = np.exp(np.concatenate([[0.0], np.cumsum(agent_log_returns)]))
    benchmark_wealth = np.exp(np.concatenate([[0.0], np.cumsum(benchmark_log_returns)]))
    
    return {
        'dates': dates,
        'agent_log_returns': agent_log_returns,
        'benchmark_log_returns': benchmark_log_returns,
        'positions': positions,
        'agent_wealth': agent_wealth,
        'benchmark_wealth': benchmark_wealth,
    }

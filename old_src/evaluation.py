import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

DATATYPE = torch.float32
HIST_LEN = 60
LOG_WEALTH_IDX = 60
POSITION_IDX = 61
DT_IDX = 62

def metrics(returns, rf=0., ann_factor=252):
    # volatility
    vol = returns.std() * np.sqrt(ann_factor)

    # sharpe ratio
    sharpe_ratio = (returns.mean() - rf) / vol * ann_factor

    # downside deviation:
    temp = np.minimum(0, returns - rf)**2
    temp_expectation = np.mean(temp)
    downside_dev = np.sqrt(temp_expectation) * np.sqrt(ann_factor)

    # Sortino ratio:
    sortino_ratio = np.mean(returns - rf) / downside_dev * ann_factor
    return vol, sharpe_ratio, downside_dev, sortino_ratio

def simulate_agent_spx(spx_df:pd.DataFrame, qfunc, action_values, int_rate=0.024, trans_cost=0.0005) -> tuple[dict[dict[str:float]], pd.DataFrame]:
    spx_df = spx_df.copy()
    spx_df = max_drawdown(spx_df, 'spx', 'spx_max_drawdown', 'spx_max_drawdown_dur')

    # renormalise spx_df
    spx_df['spx_normalised'] = spx_df['spx_normalised'] / spx_df['spx_normalised'].iloc[HIST_LEN+1]

    dt = spx_df.index.diff()
    dt = (dt.days / 365).values

    log_wealth = 0.
    position = 0.
    log_wealth_seq = [0.] # starting wealth is 1 i.e. log wealth is 0
    position_seq = [0.] # starting position is 0
    device = torch.device('cpu')

    for i in range(HIST_LEN+1, len(spx_df)): # NOTE: +1 as first log return is nan when not using start_date
        state = spx_df.iloc[i-HIST_LEN:i, spx_df.columns.get_loc('log_return')].values
        state = np.concatenate([state, np.array([log_wealth]), np.array([position]), np.array([dt[i]])])
        states = torch.tensor(state, dtype=DATATYPE, device=device)
        with torch.no_grad():
            last_q_values = qfunc(states)
        act_idx = last_q_values.argmax(dim=-1).squeeze()
        act_value = action_values[act_idx].cpu().numpy()

        delta_position = np.abs(act_value - position)
        transaction_return = -trans_cost * delta_position
        interest_return = (np.exp(int_rate * dt[i]) - 1) * (1-act_value)
        asset_return = act_value * (np.exp((spx_df.iloc[i, spx_df.columns.get_loc('log_return')]).squeeze()) - 1)
        log_return = np.log(1 + interest_return + asset_return + transaction_return)
        log_wealth = log_wealth + log_return
        position = act_value
        log_wealth_seq.append(log_wealth)
        position_seq.append(position)

    # create new columns for agent wealth
    spx_df['agent'] = np.nan # column for agent WEALTH not log wealth
    spx_df.iloc[HIST_LEN:, spx_df.columns.get_loc('agent')] = np.exp(log_wealth_seq) # NOTE: includes the starting wealth
    # create new columns for agent position
    spx_df['position'] = np.nan
    spx_df.iloc[HIST_LEN:, spx_df.columns.get_loc('position')] = position_seq
    spx_df = max_drawdown(spx_df, 'agent', 'agent_max_drawdown', 'agent_max_drawdown_dur')

    agent_returns = np.diff(log_wealth_seq)
    agent_vol, agent_sharpe, agent_downdev, agent_sortino = metrics(agent_returns)

    spx_returns = spx_df.iloc[HIST_LEN+1:, spx_df.columns.get_loc('log_return')].values
    spx_vol, spx_sharpe, spx_downdev, spx_sortino = metrics(spx_returns)
    
    results = {
        'agent': {
            'final_wealth': float(spx_df['agent'].iloc[-1]), 'max_drawdown': float(spx_df['agent_max_drawdown'].min()), 
            'sharpe': float(agent_sharpe), 'volatility': float(agent_vol), 'down_deviation': float(agent_downdev), 'sortino': float(agent_sortino)
        },
        'buy_hold': {
            'final_wealth': float(spx_df['spx_normalised'].iloc[-1]), 'max_drawdown': float(spx_df['spx_max_drawdown'].min()), 
            'sharpe': float(spx_sharpe), 'volatility': float(spx_vol), 'down_deviation': float(spx_downdev), 'sortino': float(spx_sortino)
        }
    }
    
    return (results, spx_df)

def max_drawdown(df, value_col, drawdown_col_name='max_drawdown', drawdown_dur_col_name='max_drawdown_duration', value=True, log_return=False):
    '''
    Calculates the maximum drawdown of a given value column in a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the value column and index should be ordered by date.
    value_col : str
        Name of the column containing the value to calculate the drawdown
    drawdown_col_name : str
        Name of the column to store the drawdown values
    drawdown_dur_col_name : str
        Name of the column to store the drawdown duration values
    value : bool
        If True, the value column is assumed to be a price level else it is assumed to be log returns.
    log_return : bool
        If True, the drawdown is calculated in log returns else it is calculated in absolute returns.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with max_drawdown and max_drawdown_duration columns added.
    '''

    for date in df.index:
        # calculate drawdown to future min point from current date in log returns
        if value:
            min = df.loc[date:, value_col].min()
            curr_px = df.loc[date, value_col]
            if min == curr_px:
                df.loc[date, drawdown_col_name] = np.nan
            else:
                df.loc[date, drawdown_col_name] = np.log(min / curr_px) if log_return else min / curr_px - 1
        else:
            temp_df = df.copy()
            temp_df['cum_returns'] = df[value_col].cumsum()
            min = temp_df.loc[date:, 'cum_returns'].min()
            df.loc[date, drawdown_col_name] = min - df.loc[date, 'cum_returns']

        # find date of future min point
        row = df.loc[date:,:][df.loc[date:, value_col] == min]
        # print(date.strftime('%Y-%m-%d'), curr_px, row.index[0].strftime('%Y-%m-%d'), min)
        if len(row) == 0: max_drawdown_date = np.nan
        elif len(row) >= 1: max_drawdown_date = row.index[0]
        # else: raise ValueError('Multiple min points found')

        # check that there is a max drawdown date to calculate duration
        if not max_drawdown_date == np.nan and not type(max_drawdown_date) == float:
            df.loc[date, drawdown_dur_col_name] = max_drawdown_date - date
        else:
            df.loc[date, drawdown_dur_col_name] = np.nan

    return df

def generate_graph(spx_df:pd.DataFrame, qfunc, action_values, int_rate=0.024, trans_cost=0.0005):
    spx_df = spx_df.copy()

    # renormalise spx_df
    spx_df['spx_normalised'] = spx_df['spx_normalised'] / spx_df['spx_normalised'].iloc[HIST_LEN+1]

    dt = spx_df.index.diff()
    dt = (dt.days / 365).values

    log_wealth = 0.
    position = 0.
    log_wealth_seq = [0.] # starting wealth is 1 i.e. log wealth is 0
    position_seq = [0.] # starting position is 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(HIST_LEN+1, len(spx_df)): # NOTE: +1 as first log return is nan when not using start_date
        state = spx_df.iloc[i-HIST_LEN:i, spx_df.columns.get_loc('log_return')].values
        state = np.concatenate([state, np.array([log_wealth]), np.array([position]), np.array([dt[i]])])
        states = torch.tensor(state, dtype=DATATYPE, device=device)
        with torch.no_grad():
            last_q_values = qfunc(states)
        act_idx = last_q_values.argmax(dim=-1).squeeze()
        act_value = action_values[act_idx].cpu().numpy()

        delta_position = np.abs(act_value - position)
        transaction_return = -trans_cost * delta_position
        interest_return = (np.exp(int_rate * dt[i]) - 1) * (1-act_value)
        asset_return = act_value * (np.exp((spx_df.iloc[i, spx_df.columns.get_loc('log_return')]).squeeze()) - 1)
        log_return = np.log(1 + interest_return + asset_return + transaction_return)
        log_wealth = log_wealth + log_return
        position = act_value
        log_wealth_seq.append(log_wealth)
        position_seq.append(position)

    # create new columns for agent wealth
    spx_df['agent'] = np.nan # column for agent WEALTH not log wealth
    spx_df.iloc[HIST_LEN:, spx_df.columns.get_loc('agent')] = np.exp(log_wealth_seq) # NOTE: includes the starting wealth
    
    # create new columns for agent position
    spx_df['position'] = np.nan # column for agent WEALTH not log wealth
    spx_df.iloc[HIST_LEN:, spx_df.columns.get_loc('position')] = position_seq
    #print(spx_df['position'].value_counts())

    spx_df = spx_df.iloc[HIST_LEN:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Wealth comparison (normalized)
    ax1.plot(spx_df.index, spx_df['spx_normalised'], label='Buy & Hold', alpha=0.7)
    ax1.plot(spx_df.index, spx_df['agent'], label='Agent', alpha=0.7)
    ax1.set_ylabel('Normalized Wealth')
    ax1.set_title('Agent vs Buy & Hold Wealth Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Position over time
    ax2.plot(spx_df.index, spx_df['position'], label='Position', color='blue', alpha=0.7, linewidth=0.5)
    ax2.set_ylabel('Position (Allocation)')
    ax2.set_xlabel('Date')
    ax2.set_title('Agent Position Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-1, 1])  # Position should be between 0 and 1

    plt.tight_layout()
    plt.show()

    return fig
import gymnasium as gym
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import torch

def stationary_bootstrap(returns:pd.Series, n_sims:int=1, t:int=252, avg_block_size:int=None, seed:int=None, include_noise:bool=False) -> np.array:
        rng = np.random.default_rng(seed)
        return_values = returns.values
        T = len(return_values)
        sims = np.zeros((n_sims, t))
        if avg_block_size is None:
            avg_block_size = int(np.floor(T**(1/3)))
        
        restart_prob = 1/avg_block_size
        
        for i in range(n_sims):
            path = []
            while len(path) < t:
                if len(path) == 0 or rng.random() < restart_prob:
                    start_idx = rng.integers(T)
                else:
                    start_idx = (start_idx + 1) % T #Implement circular indexing
                value = return_values[start_idx]
                
                if include_noise:
                    noise = rng.standard_normal()*np.std(return_values, ddof=1)
                    value += noise
                    
                path.append(value)
                
            sims[i] = path
            
        return sims

class PortfolioEnv(gym.Env):
    '''
    Simulator for the Agent Environment
    
    Inputs:
        start_date: the start date of the simulation
        end_date: the end date of the simulation
        rf_rate: the risk-free interest rate for continuous compounding
        trans_cost: the transaction cost and a percentage of the transaction amount
        state_len: total days of past returns to use for state
        batch_size: number of independent trading environments
        device: the device to use (cpu, gpu)
        logging: whether to log the episode data for plotting function
    '''
    
    def __init__(self, spy_log_returns:pd.Series, start_date:str='1995-01-01', end_date:str='2023-12-31', rf_rate:float=0.024, trans_cost:float=0.0005, 
                 state_len:int=60, batch_size:int=8, device:str='cpu', logging:bool=False, include_noise:bool=False, avg_block_size:int=200, seed:int=None):
        # Market Info
        self.real_returns = spy_log_returns.copy()
        self.avg_block_size = avg_block_size
        self.seed = seed
        self.include_noise = include_noise
        self.start_date = start_date
        self.end_date = end_date
        self.int_rate = rf_rate
        self.trans_cost = trans_cost
        
        # Algo Info
        self.seq_dim = 1 # Number of assets
        self.state_len = state_len
        self.batch_size = batch_size
        self.logging = logging
        self.device = device
        
        self._init_action_space()
        self._init_obs_space()
        self._init_calendar()
        super().__init__() #Required for Gym
    
    def _init_action_space(self):
        self.action_space = gym.spaces.Discrete(9)
        self.action_values = torch.linspace(-1.0, 1.0, 9, device=self.device) #-1 to 1 in steps of 0.25
    
    def _init_obs_space(self):
        obs_dim = self.state_len * self.seq_dim + self.seq_dim + 2
        self.observation_space = gym.spaces.Box(low=-1e10, high=1e10, shape=(obs_dim,), dtype=np.float32) #Expect size (63,)
    
    def _init_calendar(self, trading_calendar:str='NYSE'):
        # Create NYSE Trading Dates
        calendar = mcal.get_calendar(trading_calendar)
        schedule = calendar.schedule(start_date=self.start_date, end_date=self.end_date)
        
        # Process Dates
        calendar_dates = schedule.index.to_series()
        days_between = calendar_dates.diff()[1:].dt.days
        years_between = days_between/365
        
        t = np.zeros(len(schedule))
        t[1:] = years_between.values.cumsum() #Years in Continuous Space
        self.t = torch.tensor(t, dtype=torch.float32, device=self.device, requires_grad=False)
        self.dts = self.t.diff(dim=0) #For Interest Rate Compounding
        self.total_steps = len(self.dts)
        self.action_steps = self.total_steps - self.state_len - 500
    
    def _simulate(self, avg_block_size:int, seed:int) -> torch.Tensor:
        sim = stationary_bootstrap(self.real_returns, n_sims=self.batch_size, t=self.total_steps, seed=seed, avg_block_size=avg_block_size, include_noise=self.include_noise)
        seq = torch.tensor(sim, dtype=torch.float32, device=self.device).unsqueeze(-1)
        return seq # (batch_size, total_steps, 1)
    
    def _get_state(self) -> torch.Tensor:
        dt = self.dts[self.curr_step-1].repeat(self.batch_size, 1) # (batch_size, 1)
        seq_window = self.seq[:, self.curr_step - self.state_len : self.curr_step, :]  # (batch_size, state_len, seq_dim)
        seq = seq_window.reshape(self.batch_size, -1) # Flatten into (batch_size, state_len × seq_dim)
        next_state = torch.cat([seq, self.log_wealth, self.position, dt], dim=1) # Horizontal Append
        
        # No Batching
        if self.batch_size == 1:
            next_state = next_state.reshape(-1)

        return next_state
    
    def _check_action(self, action:torch.Tensor) -> torch.Tensor:
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32, device=self.device, requires_grad=False)
        
        if action.ndim == 1:
            if action.shape[0] == self.batch_size:
                action = action.unsqueeze(-1) #(batch_size, 1)
        
        action = action.to(torch.int32) #Index Integer
        action = self.action_values[action] #Get Index Value
        return action
    
    def _get_reward(self, action:torch.Tensor, log_return:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the reward for a given portfolio action based on asset and risk-free returns.
        
        Inputs:
            action:     Tensor of shape (batch_size, 1) representing the new portfolio weight allocated to the risky asset. The remaining (1 - action) is held as cash.
            log_return: Tensor of shape (batch_size, 1) containing the log returns of the risky asset for the current time step.
        
        Outputs:
            reward:             Logarithmic portfolio return (scalar if batch_size == 1, else (batch_size, 1)).
            interest_return:    Scalar risk-free simple return
            transaction_cost:   Transaction cost for each batch element, penalizing changes in position weights.
        """
        # Risk-Free Return
        cash_weight = 1.0 - action # (batch_size, 1)
        interest_return = torch.exp(self.int_rate* self.dts[self.curr_step]) - 1.0
        weighted_interest = cash_weight*interest_return
        
        # Asset Return
        change_in_weight = action - self.position # (batch_size, 1)
        asset_return = log_return.exp() - 1
        weighted_return = action * asset_return
        
        transaction_cost = self.trans_cost* torch.abs(change_in_weight)
        
        total_simple_return = 1 + weighted_interest + weighted_return - transaction_cost
        reward = torch.log(total_simple_return)
        
        # Check to ensure that no values are nan
        if torch.isnan(reward).any() or torch.isinf(reward).any():
            print(f"Reward: {reward}\nReturn:{total_simple_return}\nAction:{action}")
            print(f"Weighted Interest:{weighted_interest}\nWeighted Return:{weighted_return}\nTransaction Cost:{transaction_cost}")
        
        if self.batch_size == 1:
            reward = reward.squeeze() #Scalar
        
        self.position = action
        
        return reward, interest_return.repeat(self.batch_size), transaction_cost.squeeze()
    
    def reset(self) -> tuple[torch.Tensor, set]:
        # Initialize Agent State
        self.position = torch.zeros((self.batch_size, self.seq_dim), dtype=torch.float32, requires_grad=False)
        self.log_wealth = torch.zeros((self.batch_size, 1), dtype=torch.float32, requires_grad=False)
    
        # Initialize Simulation State
        self.curr_step = 500 + self.state_len
        self.seq = self._simulate(self.avg_block_size, self.seed) # (batch_size, total_steps, 1)
        next_state = self._get_state()
        
        if self.logging:
            self.episode_actions = []
            self.episode_rewards = []
            self.episode_log_returns = [next_state[:,:self.state_len*self.seq_dim]]
        
        return next_state, {}
    
    def step(self, action:torch.Tensor):
        '''
        Generate the state with internal state update
        Inputs:
            action: the action of shape (batch_size, seq_dim) signifies the action index for Discrete.
        
        Outputs:
            next_state: the next state of shape (batch_size, state_len, seq_dim)
            reward: the reward of shape (batch_size, 1)
            done: whether the episode is done
            truncated: False for the truncated input requirement of train.py
            info: the info dictionary
        '''
        
        # Take Action and Calculate Reward
        action = self._check_action(action)
        log_return = self.seq[:, self.curr_step, :]
        reward, interest, transaction_cost = self._get_reward(action=action, log_return=log_return)
        
        # Update Info
        self.log_wealth += reward
        info = (interest, transaction_cost)
        
        # Update Step
        next_state = self._get_state()
        self.curr_step += 1
        done = (self.curr_step == self.total_steps)
        done = torch.tensor(done, dtype=torch.bool, device=self.device).repeat(self.batch_size, 1)
        
        if self.logging:
            self.episode_actions.append(action)
            self.episode_rewards.append(reward)
            self.episode_log_returns.append(next_state[:,(self.state_len-1)*self.seq_dim:self.state_len*self.seq_dim])
        
        self.env_step_check = {
            'next_state': next_state,
            'reward': reward,
            'done': done,
            'info': info
        }
        return next_state, reward, done, False, info
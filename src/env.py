import gymnasium as gym
import numpy as np
import pandas_market_calendars as mcal
import torch

from src.stationary_block_bootstrap import generate_path

class PortfolioEnv(gym.Env):
    '''
    Simulator for the Agent Environment
    
    Inputs:
        asset_log_returns: a 1D numpy array of log returns for the asset to be traded.
        start_date: the start date of the simulation
        end_date: the end date of the simulation
        rf_rate: the risk-free interest rate for continuous compounding
        trans_cost: the transaction cost and a percentage of the transaction amount
        state_len: total days of past returns to use for state
        batch_size: number of independent trading environments
        logging: whether to log the episode data for plotting function
    '''
    
    def __init__(self, asset_log_returns:np.ndarray, start_date:str='1995-01-01', end_date:str='2023-12-31', rf_rate:float=0.024, trans_cost:float=0.005, 
                 state_len:int=60, batch_size:int=8, logging:bool=False, seed:int=None):
        
        super().__init__()
        
        # Market Info
        self.asset_log_returns = asset_log_returns.copy()
        self.seed = seed
        self.start_date = start_date
        self.end_date = end_date
        self.rf_rate = rf_rate
        self.trans_cost = trans_cost
        
        # Algo Info
        self.state_len = state_len
        self.batch_size = batch_size
        self.logging = logging
        
        self._init_action_space()
        self._init_obs_space()
        self._init_calendar()
    
    def _init_action_space(self):
        """
        Initializes the action space for the portfolio environment. The action space is defined as a discrete set of portfolio weights for the risky asset, ranging from -1.0 to 1.0 in increments of 0.25. 
        This allows the agent to choose from 9 possible actions, representing different levels of allocation to the risky asset (including short positions). The remaining weight (1 - action) is allocated to cash.
        """
        self.action_space = gym.spaces.Discrete(9)
        self.action_values = np.linspace(-1.0, 1.0, 9) #-1 to 1 in steps of 0.25
    
    def _init_obs_space(self):
        """
        Initializes the observation space for the portfolio environment. 
        The observation space is defined as a continuous box with dimensions corresponding to the state length (number of past returns), current portfolio return, current position, and time step information.
        """
        PORTFOLIO_RETURN_DIM, CURRENT_POSITION_DIM, TIME_DELTA_DIM = 1, 1, 1
        obs_dim = self.state_len + PORTFOLIO_RETURN_DIM + CURRENT_POSITION_DIM + TIME_DELTA_DIM
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
        self.dts = np.diff(t, prepend=0) #For Interest Rate Compounding
        self.total_steps = len(self.dts)
        self.action_steps = self.total_steps - self.state_len #Number of steps where action can be taken
    
    def _simulate(self, seed:int) -> np.ndarray:
        sim = generate_path(self.asset_log_returns, self.batch_size, self.total_steps, seed=seed) #(batch_size, total_steps)
        return sim
    
    def _get_state(self) -> np.ndarray:
        """
        Generates the next state for the portfolio environment by concatenating the past returns, current portfolio return, current position, and time step information.
        """
        dt = np.full(
            (self.batch_size, 1),
            self.dts[self.curr_step - 1],
            dtype=np.float32
        )
        seq_window = self.seq[:, self.curr_step-self.state_len : self.curr_step]  # (batch_size, state_len)
        next_state = np.concatenate([seq_window, self.log_wealth, self.position, dt], axis=1) # Horizontal Append
        
        # No Batching
        if self.batch_size == 1:
            next_state = next_state.flatten() # (state_len,)

        return next_state
    
    def _check_action(self, action:torch.Tensor) -> np.ndarray:
        if action.ndim == 1:
            if action.shape[0] == self.batch_size:
                action = action.unsqueeze(-1) #(batch_size, 1)
        
        action = action.to(torch.int32) #Index Integer
        action = self.action_values[action] #Get Index Value
        return action
    
    def _get_reward(self, action:np.ndarray, log_return:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the reward for a given portfolio action based on asset and risk-free returns.
        
        Inputs:
            action: Array of shape (batch_size, 1) representing the new portfolio weight allocated to the risky asset. The remaining (1 - action) is held as cash.
            log_return: Array of shape (batch_size, 1) containing the log returns of the risky asset for the current time step.
        
        Outputs:
            reward: Logarithmic portfolio return of shape (batch_size, 1), calculated as the log of the total return from both the risky asset and the cash position, minus transaction costs.
            interest_return: Risk-free simple return of the cash portion of the portfolio with an expected shape of (batch_size, 1).
            transaction_cost: Transaction cost for each batch element, penalizing changes in position weights, expected shape of (batch_size, 1).
        """
        # Risk-Free Return
        cash_weight = 1.0 - action # (batch_size, 1)
        interest_return = np.exp(self.rf_rate* self.dts[self.curr_step]) - 1.0
        weighted_interest = cash_weight*interest_return
        
        # Asset Return
        asset_return = np.exp(log_return) - 1.0
        weighted_return = action * asset_return # ()
        
        change_in_weight = action - self.position # (batch_size, 1)
        transaction_cost = self.trans_cost* np.abs(change_in_weight) # (batch_size, 1)
        
        total_simple_return = 1.0 + weighted_interest + weighted_return - transaction_cost
        reward = np.log(total_simple_return) # (batch_size, 1)
                
        # Check to ensure that no values are nan
        if np.isnan(reward).any() or np.isinf(reward).any():
            print(f"Reward: {reward}\nReturn:{total_simple_return}\nAction:{action}")
            print(f"Weighted Interest:{weighted_interest}\nWeighted Return:{weighted_return}\nTransaction Cost:{transaction_cost}")
        
        self.position = action.copy()
        
        interest_return = np.full(
            (self.batch_size, 1),
            interest_return,
            dtype=np.float32
        )
        
        return reward, interest_return, transaction_cost
    
    def reset(self) -> tuple[np.ndarray, set]:
        # Initialize Agent State
        self.position = np.zeros((self.batch_size, 1), dtype=np.float32)
        self.log_wealth = np.zeros((self.batch_size, 1), dtype=np.float32)
    
        # Initialize Simulation State
        self.curr_step = self.state_len
        self.seq = self._simulate(self.seed) # (batch_size, total_steps)
        next_state = self._get_state()
        
        if self.logging:
            self.episode_actions = []
            self.episode_rewards = []
            self.episode_log_returns = [next_state[:,:self.state_len]]
        
        return next_state, {}
    
    def step(self, action:torch.Tensor):
        '''
        Generate the state and update the environment based on the action taken by the agent.
        
        Inputs:
            action: the action of shape (batch_size, 1) signifies the action index for Discrete.
        
        Outputs:
            next_state: the next state of shape (batch_size, state_len).
            reward: the reward of shape (batch_size, 1)
            done: whether the episode is done.
            truncated: False for the truncated input requirement of a gym environemnt.
            info: the info dictionary that contains interest and transaction cost information.
        '''
        if not (hasattr(self, 'position') and hasattr(self, 'log_wealth')):
            raise AttributeError('Please use reset() to initialize environment.')
        
        # Take Action and Calculate Reward
        action = self._check_action(action)
        log_return = self.seq[:, self.curr_step:self.curr_step+1]
        reward, interest, transaction_cost = self._get_reward(action=action, log_return=log_return)
        
        # Update Info
        self.log_wealth += reward
        info = (interest, transaction_cost)
        
        # Update Step
        next_state = self._get_state()
        self.curr_step += 1
        done = np.full(
            (self.batch_size, 1),
            self.curr_step == self.total_steps,
            dtype=bool
        )
        
        truncated = False #Gym Environment requirement
        
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
        return next_state, reward, done, truncated, info
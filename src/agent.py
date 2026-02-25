import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from src.agent_interface import *
from src.prior_measure import *
from src.robust import *
from src.util import PORDQNProgressWriter

class ReplayBuffer:
    """
    Replay buffer to store experience tuples for training the agent. 
    Memory is stored in the CPU to prevent excessive memory overhead.
    
    Inputs:
        state_dim: Dimension of the state space.
        action_dim: Dimension of the action space.
        batch_size: Batch size for sampling.
        max_len: Maximum length of the replay buffer.
        device: Training torch device such as "cuda" or "cpu", if "cuda", pin_memory is set to True for faster transfer from CPU to GPU. 
    """
    def __init__(self, state_dim:int, action_dim:int, batch_size:int, max_len:int=1e6, device:torch.device=None, seed:int=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.max_len = int(max_len)
        self.device = torch.device('cpu') if device is None else device
        
        self.buffer_device = torch.device('cpu')
        self.pin_memory = (self.device.type == 'cuda')
        self.batch_arange = torch.arange(self.batch_size, device=self.buffer_device)
        
        self.reset(seed)
    
    def reset(self, seed:int=None):
        self.circular_ptr = 0
        self.size = 0
        
        self.generator = torch.Generator(device=self.buffer_device)
        if seed is not None:
            self.generator.manual_seed(seed)
        
        self.state = torch.empty((self.max_len, self.state_dim), dtype=torch.float32, device=self.buffer_device, pin_memory=self.pin_memory)
        self.action = torch.empty((self.max_len, self.action_dim), dtype=torch.int64, device=self.buffer_device, pin_memory=self.pin_memory)
        self.reward = torch.empty((self.max_len), dtype=torch.float32, device=self.buffer_device, pin_memory=self.pin_memory)
        self.next_state = torch.empty((self.max_len, self.state_dim), dtype=torch.float32, device=self.buffer_device, pin_memory=self.pin_memory)
        self.terminal_state = torch.empty((self.max_len), dtype=torch.bool, device=self.buffer_device, pin_memory=self.pin_memory)
        self.lambda_val = torch.empty((self.max_len), dtype=torch.float32, device=self.buffer_device, pin_memory=self.pin_memory)
        self.risk_free_rate = torch.empty((self.max_len), dtype=torch.float32, device=self.buffer_device, pin_memory=self.pin_memory)
        self.transaction_cost = torch.empty((self.max_len), dtype=torch.float32, device=self.buffer_device, pin_memory=self.pin_memory)
    
    def _to_cpu(self, item:torch.Tensor) -> torch.Tensor:
        """
        Assumes item is GPU tensor.
        """
        item = item.detach().contiguous() #contiguous handles strided tensors
        return item.to(self.buffer_device)
    
    def _to_gpu(self, item:torch.Tensor) -> torch.Tensor:
        """
        Assumes item is CPU tensor and training device is GPU.
        """
        item = item.detach().contiguous() #contiguous handles strided tensors
        return item.to(self.device, non_blocking=True)
    
    def _device_transfer(self, item:torch.Tensor, target_device:torch.device) -> torch.Tensor:
        if item.device == target_device:
            return item
        
        if target_device.type == 'cuda':
            return self._to_gpu(item)
        elif target_device.type == 'cpu':
            return self._to_cpu(item)
        else:
            raise ValueError(f"Invalid Device: {target_device.type}. Device should either be 'cuda' or 'cpu'.")
    
    def add(self, state:torch.Tensor, action:torch.Tensor, reward:torch.Tensor, next_state:torch.Tensor, terminal_state:torch.Tensor, 
            lambda_val:torch.Tensor, risk_free_rate:torch.Tensor, transaction_cost:torch.Tensor):
        """
        Add a batch of experience tensors to the replay buffer.
        
        Inputs:
            state: Tensor of shape [batch_size, state_dim].
            action: Tensor of shape [batch_size, action_dim].
            reward: Tensor of shape [batch_size]. Note that this is the reward from the environment.
            next_state: Tensor of shape [batch_size, state_dim].
            terminal_state: Tensor of shape [batch_size].
            lambda_val: Tensor of shape [batch_size].
            risk_free_rate: Tensor of shape [batch_size].
            transaction_cost: Tensor of shape [batch_size].
        """
        target_device = self.buffer_device
        
        # Add batch data to buffer
        circular_idx = (self.circular_ptr + self.batch_arange) % self.max_len
        self.state[circular_idx] = self._device_transfer(state, target_device)
        self.action[circular_idx] = self._device_transfer(action, target_device)
        self.reward[circular_idx] = self._device_transfer(reward, target_device)
        self.next_state[circular_idx] = self._device_transfer(next_state, target_device)
        self.terminal_state[circular_idx] = self._device_transfer(terminal_state, target_device)
        self.lambda_val[circular_idx] = self._device_transfer(lambda_val, target_device)
        self.risk_free_rate[circular_idx] = self._device_transfer(risk_free_rate, target_device)
        self.transaction_cost[circular_idx] = self._device_transfer(transaction_cost, target_device)
        
        # Update circular pointer and size
        self.circular_ptr = (self.circular_ptr + self.batch_size) % self.max_len
        self.size = min(self.size + self.batch_size, self.max_len)
    
    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch from the replay buffer. Note that if you have n updates, you should call this n times to get different samples.
        
        Outputs:
            A tuple of tensors.
                state: Tensor of shape [batch_size, state_dim].
                action: Tensor of shape [batch_size, action_dim].
                reward: Tensor of shape [batch_size].
                next_state: Tensor of shape [batch_size, state_dim].
                terminal_state: Tensor of shape [batch_size].
                lambda_val: Tensor of shape [batch_size].
                risk_free_rate: Tensor of shape [batch_size].
                transaction_cost: Tensor of shape [batch_size].
                idx: Tensor of shape [batch_size], indices of the sampled experiences in the buffer.
        """
        target_device = self.device
        
        idx = torch.randint(0, self.size, (self.batch_size,), device=self.buffer_device, generator=self.generator)
        state = self._device_transfer(self.state[idx], target_device)
        action = self._device_transfer(self.action[idx], target_device)
        reward = self._device_transfer(self.reward[idx], target_device)
        next_state = self._device_transfer(self.next_state[idx], target_device)
        terminal_state = self._device_transfer(self.terminal_state[idx], target_device)
        lambda_val = self._device_transfer(self.lambda_val[idx], target_device)
        risk_free_rate = self._device_transfer(self.risk_free_rate[idx], target_device)
        transaction_cost = self._device_transfer(self.transaction_cost[idx], target_device)
        
        idx = self._device_transfer(idx, target_device)
        result = (
            state, action, reward,
            next_state, terminal_state, lambda_val, 
            risk_free_rate, transaction_cost, idx
        )
        
        return result
    
    def __len__(self):
        return self.size

class PORDQN(AgentInterface):
    """
    Portfolio Optimisation Robust Deep Q-Network (PORDQN).

    PORDQN implements a robust Q-learning agent that replaces the standard Bellman target with a duality-based HQ operator under distributional
    ambiguity. The agent evaluates actions under a prior return distribution and computes robust targets using entropy bias-corrected Sinkhorn
    distances. The agent is designed for portfolio allocation problems where uncertainty in future returns is explicitly modelled via a 
    prior measure and a Wasserstein-type ambiguity set.
    
    Training Procedure for each sampled batch:
        1. Sample future returns from the prior distribution.
        2. Construct candidate next states conditioned on sampled returns.
        3. Compute rewards accounting for asset returns, risk-free allocation and transaction costs.
        4. Evaluate target Q-values.
        5. Compute Sinkhorn ambiguity radius and filter invalid samples.
        6. Optimise λ using the HQ operator neural optimisation routine.
        7. Use HQ values as robust TD targets to update the Q-network.
    
    Notes:
        1. Designed for single-asset trading where the buffer action dimension is fixed to one.
        2. Sinkhorn radius filtering removes implausible samples caused by entropy bias correction.
        3. Uses epsilon-greedy exploration during training when epsilon > 0.
        4. State contains a 60-day history of past returns, current portfolio return, current position and the time step information.
    
    Inputs:
        state_dim: Dimension of the state space.
        action_dim: Dimension of the action space.
        action_values: Discrete actions obtained using np.linspace() or through the dedicated environment (PortfolioEnv.action_values).
        batch_size: Batch size for training.
        n_updates: Number of update steps to perform when training.
        training_controller: TrainingController object to manage training steps and target network cloning, imported from agent_interface.py.
        prior_measure: A prior distribution object representing the prior distribution over returns, imported from prior_measure.py.
        duality_operator: A DualityHQOperator object that implements the HQ operator and Sinkhorn distance calculations, imported from robust.py.
        epsilon: Epsilon value for exploration, default is 0.0 which means epsilon-greedy is disabled.
        lamda_init: Initial value for lambda in the HQ optimization, default is 1.0.
        qfunc: Q-network (if None, a default network will be created).
        network_optimizer: Optimizer for the Deep Q-Network (if None, a default Adam optimizer will be created).
        network_lr: Learning rate for the network optimizer, if network_optimizer is not None, this is redundant.
        hq_optimizer: Optimizer for the HQ value optimization (if None, a default Adam optimizer will be created).
        hq_lr: Learning rate for the HQ optimizer, if hq_optimizer is not None, this is redundant.
        device: Device to run the network on, such as 'cuda' or 'cpu'.
        buffer_max_length: Maximum length of the replay buffer, default is 1e6.
        writer: TensorBoard SummaryWriter for logging (optional).
    """
    def __init__(self, state_dim:int, action_dim:int, action_values:np.ndarray, batch_size:int, n_updates:int,
                 training_controller:TrainingController, prior_measure:PriorStudentDistribution, duality_operator:DualityHQOperator, 
                 epsilon:float=0.1, lamda_init:float=0.0, qfunc:torch.nn.Module=None, network_optimizer:torch.optim.Optimizer=None, network_lr:float=1e-4,
                 hq_optimizer:torch.optim.Optimizer=None, hq_lr:float=0.02, hq_max_iter:int=100, hq_step_size:int=10, hq_gamma:float=10.0, device:torch.device=None, buffer_max_length:int=1e6, writer:PORDQNProgressWriter=None, seed:int=None):
        super().__init__()
        
        # Device
        self.device = torch.device('cpu') if device is None else device
        
        # Training Parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_values = self._modify_action_values(action_values)
        self.batch_size = batch_size
        self.n_updates = n_updates
        self.epsilon = epsilon
        
        # Objects
        self.training_controller = training_controller
        self.prior_measure = prior_measure
        self.duality_operator = duality_operator
        
        # Initialize Networks
        self.q = qfunc
        if qfunc is None:
            if self.state_dim is None:
                raise ValueError("If qfunc is None, state_dim must be provided.")
            
            self.q = nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, self.action_dim)
            )
        
        self.target_q = deepcopy(self.q)
        self.q = self.q.to(self.device)
        self.target_q = self.target_q.to(self.device)
        
        # Reproduce Results
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)
            self.seed = seed
            self.buffer_seed = seed + 1 #Seed is offset to prevent hidden correlations
        
        # Buffer
        self.buffer_action_dim = 1  # Fixed action dimension to 1 as we are using epsilon greedy and trading one asset.
        self.buffer = ReplayBuffer(self.state_dim, self.buffer_action_dim, self.batch_size, buffer_max_length, self.device, seed=self.buffer_seed) 
        
        # Loss Functions
        self.network_optimizer = torch.optim.Adam(self.q.parameters(), lr=network_lr) if network_optimizer is None else network_optimizer
        self.loss_fn = nn.MSELoss()
        self.clip_gradients = True
        
        # HQ Optimization
        self.lamda_init = lamda_init
        self.hq_optimizer = hq_optimizer
        self.hq_lr = hq_lr
        self.hq_max_iter = hq_max_iter
        self.hq_step_size = hq_step_size
        self.hq_gamma = hq_gamma
        
        # Logging Purposes
        self.writer = writer
        self.q_updates = 0 #Increment every update, so time_step*n_updates
    
    def _modify_action_values(self, action_values:np.ndarray):
        if isinstance(action_values, np.ndarray):
            return torch.from_numpy(action_values).to(self.device, dtype=torch.float32)
        elif torch.is_tensor(action_values):
            return action_values.to(self.device, dtype=torch.float32)
        else:
            raise TypeError(f"Expected a numpy array for action values. Received: {type(action_values)}")
    
    def get_action(self, observation:torch.Tensor|np.ndarray) -> torch.Tensor:
        """
        Get action from the agent given an observation. Uses epsilon-greedy exploration if enabled by setting epsilon > 0.
        
        Inputs:
            observation: Observation tensor of shape [batch_size, obs_dim]. Preprocessed to be float tensor in agent_interface.py.
        
        Outputs:
            actions: Tensor of selected actions of shape [batch_size, 1].
        """
        
        with torch.no_grad():
            # Get Q values and select action
            q_values = self.q(observation.to(self.device)) # (Batch, Num_Actions)
            actions = torch.argmax(q_values, dim=-1, keepdim=True) # For each batch, select action with highest Q value (Batch, 1)
            
            if (self.epsilon > 0) and self.training_mode:
                is_epsilon_greedy = torch.rand(actions.shape[0], device=self.device) < self.epsilon # Select which batch will explore
                total_explorers = is_epsilon_greedy.sum().item() #return an integer
                if total_explorers > 0:
                    shape = (total_explorers, 1)
                    actions[is_epsilon_greedy] = torch.randint(0, self.action_dim, shape, device=self.device, dtype=torch.long, generator=self.generator) # Random action for explorers
            
        return actions # (Batch, 1)
    
    def prepare_target_states(self, state:torch.Tensor, action:torch.Tensor, next_return_from_prior:torch.Tensor, sampled_states:torch.Tensor, realized_return:torch.Tensor) -> torch.Tensor:
        """
        Prepare the target state tensor for the target Q-network by updating the current state with the action taken and the results of that action. 
        This function is specific to our use case of selected 63 states and should be changed based on the specific state representation and how the next state should be constructed.
        
        Inputs:
            state: Current state tensor of shape [batch_size, state_dim].
            action: Tensor of shape [batch_size, action_dim].
            next_return_from_prior: Tensor of shape [batch_size, sample_size] representing the next return sampled from the prior distribution.
            sampled_states: Tensor of shape [batch_size, state_dim] representing the sampled next states from the buffer.
            realized_return: Tensor of shape [batch_size, sample_size] representing the realized return for each sampled next state.
        
        Outputs:
            next_state_expand: Tensor of shape [batch_size, sample_size, state_dim] representing the prepared next state for each sampled next return from the prior.
            reference_return: Tensor of shape [batch_size, 1] representing the reference return from the sampled states.
        """
        
        next_state = state.clone() # (batch_size, state_dim)
        next_state[:, :59] = state[:, 1:60] #Shift history by one step
        next_state[:, 61:62] = action # Update action in state
        next_state[:, 62:63] = sampled_states[:, 62:63] # Update dt in state
        
        sample_size = next_return_from_prior.shape[1]
        next_state_expand = next_state.unsqueeze(1).repeat(1, sample_size, 1) # (batch_size, sample_size, state_dim)
        next_state_expand[..., 59:60] = next_return_from_prior.unsqueeze(-1) # Update next return from prior in state (batch_size, sample_size, state_dim)
        next_state_expand[..., 60:61] += realized_return.squeeze(1).unsqueeze(-1) # Update realized return in state (batch_size, sample_size, state_dim)
        
        reference_return = sampled_states[:, 59:60].clone() # (batch_size, 1)
        return next_state_expand, reference_return
        
    def clone_q(self):
        self.target_q.load_state_dict(self.q.state_dict())
    
    def update_q(self):
        """"
        Update the Q-network by performing n_updates training steps. Note that train_batch() inputs are sampled from the buffer.
        """
        for update in range(self.n_updates):
            self.q_updates += 1
            states, actions, rewards, next_state, terminal_states, lambda_vals, risk_free_rates, transaction_costs, buffer_idx = self.buffer.sample()
            self.train_batch(states, actions, rewards, next_state, terminal_states, lambda_vals, risk_free_rates, transaction_costs, buffer_idx)
    
    def reward_fn(self, action:torch.Tensor, next_return_from_prior:torch.Tensor, risk_free_rate:torch.Tensor, transaction_cost:torch.Tensor) -> torch.Tensor:
        """
        Compute reward for a given next return from prior based on action after accounting for transaction costs and remaining cash earning arisk-free rate.
        
        Inputs:
            action: Tensor of shape [batch_size, action_dim] representing the action taken.
            next_return_from_prior: Tensor of shape [batch_size, n_samples] representing the next return sampled from the prior distribution.
            risk_free_rate: Tensor of shape [batch_size] representing the risk-free rate.
            transaction_cost: Tensor of shape [batch_size] representing the transaction cost.
        
        Outputs:
            rewards: Tensor of shape [batch_size, action_dim, n_samples] representing the rewards for each sampled next return.
        """
        #Match Shapes
        modified_next_return_from_prior = next_return_from_prior.unsqueeze(1).expand(-1, self.action_dim, -1) # (batch_size, action_dim, n_samples)
        modified_action = action.unsqueeze(-1) # (batch_size, action_dim, 1)
        modified_risk_free_rate = risk_free_rate.unsqueeze(-1).unsqueeze(-1) # (batch_size, 1, 1)
        modified_transaction_cost = transaction_cost.unsqueeze(-1).unsqueeze(-1) # (batch_size, 1, 1)
        
        #Compute Reward
        asset_return = modified_next_return_from_prior.exp() - 1.0 # (batch_size, action_dim, n_samples)
        weighted_asset_return = modified_action * asset_return # (batch_size, action_dim, n_samples)
        
        cash_weight = 1.0 - modified_action # (batch_size, action_dim, 1)
        weighted_cash_return = cash_weight * modified_risk_free_rate # (batch_size, action_dim, 1)
        simple_return_after_transaction = (weighted_asset_return + weighted_cash_return - modified_transaction_cost) + 1.0 # (batch_size, action_dim, n_samples)
        reward = simple_return_after_transaction.log() # (batch_size, action_dim, n_samples)
        return reward
    
    def _cache_lambdas(self, lamdas:torch.Tensor, buffer_idx:torch.Tensor, mask:torch.Tensor):
        buffer_ready_lamdas = self.buffer._device_transfer(lamdas, self.buffer.buffer_device)
        buffer_ready_mask = self.buffer._device_transfer(mask, self.buffer.buffer_device)
        valid_idx = buffer_idx[buffer_ready_mask]
        
        if lamdas.ndim == 1 and lamdas.shape[0] == self.batch_size:
            self.buffer.lambda_val[valid_idx] = buffer_ready_lamdas[buffer_ready_mask] #(batch_size)
        elif lamdas.ndim == 2 and lamdas.shape[1] == 1:
            self.buffer.lambda_val[valid_idx] = buffer_ready_lamdas[buffer_ready_mask].squeeze(-1) #(batch_size)
        else:
            raise ValueError("lamdas must be of shape (batch_size,) or (batch_size, 1)")
    
    def compute_loss_and_update(self, current_states:torch.Tensor, actions:torch.Tensor, hq_values:torch.Tensor, lamda_mask:torch.Tensor):
        """
        Compute the loss and update the Q-network.
        
        Inputs:
            current_states: Tensor of current states, shape (batch_size, state_dim).
            actions: Tensor of actions taken, shape (batch_size, 1).
            hq_values: Tensor of HQ values, shape (batch_size,).
            lamda_mask: Boolean mask indicating valid samples, shape (batch_size,).
        
        Outputs:
            loss: Computed loss value.
        """
        row_indices = torch.arange(self.batch_size, device=self.device) #(batch_size,)
        current_state_q = self.q(current_states)[row_indices, actions.squeeze().to(torch.int64)] #Select all rows and argmax actions
        
        # Compute loss only on valid samples if mask is provided else compute on all samples
        if (~lamda_mask).any():
            loss = self.loss_fn(current_state_q[lamda_mask], hq_values[lamda_mask])
        else:
            loss = self.loss_fn(current_state_q, hq_values)
        
        self.network_optimizer.zero_grad()
        loss.backward()
        
        if self.clip_gradients:
            torch.nn.utils.clip_grad_value_(self.q.parameters(), 1.0)
        
        if self.writer is not None:
            self.writer.log_network_updates(self.q, loss, self.q_updates)
        
        self.network_optimizer.step()
        return loss
    
    def log_indicators(self, rewards:torch.Tensor, next_states:torch.Tensor, not_terminal:torch.Tensor, targets:torch.Tensor, lambda_iters:int, lambdas:torch.Tensor, mask:torch.Tensor):
        """
        Log Lagrangian Lambda and HQ descriptive statistics to TensorBoard.
        
        Inputs:
            rewards: Tensor of shape [batch_size].
            next_states: Tensor of shape [batch_size, state_dim].
            not_terminal: Boolean tensor of shape [batch_size].
            targets: HQ Tensor of shape [batch_size].
            lambda_iters: Number of iterations taken for lambda optimization.
            lambdas: Tensor of shape [batch_size].
            mask: Boolean tensor of shape [batch_size] indicating valid samples.
        """
        
        with torch.no_grad():
            discount_rate = self.duality_operator.discount_rate # Scalar
            target_q_vals = self.target_q(next_states).max(dim=-1).values # (batch_size,)
            standard_q_targets = rewards + discount_rate * target_q_vals * not_terminal # (batch_size,)
            q_hq_diff = standard_q_targets - targets
            
            self.writer.log_hq_progress(lambda_iters, lambdas, mask, self.q_updates, q_hq_diff, targets)
    
    def train_batch(self, states:torch.Tensor, actions:torch.Tensor, rewards:torch.Tensor, next_state:torch.Tensor, terminal_states:torch.Tensor, lambda_vals:torch.Tensor, risk_free_rates:torch.Tensor, transaction_costs:torch.Tensor, buffer_idx:torch.Tensor):
        """
        Takes in a batch of experience from the replay buffer and performs a training step on the Q-network.
        For entropic bias-corrected sinkhorn distance, we exclude the ones that are implausible (negative distance).
        
        Inputs:
            states: Tensor of shape [batch_size, state_dim].
            actions: Tensor of shape [batch_size, action_dim].
            rewards: Tensor of shape [batch_size].
            next_state: Tensor of shape [batch_size, state_dim].
            terminal_states: Tensor of shape [batch_size].
            lambda_vals: Tensor of shape [batch_size].
            risk_free_rates: Tensor of shape [batch_size].
            transaction_costs: Tensor of shape [batch_size].
            buffer_idx: Tensor of shape [batch_size] representing buffer indices.
        """
        not_terminal = torch.logical_not(terminal_states)
        with torch.no_grad():
            action_weight = self.action_values[actions.squeeze(-1)].unsqueeze(-1) # (batch_size, 1)
            
            # Prior component
            next_return_from_prior = self.prior_measure.sample_from_support(self.batch_size) # (batch_size, n_samples)
            rewards_from_prior = self.reward_fn(action_weight, next_return_from_prior, risk_free_rates, transaction_costs) # (batch_size, action_dim, n_samples)
            
            # Q-learning component
            target_state, reference_return = self.prepare_target_states(states, action_weight, next_return_from_prior, next_state, realized_return=rewards_from_prior) # (batch_size, n_samples, state_dim)
            q_targets = self.target_q(target_state) # (batch_size, n_samples, action_dim)
            optimal_q_targets = q_targets.max(dim=-1).values # (batch_size, n_samples)
            
            # Entropy bias-corrected sinkhorn radius calculation and inclusion of valid samples
            cost = self.duality_operator.compute_cost(reference_return, next_return_from_prior) # (batch_size, n_samples)
            epsilon_bar = self.duality_operator.update_sinkhorn_radius(cost) # (batch_size)
            mask = epsilon_bar.gt(0) # (batch_size)
            if epsilon_bar.lt(0).any():
                print("Warning: Sinkhorn radius is negative for some batches.")
            
        hq_value, lamda_star, n_iter = hq_opt_with_nn(self.duality_operator, reference_return, next_return_from_prior, rewards_from_prior, optimal_q_targets, not_terminal, lambda_vals, mask, optimizer=self.hq_optimizer, lr=self.hq_lr, max_iter=self.hq_max_iter, step_size=self.hq_step_size, gamma=self.hq_gamma)
        
        loss = self.compute_loss_and_update(states, actions, hq_value, mask)
        self._cache_lambdas(lamda_star, buffer_idx, mask)
        
        if self.writer is not None:
            self.log_indicators(rewards, next_state, not_terminal, hq_value, n_iter, lamda_star, mask)
            self.writer.log_policy_action(action_weight, self.q_updates, decimal_places=1)
            
        return loss
    
    def _handle_terminal_state_and_info(self, is_terminal:bool, info:dict=None):
        '''
        Handle terminal state and additional info after each step in the environment.
        
        Inputs:
            is_terminal: Boolean indicating whether the current state is terminal.
            info: Additional info from the environment (for this case it is risk-free rate and the transaction cost).
        
        Outputs:
            terminal_tensor: Tensor of shape [batch_size, 1] indicating terminal states.
            risk_free_rate_tensor: Tensor of shape [batch_size, 1] representing the risk-free rate.
            transaction_cost_tensor: Tensor of shape [batch_size, 1] representing the transaction cost.
        '''
        risk_free_rate = info.get('risk_free_rate', 0.0)
        transaction_cost = info.get('transaction_cost', 0.0)
        
        terminal_tensor = torch.full(
            (self.batch_size, ),
            is_terminal,
            dtype=torch.bool,
            device=self.device
        )
        
        # Handle numpy arrays and scalar values through reshape and expand
        risk_free_rate_tensor = torch.as_tensor(
            risk_free_rate,
            dtype=torch.float32,
            device=self.device
        ).reshape(-1).expand(self.batch_size)

        transaction_cost_tensor = torch.as_tensor(
            transaction_cost,
            dtype=torch.float32,
            device=self.device
        ).reshape(-1).expand(self.batch_size)

        return terminal_tensor, risk_free_rate_tensor, transaction_cost_tensor
    
    def train_mode_actions(self, reward:torch.Tensor|np.ndarray|float, observation:torch.Tensor|np.ndarray, is_terminal:bool=False, info=None):
        '''
        Actions to take when agent in training mode i.e. adding to replay buffer, cloning target q network and training q network
        
        Inputs:
            reward: Reward tensor of shape [batch_size].
            observation: Observation tensor of shape [batch_size, obs_dim].
            terminal: Terminal tensor of shape [batch_size], indicating whether each episode has ended.
            info: Additional info (not used here).
        '''
        self.training_controller.step_increment()
        lamda_init = torch.full((self.batch_size, ), self.lamda_init, dtype=torch.float32, device=self.device)
        terminal_tensor, risk_free_rate_tensor, transaction_cost_tensor = self._handle_terminal_state_and_info(is_terminal, info)
        self.buffer.add(self.prev_state, self.prev_action, reward, observation, terminal_tensor, lamda_init, risk_free_rate_tensor, transaction_cost_tensor)
        sufficient_samples = self.training_controller.has_samples(len(self.buffer))
        
        if sufficient_samples and self.training_controller.should_clone_q(): 
            self.clone_q()
        if sufficient_samples and self.training_controller.should_train(): 
            self.update_q()
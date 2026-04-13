import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from src.agents.base import AgentInterface, TrainingController, ReplayBuffer
from src.schedulers import EpsilonGlobalScheduler
from src.utils.writer import PORDQNProgressWriter
from src.robust.dro import DualityHQOperator
from src.robust.prior import PriorStudentDistribution

class QFunc(nn.Module):
    def __init__(self, input_size:int, hidden_size:list[int], output_size:int, activation:str='tanh'):
        super().__init__()
        activation_fn = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh()
        }
        if activation not in activation_fn:
            raise ValueError(f'Activation function {activation} not supported, please use either relu or tanh.')
        else:
            chosen_activation = activation_fn[activation]
        
        #Input layer
        layers = [nn.Linear(input_size, hidden_size[0])]
        
        #Hidden Layers
        for dim in hidden_size[1:]:
            layers.extend([
                chosen_activation,
                nn.Linear(layers[-1].out_features, dim),
            ])
        
        #Output Layers
        layers.append(chosen_activation)
        layers.append(nn.Linear(hidden_size[-1], output_size))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x) -> torch.Tensor:
        return self.net(x)

class DQNBase(AgentInterface):
    """
    Pure DQN base class: Q-network management, replay buffer, epsilon-greedy action selection,
    and MSE loss computation. Contains no portfolio or DRO logic.
    
    Subclasses must implement: train_batch, prepare_target_states, reward_fn, train_mode_actions.
    
    Inputs:
        state_dim: Dimension of the state space.
        action_dim: Number of discrete actions.
        action_values: Array of action values corresponding to each action index.
        total_paths: Number of parallel environment paths used during environment interaction.
        batch_size: Batch size for sampling from the replay buffer.
        n_updates: Number of gradient update steps per training trigger.
        training_controller: TrainingController managing clone and train step intervals.
        epsilon_scheduler: EpsilonGlobalScheduler for epsilon-greedy exploration decay.
        qfunc: Optional Q-network. If None, a default 64-64 ReLU MLP is created.
        network_optimizer: Optional optimizer. If None, Adam with network_lr is created.
        network_lr: Learning rate for the default optimizer (ignored if network_optimizer provided).
        device: Torch device for network and buffer.
        buffer_max_length: Maximum replay buffer capacity.
        clip_gradients: If True, clips Q-network gradients to 1.0.
        writer: Optional PORDQNProgressWriter for TensorBoard logging.
        seed: Optional seed for reproducibility.
    """
    
    def __init__(self, state_dim:int, action_dim:int, action_values:np.ndarray, total_paths:int, batch_size:int, n_updates:int,
                 training_controller:TrainingController, epsilon_scheduler:EpsilonGlobalScheduler, qfunc:torch.nn.Module=None,
                 network_optimizer:torch.optim.Optimizer=None, network_lr:float=1e-4, device:torch.device=None,
                 buffer_max_length:int=1e6, clip_gradients:bool=False, writer:PORDQNProgressWriter=None, seed:int=None):
        
        super().__init__()
        
        self.device = torch.device('cpu') if device is None else device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_values = self._modify_action_values(action_values)
        self.total_paths = total_paths
        self.batch_size = batch_size
        self.n_updates = n_updates
        self.epsilon_scheduler = epsilon_scheduler
        self.training_controller = training_controller
        
        self._init_q_networks(qfunc)
        self._set_seed(seed)
        
        # Buffer
        self.buffer_action_dim = 1 # Fixed action dimension to 1 as we are using epsilon greedy and trading one asset.
        self.buffer = ReplayBuffer(self.state_dim, self.buffer_action_dim, self.total_paths,
                                   self.batch_size, buffer_max_length, self.device, seed=self.buffer_seed)
        
        # Loss Functions
        self.network_optimizer = (torch.optim.Adam(self.q.parameters(), lr=network_lr) if network_optimizer is None else network_optimizer)
        self.loss_fn = nn.MSELoss()
        self.clip_gradients = clip_gradients
        
        # Logging Purposes
        self.writer = writer
        self.q_updates = 0 #Increment every update, so time_step*n_updates
    
    def _init_q_networks(self, qfunc:torch.nn.Module):
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
    
    def _set_seed(self, seed:int):
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)
            self.seed = seed
            self.buffer_seed = seed + 1
        else:
            self.buffer_seed = None
    
    def _modify_action_values(self, action_values:np.ndarray) -> torch.Tensor:
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
            observation: Observation tensor of shape [batch_size or total_paths, obs_dim]. Preprocessed to be float tensor in agents/base.py.
        
        Outputs:
            actions: Tensor of selected actions of shape [batch_size, 1].
        """
        with torch.no_grad():
            q_values = self.q(observation.to(self.device))
            actions = torch.argmax(q_values, dim=-1, keepdim=True)
            
            epsilon = self.epsilon_scheduler.epsilon
            if (epsilon > 0) and self.training_mode:
                is_epsilon_greedy = torch.rand(actions.shape[0], device=self.device) < epsilon
                total_explorers = is_epsilon_greedy.sum().item()
                if total_explorers > 0:
                    actions[is_epsilon_greedy] = torch.randint(
                        0, self.action_dim, (total_explorers, 1),
                        device=self.device, dtype=torch.long, generator=self.generator)
                self.epsilon_scheduler.step()
        
        return actions
    
    def clone_q(self):
        self.target_q.load_state_dict(self.q.state_dict())
    
    def update_q(self):
        """
        To be implemented by subclass. Should perform n_updates training steps on the Q-network using batches sampled from the replay buffer.
        """
        raise NotImplementedError("update_q() must be implemented by subclass.")
    
    def compute_loss_and_update(self):
        """
        To be implemented by subclass. Should compute the loss between current Q-values and HQ targets, perform backpropagation, and update the Q-network parameters.
        """
        raise NotImplementedError("compute_loss_and_update() must be implemented by subclass.")

class RobustDQNBase(DQNBase):
    """
    Extends DQNBase with distributional robustness via the per-sample HQ operator and
    Sinkhorn distance. Implements train_batch using hq_opt_with_nn (one lambda per buffer
    sample, warm-started from cached lambda values).
    
    Subclasses must implement: prepare_target_states, reward_fn, train_mode_actions.
    
    Inputs (in addition to DQNBase, forwarded via **kwargs):
        prior_measure: Prior distribution over returns used as the adversary's support.
        duality_operator: DualityHQOperator for cost, cij, and Sinkhorn radius calculations.
        lamda_init: Initial scalar lambda stored in the buffer for new transitions.
        hq_optimizer: Unused — kept for API parity with PORDQN. Optimizer is created internally.
        hq_lr: Learning rate for the lambda Adam optimizer.
        hq_max_iter: Maximum iterations for the lambda optimization loop.
        hq_step_size: Step interval for LagrangianLambdaScheduler.
        hq_gamma: Learning rate multiplier for LagrangianLambdaScheduler.
    """
    
    def __init__(self, prior_measure:PriorStudentDistribution, duality_operator:DualityHQOperator, lamda_init:float=0.0,
                 hq_optimizer:torch.optim.Optimizer=None, hq_lr:float=0.02, hq_max_iter:int=100,
                 hq_step_size:int=10, hq_gamma:float=10.0, **kwargs):
        
        super().__init__(**kwargs)
        
        self.prior_measure = prior_measure
        self.duality_operator = duality_operator
        
        # HQ Optimization
        self.lamda_init = lamda_init
        self.hq_optimizer = hq_optimizer
        self.hq_lr = hq_lr
        self.hq_max_iter = hq_max_iter
        self.hq_step_size = hq_step_size
        self.hq_gamma = hq_gamma
    
    def _compute_sinkhorn_mask(self, reference_return:torch.Tensor, next_return_from_prior:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute entropy bias-corrected Sinkhorn radius and filter out implausible prior samples.
        Negative radius values indicate samples where the entropic correction exceeds the radius budget.
        
        Inputs:
            reference_return: Realized return from the buffer. Shape: (batch_size, 1).
            next_return_from_prior: Returns sampled from the prior. Shape: (batch_size, n_samples).
        
        Returns:
            epsilon_bar: Corrected Sinkhorn radius per batch element. Shape: (batch_size,).
            mask: Boolean mask of valid samples (epsilon_bar > 0). Shape: (batch_size,).
        """
        cost = self.duality_operator.compute_cost(reference_return, next_return_from_prior)
        epsilon_bar = self.duality_operator.update_sinkhorn_radius(cost)
        mask = epsilon_bar.gt(0)
        if epsilon_bar.lt(0).any():
            print("Warning: Sinkhorn radius is negative for some batches.")
        return epsilon_bar, mask
    
    def _hq_optimize(self, reference_return:torch.Tensor, next_return_from_prior:torch.Tensor, rewards_from_prior:torch.Tensor, optimal_q_targets:torch.Tensor,
                     not_terminal:torch.Tensor, lambda_vals:torch.Tensor, mask:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Run the HQ optimization to compute robust Bellman targets under distributional ambiguity.
        Subclasses implement this with their specific lambda optimization strategy (per-sample or shared).
        
        Contract: lamda_star must be returned as shape (batch_size,) — scalar variants must expand before returning.
        
        Returns:
            hq_value: Robust HQ targets. Shape: (batch_size,).
            lamda_star: Optimized lambdas. Shape: (batch_size,).
            n_iter: Number of optimization iterations.
        """
        raise NotImplementedError("_hq_optimize() must be implemented by subclass.")
    
    def train_batch(self, states:torch.Tensor, actions:torch.Tensor, rewards:torch.Tensor, next_state:torch.Tensor,
                    terminal_states:torch.Tensor, lambda_vals:torch.Tensor, risk_free_rates:torch.Tensor,
                    transaction_costs:torch.Tensor, buffer_idx:torch.Tensor):
        raise NotImplementedError("train_batch() must be implemented by subclass.")
    
    def _cache_lambdas(self, lamdas:torch.Tensor, buffer_idx:torch.Tensor, mask:torch.Tensor):
        buffer_ready_lamdas = self.buffer._device_transfer(lamdas, self.buffer.buffer_device)
        buffer_ready_mask = self.buffer._device_transfer(mask, self.buffer.buffer_device)
        valid_idx = buffer_idx[buffer_ready_mask]
        
        if lamdas.ndim == 1 and lamdas.shape[0] == self.batch_size:
            self.buffer.lambda_val[valid_idx] = buffer_ready_lamdas[buffer_ready_mask]
        elif lamdas.ndim == 2 and lamdas.shape[1] == 1:
            self.buffer.lambda_val[valid_idx] = buffer_ready_lamdas[buffer_ready_mask].squeeze(-1)
        else:
            raise ValueError("lamdas must be of shape (batch_size,) or (batch_size, 1)")
    
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
    
    def update_q(self):
        """
        Update the Q-network by performing n_updates training steps. Note that train_batch() inputs are sampled from the buffer.
        """
        for _ in range(self.n_updates):
            self.q_updates += 1
            states, actions, rewards, next_state, terminal_states, lambda_vals, risk_free_rates, transaction_costs, buffer_idx = self.buffer.sample()
            self.train_batch(states, actions, rewards, next_state, terminal_states, lambda_vals, risk_free_rates, transaction_costs, buffer_idx)
    
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
        row_indices = torch.arange(self.batch_size, device=self.device)
        current_state_q = self.q(current_states)[row_indices, actions.squeeze().to(torch.int64)]
        
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

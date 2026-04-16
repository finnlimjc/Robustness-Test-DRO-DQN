import numpy as np
import torch

from src.agents.dqn import RobustDQNBase
from src.robust.dro import hq_opt_with_nn, hq_opt_shared_lambda

class PORDQN(RobustDQNBase):
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
        training_controller: TrainingController object to manage training steps and target network cloning.
        prior_measure: A prior distribution object representing the prior distribution over returns.
        duality_operator: A DualityHQOperator object that implements the HQ operator and Sinkhorn distance calculations.
        epsilon: Epsilon value for exploration, default is 0.0 which means epsilon-greedy is disabled.
        lamda_init: Initial value for lambda in the HQ optimization, default is 1.0.
        qfunc: Q-network (if None, a default network will be created).
        network_optimizer: Optimizer for the Deep Q-Network (if None, a default Adam optimizer will be created).
        network_lr: Learning rate for the network optimizer, if network_optimizer is not None, this is redundant.
        hq_optimizer: Optimizer for the HQ value optimization (if None, a default Adam optimizer will be created).
        hq_lr: Learning rate for the HQ optimizer, if hq_optimizer is not None, this is redundant.
        device: Device to run the network on, such as 'cuda' or 'cpu'.
        buffer_max_length: Maximum length of the replay buffer, default is 1e6.
        clip_gradients: If set to True, neural network gradient will be clipped at a maximum value of 1.0.
        writer: TensorBoard SummaryWriter for logging (optional).
    """
    
    def reward_fn(self, action:torch.Tensor, next_return_from_prior:torch.Tensor, risk_free_rate:torch.Tensor, transaction_cost:torch.Tensor) -> torch.Tensor:
        """
        Compute reward for a given next return from prior based on action after accounting for transaction costs and remaining cash earning a risk-free rate.
        
        Inputs:
            action: Tensor of shape [batch_size, action_dim] representing the action taken.
            next_return_from_prior: Tensor of shape [batch_size, n_samples] representing the next return sampled from the prior distribution.
            risk_free_rate: Tensor of shape [batch_size] representing the risk-free rate.
            transaction_cost: Tensor of shape [batch_size] representing the transaction cost.
        
        Outputs:
            rewards: Tensor of shape [batch_size, action_dim, n_samples] representing the rewards for each sampled next return.
        """
        #Match Shapes
        modified_next_return_from_prior = next_return_from_prior.unsqueeze(1).expand(-1, self.buffer_action_dim, -1) # (batch_size, action_dim, n_samples)
        modified_action = action.unsqueeze(-1) # (batch_size, action_dim, 1)
        modified_risk_free_rate = risk_free_rate.unsqueeze(-1).unsqueeze(-1) # (batch_size, 1, 1)
        modified_transaction_cost = transaction_cost.unsqueeze(-1).unsqueeze(-1) # (batch_size, 1, 1)
        
        #Compute Reward
        asset_return = modified_next_return_from_prior.exp() - 1.0 # (batch_size, action_dim, n_samples)
        weighted_asset_return = modified_action * asset_return # (batch_size, action_dim, n_samples)
        
        cash_weight = 1.0 - modified_action # (batch_size, action_dim, 1)
        weighted_cash_return = cash_weight * modified_risk_free_rate # (batch_size, action_dim, 1)
        simple_return_after_transaction = torch.clamp(weighted_asset_return + weighted_cash_return - modified_transaction_cost + 1.0, min=1e-6) # (batch_size, action_dim, n_samples)
        reward = simple_return_after_transaction.log() # (batch_size, action_dim, n_samples)
        
        return reward
    
    def prepare_target_states(self, state:torch.Tensor, action:torch.Tensor, next_return_from_prior:torch.Tensor, sampled_states:torch.Tensor, realized_return:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the target state tensor for the target Q-network by updating the current state with the action taken and the results of that action.
        This function is specific to our use case of selected 63 states and should be changed based on the specific state representation and how the next state should be constructed.
        
        Inputs:
            state: Current state tensor of shape [batch_size, state_dim].
            action: Tensor of shape [batch_size, action_dim].
            next_return_from_prior: Tensor of shape [batch_size, sample_size] representing the next return sampled from the prior distribution.
            sampled_states: Tensor of shape [batch_size, state_dim] representing the sampled next states from the buffer.
            realized_return: Tensor of shape [batch_size, action_dim, sample_size] representing the realized return for each sampled next state.

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
    
    def _handle_terminal_state_and_info(self, is_terminal:bool, info:dict=None):
        """
        Handle terminal state and additional info after each step in the environment.
        
        Inputs:
            is_terminal: Boolean indicating whether the current state is terminal.
            info: Additional info from the environment (for this case it is risk-free rate and the transaction cost).
        
        Outputs:
            terminal_tensor: Tensor of shape [batch_size, 1] indicating terminal states.
            risk_free_rate_tensor: Tensor of shape [batch_size, 1] representing the risk-free rate.
            transaction_cost_tensor: Tensor of shape [batch_size, 1] representing the transaction cost.
        """
        risk_free_rate = info.get('risk_free_rate', 0.0)
        transaction_cost = info.get('transaction_cost', 0.0)
        
        terminal_tensor = torch.full((self.total_paths,), is_terminal, dtype=torch.bool, device=self.device)
        
        # Handle numpy arrays and scalar values through reshape and expand
        risk_free_rate_tensor = torch.as_tensor(
            risk_free_rate, dtype=torch.float32, device=self.device
        ).reshape(-1).expand(self.total_paths)
        
        transaction_cost_tensor = torch.as_tensor(
            transaction_cost, dtype=torch.float32, device=self.device
        ).reshape(-1).expand(self.total_paths)
        
        return terminal_tensor, risk_free_rate_tensor, transaction_cost_tensor
    
    def _hq_optimize(self, reference_return:torch.Tensor, next_return_from_prior:torch.Tensor,
                     rewards_from_prior:torch.Tensor, optimal_q_targets:torch.Tensor,
                     not_terminal:torch.Tensor, lambda_vals:torch.Tensor,
                     mask:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        return hq_opt_with_nn(
            self.duality_operator, reference_return, next_return_from_prior, rewards_from_prior,
            optimal_q_targets, not_terminal, lambda_vals, mask,
            optimizer=self.hq_optimizer, lr=self.hq_lr, max_iter=self.hq_max_iter,
            step_size=self.hq_step_size, gamma=self.hq_gamma)
    
    def train_mode_actions(self, reward:torch.Tensor|np.ndarray|float, observation:torch.Tensor|np.ndarray, is_terminal:bool=False, info=None):
        """
        Actions to take when agent in training mode i.e. adding to replay buffer, cloning target q network and training q network
        
        Inputs:
            reward: Reward tensor of shape [total_paths].
            observation: Observation tensor of shape [total_paths, obs_dim].
            terminal: Terminal tensor of shape [total_paths], indicating whether each episode has ended.
            info: Additional info (not used here).
        """
        self.training_controller.step_increment()
        lamda_init = torch.full((self.total_paths,), self.lamda_init, dtype=torch.float32, device=self.device)
        terminal_tensor, risk_free_rate_tensor, transaction_cost_tensor = self._handle_terminal_state_and_info(is_terminal, info)
        self.buffer.add(
            self.prev_state, self.prev_action, reward, observation, terminal_tensor,
            lamda_init, risk_free_rate_tensor, transaction_cost_tensor
        )
        sufficient_samples = self.training_controller.has_samples(len(self.buffer))
        
        if sufficient_samples and self.training_controller.should_clone_q():
            self.clone_q()
        if sufficient_samples and self.training_controller.should_train():
            self.update_q()
    
    def train_batch(self, states:torch.Tensor, actions:torch.Tensor, rewards:torch.Tensor, next_state:torch.Tensor,
                    terminal_states:torch.Tensor, lambda_vals:torch.Tensor, risk_free_rates:torch.Tensor,
                    transaction_costs:torch.Tensor, buffer_idx:torch.Tensor):
        not_terminal = torch.logical_not(terminal_states)
        with torch.no_grad():
            action_weight = self.action_values[actions.squeeze(-1)].unsqueeze(-1)
            next_return_from_prior = self.prior_measure.sample_from_support(self.batch_size)
            rewards_from_prior = self.reward_fn(action_weight, next_return_from_prior, risk_free_rates, transaction_costs)
            target_state, reference_return = self.prepare_target_states(
                states, action_weight, next_return_from_prior, next_state, realized_return=rewards_from_prior)
            optimal_q_targets = self.target_q(target_state).max(dim=-1).values
            epsilon_bar, mask = self._compute_sinkhorn_mask(reference_return, next_return_from_prior)
        
        hq_value, lamda_star, n_iter = self._hq_optimize(
            reference_return, next_return_from_prior, rewards_from_prior,
            optimal_q_targets, not_terminal, lambda_vals, mask)
        
        loss = self.compute_loss_and_update(states, actions, hq_value, mask)
        self._cache_lambdas(lamda_star, buffer_idx, mask)
        if self.writer is not None:
            self.log_indicators(rewards, next_state, not_terminal, hq_value, n_iter, lamda_star, mask)
            self.writer.log_corrected_radius(epsilon_bar, self.q_updates)
            self.writer.log_policy_action(action_weight, self.q_updates, decimal_places=1)
        return loss

class SharedLambdaPORDQN(PORDQN):
    """
    Extends PORDQN to optimize a single shared scalar lambda over the batch, rather than one
    lambda per buffer sample. The scalar lambda is updated each training step and serves as the
    warm start for the next call, replacing the per-sample lambda caching in the replay buffer.
    
    Inputs (in addition to PORDQN, forwarded via **kwargs):
        loss_tol: Tolerance for convergence in the HQ optimization routine.
    """
    
    def __init__(self, loss_tol:float=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.loss_tol = loss_tol
        self.lambda_val = torch.tensor(float(self.lamda_init), dtype=torch.float32, device=self.device)
    
    def _hq_optimize(self, reference_return:torch.Tensor, next_return_from_prior:torch.Tensor, rewards_from_prior:torch.Tensor, optimal_q_targets:torch.Tensor,
                     not_terminal:torch.Tensor, lambda_vals:torch.Tensor, mask:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        hq_value, lamda_star, n_iter = hq_opt_shared_lambda(
            self.duality_operator, reference_return, next_return_from_prior, rewards_from_prior,
            optimal_q_targets, not_terminal, self.lambda_val, mask,
            lr=self.hq_lr, max_iter=self.hq_max_iter, step_size=self.hq_step_size, gamma=self.hq_gamma, loss_tol=self.loss_tol)
        
        # Expand scalar to (batch_size,) for uniform interface with PORDQN
        return hq_value, lamda_star.expand(self.batch_size), n_iter
    
    def _cache_lambdas(self, lamdas:torch.Tensor, buffer_idx:torch.Tensor, mask:torch.Tensor):
        self.lambda_val = lamdas[0]  # all elements identical; store scalar warm-start for next step

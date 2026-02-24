import torch
import torch.nn as nn
from src.schedulers import *

class DualityHQOperator:
    """
    Calculates the components required for the duality-based HQ operator.
    
    Inputs:
        discount_rate: Discount factor for future rewards.
        delta: Entropy regularization coefficient for the Sinkhorn Ball.
        sinkhorn_radius: Sinkhorn radius (epsilon).
    """
    def __init__(self, discount_rate:float, delta:float, sinkhorn_radius:float, norm_order:int=1):
        self.discount_rate = discount_rate
        self.delta = delta
        self.sinkhorn_radius = sinkhorn_radius
        self.norm_order = norm_order
    
    def compute_cost(self, reference_r:torch.Tensor, prior_r:torch.Tensor) -> torch.Tensor:
        """
        Computes the norm cost between reference and prior returns.
        
        Inputs:
            reference_r: Returns sampled from the next_state in the buffer, note that we only need the returns not the full state, expected shape of (batch_size, 1).
            prior_r: Returns sampled from the support of the prior distribution, expected shape of (batch_size, n_samples).
            norm_order: Order of the norm to compute cost, where 1 is L1 norm and 2 is L2 norm.
        
        Outputs:
            cost: Norm cost between reference and prior returns, shape of (batch_size, n_samples).
        """
        distance = (reference_r - prior_r).unsqueeze(-1) #(batch_size, n_samples, 1)
        cost = torch.linalg.norm(distance, ord=self.norm_order, dim=-1) #(batch_size, n_samples)
        
        expected_shape = (reference_r.size(0), prior_r.size(1))
        assert cost.shape == expected_shape, f"Cost shape {cost.shape} does not match expected shape {expected_shape}."
        
        return cost #(batch_size, n_samples)
    
    def compute_cij(self, prior_reward:torch.Tensor, q_max:torch.Tensor, not_terminal:torch.Tensor, lamda:torch.Tensor, cost:torch.Tensor) -> torch.Tensor:
        """
        Calculates the exponential term cij used in the inner expectation of the HQ operator.
        
        Inputs:
            prior_reward: Returns sampled from the support of the prior distribution fed into the reward function, expected shape of (batch_size, action_dim, n_samples).
            q_max: Maximum Q-values for the sampled next states, expected shape of (batch_size, n_samples).
            not_terminal: Tensor indicating whether the next state is terminal, expected shape of (batch_size).
            lamda: Lagrangian multiplier lambda for the HQ operator, expected shape of (batch_size).
            cost: Norm cost between reference and prior returns, expected shape of (batch_size, n_samples)
        
        Outputs:
            cij: Exponential term used in the inner expectation, shape of (batch_size, n_samples).
        """
        if torch.isnan(prior_reward).any() or torch.isinf(prior_reward).any():
            raise ValueError(f"Numerical instability in prior_reward, max: {prior_reward.max()}, min: {prior_reward.min()}")
        assert prior_reward.shape[1] == 1, "Current implementation for Prior Reward only accept trading of one asset, expect size of (batch_size, 1, n_samples)"
        
        # Prevent silent broadcasting errors
        not_terminal = not_terminal.unsqueeze(-1) #(batch_size, 1)
        lamda = lamda.unsqueeze(-1) #(batch_size, 1)
        prior_reward = prior_reward.squeeze(1)
        
        discounted_return = prior_reward + self.discount_rate*q_max*not_terminal # (batch_size, n_samples)
        first_second_term = - discounted_return / (self.delta * lamda)
        third_term = cost/self.delta #lambda cancels out for the third term
        cij = first_second_term - third_term
        
        expected_shape = cost.shape
        assert cij.shape == expected_shape, f"Exponential term (cij) shape {cij.shape} does not match expected shape {expected_shape}."
        
        return cij # (batch_size, n_samples)
    
    def inner_expectation(self, val:torch.Tensor) -> torch.Tensor:
        """
        Computes the inner expectation using log-sum-exp for numerical stability and deducting the log of the number of samples due to log rules.
        
        Inputs:
            val: Exponential term for the update of the Sinkhorn radius (epsilon) or HQ operator, expected shape of (batch_size, n_samples).
        """
        total_samples = torch.tensor(val.size(1), dtype=torch.int64, device=val.device)
        log_mean_exp = torch.logsumexp(val, dim=1) - torch.log(total_samples)
        if torch.isnan(log_mean_exp).any() or torch.isinf(log_mean_exp).any():
            raise ValueError(f'Numerical instability in inner_expectation, max: {log_mean_exp.max()}, min: {log_mean_exp.min()}')
        
        return log_mean_exp # (batch_size)
    
    def outer_expectation(self):
        """
        This is not required as we assume that reference returns come from one probability distribution, so there is no mixture weights to calculate the expectation.
        """
        pass
    
    def update_sinkhorn_radius(self, cost:torch.Tensor) -> torch.Tensor:
        """
        Updates the Sinkhorn radius (epsilon) to correct for entropic bias. This mainly acts as a validity constraint to ensure that the entropy bias corrected sinkhorn radius is valid.
        Essentially, this cannot be negative as a negative radius does not make sense in the context of a Sinkhorn ball.
        
        Inputs:
            cost: Norm cost between reference and prior returns, expected shape of (batch_size, n_samples).
        
        Outputs:
            epsilon_bar: Updated Sinkhorn radius after entropy bias correction, shape of (batch_size).
        """
        
        exp_term = -cost/self.delta #(batch_size, n_samples)
        log_mean_exp = self.inner_expectation(exp_term) #(batch_size)
        epsilon_bar = self.sinkhorn_radius + self.delta*log_mean_exp #(batch_size)
        return epsilon_bar
    
    def hq_value(self, lamda_plus:torch.Tensor, inner_exp:torch.Tensor) -> torch.Tensor:
        """
        Calculates the HQ value.
        
        Inputs:
            lamda_plus: Positive part of the Lagrangian multiplier lambda for the HQ operator given by log[1 + exp(lamda)], expected shape of (batch_size).
            inner_exp: Inner expectation value retrieved from self.inner_expectation(), expected shape of (batch_size).
        
        Outputs:
            hq_value: HQ value, shape of (batch_size).
        """
        val = -lamda_plus* (self.sinkhorn_radius + self.delta*inner_exp) # (batch_size)
        return val

class DualObjective(nn.Module): 
    """
    Neural network module to compute the dual objective (lambda^+) for the HQ operator.
    
    Inputs:
        duality_operator: DualityHQOperator object to compute components of the HQ operator.
        reference_r: Returns sampled from the next_state in the buffer, note that we only need the returns not the full state, expected shape of (batch_size, 1).
        prior_r: Returns sampled from the support of the prior distribution, expected shape of (batch_size, n_samples).
        prior_reward: Reward of prior_r, expected shape of # (batch_size, action_dim, n_samples).
        q_max: Maximum Q-values for the sampled next states, expected shape of (batch_size, n_samples).
        not_terminal: Tensor indicating whether the next state is terminal, expected shape of (batch_size).
        norm_order: Order of the norm to compute cost, where 1 is L1 norm and 2 is L2 norm.
    
    Outputs:
        hq_value: HQ value, shape of (batch_size).
    """
    def __init__(self, duality_operator:DualityHQOperator, reference_r:torch.Tensor, prior_r:torch.Tensor, prior_reward:torch.Tensor, q_max:torch.Tensor, not_terminal:torch.Tensor):
        super().__init__()
        self.duality_operator = duality_operator
        self.reference_r = reference_r
        self.prior_r = prior_r
        self.prior_reward = prior_reward
        self.q_max = q_max
        self.not_terminal = not_terminal
        
        self.softplus = nn.Softplus()
    
    def forward(self, lamda:torch.Tensor):
        lamda_plus = self.softplus(lamda) #(batch_size)
        cost = self.duality_operator.compute_cost(self.reference_r, self.prior_r) #(batch_size, n_samples)
        cij = self.duality_operator.compute_cij(self.prior_reward, self.q_max, self.not_terminal, lamda_plus, cost) #(batch_size, n_samples)
        inner_exp = self.duality_operator.inner_expectation(cij) #(batch_size)
        hq_value = self.duality_operator.hq_value(lamda_plus, inner_exp) #(batch_size)
        return hq_value

class OptimizeLamda:
    """
    Optimizer for the Lagrangian multiplier lambda in the HQ operator using gradient ascent.
    
    Inputs:
        dual_objective: DualObjective module to compute the HQ value.
        lr: Learning rate for the optimizer.
        max_iter: Maximum number of iterations for optimization.
        step_size: Step size for the learning rate scheduler.
        gamma: Decay factor for the learning rate scheduler.
    """
    def __init__(self, dual_objective:DualObjective, lr:float, max_iter:int, step_size:int, gamma:float):
        self.dual_objective = dual_objective
        self.lr = lr
        self.max_iter = max_iter
        self.step_size = step_size
        self.gamma = gamma
    
    def optimize(self, lamda_from_buffer:torch.Tensor, lamda_mask:torch.Tensor, optimizer:torch.optim.Optimizer=None) -> tuple[torch.Tensor, int]:
        """
        Performs gradient ascent to optimize lambda and returns the optimized lambda value.
        
        Inputs:
            lamda_from_buffer: Initial lambda values from the replay buffer, shape of (batch_size).
            lamda_mask: Boolean mask indicating which lambda values are still being optimized, shape of (batch_size).
            optimizer: Optional optimizer to use for gradient ascent. If None, Adam optimizer will be used.
        
        Outputs:
            target: Optimized lambda value.
            iter: Number of iterations performed.
        """
        target = nn.Parameter(lamda_from_buffer.clone())
        if optimizer is None:
            optimizer = torch.optim.Adam([target], lr=self.lr)
        scheduler = LagrangianLambdaScheduler(optimizer, step_size=self.step_size, gamma=self.gamma, init_lr=self.lr, init_lamda=None)
        
        prev_grad = None #Early stopping if gradient changes sign
        
        for iter in range(self.max_iter):
            hq = self.dual_objective(target)[lamda_mask]
            loss = -hq.sum() #We use negative as we want to maximize hq_value
            optimizer.zero_grad()
            loss.backward()
            grad = target.grad.detach().clone()
            
            if prev_grad is not None:
                change_sign = (grad * prev_grad < 0)
                active = (lamda_mask & change_sign & (target > -6))
                lamda_mask = active #Tensor of shaoe (batch_size)
            
            if not lamda_mask.any():
                break
            
            prev_grad = grad
            optimizer.step()
            scheduler.step()
        
        return target.detach(), iter+1

def hq_opt_with_nn(duality_operator:DualityHQOperator, reference_r:torch.Tensor, prior_r:torch.Tensor, prior_reward:torch.Tensor, q_max:torch.Tensor, not_terminal:torch.Tensor, 
                   lamda_from_buffer:torch.Tensor, lambda_mask:torch.Tensor, optimizer:torch.optim.Optimizer=None,
                   lr:float=0.02, max_iter:int=100, step_size:int=10, gamma:float=10.0) -> torch.Tensor:
    """
    HQ Optimizer by optimizing the Lagrangian Lambda using a neural network and a scheduler.
    This is called per update step which happens at each time step for n episodes.
    Therefore, this resets to the original state at each time step.
    """
    
    dual_obj = DualObjective(duality_operator, reference_r, prior_r, prior_reward, q_max, not_terminal)
    opt = OptimizeLamda(dual_obj, lr=lr, max_iter=max_iter, step_size=step_size, gamma=gamma)
    lamda_star, n_iter = opt.optimize(lamda_from_buffer, lambda_mask, optimizer=optimizer)
    
    with torch.no_grad():
        hq_value = dual_obj(lamda_star)
    
    return hq_value, lamda_star, n_iter
    
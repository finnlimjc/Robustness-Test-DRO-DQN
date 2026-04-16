import torch
import torch.nn as nn

from src.schedulers import LagrangianLambdaScheduler

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
    
    def _check_numerical_stability(self, tensor:torch.Tensor, name:str):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError(f"Numerical instability in {name}, max: {tensor.max()}, min: {tensor.min()}")
    
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
        self._check_numerical_stability(prior_reward, "prior_reward")
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
        self._check_numerical_stability(log_mean_exp, "inner_expectation")
        
        return log_mean_exp # (batch_size)
    
    def outer_expectation(self):
        """
        Replay buffer sample is already an approximation for the outer expectation.
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
    
    def _compute_hq_components(self, lamda:torch.Tensor) -> tuple:
        lamda_plus = self.softplus(lamda) #(batch_size)
        cost = self.duality_operator.compute_cost(self.reference_r, self.prior_r) #(batch_size, n_samples)
        cij = self.duality_operator.compute_cij(self.prior_reward, self.q_max, self.not_terminal, lamda_plus, cost) #(batch_size, n_samples)
        inner_exp = self.duality_operator.inner_expectation(cij) #(batch_size)
        return lamda_plus, inner_exp
    
    def forward(self, lamda:torch.Tensor):
        lamda_plus, inner_exp = self._compute_hq_components(lamda)
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
    
    def _build_optimizer_and_scheduler(self, params:list):
        optimizer = torch.optim.Adam(params, lr=self.lr)
        scheduler = LagrangianLambdaScheduler(optimizer, step_size=self.step_size, gamma=self.gamma, init_lr=self.lr)
        return optimizer, scheduler
    
    def optimize(self, lamda_from_buffer: torch.Tensor, lamda_mask: torch.Tensor, optimizer=None):
        batch_size = lamda_from_buffer.shape[0]
        
        # Create scalar parameters
        lamda = [
            nn.Parameter(lamda_from_buffer[i].clone().detach(),
                         requires_grad=True)
            for i in range(batch_size)
        ]
        
        # Per-parameter optimizer groups
        optim_input = [{'params': [p]} for p in lamda]
        optimizer, scheduler = self._build_optimizer_and_scheduler(optim_input)
        
        # Boolean mask (torch, same device)
        lamda_opt = lamda_mask.clone()
        prev_grad = None
        iter_count = 0
        
        while lamda_opt.any():
            lamda_tensor = torch.stack(lamda)
            
            # Compute HQ
            hq = self.dual_objective(lamda_tensor)
            loss = (-hq[lamda_opt]).sum()
            optimizer.zero_grad()
            loss.backward()
            
            # Collect gradients (torch tensors)
            grads = torch.stack([
                lamda[i].grad if lamda[i].grad is not None
                else torch.zeros_like(lamda[i])
                for i in range(batch_size)
            ])
            
            # Stopping condition
            if prev_grad is not None:
                same_sign = grads * prev_grad > 0
                lower_bound_ok = (lamda_tensor > -6) | (prev_grad < 0)
                lamda_opt = lamda_opt & same_sign & lower_bound_ok
            
            # Freeze converged lambdas
            for i in range(batch_size):
                if not lamda_opt[i]:
                    lamda[i].requires_grad = False
            
            prev_grad = grads.detach()
            optimizer.step()
            scheduler.step()
            
            iter_count += 1
            if iter_count >= self.max_iter:
                break
        
        lamda_final = torch.stack(lamda).detach()
        
        return lamda_final, iter_count

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

class SharedLambdaDualObjective(DualObjective):
    """
    Variant of DualObjective that treats the batch as a single empirical distribution.
    Returns a scalar HQ value by averaging inner_exp over valid batch samples (mask=True),
    rather than returning per-sample HQ values of shape (batch_size,).
    
    Inputs:
        mask: Boolean tensor of shape (batch_size,) indicating valid samples (epsilon_bar > 0).
        All other inputs are identical to DualObjective.
    """
    
    def __init__(self, duality_operator:DualityHQOperator, reference_r:torch.Tensor, prior_r:torch.Tensor,
                 prior_reward:torch.Tensor, q_max:torch.Tensor, not_terminal:torch.Tensor, mask:torch.Tensor):
        super().__init__(duality_operator, reference_r, prior_r, prior_reward, q_max, not_terminal)
        self.mask = mask
    
    def forward(self, lamda:torch.Tensor):
        """
        Inputs:
            lamda: Scalar tensor (0-dim). Broadcasts correctly in compute_cij via unsqueeze(-1) -> shape (1,).
        Outputs:
            hq_value: Scalar HQ value representing the batch-level empirical dual objective.
        """
        lamda_plus, inner_exp = self._compute_hq_components(lamda)
        mean_inner_exp = inner_exp[self.mask].mean() #scalar — valid samples only
        return self.duality_operator.hq_value(lamda_plus, mean_inner_exp) #scalar

class SharedLambdaOptimizer(OptimizeLamda):
    """
    Variant of OptimizeLamda that optimizes a single shared scalar lambda over the full batch.
    Uses SharedLambdaDualObjective (scalar HQ) as the objective.
    
    Convergence: stops when the change in loss between iterations falls below loss_tol.
    """
    
    def __init__(self, dual_objective:SharedLambdaDualObjective, lr:float, max_iter:int, step_size:int, gamma:float, loss_tol:float=1e-6):
        super().__init__(dual_objective, lr, max_iter, step_size, gamma)
        self.loss_tol = loss_tol
    
    def optimize(self, lamda_from_buffer:torch.Tensor, optimizer=None):
        """
        Inputs:
            lamda_from_buffer: Scalar tensor used as warm start (agent-level self.lambda_val).
            optimizer: Unused — a new Adam optimizer is created internally each call.
        Outputs:
            lamda_final: Optimized scalar tensor.
            iter_count: Number of iterations taken.
        """
        lamda = nn.Parameter(lamda_from_buffer.clone().detach(), requires_grad=True)
        optimizer, scheduler = self._build_optimizer_and_scheduler([{'params': [lamda]}])
        
        prev_loss = None
        iter_count = 0
        
        while True:
            hq = self.dual_objective(lamda) #scalar
            loss = -hq
            optimizer.zero_grad()
            loss.backward()
            
            if prev_loss is not None and abs(loss.item() - prev_loss) < self.loss_tol:
                break
            
            prev_loss = loss.item()
            optimizer.step()
            scheduler.step()
            iter_count += 1
            if iter_count >= self.max_iter:
                break
        
        return lamda.detach(), iter_count

def hq_opt_shared_lambda(duality_operator:DualityHQOperator, reference_r:torch.Tensor, prior_r:torch.Tensor,
                         prior_reward:torch.Tensor, q_max:torch.Tensor, not_terminal:torch.Tensor,
                         lamda_from_buffer:torch.Tensor, lambda_mask:torch.Tensor, optimizer:torch.optim.Optimizer=None,
                         lr:float=0.02, max_iter:int=100, step_size:int=10, gamma:float=10.0, loss_tol:float=1e-3):
    """
    HQ Optimizer treating the batch as a single empirical distribution with one shared lambda.
    Optimizes lambda over the batch-level dual objective (average over valid samples), then
    recomputes per-sample HQ_i(lambda*) values to use as individual Q-network TD targets.
    
    Outputs:
        hq_values: Per-sample HQ values of shape (batch_size,) for Q-network loss.
        lamda_star: Optimized scalar lambda tensor.
        n_iter: Number of optimization iterations taken.
    """
    dual_obj = SharedLambdaDualObjective(duality_operator, reference_r, prior_r, prior_reward, q_max, not_terminal, lambda_mask)
    opt = SharedLambdaOptimizer(dual_obj, lr=lr, max_iter=max_iter, step_size=step_size, gamma=gamma, loss_tol=loss_tol)
    lamda_star, n_iter = opt.optimize(lamda_from_buffer, optimizer=optimizer)
    
    with torch.no_grad():
        lamda_plus, inner_exp = dual_obj._compute_hq_components(lamda_star)
        hq_values = duality_operator.hq_value(lamda_plus, inner_exp) #(batch_size,)
    
    return hq_values, lamda_star, n_iter

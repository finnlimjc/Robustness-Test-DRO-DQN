import torch
from typing import Callable, Optional, List
import numpy as np
import warnings
import torch.nn as nn

def inner_expectation(lamda:torch.Tensor,
                      delta:float,
                      discount:float,
                      r:torch.Tensor,
                      q_max:torch.Tensor,
                      cost:torch.Tensor,
                      not_terminal:torch.Tensor=None,
                      modified_method:bool=True):
    '''
    Calculates the inner expectation of the HQ operator
    Parameters:
        lamda: torch.Tensor, shape (batch_size)
        delta: float, entropy regularisation coefficient
        discount: float, discount factor
        r: torch.Tensor, shape (batch_size, n_outer, n_inner)
        q_max: torch.Tensor, shape (batch_size, n_outer, n_inner)
        cost: torch.Tensor, shape (batch_size, n_outer, n_inner)
        not_terminal: torch.Tensor, shape (batch_size, 1)
        modified_method: If True, uses logsumexp to calculate result
    '''
    not_terminal = not_terminal.unsqueeze(-1)
    lamda = lamda.unsqueeze(-1).unsqueeze(-1)
    
    # lambda cancels out for delta_cost
    delta_cost = (cost / delta)
    cij = (-r - discount*q_max*not_terminal) / (delta * lamda) - delta_cost
    
    if modified_method:
        log_meanexp = torch.logsumexp(cij, dim=2, keepdim=True) - np.log(cij.size(2))
        result = log_meanexp.mean(dim=1)
    
    else:
        c = cij.amax(dim=2, keepdim=True)  # amax is faster than max()[0]
        exponent = torch.exp(cij - c)
        inner_exp = exponent.mean(dim=2) # average across N_nu
        result = (torch.log(inner_exp) + c.squeeze(-1)).mean(dim=1)
    
        if torch.isnan(inner_exp).any() or torch.isinf(inner_exp).any():
            raise ValueError(f'Numerical instability in inner_expectation, max: {inner_exp.abs().max()}, min: {inner_exp.abs().min()}, c: {c.abs().max()}, cij: {cij.abs().max()}, lambda: {lamda.abs().max()}, delta_cost: {delta_cost.abs().max()}')

    return result


def hq_max(lamda:List[nn.parameter.Parameter],
           epsilon:float,
           delta:float,
           discount:float,
           r:torch.tensor,
           q_max:torch.tensor,
           cost:torch.tensor,
           not_terminal:torch.tensor=None,
           lamda_opt=None,
           modified_method:bool=True):
    '''
    Calculates the HQ value to be maximised
    Parameters:
        lamda: torch.Tensor, list of lambda parameters of length batch_size
        epsilon: float, sinkhorn distance
        delta: float, entropy regularisation coefficient
        discount: float, discount factor
        r: torch.Tensor, shape (batch_size, n_outer, n_inner)
        q_max: torch.Tensor, shape (batch_size, n_outer, n_inner)
        cost: torch.Tensor, shape (batch_size, n_outer, n_inner)
        not_terminal: torch.Tensor, shape (batch_size, 1)
        lamda_opt: torch.Tensor, shape (batch_size), boolean mask for which lambdas to optimise
        modified_method: If True, uses logsumexp to calculate result
    '''
    if lamda_opt is None:
        lamda_opt = np.array([True for _ in range(len(lamda))])
    lamda_plus = torch.nn.Softplus()(torch.stack(lamda)[lamda_opt]) # enforce positivity
    outer_exp = inner_expectation(lamda_plus, delta, discount, r[lamda_opt], q_max[lamda_opt], cost[lamda_opt], not_terminal[lamda_opt], modified_method=modified_method)
    hq = -lamda_plus * (epsilon + delta * outer_exp)
    return hq


def hq_opt(
    qfunc: torch.nn.Module,
    states: torch.Tensor, # (batch_size, state_dim)
    act_values: torch.Tensor, # (batch_size, action_dim),
    next_states: torch.Tensor,
    nu: Callable,
    reward_fn: Callable,
    epsilon: float,
    delta: float,
    n_outer: int,
    n_inner: int,
    discount: float=1.0,
    lamda_init: float=1.0,
    lamda_max_iter: int=1000,
    lr: float=0.1,
    step_size: int=100,
    gamma: float=0.1,
    norm_ord: int=1, # order of norm for cost function
    not_terminal: Optional[torch.Tensor]=None,
    modify_states: Optional[Callable]=None,
    device='cpu',
    modified_method:bool=True
):
    '''
    Optimises the HQ operator for a given q-function, states, action_space, nu, reward_fn, transition_fn
    Parameters:
        qfunc: torch.nn.Module, q-function neural network
        states: torch.Tensor, shape (batch_size, state_dim)
        act_values: torch.Tensor, shape (batch_size, action_dim), provide action values
        nu: function, sampling function for optimisation in dual space: states, actions, n_outer, n_inner -> nu_states
        reward_fn: function, reward function: states, actions, next_states -> rewards
        epsilon: float, sinkhorn distance
        delta: float, sinkhorn entropy regularisation coefficient
        n_outer: int, number of samples in outer expectation
        n_inner: int, number of samples in inner expectation
        discount: float, discount factor
        lamda_init: float, initial value of lambda
        lamda_max_iter: int, maximum number of iterations for lambda optimisation
        lr: float, learning rate for lambda
        step_size: int, step size for learning rate scheduler
        gamma: float, gamma for learning rate scheduler
        norm_ord: int, order of norm for cost function
        not_terminal: whether the state is not a terminal state, torch.Tensor, shape (batch_size, 1)
        next_states: torch.Tensor, shape (batch_size, n_outer, state_dim), provide next states which will be used instead of transition_fn
        modify_states: function, function to expand states for q-function in mmd environment
        device: str, device to run optimisation
        modified_method: If True, uses logsumexp to calculate result
    '''
    batch_size = states.shape[0]

    with torch.no_grad():
        # sample from Student t-distribution
        nu_states = nu(states, act_values, n_outer, n_inner).to(device) # (batch_size, n_outer, n_inner, 1)
        nu_rewards = reward_fn(states, act_values, nu_states).to(device).squeeze(-1) # (batch_size, n_outer, n_inner)
        
        # calculate the next state
        modified_states, next_states = modify_states(states, act_values, next_states, nu_states, nu_rewards)
        
        # Qnn(Qtarget; Xij; b)
        nu_q_value = qfunc(modified_states.to(device)) # (batch_size, n_outer, n_inner, action_dim)
        
        # sup Qnn(Qtarget; Xij; b)
        nu_act_idx = nu_q_value.argmax(dim=-1).unsqueeze(-1) # (batch_size, n_outer, n_inner, 1)
        q_max = torch.gather(nu_q_value, -1, nu_act_idx).squeeze(-1) # (batch_size, n_outer, n_inner)
        
        # || Xi - Xij ||
        cost = torch.linalg.norm(next_states.unsqueeze(2) - nu_states, ord=norm_ord, dim=-1) # (batch_size, n_outer, n_inner)
        
        # raise a warning if epsilon bar < 0
        frac = -(cost/delta)
        c = frac.amax(dim=2, keepdim=True) # max across N_nu dimensions
        ebar = epsilon + delta * (c.squeeze(-1) + torch.log(torch.exp(frac-c).mean(dim=2))).mean(dim=1)
        ebar_pos = (ebar > 0).squeeze()
        
        if torch.any(ebar < 0):
            warnings.warn("epsilon bar has negative values!")
    
    lamda = []
    if isinstance(lamda_init, float):
        for _ in range(batch_size):
            lamda.append(nn.Parameter(torch.tensor(lamda_init, device=device), requires_grad=True))
    elif isinstance(lamda_init, torch.Tensor): # using cached lambdas
        if lamda_init.shape[0] != batch_size:
            raise ValueError(f'lamda_init shape {lamda_init.shape} does not match batch size {batch_size}')
        for i in range(batch_size):
            lamda.append(nn.Parameter(lamda_init[i].squeeze(), requires_grad=True))
    else:
        raise ValueError(f'lamda_init type {type(lamda_init)} not supported')
    
    optim_input = [{'params': [p]} for p in lamda]
    lamda_optim = torch.optim.Adam(optim_input, lr=lr)
    lamda_sched = LambdaScheduler(lamda_optim, step_size=step_size, gamma=gamma, init_lr=lr)
    
    # remove lambdas where epsilon bar is negative
    lamda_opt = ebar_pos.clone().numpy()
    if (~lamda_opt).any():
        for i in range(batch_size):
            if not lamda_opt[i]:
                lamda[i].requires_grad = False
    
    # optimise lambdas
    iter_count = 0
    lamda_grad = None
    while lamda_opt.any():
        hq = hq_max(lamda, epsilon, delta, discount, nu_rewards, q_max, cost, not_terminal, lamda_opt=lamda_opt, modified_method=modified_method)
        loss_sum = (-hq).sum()
        lamda_optim.zero_grad()
        loss_sum.backward()
        
        # get the gradient of lambda
        new_lamda_grad = [lamda[i].grad.clone().detach().cpu() if lamda[i].grad is not None else None for i in range(batch_size)]
        
        # determine which lambdas still require optimisation
        # use change of the sign in the gradient as a stopping criterion
        # gradient of lambda gets close to zero when lambda < 0
        if lamda_grad is not None:
            lamda_opt = np.array([new_lamda_grad[i] is not None and lamda_grad[i] * new_lamda_grad[i] > 0. and (lamda[i] > -6. or lamda_grad[i] < 0.) for i in range(batch_size)])
        
        # determine which lambdas have just converged and need to be frozen
        requires_mod = (~lamda_opt) & [lamda[i].requires_grad for i in range(batch_size)]

        lamda_grad = new_lamda_grad
        lamda_optim.step()

        # freeze lambdas that have converged
        if requires_mod.any():
            for i in range(batch_size):
                if requires_mod[i]:
                    lamda[i].requires_grad = False
        
        lamda_sched.step()
        iter_count += 1
        
        # maximum iteration limit
        if iter_count > lamda_max_iter:
            break
        
    with torch.no_grad():
        hq = hq_max(lamda, epsilon, delta, discount, nu_rewards, q_max, cost, not_terminal, modified_method=modified_method)

    return hq, torch.stack(lamda).clone().detach().cpu(), iter_count, ebar_pos


class LambdaScheduler:
    def __init__(self, optimizer, step_size, gamma, init_lr, init_lamda=None):
        self.optimizer = optimizer
        self.step_size = step_size # interval at which change in learning rate occurs
        self.gamma = gamma # multiplication factor for increasing the learning rate
        self.init_lr = init_lr # base (small) learning rate used initially
        self.init_lamda = init_lamda # starting value for lambda parameters
        self.last_epoch = 1

    def step(self):
        '''
        Use small init_lr for first step_size epochs to refine cached lambdas
        EXCEPT for the following which are set to init_lr * gamma:
        1. For cached lambdas that are < -1 (equivalent to lambda_plus=0.31) and headed down, i.e. param.grad > 0
        2. For cached lambdas that are < -3 (equivalent to lambda_plus=0.05)
        3. Non-cached lambdas i.e. set to init_lamda
        After step_size epochs, increase lr to init_lr * gamma for all lambdas
        Thereafter, increase lr by a factor of gamma for lambdas that are < 0 and headed down every step_size epochs
        '''
        if self.last_epoch == 1:
            for param_group in self.optimizer.param_groups:
                param = param_group['params'][0]
                if not param.requires_grad: # if parameter is frozen, skip it
                    continue
                if param.dim() == 0:
                    if (self.init_lamda is not None and param.data == self.init_lamda) or (param.data < -3.) or (param.data < -1. and param.grad > 0.):
                        param_group['lr'] = self.init_lr * self.gamma
        elif self.last_epoch == self.step_size:
            for param_group in self.optimizer.param_groups:
                param = param_group['params'][0]
                if not param.requires_grad:
                    continue
                param_group['lr'] = self.init_lr * self.gamma
        elif self.last_epoch % self.step_size == 0:
            for param_group in self.optimizer.param_groups:
                param = param_group['params'][0]
                if not param.requires_grad:
                    continue
                if param.dim() == 0 and param.data < 0. and param.grad > 0.:
                    param_group['lr'] *= self.gamma
        self.last_epoch += 1
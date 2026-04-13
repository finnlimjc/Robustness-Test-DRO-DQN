import numpy as np
import torch

from src.utils.validation import check_modify_obs, check_modify_reward

class AgentInterface:
    def __init__(self):
        self.training_mode = True
        self.prev_state = None
        self.prev_action = None
    
    def agent_start(self, observation:torch.Tensor|np.ndarray) -> torch.Tensor:
        observation = check_modify_obs(observation)
        action = self.get_action(observation)
        
        #Initialize previous state and action as current
        self.prev_state = observation
        self.prev_action = action
        return action # Size depends on get_action()
    
    def agent_step(self, reward:torch.Tensor|np.ndarray|float, observation:torch.Tensor|np.ndarray, info=None) -> torch.Tensor:
        IS_TERMINAL = False
        observation = check_modify_obs(observation)
        reward = check_modify_reward(reward)
        action = self.get_action(observation)
        
        if self.training_mode:
            self.train_mode_actions(reward, observation, IS_TERMINAL, info)
        
        # Update previous state and action
        self.prev_state = observation
        self.prev_action = action
        return action # Size depends on get_action()
    
    def agent_end(self, reward:torch.Tensor|np.ndarray|float, observation:torch.Tensor|np.ndarray, info=None):
        IS_TERMINAL = True
        observation = check_modify_obs(observation)
        reward = check_modify_reward(reward)
        
        if self.training_mode:
            self.train_mode_actions(reward, observation, IS_TERMINAL, info)
    
    # ---- Implement by child class ----
    def get_action(self, observation:torch.Tensor|np.ndarray) -> torch.Tensor:
        raise NotImplementedError("get_action() not implemented.")
    
    def prepare_target_states(self, sampled_states:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        """
        Prepare the target state tensor for the target Q-network by updating the sampled states with the action taken and the results of that action.
        """
        raise NotImplementedError("prepare_target_state() not implemented.")
    
    def reward_fn(self) -> torch.Tensor:
        """
        Reward function should depend on the action taken and treating the sampled next state as the actual next state.
        This is a placeholder function and should be implemented based on the specific environment and reward structure.
        """
        raise NotImplementedError("reward_fn() not implemented.")
    
    def train_mode_actions(self, reward:torch.Tensor|np.ndarray|float, observation:torch.Tensor|np.ndarray, is_terminal:bool, info=None):
        """
        Actions to take when agent in training mode i.e. adding to replay buffer, cloning target q network and training q network
        """
        raise NotImplementedError("train_mode_actions() not implemented.")

class TrainingController:
    """
    Controller for training steps, cloning target q network and checking buffer length. If you do not wish to use all features, you can set the remaining input to be None.
    
    Inputs:
        train_steps: Number of steps between training the network.
        clone_steps: Number of steps between cloning the target network.
        batch_size: Batch size for training.
        n_updates: Number of batches for training.
    
    """
    
    def __init__(self, train_steps:int=None, clone_steps:int=None, batch_size:int=None, n_updates:int=None):
        self.train_steps = train_steps
        self.clone_steps = clone_steps
        self.batch_size = batch_size
        self.n_updates = n_updates
        self.steps = 0
    
    def step_increment(self):
        self.steps += 1
    
    def has_samples(self, buffer_len:int) -> bool:
        return buffer_len >= self.batch_size * self.n_updates
    
    def should_train(self) -> bool:
        is_train_step = (self.steps % self.train_steps == 0)
        return is_train_step
    
    def should_clone_q(self) -> bool:
        is_clone_step = (self.steps % self.clone_steps == 0)
        return is_clone_step

class ReplayBuffer:
    """
    Replay buffer to store experience tuples for training the agent.
    Memory is stored in the CPU to prevent excessive memory overhead.
    
    Inputs:
        state_dim: Dimension of the state space.
        action_dim: Dimension of the action space.
        total_paths: Number of synthetic paths created in the environment, which affects add().
        batch_size: Batch size for sampling.
        max_len: Maximum length of the replay buffer.
        device: Training torch device such as "cuda" or "cpu", if "cuda", pin_memory is set to True for faster transfer from CPU to GPU.
    """
    
    def __init__(self, state_dim:int, action_dim:int, total_paths:int, batch_size:int, max_len:int=1e6, device:torch.device=None, seed:int=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.total_paths = total_paths
        self.batch_size = batch_size
        self.max_len = int(max_len)
        self.device = torch.device('cpu') if device is None else device
        
        self.buffer_device = torch.device('cpu')
        self.pin_memory = (self.device.type == 'cuda')
        self.total_paths_arange = torch.arange(self.total_paths, device=self.buffer_device)
        
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
            state: Tensor of shape [total_paths, state_dim].
            action: Tensor of shape [total_paths, action_dim].
            reward: Tensor of shape [total_paths]. Note that this is the reward from the environment.
            next_state: Tensor of shape [total_paths, state_dim].
            terminal_state: Tensor of shape [total_paths].
            lambda_val: Tensor of shape [total_paths].
            risk_free_rate: Tensor of shape [total_paths].
            transaction_cost: Tensor of shape [total_paths].
        """
        target_device = self.buffer_device
        
        # Add batch data to buffer
        circular_idx = (self.circular_ptr + self.total_paths_arange) % self.max_len
        self.state[circular_idx] = self._device_transfer(state, target_device)
        self.action[circular_idx] = self._device_transfer(action, target_device)
        self.reward[circular_idx] = self._device_transfer(reward, target_device)
        self.next_state[circular_idx] = self._device_transfer(next_state, target_device)
        self.terminal_state[circular_idx] = self._device_transfer(terminal_state, target_device)
        self.lambda_val[circular_idx] = self._device_transfer(lambda_val, target_device)
        self.risk_free_rate[circular_idx] = self._device_transfer(risk_free_rate, target_device)
        self.transaction_cost[circular_idx] = self._device_transfer(transaction_cost, target_device)
        
        # Update circular pointer and size
        self.circular_ptr = (self.circular_ptr + self.total_paths) % self.max_len
        self.size = min(self.size + self.total_paths, self.max_len)
    
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
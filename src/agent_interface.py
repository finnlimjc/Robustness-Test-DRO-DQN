import numpy as np
import torch

from src.check_input import *

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
        '''
        Actions to take when agent in training mode i.e. adding to replay buffer, cloning target q network and training q network
        '''
        raise NotImplementedError("train_mode_actions() not implemented.")

class TrainingController:
    """
    Controller for training steps, cloning target q network and checking buffer length. If you do not wish to use all features, you can set the remaining input to be None.
    
    Inputs:
        train_steps: Number of steps between training the network.
        clone_steps: Number of steps between cloning the target network.
        batch_size: Batch size for training.
        n_batches: Number of batches for training.
    
    """
    def __init__(self, train_steps:int=None, clone_steps:int=None, batch_size:int=None, n_batches:int=None):
        self.train_steps = train_steps
        self.clone_steps = clone_steps
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.steps = 0
    
    def step_increment(self):
        self.steps += 1
    
    def has_samples(self, buffer_len:int) -> bool:
        return buffer_len >= self.batch_size * self.n_batches
    
    def should_train(self) -> bool:
        is_train_step = (self.steps % self.train_steps == 0)
        return is_train_step
    
    def should_clone_q(self) -> bool:
        is_clone_step = (self.steps % self.clone_steps == 0)
        return is_clone_step    
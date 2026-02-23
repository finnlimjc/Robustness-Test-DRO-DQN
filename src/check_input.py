import numpy as np
import torch

def check_modify_obs(observation:torch.Tensor|np.ndarray) -> torch.Tensor:
    """
    Check and reshape observation input to tensor of shape [batch_size, features] of type float.
    
    Input:
        observation: Observation input to check and reshape.
        
    Output:
        observation: Reshaped observation tensor of type float.
    """
    #Convert to tensor of type float
    if isinstance(observation, np.ndarray):
        observation = torch.from_numpy(observation).float()
    elif isinstance(observation, torch.Tensor):
        observation = observation.float()
    else:
        raise ValueError(f'Invalid observation type: {type(observation)}')
    
    if observation.ndim == 1:
        observation = observation.unsqueeze(0) #Add batch dimension (Batch, Features)
    
    elif observation.ndim > 2:
        batch_size = observation.shape[0] #Assume first dimension is batch
        observation = observation.view(batch_size, -1) #(Batch, Features)
    
    assert observation.ndim == 2, f'Observation cannot be reshaped to 2D (Batch, Features): {observation.shape}'
    return observation

def check_modify_reward(reward:torch.Tensor|np.ndarray|float) -> torch.Tensor:
    """
    Check and reshape reward input to tensor of shape [batch_size] of type float.
    
    Input:
        reward: Reward input to check and reshape.
    
    Output:
        reward: Reshaped reward tensor of type float.
    """
    
    #Convert to tensor of type float
    if isinstance(reward, float):
        reward = torch.tensor([reward], dtype=torch.float32)
    elif isinstance(reward, np.ndarray):
        reward = torch.from_numpy(reward).float()
    elif isinstance(reward, torch.Tensor):
        reward = reward.float()
    else:
        raise ValueError(f'Invalid reward type: {type(reward)}')
    
    #Reshape to 1D tensor
    is_scalar = (reward.ndim == 0)
    if is_scalar:
        reward = reward.unsqueeze(-1) # (Batch, )
    
    elif reward.ndim == 2 and reward.shape[1] == 1:
        reward = reward.view(-1) # (Batch, )
    
    assert reward.ndim == 1, f'Reward cannot be reshaped to 1D (Batch, ): {reward.shape}'
    return reward
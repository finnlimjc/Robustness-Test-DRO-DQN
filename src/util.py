import json
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def atomic_torch_save(obj, file_path:str):
    """
    Save temporary file to prevent file loss during saving errors.
    
    Inputs:
        obj: Object to save, such as the state dictionary of the Torch Network.
        file_path: Location to save object.
    """
    tmp_path = file_path+'.tmp'
    torch.save(obj, tmp_path) #Save temporary file first
    os.replace(tmp_path, file_path) #Replace existing file

class Config:
    def __init__(self):
        pass
    
    def build_config_path(self, model_name:str, runs_folder_name:str="runs", config_file_name:str='config.json'):
        file_path = os.path.join(os.getcwd(), runs_folder_name, model_name, config_file_name)
        return file_path

    def download_config(self, configs:dict[dict], config_path:str):
        with open(config_path, "w") as f:
            json.dump(configs, f, indent=4)

    def load_config(self, config_path:str) -> tuple[dict]:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
            return(
                config.get('stock_params', {}),
                config.get('env_params', {}),
                config.get('training_controller_params', {}),
                config.get('duality_params', {}),
                config.get('q_params', {}),
                config.get('dqn_params', {}),
                config.get('other_params', {})
            )

class PORDQNProgressWriter:
    def __init__(self, model_name:str, timestamp_folder_name:str=None, overwrite_existing_checkpoint_file:bool=True):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp_folder_name is None else timestamp_folder_name
        self.base_file_path = os.path.join("runs", model_name, self.timestamp)
        self.checkpoint_dir = os.path.join(self.base_file_path, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=overwrite_existing_checkpoint_file)
        
        self.writer = SummaryWriter(self.base_file_path)
    
    def _save_buffer(self, buffer) -> dict:
        buffer_vals = {
            "state": buffer.state,
            "action": buffer.action,
            "reward": buffer.reward,
            "next_state": buffer.next_state,
            "terminal_state": buffer.terminal_state,
            "lambda_val": buffer.lambda_val,
            "risk_free_rate": buffer.risk_free_rate,
            "transaction_cost": buffer.transaction_cost,
            "size": buffer.size,
            "ptr": buffer.circular_ptr
        }
        return buffer_vals 
    
    def _build_checkpoint(self, epoch:int, agent) -> dict:
        checkpoint = {
            #Progress
            "epoch": epoch,
            "agent_steps": agent.training_controller.steps,
            "agent_q_updates": agent.q_updates,
            
            #Networks and Optimizers
            "q_network": agent.q.state_dict(),
            "target_q_network": agent.target_q.state_dict(),
            "network_optimizer": agent.network_optimizer.state_dict(),
            "network_rng": agent.generator.get_state() if hasattr(agent, 'generator') else None,
            
            #Buffer
            "prev_state": agent.prev_state,
            "prev_action": agent.prev_action,
            "buffer": self._save_buffer(agent.buffer),
            "buffer_rng": agent.buffer.generator.get_state() if hasattr(agent.buffer, 'generator') else None,
            
            #Global RNG
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            
            #Metadata
            "device": str(agent.device),
            "torch_version": torch.__version__
        }
        return checkpoint
    
    def save_model_params_periodically(self, epoch:int, agent, checkpoint_interval:int=1000):
        """
        Checks agent training steps, if it is a multiple of checkpoint_interval, a checkpoint save will be attempted.
        
        Inputs:
            epoch: Current training episode.
            agent: PORDQN agent with a TrainingController object.
            checkpoint_interval: n-th training step to attempt a checkpoint save.
        """
        steps = agent.training_controller.steps
        if steps % checkpoint_interval == 0:
            checkpoint = self._build_checkpoint(epoch, agent)
            step_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{steps}.pt")
            atomic_torch_save(checkpoint, step_path)
    
    def log_hq_progress(self, lambda_iters:int, lambdas:torch.Tensor, mask:torch.Tensor, q_updates:int, q_hq_diff:torch.Tensor, targets:torch.Tensor):
        with torch.no_grad():
            lambdas_cpu = lambdas.detach().cpu()
            q_hq_diff_cpu = q_hq_diff.detach().cpu()
            targets_cpu = targets.detach().cpu()
            mask_cpu = mask.detach().cpu()
            
            stats = {
                "Lambda/iterations": lambda_iters,
                "Lambda/max": lambdas_cpu.max().item(),
                "Lambda/min": lambdas_cpu.min().item(),
                "Lambda/median": lambdas_cpu.median().item(),
                "Lambda/total_neg_ebar": (~mask_cpu).sum().item(),
                "HQ/min_delta": q_hq_diff_cpu.min().item(),
                "HQ/max_delta": q_hq_diff_cpu.max().item(),
                "HQ/mean_delta": q_hq_diff_cpu.mean().item(),
                "HQ/mean_value": targets_cpu.mean().item(),
            }
        
        for tag, value in stats.items():
            self.writer.add_scalar(tag, value, q_updates)
        
        if q_updates % 100 == 0:
            self.writer.add_histogram(
                "Lambda/distribution",
                lambdas.detach().cpu().flatten(),
                q_updates
            )
            self.writer.flush()
    
    def _calculate_gradient(self, qfunc:torch.nn.Module) -> torch.Tensor:
        total_norm = 0.0
        for p in qfunc.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().norm(2).pow(2)
                total_norm += param_norm
        
        total_norm = torch.sqrt(total_norm)
        return total_norm
    
    def log_network_updates(self, qfunc:torch.nn.Module, loss:torch.Tensor, q_updates:int):
        self.writer.add_scalar("Network/loss", loss.item(), q_updates)
        
        total_norm = self._calculate_gradient(qfunc)
        self.writer.add_scalar("Network/gradient", total_norm.item(), q_updates)
    
    def _calculate_simpson_index(self, actions:np.ndarray, decimal_places:int) -> float:
        """
        Calculates the Simpson Dominance Index of an infinite community that measures concentration of actions.
        Values approaching 1 indicate strong policy concentration.
        
        Inputs:
            actions: An array of discrete/continuous actions taken by the agent. The format does not matter as only the values will be taken.
            decimal_places: Number of decimal places to round to. This allows us to handle both discrete and continuous actions through np.unique().
            
        Outputs:
            simpson_index: A float that ranges from 0 to 1 where 1 indicates that one or two actions dominate.
        """
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        if not isinstance(actions, np.ndarray):
            raise TypeError(f"Expected a numpy array, received: {type(actions)}")
        if actions.size == 0:
            raise ValueError(f"There should be at least one action. Your input: {actions}")
        
        one_dim_actions = actions.ravel().astype(np.float32)
        one_dim_actions = np.round(one_dim_actions, decimal_places)
        _, counts = np.unique(one_dim_actions, return_counts=True)
        p = counts/counts.sum()
        p = p[p>0] #Avoid zero bins
        
        simpson_index = np.sum(p**2)
        return simpson_index
    
    def log_policy_action(self, actions:np.ndarray, q_updates:int, decimal_places:int=1):
        simpson_index = self._calculate_simpson_index(actions, decimal_places)
        epsilon = 1e-12 #Prevent explosion
        reciprocal_simpson = 1/(simpson_index + epsilon)#Number of unique actions where 1.13 means that 1 action dominates
        self.writer.add_scalar('Network/effective_actions', reciprocal_simpson, q_updates)
    
    def save_latest_model_params(self, epoch:int, agent, file_name:str=None):
        file_name = f"checkpoint_ep{epoch}.pt" if file_name is None else file_name
        latest_path = os.path.join(self.checkpoint_dir, file_name)
        checkpoint = self._build_checkpoint(epoch, agent)
        atomic_torch_save(checkpoint, latest_path)
    
    def close_writer(self):
        self.writer.flush()
        self.writer.close()

class LoadModel:
    def __init__(self, runs_folder_name:str="runs", checkpoint_folder_name:str="checkpoints"):
        self.device = torch.device("cpu")
        self.runs_folder_name = runs_folder_name
        self.checkpoint_folder_name = checkpoint_folder_name
    
    def _build_path(self, model_name:str, checkpoint_file_name:str, runtime_folder:str):
        file_path = os.path.join(os.getcwd(), self.runs_folder_name, model_name, runtime_folder, self.checkpoint_folder_name, checkpoint_file_name)
        return file_path
        
    def _load_checkpoint(self, file_path:str):
        checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
        return checkpoint
    
    def _load_buffer(self, checkpoint, buffer):
        buffer_checkpoint = checkpoint['buffer']
        
        buffer.state = buffer_checkpoint['state']
        buffer.action = buffer_checkpoint['action']
        buffer.reward = buffer_checkpoint['reward']
        buffer.next_state = buffer_checkpoint['next_state']
        buffer.terminal_state = buffer_checkpoint['terminal_state']
        buffer.lambda_val = buffer_checkpoint['lambda_val']
        buffer.risk_free_rate = buffer_checkpoint['risk_free_rate']
        buffer.transaction_cost = buffer_checkpoint['transaction_cost']
        buffer.size = buffer_checkpoint['size']
        buffer.circular_ptr = buffer_checkpoint['ptr']
        
        buffer.generator.set_state(checkpoint['buffer_rng'])
        return buffer
    
    def _transfer_optimizer_tensors_device(self, optimizer, target_device:torch.device):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    if v.device != target_device:
                        state[k] = v.to(target_device)
        return optimizer
    
    def restore_agent_state(self, agent, model_name:str, checkpoint_file_name:str, runtime_folder:str, target_device:torch.device=None):
        file_path = self._build_path(model_name, checkpoint_file_name, runtime_folder)
        checkpoint = self._load_checkpoint(file_path)
        
        #Restore States
        agent.training_controller.steps = checkpoint['agent_steps']
        agent.q_updates = checkpoint['agent_q_updates']
        
        #Networks and Optimizers
        agent.q.load_state_dict(checkpoint['q_network'])
        agent.target_q.load_state_dict(checkpoint['target_q_network'])
        agent.network_optimizer.load_state_dict(checkpoint['network_optimizer'])
        
        #Buffer
        agent.prev_state = checkpoint['prev_state']
        agent.prev_action = checkpoint['prev_action']
        agent.buffer = self._load_buffer(checkpoint, agent.buffer)
        
        #Global RNG
        torch.set_rng_state(checkpoint['torch_rng_state'])
        
        device = checkpoint['device']
        if isinstance(device, str):
            device = torch.device(device)
        if target_device is None:
            target_device = device 
        
        if device.type == target_device.type:
            agent.generator.set_state(checkpoint['network_rng'])
        
        if torch.cuda.is_available() and target_device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
            agent.device = target_device
            agent.q = agent.q.to(target_device)
            agent.target_q = agent.target_q.to(target_device)
            agent.prev_action = agent.prev_action.to(target_device)
            self._transfer_optimizer_tensors_device(agent.network_optimizer, target_device)
        
        current_episode = checkpoint['epoch']
        return agent, current_episode
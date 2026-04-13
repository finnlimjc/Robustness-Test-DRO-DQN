import torch

from src.utils.writer import atomic_torch_save

class LoadModel:
    def __init__(self, runs_folder_name:str="runs", checkpoint_folder_name:str="checkpoints"):
        self.device = torch.device("cpu")
        self.runs_folder_name = runs_folder_name
        self.checkpoint_folder_name = checkpoint_folder_name
    
    def _build_path(self, model_name:str, checkpoint_file_name:str, runtime_folder:str):
        import os
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
        agent.action_values = checkpoint["action_values"]
        
        #Epsilon Scheduler
        agent.epsilon_scheduler.epsilon = checkpoint["current_epsilon"]
        agent.epsilon_scheduler.timestep = checkpoint['current_timestep']
        
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
            agent.action_values = agent.action_values.to(target_device)
            self._transfer_optimizer_tensors_device(agent.network_optimizer, target_device)
        
        current_episode = checkpoint['epoch']
        return agent, current_episode
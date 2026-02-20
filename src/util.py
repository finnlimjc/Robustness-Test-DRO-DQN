import json
import torch
from torch.utils.tensorboard import SummaryWriter
from src.agent import ReplayBuffer, PORDQN

def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))

def start_writer(model_params, model_name):
    
    model_params['checkpoint_path'] = f'{writer.log_dir}/checkpoint.pt'
    writer.add_text('Model parameters', pretty_json(model_params))
    writer.flush()
    return writer

class ProgressWriter:
    def __init__(self, model_name:str):
        self.writer = SummaryWriter(f'./runs/{model_name}')
        self.base_file_path = self.writer.log_dir
    
    def _save_buffer(self, buffer:ReplayBuffer):
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
            "ptr": buffer.circular_ptr,
        }
        return buffer_vals 
    
    def save_model_params(self, epoch:int, agent:PORDQN):
        file_path = f"./{self.base_file_path}/checkpoint.pt"
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
            "buffer": self._save_buffer(agent.buffer),
            "buffer_rng": agent.buffer.generator.get_state() if hasattr(agent.buffer, 'generator') else None,
        
            #Metadata
            "device": str(agent.device),
            "torch_version": torch.__version__
        }
        
        torch.save(checkpoint, file_path)
    
    def log_hq_progress(self, lambda_iters:int, lambdas:torch.Tensor, mask:torch.Tensor, q_updates:int, q_hq_diff:torch.Tensor, targets:torch.Tensor):
        with torch.no_grad():
            stats = {
                "Lambda/iterations": lambda_iters,
                "Lambda/max": lambdas.max().item(),
                "Lambda/min": lambdas.min().item(),
                "Lambda/median": lambdas.median().item(),
                "Lambda/total_neg_ebar": (~mask).sum().item(),
                "HQ/min_delta": q_hq_diff.min().item(),
                "HQ/max_delta": q_hq_diff.max().item(),
                "HQ/mean_delta": q_hq_diff.mean().item(),
                "HQ/mean_value": targets.mean().item(),
            }
        
        for tag, value in stats.items():
            self.writer.add_scalar(tag, value, q_updates)
        
        if q_updates % 100 == 0:
            self.writer.add_histogram(
                "Lambda/distribution",
                lambdas.detach().cpu(),
                q_updates
            )
    
    def close_writer(self):
        self.writer.close()
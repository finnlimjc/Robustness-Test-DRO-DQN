import json
import os

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
                config.get('eps_scheduler_params', {}),
                config.get('prior_params', {}),
                config.get('other_params', {})
            )
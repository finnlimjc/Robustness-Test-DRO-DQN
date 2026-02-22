import numpy as np
from tqdm import tqdm

def train_agent(env, agent, current_epoch:int, n_epochs:int, writer=None) -> list[np.ndarray]:
    if current_epoch < 0:
        raise ValueError(f"Invalid value for current_epoch: {current_epoch}. Current episode should start from 1.")
    if current_epoch > n_epochs:
        raise ValueError(f"Current episode ({current_epoch}) cannot exceed total episodes ({n_epochs}).")
    
    all_cum_rewards = []
    batch_size = agent.batch_size
    for epoch in range(current_epoch, n_epochs+1):
        cum_rewards = np.zeros(shape=(batch_size, 1))
        observation, _ = env.reset()
        action_idx = agent.agent_start(observation)
        steps = env.action_steps
        done = np.zeros(batch_size, dtype=bool)
        
        with tqdm(total=steps, desc=f"Episode {epoch}", mininterval=2) as step_bar:
            while not done.any():
                next_state, reward, done, truncated, info = env.step(action_idx)
                cum_rewards += reward #Reward is log(1+R)
                if done.any():
                    agent.agent_end(reward=reward, observation=next_state, info=info)
                else:
                    action_idx = agent.agent_step(reward=reward, observation=next_state, info=info)
                
                step_bar.update(1)
            
                if writer is not None:
                    writer.save_model_params_periodically(epoch, agent, checkpoint_interval=1000)
        
        if writer is not None:
            writer.save_latest_model_params(epoch, agent)
            writer.writer.flush()
        
        modified_expected_cum_reward = np.expm1(cum_rewards.mean())
        print(f'Episode {epoch} mean of summed rewards: {modified_expected_cum_reward:.4%}')
        all_cum_rewards.append(cum_rewards)
    
    if writer is not None:
        writer.close_writer()
    
    return all_cum_rewards
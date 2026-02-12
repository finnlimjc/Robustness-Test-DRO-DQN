from old_src.q import *
from old_src.DQN import *
from old_src.env import *
from old_src.train import *

def load_checkpt(file_name:str):
    checkpoint_path = f"./models/{file_name}/checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=torch.device('cpu'))
    return checkpoint

class LoadModel:
    def __init__(self, checkpt):
        self.checkpoint = checkpt
        self.device = 'cpu'
        self._init_action_values()
    
    def _init_action_values(self):
        action_values = torch.linspace(-1., 1., 9, device=self.device)
        self.action_values = action_values
    
    def _load_qfunc(self):
        #Default
        self.state_len = 60
        self.other_state_vars = ['log_wealth', 'positions', 'dt']
        architecture = [64, 64]
        self.robustq = QFunc(self.state_len+len(self.other_state_vars), architecture, self.action_values.shape[0]).to(self.device)
    
    def _load_dqn(self):
        #Default
        obs_dim = self.state_len + len(self.other_state_vars)
        num_actions = len(self.action_values)
        discount = 0.99
        nu_scale = 0.03
        nu_df = 2
        
        epsilon = 0.003 # Sinkhorn distance
        delta = 1e-4 # regularisation parameter for Sinkhorn distance
        n_inner = 1000 # number of samples from nu to calc inner expectations
        
        lamda_init = 0. # initial lambda
        lamda_lr = [0.02,0.1,0.25,0.4,0.8,1.5,3,100,1000,10000,100000]
        lamda_max_iter = 100
        lamda_step_size = 10 # step size for learning rate scheduler
        lamda_gamma = 0 # gamma for learning rate scheduler
        norm_ord = 1
        
        eps_greedy = 0.1 # epsilon greedy parameter
        buffer_max_length = int(1e5)
        clone_steps = 50
        train_steps = 1
        agent_batch_size = 128
        n_batches = 1
        n_epochs = 1
        robustq_lr = 1e-4
        
        seed = 0
        writer = None
        
        self.robustdqn_agent = PORDQN(obs_dim, num_actions, discount, nu_scale, nu_df, self.action_values, epsilon, delta, n_inner, lamda_init,lamda_lr, lamda_max_iter, 
                         lamda_step_size, lamda_gamma, norm_ord, self.robustq, eps_greedy, buffer_max_length, clone_steps, train_steps, agent_batch_size, 
                         n_batches, n_epochs, robustq_lr, device=self.device, seed=seed, writer=writer)
    
    def _input_checkpt(self):
        self.robustdqn_agent.training_mode = False
        self.robustdqn_agent.steps = self.checkpoint['agent_steps']
        self.robustdqn_agent.q_updates = self.checkpoint['agent_q_updates']
        self.robustdqn_agent.q.load_state_dict(self.checkpoint['agent_state_dict'])
        self.robustdqn_agent.target_q.load_state_dict(self.checkpoint['agent_target_q'])
        self.robustdqn_agent.optimizer.load_state_dict(self.checkpoint['agent_optimizer'])
        self.robustdqn_agent.buffer = self.checkpoint['agent_buffer']
        self.robustdqn_agent.rng.bit_generator.state = self.checkpoint['numpy_rng_state']
        self.robustdqn_agent.epsilon = self.checkpoint['agent_eps_greedy']
        random.setstate(self.checkpoint['random_state'])
        torch.set_rng_state(self.checkpoint['torch_rng_state'])
    
    def load_model(self):
        self._load_qfunc()
        self._load_dqn()
        self._input_checkpt()
        return self.robustdqn_agent, self.action_values
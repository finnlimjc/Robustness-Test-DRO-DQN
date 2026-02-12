import torch
import torch.nn as nn

class QFunc(nn.Module):
    def __init__(self, input_size:int, hidden_size:list[int], output_size:int, activation:str='tanh'):
        super().__init__()
        activation_fn = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh()
        }
        if activation not in activation_fn:
            raise ValueError(f'Activation function {activation} not supported, please use either relu or tanh.')
        
        #Input layer
        layers = [nn.Linear(input_size, hidden_size[0])]
        
        #Hidden Layers
        for dim in hidden_size[1:]:
            layers.extend([
                activation_fn,
                nn.Linear(layers[-1].out_features, dim),
            ])
        
        #Output Layers
        layers.append(activation_fn)
        layers.append(nn.Linear(hidden_size[-1], output_size))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x) -> torch.Tensor:
        return self.net(x)

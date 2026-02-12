import numpy as np
import torch
from scipy.special import stdtrit

class PriorStudentDistribution:
    """
    Deterministic support approximation of a 1D prior distribution using inverse-CDF (quantile) discretization.
    
    Inputs:
        n_samples: Number of samples to approximate the distribution.
        mu: Mean of the student-t distribution.
        scale: Scale of the student-t distribution.
        ddof: Degrees of freedom of the student-t distribution.
        device: Torch device to store the tensors, such as "cuda" or "cpu".
    """
    def __init__(self, n_samples:int=5000, mu:float=0.0, scale:float=0.03, ddof:int=2, device:torch.device=None):
        self.n_samples = int(n_samples)
        self.mu = mu 
        self.scale = scale
        self.ddof = ddof
        self.device = torch.device('cpu') if device is None else device
    
        self.support = self._build_support()
    
    def _build_support(self) -> torch.Tensor:
        x = np.linspace(0, 1, self.n_samples + 2) #Add 2 as we remove first and last element in the next line
        x = x[1:-1] # Avoid 0 and 1 exactly (numerical safety)
        
        scaled_samples = stdtrit(self.ddof, x) * self.scale + self.mu # stdtrit assumes a mean of 0 and scale of 1, so we adjust accordingly
        return torch.tensor(scaled_samples, dtype=torch.float32, device=self.device)
    
    def sample_from_support(self, batch_size:int) -> torch.Tensor:
        next_return_from_prior = self.support.repeat(batch_size, 1) #(batch_size, n_samples)
        return next_return_from_prior
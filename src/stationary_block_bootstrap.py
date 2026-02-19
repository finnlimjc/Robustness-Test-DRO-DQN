import numpy as np
from statsmodels.tsa.stattools import acf

def stationary_bootstrap(data:np.ndarray, n_sims:int=1, t:int=252, avg_block_size:int=None, seed:int=None) -> np.ndarray:
    """
    Generates bootstrap samples using the stationary block bootstrap method. 
    This method creates bootstrap samples by randomly selecting blocks of consecutive data points from the original dataset, with the block lengths following a geometric distribution. 
    
    Inputs:
        data: The original time series data as a 1D numpy array.
        n_sims: The number of bootstrap samples to generate.
        t: The length of each bootstrap sample (number of time steps).
        avg_block_size: The average block size for the geometric distribution of block lengths. If None, it will be set to the cube root of the length of the data.
        seed: Random seed for reproducibility.
    
    Outputs:
        sims: An array of shape (n_sims, t) containing the bootstrap samples.
    """
    rng = np.random.default_rng(seed)
    T = len(data)
    sims = np.zeros((n_sims, t))
    if avg_block_size is None:
        avg_block_size = int(np.floor(T**(1/3)))
    
    restart_prob = 1/avg_block_size
    
    for i in range(n_sims):
        array_idx = 0
        while array_idx < t:
            if array_idx == 0 or rng.random() < restart_prob:
                idx = rng.integers(T)
            else:
                idx = (idx + 1) % T #Implement circular indexing
            
            sims[i, array_idx] = data[idx]
            array_idx += 1
    
    return sims

class OptimalBlockSize:
    def __init__(self, data:np.ndarray):
        self.data = data.copy()
        self.n = len(data)
    
    def _adaptive_bandwidth(self, c:int=2, raise_error:bool=True) -> float:
        K = max(5, np.sqrt(np.log10(self.n)))
        max_M = np.ceil(np.sqrt(self.n)) + K #Add K-lags to give some leeway
        
        threshold = c*np.sqrt(np.log10(self.n) / self.n)
        p = acf(self.data, nlags=max_M)[1:] #Drop lag 0
        abs_p = np.abs(p)
        is_below_threshold = abs_p < threshold
        
        for m in range(int(max_M - K)):
            if is_below_threshold[m:m+K].all():
                m_hat = m+1
                break #If break does not occur, run the else statement
        else:
            p_above_threshold = np.where(abs_p >= threshold)[0] #Index 0 selects the inner array
            m_hat = p_above_threshold.max() + 1 if p_above_threshold.size else 0 #Return the index + 1, else if there are no values >= threshold return 0
        
        if raise_error and m_hat <= 0:
            raise ValueError(f"Estimated m_hat is less than or equal to 0: {m_hat}")
        
        M = min(2*m_hat, max_M)
        return M
    
    def _flat_top_lag_window(self, t:np.ndarray) -> np.ndarray:
        abs_t = np.abs(t)
        window = np.zeros_like(abs_t, dtype=float)
        window[abs_t<=0.5] = 1.0
        
        mask = (abs_t > 0.5) & (abs_t<=1.0)
        window[mask] = 2.0*(1.0-abs_t[mask])
        return window
    
    def _sample_autocovariance(self, k:np.ndarray) -> np.ndarray:
        x_bar = self.data.mean()
        centered_x = self.data - x_bar
        
        R = np.zeros_like(k, dtype=float)
        for idx, lag in enumerate(k):
            abs_lag = np.abs(lag)
            R[idx] = centered_x[:self.n-abs_lag]@ centered_x[abs_lag:]/ self.n
        return R

    def _long_run_variance_components(self, M:int) -> tuple[float, float]:
        k = np.arange(-M, M+1, dtype=int)
        lambda_vals = self._flat_top_lag_window(k/M)
        R_hat = self._sample_autocovariance(k)
        
        g_hat = np.sum(lambda_vals*R_hat) #cos(0) = 1
        G_hat = np.sum(lambda_vals* np.abs(k)* R_hat)
        D_hat = 2.0* g_hat**2
        return G_hat, D_hat

    def optimal_stationary_block_size(self, c:int=2, raise_error:bool=True) -> float:
        M = self._adaptive_bandwidth(c=c, raise_error=raise_error)
        G_hat, D_hat = self._long_run_variance_components(M)
        block_size = (2* G_hat**2 / D_hat)**(1/3)* self.n**(1/3)
        max_block_size = np.ceil(min(3*np.sqrt(self.n), self.n/3)) #Upper bound on estimated optimal block length
        return min(block_size, max_block_size)

def generate_path(data:np.ndarray, n_sims:int, t:int, seed:int=None) -> np.ndarray:
    optimal_block_size = OptimalBlockSize(data).optimal_stationary_block_size()
    optimal_block_size = int(np.ceil(optimal_block_size))
    path = stationary_bootstrap(data, n_sims=n_sims, t=t, avg_block_size=optimal_block_size, seed=seed)
    return path #(n_sims, t)
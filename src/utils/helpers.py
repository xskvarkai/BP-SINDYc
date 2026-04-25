import numpy as np
import pysindy as ps
from typing import List, Callable
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.signal import savgol_filter

def compute_time_vector(x: np.ndarray|List[np.ndarray], dt: float):
    """
    Computes time vector suitable for data shape.
    """
    if isinstance(x, list):
        time_vec = (np.arange(x[0].shape[0]) * dt)
    elif isinstance(x, int):
        time_vec = (np.arange(x) * dt)
    else:
        time_vec = (np.arange(x.shape[0]) * dt)

    return time_vec

def rk4_integrator(x: np.ndarray, u: float, dt: float, dxdt_func: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:  
    """  
    Integrates the state x over time dt using Runge-Kutta 4th order method.  
    """  
    k1 = dxdt_func(x, u)  
    k2 = dxdt_func(x + 0.5 * dt * k1, u)  
    k3 = dxdt_func(x + 0.5 * dt * k2, u)  
    k4 = dxdt_func(x + dt * k3, u)  
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def evaluate_simulation(x_ref, x_sim, dt):
    min_len = min(len(x_ref), len(x_sim))
    rmse = root_mean_squared_error(x_ref[:min_len], x_sim[:min_len])
    r2 = r2_score(x_ref[:min_len], x_sim[:min_len])
    
    return rmse, r2
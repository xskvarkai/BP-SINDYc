import numpy as np
from typing import Optional, Tuple
from scipy.signal import savgol_filter

from utils import constants
from utils.vizualization import vizualize_trajectory
from utils.helpers import compute_time_vector

def split_data(
    x: np.ndarray,
    u: Optional[np.ndarray] = None,
    val_size: float = 0.2,
    test_size: float = 0.2,
    dt: float = None,
    filter_data: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], 
           Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:

    # Rozdelenie casovych radov 
    # musime zachovat casovu naslednost dynamiky:  
    # Train: [0 ... t1], Val: [t1 ... t2], Test: [t2 ... end]

    # Ziskanie poctu vzoriek a poctu pre validacnu a testovaciu sadu
    num_samples = x.shape[0]
    val_count = int(np.floor(num_samples * val_size)) if val_size > 0 else 0
    test_count = int(np.floor(num_samples * test_size)) if test_size > 0 else 0

    # Vypocet indexov na rozdelenie dat
    # train_index je bod zlomu medzi treningovou a (validacnou alebo testovacou) sadou
    train_index = num_samples - val_count - test_count
    val_index = train_index + val_count

    # Rozdelenie stavovych premennych x
    if filter_data:
        x_smooth = savgol_filter(x, constants.SAVGOL_WINDOW_LENGTH, constants.SAVGOL_POLYORDER, axis=0)
        vizualize_trajectory(compute_time_vector(x, dt), x, x_smooth, u)
        x = x_smooth
        x_smooth = None 

    x_train = x[:train_index]
    x_val = x[train_index:val_index] if val_count > 0 else None
    x_test = x[val_index:] if test_count > 0 else None

    # Rozdelenie vstupneho signalu u, ak existuje
    if not np.any(u) or u is None:
        u_train = u_val = u_test = None
    else:
        u_train = u[:train_index]
        u_val = u[train_index:val_index] if val_count > 0 else None
        u_test = u[val_index:] if test_count > 0 else None
        
    return x_train, x_val, x_test, u_train, u_val, u_test

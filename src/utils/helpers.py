import numpy as np
import pysindy as ps
from typing import List, Union

def compute_time_vector(x: Union[np.ndarray, List[np.ndarray]], dt: float):
    """
    Computes time vector suitable for data shape.
    """
    if isinstance(x, list):
        time_vec = (np.arange(x[0].shape[0]) * dt)
    else:
        time_vec = (np.arange(x.shape[0]) * dt)

    return time_vec
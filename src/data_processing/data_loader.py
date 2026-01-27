import pandas as pd
import numpy as np
from typing import Optional, List, Tuple

def load_data(
    file_path: str,
    time: Optional[np.ndarray | float] = None,
    include_control_input: bool = True,
    column_indices: Optional[List[int] | Tuple[int, ...]] = None
) -> Tuple[np.ndarray, np.ndarray, float]:

    data_csv = pd.read_csv("data/raw/" + file_path + ".csv")
    data = data_csv.to_numpy()

    dt_value: float

    if time is None:
        if data.shape[1] > 0:
            dt_value = np.round(np.median(np.diff(data[:, 0])), decimals=5)
            print(f"\nEstimated time step (dt): {dt_value}")
        else:
            raise ValueError("Data are empty, the time step from the first data column cannot be estimated..")
    elif isinstance(time, np.ndarray):
        if time.size > 1:
            dt_value = np.round(np.median(np.diff(time)), decimals=5)
            print(f"\nEstimated time step (dt): {dt_value}")
        else:
            raise ValueError("The time field must contain at least two elements to estimate the time step.")
    elif isinstance(time, float):
        dt_value = time
        print(f"\nProvided time step (dt): {dt_value}")
    else:
        raise ValueError(f"The 'time' parameter must be None, np.ndarray, or float. Accepted type: {type(time)}")

    if not isinstance(column_indices, (list, tuple)) or not all(isinstance(i, int) for i in column_indices):
        raise TypeError("column_indices must be a list or an n-tuple of integers.")

    if not column_indices:
        raise ValueError("column_indices cannot be an empty list.")

    X_cols: List[int]
    U_data: np.ndarray = np.array([])

    if include_control_input:
        if len(column_indices) < 2:
            raise ValueError("If include_control_input is True, column_indices must contain at least two elements (one for X, one for U).")

        u_col_idx = column_indices[-1]
        X_cols = list(column_indices[:-1])

        if not (0 <= u_col_idx < data.shape[1]):
            raise IndexError(f"The index of column U ({u_col_idx}) is out of range for data with {data.shape[1]} columns.")
        U_data = np.stack((data[:, u_col_idx]), axis=-1)
    else:
        X_cols = list(column_indices)
        U_data = np.array([])

    if not all(0 <= idx < data.shape[1] for idx in X_cols):
        raise IndexError(f"One or more indexes of columns X ({X_cols}) are out of range for data with {data.shape[1]} columns.")

    X = np.stack([data[:, col_idx] for col_idx in X_cols], axis=-1)

    return X, U_data, dt_value

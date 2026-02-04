import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path
from scipy.signal import savgol_filter

# Import local modules
from utils.helpers import compute_time_vector
from utils.config_manager import ConfigManager
from utils.plots import plot_trajectory

class TimeSeriesSplitter:
    """
    A class for splitting time-series data into training, validation, and test sets,
    with optional Savitzky-Golay filtering.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        X_raw: np.ndarray,
        dt: float,
        U_raw: Optional[np.ndarray] = None,
    ):
        """
        Initializes the TimeSeriesSplitter.
        """
        
        if not isinstance(X_raw, np.ndarray) or X_raw.ndim != 2:
            raise ValueError("X_raw must be a 2D NumPy array.")
        if not isinstance(U_raw, np.ndarray) or U_raw.ndim != 2:
            raise ValueError("U_raw must be a 2D NumPy array.")
        if X_raw.shape[0] != U_raw.shape[0] and U_raw.shape[1] > 0:
             raise ValueError("X_raw and U_raw must have the same number of samples (rows) if U_raw is not empty.")
        if not isinstance(dt, (int, float)) or dt <= 0:
            raise ValueError("dt must be a positive number.")

        self.config_manager = config_manager
        self.X_raw = X_raw
        self.U_raw = U_raw
        self.dt = dt

        self.config_manager.load_config("data_config")
        # Nacitanie parametrov pre Savitzky-Golay filter z konfiguracie
        self.savgol_window_length = self.config_manager.get_param(
            'data_preprocessing.savgol_window_length', default=21
        )
        self.savgol_polyorder = self.config_manager.get_param(
            'data_preprocessing.savgol_polyorder', default=2
        )
        
        # Validacia pre Savitzky-Golay filter
        if self.savgol_window_length % 2 == 0 or self.savgol_window_length < 1:
            raise ValueError("Savitzky-Golay window_length must be odd and positive.")
        if self.savgol_polyorder >= self.savgol_window_length:
            raise ValueError("Savitzky-Golay polyorder must be less than window_length.")
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None
    
    def split_data(
            self,
            train_ratio: float = 0.6,
            val_ratio: float = 0.2,
            apply_savgol_filter: bool = False,
            plot_data: bool = False,
            verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        """
        Splits the data into training, validation, and test sets.
        Optionally applies a Savitzky-Golay filter to the state variables (X).
        """

        if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1):
            raise ValueError("Invalid train_ratio or val_ratio. Ratios must be between 0 and 1, and their sum must be less than 1.")

        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio < 0: # Ak by sa nieco pokazilo pri prvej validacii
            raise ValueError("Calculated test_ratio is negative. Check train_ratio and val_ratio.")
        
        num_samples = self.X_raw.shape[0]
        if num_samples < (self.savgol_window_length if apply_savgol_filter else 1):
            raise ValueError(f"Insufficient data samples ({num_samples}) for splitting or Savitzky-Golay filtering (requires at least {self.savgol_window_length} samples).")
        
        if verbose:
            print(f"\nSplitting data into train ({train_ratio:.1%}), validation ({val_ratio:.1%}), test ({test_ratio:.1%}).")

        X_processed = self.X_raw
        if apply_savgol_filter:
            if verbose:
                print(f"Applying Savitzky-Golay filter with window_length={self.savgol_window_length}, polyorder={self.savgol_polyorder}.")
            
            # Aplikacia Savitzky-Golay filtera a typova konverzia
            X_processed = savgol_filter(self.X_raw, self.savgol_window_length, self.savgol_polyorder, axis=0)
            
        # Ziskanie poctu vzoriek a poctu pre validacnu a testovaciu sadu
        train_end_index = int(num_samples * train_ratio)
        val_end_index = int(num_samples * (train_ratio + val_ratio)) 

        # Rozdelenie na 
        X_train = X_processed[:train_end_index]
        X_val = X_processed[train_end_index:val_end_index] if val_ratio > 0 else None
        X_test = X_processed[val_end_index:] if test_ratio > 0 else None

        U_train: Optional[np.ndarray] = None
        U_val: Optional[np.ndarray] = None
        U_test: Optional[np.ndarray] = None

        if self.U_raw is not None:
            U_train = self.U_raw[:train_end_index]
            U_val = self.U_raw[train_end_index:val_end_index] if val_ratio > 0 else None
            U_test = self.U_raw[val_end_index:] if test_ratio > 0 else None

        if plot_data:
            time_vector = compute_time_vector(X_processed, self.dt)
            plot_trajectory(time_vector[:train_end_index], X_train, input_signal=(U_train if U_train is not None else None), title="Train Data")
            plot_trajectory(time_vector[train_end_index:val_end_index], X_val, input_signal=(U_val if U_val is not None else None), title="Validation Data")
            plot_trajectory(time_vector[val_end_index:], X_test, input_signal=(U_test if U_test is not None else None), title="Test Data")

        return X_train, X_val, X_test, U_train, U_val, U_test
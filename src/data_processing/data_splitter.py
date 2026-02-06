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
    with optional Savitzky-Golay filtering and input signal perturbing.
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
        if U_raw is not None and (not isinstance(U_raw, np.ndarray) or U_raw.ndim != 2):
            raise ValueError("U_raw must be a 2D NumPy array.")
        if X_raw.shape[0] != U_raw.shape[0] and U_raw.shape[1] > 0:
             raise ValueError("X_raw and U_raw must have the same number of samples (rows) if U_raw is not empty.")
        if not isinstance(dt, (int, float)) or dt <= 0:
            raise ValueError("dt must be a positive number.")

        self.config_manager = config_manager
        self.X_raw = X_raw
        self.U_raw = U_raw
        self.dt = dt

        self.config_manager.load_config("settings")
        # Nacitanie parametru na perturbaciu vstupneho signalu
        self._minimal_noise_value = self.config_manager.get_param(
            'settings.constants.values.minimal_noise_value'
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None
    
    def split_data(
            self,
            train_ratio: float = 0.6,
            val_ratio: float = 0.2,
            perturb_input_signal_ratio: Optional[float] = None,
            rng: Optional[np.random.RandomState] = np.random.RandomState(42),
            plot_data: bool = False,
            verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits the data into training, validation, and test sets.
        """

        if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1):
            raise ValueError("Invalid train_ratio or val_ratio. Ratios must be between 0 and 1, and their sum must be less than 1.")

        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio < 0: # Ak by sa nieco pokazilo pri prvej validacii
            raise ValueError("Calculated test_ratio is negative. Check train_ratio and val_ratio.")
        
        num_samples = self.X_raw.shape[0]
        if num_samples < 1:
            raise ValueError(f"Insufficient data samples ({num_samples}) for splitting.")
        
        if verbose:
            print(f"\nSplitting data into train ({train_ratio:.1%}), validation ({val_ratio:.1%}), test ({test_ratio:.1%}).")

        # Ziskanie poctu vzoriek a poctu pre validacnu a testovaciu sadu
        train_end_index = int(num_samples * train_ratio)
        val_end_index = int(num_samples * (train_ratio + val_ratio)) 

        # Rozdelenie na 
        X_train = self.X_raw[:train_end_index]
        X_val = self.X_raw[train_end_index:val_end_index] if val_ratio > 0 else None
        X_test = self.X_raw[val_end_index:] if test_ratio > 0 else None

        U_train: Optional[np.ndarray] = None
        U_val: Optional[np.ndarray] = None
        U_test: Optional[np.ndarray] = None

        if self.U_raw is not None:
            U_train = self.U_raw[:train_end_index]
            U_val = self.U_raw[train_end_index:val_end_index] if val_ratio > 0 else None
            U_test = self.U_raw[val_end_index:] if test_ratio > 0 else None
            # Pozadovana perturbacia vstupu
            if (perturb_input_signal_ratio is not None 
            and isinstance(perturb_input_signal_ratio, (float, int)) and perturb_input_signal_ratio > 0):
                U_train = self._perturb_input_signal(U_train, perturb_input_signal_ratio, rng)

        if plot_data:
            time_vector = compute_time_vector(self.X_raw, self.dt)
            plot_trajectory(time_vector[:train_end_index], X_train, input_signal=U_train, title="Train Data")
            plot_trajectory(time_vector[train_end_index:val_end_index], X_val, input_signal=U_val, title="Validation Data")
            plot_trajectory(time_vector[val_end_index:], X_test, input_signal=U_test, title="Test Data")

        return X_train, X_val, X_test, U_train, U_val, U_test

    def _perturb_input_signal(self, U: np.ndarray, perturb_ratio: float, rng: np.random.RandomState) -> np.ndarray:
        """
        Adds noise to all input signal columns.
        """
        # Pouzivame RandomState pre reprodukovatelnost
        # Aplikujeme sum na vsetky stlpce vstupneho signalu
        for i in range(U.shape[1]):
            noise_level = max(perturb_ratio * np.std(U[:, i]), self._minimal_noise_value)
            noise = rng.normal(0, noise_level, U[:, i].shape)
            U[:, i] += noise

        return U


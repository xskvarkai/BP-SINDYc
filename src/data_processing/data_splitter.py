import numpy as np
from typing import Optional, List, Tuple
import warnings
from scipy.signal import savgol_filter

# Import local modules
from utils.helpers import compute_time_vector
from utils.config_manager import ConfigManager
from utils.plots import plot_trajectory

class TimeSeriesSplitter:
    """
    A class for splitting time-series data into training, validation, and test sets.
    It supports optional Savitzky-Golay filtering and perturbation of the input signal.
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

        Args:
            config_manager (ConfigManager): An instance of ConfigManager for accessing configurations.
            X_raw (np.ndarray): The raw state variables (features) as a 2D NumPy array.
            U_raw (Optional[np.ndarray]): The raw control inputs as a 2D NumPy array, or None if not available.
            dt (float): The time step of the data.

        Raises:
            ValueError: If X_raw is not a 2D array, dt is not positive, or if
                        X_raw and U_raw have inconsistent numbers of samples.
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
        self._minimal_noise_value = self.config_manager.get_param("settings.constants.values.minimal_noise_value", default=1e-3) # Load the minimal noise value for perturbing input data from configuration

    def __enter__(self):
        """
        Enters the runtime context related to this object.
        Returns the instance itself for use in 'with' statements.
        """

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the runtime context related to this object.
        Handles any exceptions that occurred within the 'with' block.
        """

        return None

    def split_data(
            self,
            train_ratio: float = 0.6,
            val_ratio: float = 0.2,
            perturb_input_signal_ratio: Optional[float] = None,
            rng: Optional[np.random.RandomState] = np.random.RandomState(42),
            apply_savgol_filter: bool = False,
            filtered_set_names: List[str] = None,
            savgol_window_length: Optional[int] = None,
            savgol_polyorder: Optional[int] = None,
            plot_data: bool = False,
            verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits the data into training, validation, and test sets.
        Optionally applies a Savitzky-Golay filter and perturbs the input signal.

        Args:
            train_ratio (float): The proportion of the dataset to include in the train split.
            val_ratio (float): The proportion of the dataset to include in the validation split.
            perturb_input_signal_ratio (Optional[float]): The ratio by which to perturb the input signal (U).
                                                          If None, no perturbation is applied.
            rng (Optional[np.random.RandomState]): Random number generator for reproducibility, especially for perturbation.
            apply_savgol_filter (bool): Whether to apply the Savitzky-Golay filter to specified sets.
            filtered_set_names (List[str]): A list of strings indicating which sets to filter (e.g., ["train", "val", "test"]).
                                            Only relevant if apply_savgol_filter is True.
            savgol_window_length (Optional[int]): The window length for the Savitzky-Golay filter. Must be odd and positive.
            savgol_polyorder (Optional[int]): The polynomial order for the Savitzky-Golay filter.
            plot_data (bool): If True, plots the trajectories of the split data.
            verbose (bool): If True, prints detailed messages about the splitting process.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray],
                  np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]: A tuple containing:
                - X_train (np.ndarray): State variables for the training set.
                - U_train (Optional[np.ndarray]): Control inputs for the training set.
                - X_val (Optional[np.ndarray]): State variables for the validation set (None if val_ratio is 0).
                - U_val (Optional[np.ndarray]): Control inputs for the validation set (None if val_ratio is 0).
                - X_test (Optional[np.ndarray]): State variables for the test set (None if test_ratio is 0).
                - U_test (Optional[np.ndarray]): Control inputs for the test set (None if test_ratio is 0).

        Raises:
            ValueError: If ratios are invalid, insufficient data, or Savitzky-Golay filter parameters are incorrect.
        """

        if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio <= 1):
            raise ValueError("Invalid train_ratio or val_ratio. Ratios must be between 0 and 1, and their sum must be less than or equal 1.")

        test_ratio = float(np.round(1.0 - train_ratio - val_ratio, decimals=5))
        if test_ratio < 0: # This should ideally be caught by the first validation, but as a safeguard.
            raise ValueError("Calculated test_ratio is negative. Check train_ratio and val_ratio.")

        num_samples = self.X_raw.shape[0]
        if num_samples < 1:
            raise ValueError(f"Insufficient data samples ({num_samples}) for splitting.")

        if verbose:
            print(f"\nSplitting data into train ({train_ratio:.1%}), validation ({val_ratio:.1%}), test ({test_ratio:.1%}).")

        # Determine indices for splitting
        train_end_index = int(num_samples * train_ratio)
        val_end_index = int(num_samples * (train_ratio + val_ratio))

        # Split X (state variables)
        X_train = self.X_raw[:train_end_index]
        X_val = self.X_raw[train_end_index:val_end_index] if val_ratio > 0 else None
        X_test = self.X_raw[val_end_index:] if test_ratio > 0 else None

        # Initialize U splits
        U_train: Optional[np.ndarray] = None
        U_val: Optional[np.ndarray] = None
        U_test: Optional[np.ndarray] = None

        if apply_savgol_filter: # Validate Savitzky-Golay filter parameters if filtering is requested
            if savgol_window_length is None or savgol_polyorder is None:
                raise ValueError("Savitzky-Golay filter parameters must be provided when apply_savgol_filter is True.")
            if savgol_window_length % 2 == 0 or savgol_window_length < 1:
                raise ValueError("Savitzky-Golay window_length must be odd and positive.")
            if savgol_polyorder >= savgol_window_length:
                raise ValueError("Savitzky-Golay polyorder must be less than window_length.")

            if verbose:
                print(f"Applying Savitzky-Golay filter to {', '.join(filtered_set_names)} sets with window_length={savgol_window_length}, polyorder={savgol_polyorder}.")

            if filtered_set_names is not None:
                if "train" in filtered_set_names: # Apply Savitzky-Golay filter to specified sets
                    X_train = savgol_filter(X_train, savgol_window_length, savgol_polyorder, axis=0)
                if "val" in filtered_set_names:
                    X_val = savgol_filter(X_val, savgol_window_length, savgol_polyorder, axis=0) if val_ratio > 0 else None
                if "test" in filtered_set_names:
                    X_test = savgol_filter(X_test, savgol_window_length, savgol_polyorder, axis=0) if test_ratio > 0 else None

            if filtered_set_names is None: # Default to filtering all sets
                X_train = savgol_filter(X_train, savgol_window_length, savgol_polyorder, axis=0)
                X_val = savgol_filter(X_val, savgol_window_length, savgol_polyorder, axis=0) if val_ratio > 0 else None
                X_test = savgol_filter(X_test, savgol_window_length, savgol_polyorder, axis=0) if test_ratio > 0 else None

        if self.U_raw is not None: # Split U (control inputs) if available
            U_train = self.U_raw[:train_end_index]
            U_val = self.U_raw[train_end_index:val_end_index] if val_ratio > 0 else None
            U_test = self.U_raw[val_end_index:] if test_ratio > 0 else None
            if perturb_input_signal_ratio is not None and isinstance(perturb_input_signal_ratio, (float, int)) and perturb_input_signal_ratio > 0: # Perturb input signal if requested
                U_train = self._perturb_input_signal(U_train, perturb_input_signal_ratio, rng, verbose)

        if plot_data: # Plot data if requested
            time_vector = compute_time_vector(self.X_raw, self.dt)
            plot_trajectory(time_vector[:train_end_index], X_train, input_signal=U_train, title="Train Data")
            plot_trajectory(time_vector[train_end_index:val_end_index], X_val, input_signal=U_val, title="Validation Data") if X_val is not None else None
            plot_trajectory(time_vector[val_end_index:], X_test, input_signal=U_test, title="Test Data") if X_test is not None else None

        return X_train, X_val, X_test, U_train, U_val, U_test

    def _perturb_input_signal(self, U: np.ndarray, perturb_ratio: float, rng: np.random.RandomState, verbose: bool) -> np.ndarray:
        """
        Internal method to perturb a given signal by adding random noise.

        Args:
            signal (np.ndarray): The input signal to perturb.
            ratio (float): The ratio by which to perturb the signal (e.g., 0.01 for 1%).
            rng (np.random.RandomState): Random number generator.
            verbose (bool): If True, print detailed messages.

        Returns:
            np.ndarray: The perturbed signal.
        """

        if rng is None:
            warnings.warn("RNG not provided for perturbing input signal. Using default np.random.RandomState(42).")
            rng = np.random.RandomState(42)

        for i in range(U.shape[1]):
            noise_level = max(perturb_ratio * np.std(U[:, i]), self._minimal_noise_value) # Calculate a noise level based on the signal's magnitude and the perturbation ratio
            noise = rng.normal(0, noise_level, U[:, i].shape) # Generate noise from a normal distribution
            U[:, i] += noise

        if verbose:
            print(f"Perturbing signal with ratio {perturb_ratio:.2%} and noise level approximately {noise_level:.4f}.")

        return U
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path
from scipy.signal import savgol_filter

# Import local modules
from utils.helpers import compute_time_vector
from utils.config_manager import ConfigManager
from utils.plots import plot_trajectory

class DataLoader:
    """
    A class for loading and processing data from CSV files.
    It provides methods to extract state variables (X), control inputs (U),
    and determine the time step (dt). Supports optional Savitzky-Golay filtering
    for noise reduction and visualization of loaded data.
    """

    def __init__(self, config_manager: ConfigManager, data_dir: str = "data_processed_dir"):
        """
        Initializes the DataLoader with a ConfigManager instance.

        Args:
            config_manager (ConfigManager): An instance of ConfigManager to access configuration settings.
            data_dir (str): The directory key in the configuration where data files are located.
        """
        self.config_manager = config_manager
        self.config_manager.load_config("settings")
        self.data_load_path = Path(self.config_manager.get_path("settings.paths." + data_dir)) # Load the data path from configuration settings
        self.minimal_noise_value = self.config_manager.get_param("settings.constants.values.minimal_noise_value", default=1e-3) # Load the minimal noise value for perturbing input data

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

    def load_csv_data(
            self,
            file_name,
            state_column_indices: List[int],
            time_column_index: Optional[int] = None,
            time: Optional[float|np.ndarray] = None,
            control_input_column_indices: Optional[List[int]] = None,
            apply_savgol_filter: bool = False,
            savgol_window_length: Optional[int] = None,
            savgol_polyorder: Optional[int] = None,
            plot_data: bool = False,
            verbose: bool = True,
        ) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """
        Loads data from a specified CSV file, processes it, and optionally applies
        a Savitzky-Golay filter.

        Args:
            file_name (str): The name of the CSV file to load (without extension).
            state_column_indices (List[int]): A list of column indices corresponding to state variables (X).
            time_column_index (Optional[int]): The column index for time. If None, 'time' parameter must be provided.
            time (Optional[float | np.ndarray]): The time step (dt) if time_column_index is None,
                                                 or a time vector.
            control_input_column_indices (Optional[List[int]]): A list of column indices for control inputs (U).
            apply_savgol_filter (bool): Whether to apply the Savitzky-Golay filter.
            savgol_window_length (Optional[int]): The window length for the Savitzky-Golay filter.
            savgol_polyorder (Optional[int]): The polynomial order for the Savitzky-Golay filter.
            plot_data (bool): Whether to plot the loaded and filtered data.
            verbose (bool): If True, print detailed messages.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray], float]: A tuple containing:
                - X (np.ndarray): The processed state variables.
                - U (Optional[np.ndarray]): The processed control inputs, or None if not specified.
                - dt (float): The determined time step.

        Raises:
            FileNotFoundError: If the specified CSV file does not exist.
            ValueError: If there are issues loading the CSV, empty data,
                        invalid Savitzky-Golay filter parameters, or missing column definitions.
        """

        filepath = self.data_load_path / f"{file_name}.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"File '{filepath}' not found.")

        all_cols_to_load = self._comprehend_and_validate(state_column_indices, time_column_index, control_input_column_indices) # Input validation

        try: # Load data from CSV
            data_csv = pd.read_csv(filepath, usecols=all_cols_to_load)
        except Exception as e:
            raise ValueError(f"Error loading CSV file '{filepath}': {e}")

        if data_csv.empty:
            raise ValueError(f"File '{filepath}' is empty or contains no data in the specified columns.")

        data = data_csv.to_numpy() # Convert to NumPy array
        data_csv = None # Clear data_csv

        num_samples, _ = data.shape
        if num_samples == 0:
            raise ValueError(f"Data is empty after conversion to numpy array.")

        dt = self._determine_dt(time_column_index, time, data, all_cols_to_load, verbose) # Determine the time step
        original_to_numpy_index_map = {original_index: numpy_index for numpy_index, original_index in enumerate(all_cols_to_load)} # Create a map for NumPy array indices

        X_numpy_column_indices: List[int] = []
        U_numpy_column_indices: List[int] = []

        # Determine which NumPy columns are X and U
        X_numpy_column_indices, U_numpy_column_indices = (
            self._determine_numpy_columns(state_column_indices, original_to_numpy_index_map, time_column_index, control_input_column_indices, verbose)
        )

        X = data[:, X_numpy_column_indices] # Create X array
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Validation for Savitzky-Golay filter
        if apply_savgol_filter:
            if savgol_window_length is None or savgol_polyorder is None:
                raise ValueError("Savitzky-Golay filter parameters must be provided when apply_savgol_filter is True.")
            if savgol_window_length % 2 == 0 or savgol_window_length < 1:
                raise ValueError("Savitzky-Golay window_length must be odd and positive.")
            if savgol_polyorder >= savgol_window_length:
                raise ValueError("Savitzky-Golay polyorder must be less than window_length.")

            if verbose:
                print(f"Applying Savitzky-Golay filter with window_length={savgol_window_length}, polyorder={savgol_polyorder}.")

            X = savgol_filter(X, savgol_window_length, savgol_polyorder, axis=0) # Apply Savitzky-Golay filter

        U: Optional[np.ndarray] = None # Create U array
        if U_numpy_column_indices: # If control input signal exists
            U = data[:, U_numpy_column_indices]
            if U.ndim == 1:
                U = U.reshape(-1, 1)

        data = None # Clear data from memory

        if plot_data: # Plot the trajectory if requested
            time_vector = compute_time_vector(X, dt)
            plot_trajectory(time_vector, X, input_signal=(U if U.shape[1] > 0 else None), title=f"Loaded Data from {file_name}")

        return X, U, dt

    def _comprehend_and_validate(
            self,
            state_column_indices: List[int],
            time_column_index: Optional[int] = None,
            control_input_column_indices: Optional[List[int]] = None
    ) -> List[int]:
        """
        Internal method to validate and combine all column indices to be loaded.

        Args:
            state_column_indices (List[int]): Indices for state variables.
            time_column_index (Optional[int]): Index for the time column.
            control_input_column_indices (Optional[List[int]]): Indices for control inputs.

        Returns:
            List[int]: A sorted list of all unique column indices to load.

        Raises:
            ValueError: If any column index is duplicated or invalid.
        """

        # Implementation of column validation
        if not isinstance(state_column_indices, (list, tuple)) or not all(isinstance(i, int) for i in state_column_indices):
            raise TypeError("`state_column_indices` must be a list or tuple of integers.")
        if not state_column_indices:
            raise ValueError("`state_column_indices` cannot be an empty list.")
        if control_input_column_indices is not None and (
            not isinstance(control_input_column_indices, (list, tuple)) or not all(isinstance(i, int) for i in control_input_column_indices)
        ):
            raise TypeError("`control_input_column_indices` must be a list or tuple of integers, or None.")

        # Implementation of combination logic
        all_cols_to_load_set = set(state_column_indices)
        if time_column_index is not None:
            all_cols_to_load_set.add(time_column_index)
        if control_input_column_indices is not None:
            all_cols_to_load_set.update(control_input_column_indices)

        all_cols_to_load = sorted(list(all_cols_to_load_set))
        return all_cols_to_load

    def _determine_dt(
            self,
            time_column_index: Optional[int],
            time: Optional[float|np.ndarray],
            data: np.ndarray,
            all_cols_to_load: List[int],
            verbose: bool
        ) -> float:
        """
        Internal method to determine the time step (dt) from the loaded data or provided parameter.

        Args:
            time_column_index (Optional[int]): The original index of the time column.
            time (Optional[float | np.ndarray]): The time step or time vector provided by the user.
            data (np.ndarray): The loaded numerical data.
            all_cols_to_load (List[int]): All column indices that were loaded from the CSV.
            verbose (bool): If True, print detailed messages.

        Returns:
            float: The calculated time step (dt).

        Raises:
            ValueError: If dt cannot be determined.
        """

        num_samples, _ = data.shape
        dt: float

        # Implementation of dt determination logic
        if time is None: # Time vector or time step was not provided
            if time_column_index is not None: # Index of time vector is provided
                time_column_numpy_index = all_cols_to_load.index(time_column_index)
                time = data[:, time_column_numpy_index]
                if time.size > 1:
                    dt = np.round(np.median(np.diff(time)), decimals=4)
                    time_print = f"\nEstimated time step (dt): {dt}"
            else: # For ensure if something goes wrong
                raise ValueError("Data are empty, the time step from the first data column cannot be estimated and time is None.")
        elif isinstance(time, (np.ndarray, float, int)): # Time vector or time step was provided
            if isinstance(time, np.ndarray): # Time is time vector
                if time.size > 1:
                    dt = np.round(np.median(np.diff(time)), decimals=4)
                    time_print = f"\nEstimated time step (dt): {dt}"
                else: # Field does not have sufficient length
                    raise ValueError("The time field must contain at least two elements to estimate the time step.")

            elif isinstance(time, (float, int)): # Time is time step
                dt = time
                time_print = f"\nProvided time step (dt): {dt}"
        else:
            if num_samples > 1: # If everything flase, try to load from data_config
                self.config_manager.load_config("data_config")
                dt = self.config_manager.get_param("data_config.dt")
                time_print = f"\nUsing default time step (dt) from configuration: {dt}"
            else:
                raise ValueError("Insufficient data to determine time step. Provide " \
                                "`dt_override`, `time_column_index`, or configure `data_config.dt`.")

        if verbose:
            print(time_print)

        return dt

    def _determine_numpy_columns(
            self,
            state_column_indices: List[int],
            original_to_numpy_index_map = List[int],
            time_column_index: Optional[int] = None,
            control_input_column_indices: Optional[List[int]] = None,
            verbose: bool = True,
    ) -> Tuple[List[int], List[int]]:
        """
        Internal method to map original column indices to NumPy array indices for X and U.

        Args:
            state_column_indices (List[int]): Original indices for state variables.
            original_to_numpy_index_map (dict): Mapping from original CSV indices to loaded NumPy array indices.
            time_column_index (Optional[int]): Original index for the time column.
            control_input_column_indices (Optional[List[int]]): Original indices for control inputs.
            verbose (bool): If True, print detailed messages.

        Returns:
            Tuple[List[int], List[int]]: A tuple containing lists of NumPy indices for X and U.

        Raises:
            ValueError: If state variables (X) are not selected.
        """

        X_numpy_column_indices: List[int] = []
        U_numpy_column_indices: List[int] = []

        # Determine which NumPy columns are X and U
        for original_index in state_column_indices:
            if original_index == time_column_index: # Time vector will not be in X
                if verbose:
                    print(f"Warning: Time column index {original_index} specified in "
                           "`state_column_indices` will be ignored for X.")
                continue
            if control_input_column_indices is not None and original_index in control_input_column_indices: # If column is defined as X and U
                if verbose:
                    print(f"Warning: Column index {original_index} specified in both "
                           "`state_column_indices` and `control_input_column_indices`. "
                           "It will be treated as a control input (U).")
                continue # Take it as U
            if original_index in original_to_numpy_index_map: # Index exist in NumPy map
                X_numpy_column_indices.append(original_to_numpy_index_map[original_index])
            else:
                raise ValueError(f"Original column index {original_index} for state variable X not found in loaded data.")

        if control_input_column_indices is not None: # If U columns indicies are provided
            for original_index in control_input_column_indices:
                if original_index in original_to_numpy_index_map:
                    U_numpy_column_indices.append(original_to_numpy_index_map[original_index])
                else:
                    raise ValueError(f"Original column index {original_index} for control input U not found in loaded data.")

        if not X_numpy_column_indices:
            raise ValueError("No state variables (X) were selected. Check " \
                             "`state_column_indices` and avoid overlaps with `time_column_index` or `control_input_column_indices`.")

        return X_numpy_column_indices, U_numpy_column_indices

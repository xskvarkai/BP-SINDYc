import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path

# Import local modules
from utils.helpers import compute_time_vector
from utils.config_manager import ConfigManager
from utils.plots import plot_trajectory

class DataLoader:
    """
    A class for loading data from files. It provides methods to extract
    state variables (X), control inputs (U), and determine the time step (dt).
    """

    def __init__(self, config_manager: ConfigManager, data_dir: str = "data_processed_dir"):
        """
        Initializes the DataLoader with a ConfigManager instance.
        """
        self.config_manager = config_manager
        self.config_manager.load_config("settings")
        # Nacitanie cesty pre data z konfiguracie, ak cesta existuje
        self.data_load_path = Path(self.config_manager.get_path("settings.paths." + data_dir))
        # Nacitanie minimal_noise_value z konfiguracie pre perturbovanie vstupny dat
        self.minimal_noise_value = self.config_manager.get_param(
            "settings.defaults.constants.values.minimal_noise_value", default=1e-3
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def load_csv_data(
            self,
            file_name,
            state_column_indices: List[int],
            time_column_index: Optional[int] = None,
            time: Optional[float|np.ndarray] = None,
            control_input_column_indices: Optional[List[int]] = None,
            plot_data: bool = False,
            perturb_input_signal: bool = False,
            verbose: bool = True,
        ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Loads data from a CSV file, extracts state variables (X),
        control inputs (U), and estimates the time step (dt).
        """

        filepath = self.data_load_path / f"{file_name}.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"File '{filepath}' not found.")

        # Validacia vstupu 
        all_cols_to_load = self._comprehend_and_validate(state_column_indices, time_column_index, control_input_column_indices)

        # ---------- Nacitanie dat z csv ----------
        try:
            data_csv = pd.read_csv(filepath, usecols=all_cols_to_load)
        except Exception as e:
            raise ValueError(f"Error loading CSV file '{filepath}'")
        
        if data_csv.empty:
            raise ValueError(f"File '{filepath}' is empty or contains no data in the specified columns.")
        
        # Konverzia na NumPy array a vymazanie data_csv (Optimalizacia RAM)
        data = data_csv.to_numpy()
        data_csv = None # Vymazanie data_csv

        num_samples, _ = data.shape
        if num_samples == 0:
            raise ValueError(f"Data is empty after conversion to numpy array.")
        
        # Zistenie casoveho kroku
        dt = self._determine_dt(time_column_index, time, data, all_cols_to_load, verbose)

        # Vytvorenie mapy pre NumPy polia
        original_to_numpy_index_map = {original_index: numpy_index for numpy_index, original_index in enumerate(all_cols_to_load)}

        X_numpy_column_indices: List[int] = []
        U_numpy_column_indices: List[int] = []

        # Urcenie, ktore NumPy stlpce su X a U
        X_numpy_column_indices, U_numpy_column_indices = (
            self._determine_numpy_columns(state_column_indices, original_to_numpy_index_map, time_column_index, control_input_column_indices, verbose)
        )
        
        # ---------- Vytvorenie X array ----------
        X = data[:, X_numpy_column_indices]
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # ---------- Vytvorenie U array ----------
        U: Optional[np.ndarray] = None
        if U_numpy_column_indices: # Ak existuje vstupny signal
            U = data[:, U_numpy_column_indices]
            if U.ndim == 1:
                U = U.reshape(-1, 1)
        
        data = None # Vymazanie data

        # Vykreslenie grafu
        if plot_data:
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
        Valides columns indices inputs,
        calculates and sorts all indices columns. 
        """

        # Validacia vstupnych udajov
        if not isinstance(state_column_indices, (list, tuple)) or not all(isinstance(i, int) for i in state_column_indices):
            raise TypeError("`state_column_indices` must be a list or tuple of integers.")
        if not state_column_indices:
            raise ValueError("`state_column_indices` cannot be an empty list.")
        if control_input_column_indices is not None and (
            not isinstance(control_input_column_indices, (list, tuple)) or not all(isinstance(i, int) for i in control_input_column_indices)
        ):
            raise TypeError("`control_input_column_indices` must be a list or tuple of integers, or None.")

        # Zistenie poctu stlpcov
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
        ) -> Tuple[float, str]:
        """
        Determines the time step (dt) based on time, time column, or configuration.
        """

        num_samples, _ = data.shape
        dt: float

        if time is None: # Nebol zadany casovy vektor ani casovy krok
            if time_column_index is not None: # Zadany je index casoveho vektora
                time_column_numpy_index = all_cols_to_load.index(time_column_index)
                time = data[:, time_column_numpy_index]
                if time.size > 1:
                    dt = np.round(np.median(np.diff(time)), decimals=4)
                    time_print = f"\nEstimated time step (dt): {dt}"
            else: # Nebol zadany ani casovy krok ani index casoveho vektora
                raise ValueError("Data are empty, the time step from the first data column cannot be estimated.")
        elif isinstance(time, (np.ndarray, float, int)): # Zadany casovy krok alebo vektor
            if isinstance(time, np.ndarray): # Zadany je vektor
                if time.size > 1:
                    dt = np.round(np.median(np.diff(time)), decimals=4)
                    time_print = f"\nEstimated time step (dt): {dt}"
                else: # Vektor je mensi nanajvys rovny 1 
                    raise ValueError("The time field must contain at least two elements to estimate the time step.")
            
            elif isinstance(time, (float, int)): # Zadany je casovy krok
                dt = time
                time_print = f"\nProvided time step (dt): {dt}"
        else:
            if num_samples > 1: # Pokus o nacitanie z konfiguracie
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
        Determines which columns are X_numpy_column_indicies and U_numpy_column_indices.
        """
                
        X_numpy_column_indices: List[int] = []
        U_numpy_column_indices: List[int] = []

        # Urcenie, ktore stlpce su X a U
        for original_index in state_column_indices:
            if original_index == time_column_index: # Casovy stlpec nebude v datach X
                if verbose:
                    print(f"Warning: Time column index {original_index} specified in "
                           "`state_column_indices` will be ignored for X.")
                continue
            if control_input_column_indices is not None and original_index in control_input_column_indices:
                # Ak je jeden stlpec definovany ako X aj U, uprednostineme U
                if verbose:
                    print(f"Warning: Column index {original_index} specified in both "
                           "`state_column_indices` and `control_input_column_indices`. "
                           "It will be treated as a control input (U).")
                continue # Zoberieme ako U
            if original_index in original_to_numpy_index_map:
                # Index existuje v mape pre NumPy data
                X_numpy_column_indices.append(original_to_numpy_index_map[original_index])
            else:
                raise ValueError(f"Original column index {original_index} for state variable X not found in loaded data.")

        if control_input_column_indices is not None: # Vyber U ak su zadane indexy pre vstupny signal
            for original_index in control_input_column_indices:
                if original_index in original_to_numpy_index_map:
                    U_numpy_column_indices.append(original_to_numpy_index_map[original_index])
                else:
                    raise ValueError(f"Original column index {original_index} for control input U not found in loaded data.")
              
        if not X_numpy_column_indices:
            raise ValueError("No state variables (X) were selected. Check " \
                             "`state_column_indices` and avoid overlaps with `time_column_index` or `control_input_column_indices`.")
        
        return X_numpy_column_indices, U_numpy_column_indices

import pysindy as ps
import numpy as np
from datetime import datetime
from pathlib import Path
import traceback
from typing import List, Dict, Any, Optional, Union
import concurrent.futures
from pebble import ProcessPool
from concurrent.futures import TimeoutError as FuturesTimeoutError

import json
import pickle
import multiprocessing
import tempfile
import os
import gc
import warnings

from models.base import BaseSindyEstimator
from utils.config_manager import ConfigManager
from utils.plots import plot_trajectory, plot_pareto
import utils.sindy_helpers as sindy_helpers
from scripts.sindy_run_configuration import run_config

#PysindyConfigObject = Union[ps.optimizers.BaseOptimizer, ps.feature_library.base.BaseFeatureLibrary, ps.differentiation.BaseDifferentiation]

class SindyEstimator(BaseSindyEstimator):
    """
    A class for estimating SINDy models. It manages configurations, performs
    parallel searches to find optimal models, It inherits from BaseSindyEstimator
    for configuration management.
    """

    def __init__(self, config_manager: ConfigManager):
        """
        Initializes the SindyEstimator with a ConfigManager instance.

        Args:
            config_manager (ConfigManager): An instance of ConfigManager to access
                                            configuration settings.
        """

        self.pareto_front: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []
        self.best_config: Optional[Dict[str, Any]] = None
        self.results_file_name: Optional[str] = None # Name of the temporary file for results

        self.config_manager = config_manager
        config_manager.load_config("settings")
        self.data_export_path = Path(self.config_manager.get_path("settings.paths.data_export_dir"))
        self._default_constraints: Dict[str, Any] = self.config_manager.get_param("settings.valid_methods.search_constraints", default={})

        super().__init__(config_manager)

    def __enter__(self):
        """
        Enters the runtime context related to this object.
        Returns the instance itself for use in 'with' statements.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the runtime context related to this object.
        Handles any exceptions that occurred within the 'with' block,
        and cleans up temporary files.
        """

        self._delete_tempfiles()
        self.differentiation_methods.clear()
        self.optimizers.clear()
        self.feature_libraries.clear()
        self.configurations.clear()
        self.results.clear()

        return None

    # Paralelne hladanie najlepsej konfiguracie
    def search_configurations(
        self,
        x_train: np.ndarray|List[np.ndarray],
        x_val: np.ndarray|List[np.ndarray],
        u_train: Optional[np.ndarray|List[np.ndarray]] = None,
        u_val: Optional[np.ndarray|List[np.ndarray]] = None,
        dt: float = None,
        n_processes: int = 4,
        log_file_name: str = "worker_results",
        verbose: bool = True,
        timeout_per_config: Optional[int] = 300, # v sekundach
        **constraints: Any
    ):
        """
        Performs a parallel search for optimal SINDy model configurations.
        It evaluates each configuration using training and validation data,
        and stores the results in a temporary file to manage memory.

        Args:
            x_train (np.ndarray): Training state variables.
            x_val (np.ndarray): Validation state variables.
            u_train (Optional[np.ndarray]): Training control inputs.
            u_val (Optional[np.ndarray]): Validation control inputs.
            dt (float): The time step of the data.
            n_processes (int): Number of parallel processes to use.
            log_file_name (str): Name of the log file for worker results.
            timeout_per_config (int): Maximum time (in seconds) allowed for
                                      each configuration to run.
            **constraints (Any): Additional constraints for filtering models.

        Raises:
            ValueError: If there are not enough validation samples.
        """

        if not self.configurations:
            raise ValueError("No configurations defined. Use generate_configurations() first.")

        if constraints:
            self._default_constraints.update(constraints) # Update default constraints with any provided in kwargs

        total_val_samples = x_val.shape[0]

        min_validation_sim_steps = self.config_manager.get_param("settings.constants.values.min_validation_sim_steps", 50)
        if self._default_constraints.get("sim_steps") <= min_validation_sim_steps:
            self._default_constraints["sim_steps"] = min_validation_sim_steps
            warnings.warn(f"Minimum required simulation steps are {min_validation_sim_steps},"
                          f"validation steps increased/decreased automatically to match this requirement.")
        min_validation_sim_steps = None

        if total_val_samples < self._default_constraints.get("sim_steps"): # Check if enough validation samples are available for simulation
            raise ValueError(f"Not enough validation samples. Decrease validation steps to {total_val_samples} or increase validation size.")

        configurations_and_data = [ # Prepare arguments for parallel processing
                (index, config, x_train, x_val, u_train, u_val, dt, self._default_constraints)
                for index, config in enumerate(self.configurations)
            ]
        total_configurations = len(configurations_and_data)

        self.configurations.clear() # Clear configurations after they are prepared for processing

        if verbose:
            start_time = datetime.now()
            print(f"\nParameter search started...")
            print(f"Total configurations to explore: {total_configurations}")
            print(f"Using {n_processes} parallel processes")
            print(f"Start time: {start_time.strftime("%H:%M:%S")}")

        log_filepath = self.data_export_path / f"{log_file_name}.log" # Define file paths for logging and temporary results
        if os.path.exists(log_filepath):
            os.remove(log_filepath)

        with tempfile.NamedTemporaryFile(delete=False, mode="wb", prefix="sindyR", suffix=".pkl") as results_file, \
                open(log_filepath, "a", encoding="utf-8") as f_log: # Use temporary file to store results to avoid memory issues

            self.results_file_name = results_file.name

            with ProcessPool(max_workers=n_processes) as pool: # Use Pebble's ProcessPool for parallel execution with timeouts
                future_to_index = { # Schedule tasks and map futures to their original index
                    pool.schedule(run_config, args=(config_data,), timeout=timeout_per_config): index
                    for index, config_data in enumerate(configurations_and_data, 1)
                }

                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_index): # Process results as they complete
                    index = future_to_index[future]
                    completed_count += 1

                    try:
                        current_config_result = future.result()

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_output = current_config_result.copy()
                        log_output = self._prepare_log_output(log_output)
                        f_log.write(f"[{timestamp}] Result {index}: {str(log_output)}\n{'-'*180}\n\n")
                        f_log.flush()

                        if not current_config_result.get("error"): # Store valid results in the temporary file
                            current_config_result["index"] = index
                            pickle.dump(current_config_result, results_file)

                    except FuturesTimeoutError: # Handle configurations that exceeded the timeout
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f_log.write(f"[{timestamp}] Result {index}: KILLED (Reach timeout limit {timeout_per_config}s)\n{'-'*180}\n\n")
                        f_log.flush()

                    except Exception as e: # Handle other exceptions during configuration processing
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f_log.write(f"[{timestamp}] Error (Index {index}): {str(e)}\n{'-'*180}\n\n")
                        f_log.flush()
                        warnings.warn(f"Chyba v konfigurácii {index}: {str(e)}")

                    gc.collect() # Trigger garbage collection to free up memory

                    if verbose: # UI/UX - progress bar
                        print(f"Processing configuration {completed_count}/{total_configurations} "
                                f"({(completed_count/total_configurations)*100:.2f}%)", end="\r", flush=True)
                print() # Print separator

        warnings.filterwarnings("default", category=UserWarning)
        configurations_and_data.clear() # Clear configuration_and_data

        with open(self.results_file_name, "rb") as f: # Load all results from the temporary file after all tasks are completed
            try:
                while True:
                    self.results.append(pickle.load(f))
            except EOFError:
                pass

        if verbose: # UI/UX
            print("\nParameter search complete.")
            duraction = datetime.now() - start_time
            seconds = duraction.total_seconds()
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"The process took {int(hours):02}:{int(minutes):02}:{int(seconds):02} hours")
            valid_configs = sum(1 for result in self.results if result is not None)
            print(f"Valid configurations found: {valid_configs} out of {total_configurations}")

        if self.results: # Compute Pareto front and identify the best configuration
            self.best_config = self._select_best_config(self.results)
            self.pareto_front = self._compute_pareto_front(self.results)
            self.results.clear() # Clear results

        return None

    def validate_on_test(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        u_train: Optional[np.ndarray] = None,
        u_test: Optional[np.ndarray] = None,
        dt: float = 0.01,
        plot: bool = True,
        **constraints
    ):
        """
        Validates the best SINDy model (found by `search_configurations`)
        on the unseen test dataset.

        Args:
            x_train (np.ndarray): Training state variables (used for model construction if needed).
            x_test (np.ndarray): Test state variables.
            u_train (Optional[np.ndarray]): Training control inputs.
            u_test (Optional[np.ndarray]): Test control inputs.
            dt (float): The time step of the data.
            plot (bool): If True, plots the simulated vs. actual trajectories on test data.
            **constraints (Any): Additional constraints to pass to model evaluation.

        Raises:
            ValueError: If no best configuration has been found.
        """

        if self.best_config is None:
            warnings.warn("No best configuration found. Run search_configurations() first.")
            return None

        if constraints: # Update default constraints with any provided in kwargs
            self._default_constraints.update(constraints)

        data = { # Prepare data for model reconstruction and evaluation
            "x_train": x_train,
            "x_ref": x_test,
            "u_train": u_train,
            "u_ref": u_test,
            "dt": dt
        }

        config = self.best_config["configuration"]

        # Ignorovanie warningov pocas testovania
        warnings.filterwarnings("ignore", module="pysindy")

        if self.best_config.get("coefficients").any(): # Reconstruct the best model using their coeffs
            model = sindy_helpers.copy_coeffs(config, data, self.best_config["coefficients"])
        else: # Reconstruct the best model using the best configuration
            model = sindy_helpers.model_costruction(config, data, self.best_config.get("random_seed", 42), constraints.get("coeff_precision"))

        print("\nStarting validation on test data...")
        x_sim, rmse, r2, _ = sindy_helpers.evaluate_model( # Evaluate the reconstructed model on the test data
            model,
            data,
            start_index=0,
            current_steps=x_test.shape[0],
            ksteps=constraints.get("ksteps"),
            integrator_kwargs={"rtol": 1e-6,"atol": 1e-6}
        )
        warnings.filterwarnings("default", category=UserWarning)

        min_len = min(len(x_test), len(x_sim))

        print(f"Best model R2 score: {r2:.3%}") # Print validation metrics
        self.best_config["test_metrics"] = {
            "rmse": np.round(rmse, 5),
            "r2": np.round(r2, 5),
            "simulation_length": min_len
        }

        if plot: # Plot if requested
            t_test = np.arange(min_len) * dt
            x_ref_cut = x_test[:min_len]
            x_sim_cut = x_sim[:min_len]
            plot_trajectory(t_test, x_ref_cut, x_sim_cut, title="Validation on test data")

        return None

    def export_data(self, data: dict = None, export_file_name: str = "data"):
        """
        Exports the best configuration, Pareto front, and optional user data
        to a JSON file.

        Args:
            data (Optional[Dict[str, Any]]): Optional dictionary of additional
                                             user data to export.
            export_file_name (str): The name of the file to export data to
                                     (without extension).
        """

        if self.pareto_front is None:
            warnings.warn("Pareto front is None")
        if self.best_config is None:
            warnings.warn("Best configuration is None")

        for result in self.pareto_front:
            if result.get("coefficients") is not None:
                del result["coefficients"]

        payload = {
            "best_result": self.best_config,
            "pareto_front": self.pareto_front,
            "user_data": data
        }

        try:
            filepath = self.data_export_path / f"{export_file_name}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=5, default=str)
        except Exception as e:
            warnings.warn(str(e))

        return None


    def plot_pareto(self):
        """
        Plots the Pareto front if available, visualizing the trade-off
        between model error (RMSE) and complexity.
        """

        if self.pareto_front is None:
            return None

        errs = np.array([r.get("rmse") for r in self.pareto_front], dtype=float)
        spars = np.array([r.get("complexity") for r in self.pareto_front], dtype=float)

        plot_pareto(errs, spars)
        return None

    def _compute_pareto_front(self, results: List[Dict]) -> List[Dict]:
        """
        Computes the Pareto front from a list of SINDy model results.
        The Pareto front represents configurations that offer the best trade-off
        between model error (RMSE) and complexity (number of non-zero coefficients).

        Args:
            results (List[Dict]): A list of dictionaries, where each dictionary
                                  represents the results of a SINDy configuration.

        Returns:
            List[Dict]: A list of dictionaries representing the Pareto optimal configurations.
        """

        valid_results = [result for result in results if result is not None] # Filter out invalid (None) results
        if not valid_results:
            warnings.warn("No valid configurations found. All configurations were filtered out.")
            return None

        sorted_results = sorted(valid_results, key=lambda x: x["rmse"])  # Sort results by RMSE (ascending) and then by complexity (ascending)

        pareto_front = [sorted_results[0]]
        # Check if the current result is dominated by an existing Pareto result
        # A result is dominated if another result has lower or equal RMSE AND
        # lower or equal complexity, with at least one strictly lower.
        for candidate in sorted_results[1:]:
            if candidate["complexity"] < pareto_front[-1]["complexity"]:
                    pareto_front.append(candidate)

        return pareto_front

    def _select_best_config(self, results: List[Dict]) -> Dict[str, Any]:
        valid_results = [result for result in results if result is not None] # Filter out invalid (None) results
        if not valid_results:
            warnings.warn("No valid configurations with AIC found. Cannot select best configuration.")
            return None

        sorted_results = sorted(results, key=lambda x: x["aic"]) # Select base on AIC (Akaike Information Criterion)
        best_model = sorted_results[0]

        return best_model

    def _prepare_log_output(self, log: Dict[str, Any]):

        if log.get("coefficients") is not None:
            del log["coefficients"]

        return log

    def _delete_tempfiles(self):
        if self.results_file_name and os.path.exists(self.results_file_name):
            os.remove(self.results_file_name)

        return None
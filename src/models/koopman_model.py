import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Union
import warnings
import json
warnings.filterwarnings("ignore", module="pykoopman")
import pykoopman
from utils.plots import plot_trajectory, plot_koopman_spectrum
from utils.helpers import compute_time_vector
from utils.config_manager import ConfigManager
import utils.koopman_helpers as koopman_helpers
from sklearn.preprocessing import MinMaxScaler

class KoopmanModel():
    """
    A class for modeling dynamical systems using the Koopman operator framework.
    It handles data scaling, model training, performance evaluation, and simulation
    of future states. The class also includes methods for interpreting eigenvalues
    to assess system stability and oscillation characteristics, and for exporting
    model parameters.
    """

    def __init__(
            self,
            config_manager: ConfigManager,
            config: Dict[str, Any],
            X_train: np.ndarray,
            U_train: np.ndarray,
            X_test: np.ndarray,
            U_test: np.ndarray,
            dt: Union[float, int]
        ):
        """
        Initializes the KoopmanModel.

        Args:
            config_manager (ConfigManager): An instance of ConfigManager to access configuration settings.
            config (Dict[str, Any]): A dictionary containing configuration parameters for the Koopman model,
                                     including 'observables' and 'regressor'.
            X_train (np.ndarray): Training state variables.
            U_train (Optional[np.ndarray]): Training control inputs.
            X_test (np.ndarray): Test state variables.
            U_test (Optional[np.ndarray]): Test control inputs.
            dt (float): The time step of the data.
        """

        self.scaler_X = MinMaxScaler().fit(X_train) # Scaler for state variables
        self.scaler_U = MinMaxScaler().fit(U_train) # Scaler for control inputs

        self.config_manager = config_manager
        self.data_export_path = config_manager.load_config("settings")
        self.data_export_path = Path(self.config_manager.get_path("settings.paths.data_export_dir")) # Load data export path from configuration

        self.config = config
        self.data = { # Store scaled data
            "x_train": self.scaler_X.transform(X_train),
            "u_train": self.scaler_U.transform(U_train),
            "x_ref": self.scaler_X.transform(X_test),
            "u_ref": self.scaler_U.transform(U_test),
            "dt": dt
        }

        self.model = koopman_helpers.make_model(self.config, self.data) # Train model during initialization

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

    def evaluateModel(self, print_metrics: bool = False, plot: bool = True, u_plot: np.ndarray = None) -> Tuple[np.ndarray, float, float]:
        """
        Evaluates the trained Koopman model on the test set.

        Args:
            start_index (int): The starting index for evaluation in the test set.
            print_metrics (bool): If True, prints RMSE and R2 scores.
            plot (bool): If True, plots the simulated vs. actual trajectories.
            u_plot (Optional[np.ndarray]): Control input signal to use for plotting, if different from U_test_raw.

        Returns:
            Dict[str, float]: A dictionary containing RMSE and R2 scores.
        """

        if self.model is None:
            warnings.warn("Koopman model not trained. Please call _make_model first.")
            return None

        x_sim, rmse, r2 = koopman_helpers.evaluate_model(self.model, self.data, 0, self.data.get("x_ref").shape[0])

        if print_metrics:
            print(f"Koopman model state R2 score: {r2:.3%}")
            print(f"Koopman model state RMSE: {rmse:.5f}")
        x_sim = self.scaler_X.inverse_transform(x_sim) # Inverse transform scaled data for plotting and metric interpretation
        x_ref = self.scaler_X.inverse_transform(self.data.get("x_ref"))

        u_sim = self.scaler_U.inverse_transform(self.data.get("u_ref"))
        if u_plot is not None:
            u_sim = u_plot

        if plot:
            plot_trajectory(compute_time_vector(x_sim, self.data.get("dt")), x_ref, x_sim, u_sim, title="Validation on test data")

        return (x_sim, rmse, r2)

    def plot_koopman_spectrum(self):
        plot_koopman_spectrum(self.model.lamda_array)

    def koopman_simulate(self, x_ref: np.ndarray, dt: float, u_ref: np.ndarray = None) -> np.ndarray:

        """
        Simulates the Koopman model forward in time given initial conditions and control inputs.

        Args:
            X_ref (np.ndarray): Reference state variables for initial conditions.
            dt (float): Time step.
            U_ref (Optional[np.ndarray]): Reference control inputs.

        Returns:
            Optional[np.ndarray]: The simulated trajectory (unscaled), or None if simulation fails.
        """

        u_ref = self.scaler_U.transform(u_ref)
        x_ref = self.scaler_X.transform(x_ref)
        data = {
            "x_ref": x_ref,
            "u_ref": u_ref,
            "dt": dt
        }

        x_sim = koopman_helpers.model_simulate(self.model, data, 0, data.get("x_ref").shape[0])

        if isinstance(x_sim, str):
            warnings.warn(x_sim)
            return None

        x_sim = self.scaler_X.inverse_transform(x_sim)

        return x_sim

    def export_data(self, export_file_name: str = "Koopman_opherator"):
        """
        Exports the Koopman model's parameters (A, B matrices, observables, spectral decomposition,
        and mode decomposition) to a JSON file.

        Args:
            export_file_name (str): The name of the file to export data to (without extension).
        """

        a_matrix_export = self._format_matrix(self.model.A, self.model.observables.get_feature_names())
        b_matrix_export = self._format_matrix(self.model.B, self.model.observables.get_feature_names())
        w_matrix_export = self.model.W.tolist() if isinstance(self.model.W, np.ndarray) else self.model.W
        w_matrix_export = [[str(val) for val in row] for row in w_matrix_export]

        spectral_decomposition_data = self._intepret_eigenvalues(self.model.lamda_array)

        payload = {
            "observables": self.model.observables.get_feature_names(),
            "A matrix": a_matrix_export,
            "B_matrix": b_matrix_export,
            "Spectral decomposition": spectral_decomposition_data,
            "Mode decomposition": w_matrix_export
        }

        try:
            filepath = self.data_export_path / f"{export_file_name}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=5, default=str)
        except Exception as e:
            warnings.warn(str(e))

    def _intepret_eigenvalues(self, lambda_array: np.ndarray) -> list:
        """
        Interprets a NumPy array of complex eigenvalues and returns a list of dictionaries,
        each containing the eigenvalue's details and stability/oscillation status.
        """

        interpreted_eigenvalues = []
        for i, lambda_val in enumerate(lambda_array):
            magnitude = np.abs(lambda_val)
            phase_rad = np.angle(lambda_val)

            stability_status = ""
            if magnitude < 1 - 1e-9:
                stability_status += "Dampened (stable)"
            elif magnitude > 1 + 1e-9:
                stability_status += "Growing (unstable)"
            else:
                stability_status += "Stable (on unit circle)"

            if np.abs(phase_rad) > 1e-9:
                stability_status += ", Oscillating"
            else:
                stability_status += ", Non-oscillating"

            eigenvalue_info = {
                "id": f"lambda_{i+1}",
                "complex_value": str(lambda_val),
                "real_part": lambda_val.real,
                "imag_part": lambda_val.imag,
                "magnitude": magnitude,
                "phase_rad_per_sample": phase_rad,
                "interpretation": stability_status
            }
            interpreted_eigenvalues.append(eigenvalue_info)
        return interpreted_eigenvalues

    def _format_matrix(self, matrix, observables):
        """
        Formats a matrix (A or B) into a list of dictionaries for JSON export,
        mapping each row and column to its corresponding feature name.

        Args:
            matrix (np.ndarray): The matrix to format.
            feature_names (list): A list of feature names corresponding to rows/columns.

        Returns:
            list: A list of dictionaries representing the formatted matrix.
        """

        matrix = matrix.tolist() if isinstance(matrix, np.ndarray) else matrix

        max_width = 0
        for row in matrix:
            for val in row:
                max_width = max(max_width, len(f"{val}"))

        matrix_formatted = []
        for i, row in enumerate(matrix):
            formatted_row = [f"{x:{max_width}}" for x in row]
            matrix_formatted.append({
                "observable": observables[i],
                "row_values": formatted_row
            })

        return matrix_formatted
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
        
        self.scaler_X = MinMaxScaler().fit(X_train)
        self.scaler_U = MinMaxScaler().fit(U_train)

        self.config_manager = config_manager
        self.data_export_path = config_manager.load_config("settings")
        self.data_export_path = Path(self.config_manager.get_path("settings.paths.data_export_dir"))

        self.config = config
        self.data = {
            "x_train": self.scaler_X.transform(X_train),
            "u_train": self.scaler_U.transform(U_train),
            "x_ref": self.scaler_X.transform(X_test),
            "u_ref": self.scaler_U.transform(U_test),
            "dt": dt
        }

        self.model = koopman_helpers.make_model(self.config, self.data)
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def returnModel(self) -> pykoopman.Koopman:
        return self.model

    def evaluateModel(self, print_metrics: bool = False, plot: bool = True, u_plot: np.ndarray = None) -> Tuple[np.ndarray, float, float]:
        """
        Evaluate the model by simulating it and computing performance metrics (RMSE, R2 score).
        Returns the simulated trajectory, RMSE, R2 score. Optionally plots predicted trajectory
        and prints metrics.
        """
        
        x_sim, rmse, r2 = koopman_helpers.evaluate_model(self.model, self.data, 0, self.data.get("x_ref").shape[0])

        if print_metrics:
            print(f"Koopman model state R2 score: {r2:.3%}")
            print(f"Koopman model state RMSE: {rmse:.5f}")
        x_sim = self.scaler_X.inverse_transform(x_sim)
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
        """ """

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
                # default=str zabezpeci serializaciu objektov ako NumPy polia
                json.dump(payload, f, indent=5, default=str)
        except Exception as e:
            warnings.warn(str(e))

    def _intepret_eigenvalues(self, lambda_array: np.ndarray) -> list:  
        """  
        Interprets a NumPy array of complex eigenvalues and returns a list of dictionaries,  
        each containing the eigenvalue"s details and stability/oscillation status.  
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
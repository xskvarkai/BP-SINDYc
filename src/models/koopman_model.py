import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
import warnings
warnings.filterwarnings("ignore", module="pykoopman")
import pykoopman
from utils.plots import plot_trajectory, plot_koopman_spectrum
from utils.helpers import compute_time_vector
from utils.config_manager import ConfigManager
import utils.koopman_helpers as koopman_helpers

class KoopmanModel():
    def __init__(
            self,
            config_manager: ConfigManager,
            config: Dict[str, Any],
            X_train: np.ndarray,
            Y_train: Optional[np.ndarray],
            U_train: np.ndarray,
            X_test: np.ndarray,
            U_test: np.ndarray,
            dt: Union[float, int]
        ):
        
        self.config_manager = config_manager
        self.config = config
        self.data = {
            "x_train": X_train,
            "u_train": U_train,
            "y_train": Y_train,
            "x_ref": X_test,
            "u_ref": U_test,
            "dt": dt
        }
        self.model = koopman_helpers.make_model(self.config, self.data)
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def returnModel(self) -> pykoopman.Koopman:
        return self.model

    def evaluateModel(self, x_test: np.ndarray = None, u_test: np.ndarray = None, print_metrics: bool = False, plot: bool = True) -> Tuple[np.ndarray, float, float]:
        """
        Evaluate the model by simulating it and computing performance metrics (RMSE, R2 score).
        Returns the simulated trajectory, RMSE, R2 score. Optionally plots predicted trajectory
        and prints metrics.
        """
        if x_test is not None:
            self.data["x_ref"] = x_test
        if u_test is not None:
            self.data["u_ref"] = u_test
          
        x_sim, rmse, r2 = koopman_helpers.evaluate_model(self.model, self.data, 0, self.data.get("x_ref").shape[0])
          
        if print_metrics:
            print(f"\nKoopman model R2 score on states: {r2:3%}")
            print(f"Koopman model RMSE: {rmse:.5f}")
        
        if plot:
            plot_trajectory(compute_time_vector(x_sim, self.data.get("dt")), self.data.get("x_ref"), x_sim, self.data.get("u_ref"), title="Validation on test data")

        return (x_sim, rmse, r2)
    
    def plot_koopman_spectrum(self, print_K: bool = True):
        K = self.model.lamda_array
        if print_K:
            print(f"\nKoopman operator lambda matrix: \n{K}")
        plot_koopman_spectrum(K)
from utils.config_manager import ConfigManager
import numpy as np
import pandas as pd
from pathlib import Path
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from typing import Any, Callable, Dict, Optional, Union
from utils.helpers import rk4_integrator

# --- Helper Functions ---
def make_measurement_functions(measure_x1: bool = True, measure_x2: bool = False, measure_x3: bool = False):
    if not measure_x1:
        raise ValueError("At least state x1 must be chosen for measurement.")

    measured_indices = []
    if measure_x1:
        measured_indices.append(0)
    if measure_x2:
        measured_indices.append(1)
    if measure_x3:
        measured_indices.append(2)

    dim_z = len(measured_indices)
    
    def hx(x: np.ndarray) -> np.ndarray:
        return x[measured_indices]

    return hx, dim_z

# --- UKF Estimator Class ---

class SindyUKFEstimator:
    def __init__(self, config_manager: ConfigManager, sindy_dxdt_func: Callable[[np.ndarray, float], np.ndarray], dt: Union[float, int]):
        self.config_manager = config_manager
        config_manager.load_config("ukf_params")

        self.sindy_dxdt_func = sindy_dxdt_func

        # Load UKF parameters from the config manager
        self.dt: float = dt
        self.measure_x2: bool = self.config_manager.get_param("ukf_params.measure_x2", False)
        self.measure_x3: bool = self.config_manager.get_param("ukf_params.measure_x3", False)

        self.config_manager.load_config("settings")
        csv_output_filename: str = self.config_manager.get_param("ukf_params.csv_output_filename", "x3_estimates.csv")
        self.data_export_path = Path(self.config_manager.get_path("settings.paths.data_processed_dir") / csv_output_filename)
        
        # Initialize matrices. Config values (e.g., list of lists from YAML) are converted to np.array.
        self.P0: np.ndarray = np.array(self.config_manager.get_param("ukf_params.P0", [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]))
        self.Q: np.ndarray = np.array(self.config_manager.get_param("ukf_params.Q", [[1e-4, 0, 0], [0, 1e-4, 0], [0, 0, 1e-4]]))
        
        # R_explicit can be None or an explicit matrix from config
        self.R_explicit_config: Optional[Union[list, np.ndarray]] = self.config_manager.get_param("ukf_params.R_explicit", None)
        if self.R_explicit_config is not None:
            # Ensure R_explicit is 2D array, even if 1x1 list is provided
            if isinstance(self.R_explicit_config, list) and not isinstance(self.R_explicit_config[0], list):
                self.R_explicit = np.array([self.R_explicit_config]) # Wrap for 1D config list like [val]
            else:
                self.R_explicit = np.array(self.R_explicit_config)
        else:
            self.R_explicit = None

        self.initial_x2_guess: float = self.config_manager.get_param("ukf_params.initial_x2_guess", 0.0)
        self.initial_x3_guess: float = self.config_manager.get_param("ukf_params.initial_x3_guess", 0.0)

        self.alpha = self.config_manager.get_param("ukf_params.sigma_points.alpha", 1.0)
        self.beta = self.config_manager.get_param("ukf_params.sigma_points.beta", 2.0)
        self.kappa = self.config_manager.get_param("ukf_params.sigma_points.kappa", 0.0)
        self.points = MerweScaledSigmaPoints(n=3, alpha=self.alpha, beta=self.beta, kappa=self.kappa) # Inicializácia sigma bodov

        # These will be set during _init_ukf_filterpy
        self.hx: Optional[Callable] = None
        self.dim_z: Optional[int] = None

        print("SindyUKFEstimator initialized with parameters from configuration.")
        print(f"  dt: {self.dt}, measure_x2: {self.measure_x2}, measure_x3: {self.measure_x3}")
        print(f"  P0:\n{self.P0}\n  Q:\n{self.Q}")
        if self.R_explicit is not None:
            print(f"  R_explicit:\n{self.R_explicit}")
        else:
            print("  R will be estimated from measurement noise.")
        print(f"  Initial x3 guess: {self.initial_x3_guess}")
        print(f"  Sigma Points (alpha={self.alpha}, beta={self.beta}, kappa={self.kappa})")

    def _estimate_noise_std_dev(self, signal: np.ndarray, min_std: float = 1e-4) -> float:
        if len(signal) < 2:
            return min_std
        std_dev = float(np.std(np.diff(signal)) / np.sqrt(2))
        return max(std_dev, min_std)

    def _fx_ukf(self, x: np.ndarray, dt: float, u_val: float) -> np.ndarray:
        return rk4_integrator(x, u_val, dt, self.sindy_dxdt_func)
    
    def _init_ukf_filterpy(self, measured_x1: np.ndarray, measured_x2: Optional[np.ndarray], 
                         measured_x3: Optional[np.ndarray]) -> UnscentedKalmanFilter:
        
        actual_measure_x1 = True # Always assumed to be measured
        actual_measure_x2 = self.measure_x2 and (measured_x2 is not None)
        actual_measure_x3 = self.measure_x3 and (measured_x3 is not None)

        self.hx, self.dim_z = make_measurement_functions(
            measure_x1=actual_measure_x1,
            measure_x2=actual_measure_x2,
            measure_x3=actual_measure_x3
        )

        if self.R_explicit is None:
            noise_std_devs = []
            if actual_measure_x1:
                noise_std_devs.append(self._estimate_noise_std_dev(measured_x1, min_std=1e-4))
            if actual_measure_x2 and measured_x2 is not None:
                noise_std_devs.append(self._estimate_noise_std_dev(measured_x2, min_std=1e-4))
            if actual_measure_x3 and measured_x3 is not None:
                noise_std_devs.append(self._estimate_noise_std_dev(measured_x3, min_std=1e-4))
            
            if not noise_std_devs: 
                raise ValueError("Cannot estimate R matrix without any measurements.")

            R = np.diag([s**2 for s in noise_std_devs])
        else:
            R = self.R_explicit
            if R.shape != (self.dim_z, self.dim_z):
                raise ValueError(f"Explicit R matrix shape {R.shape} does not match expected "
                                 f"measurement dimension ({self.dim_z}, {self.dim_z}) "
                                 f"given config: measure_x2={self._config_measure_x2}, "
                                 f"measure_x3={self._config_measure_x3}, "
                                 f"actual inputs: measured_x2={'None' if measured_x2 is None else 'provided'}, "
                                 f"measured_x3={'None' if measured_x3 is None else 'provided'}.")

        x0 = np.array([
            measured_x1[0],
            measured_x2[0] if actual_measure_x2 else self.initial_x2_guess,
            measured_x3[0] if actual_measure_x3 else self.initial_x3_guess
        ])

        ukf = UnscentedKalmanFilter(dim_x=3, dim_z=self.dim_z, dt=self.dt, fx=self._fx_ukf, hx=self.hx, points=self.points)
        ukf.x = x0.astype(float)
        ukf.P = self.P0.astype(float)
        ukf.Q = self.Q.astype(float)
        ukf.R = R

        return ukf

    def estimate(self, time: np.ndarray, measured_x1: np.ndarray,
                 measured_x2: Optional[np.ndarray], measured_x3: Optional[np.ndarray],
                 control_u: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        
        N = len(time)
        if N == 0:
            raise ValueError("Input 'time' array cannot be empty.")
            
        ukf = self._init_ukf_filterpy(measured_x1, measured_x2, measured_x3)

        x_est_history = np.zeros((N, 3))

        actual_measure_x2_for_loop = self.measure_x2 and (measured_x2 is not None)
        actual_measure_x3_for_loop = self.measure_x3 and (measured_x3 is not None)

        for i in range(N):
            u_i = float(control_u[i])

            ukf.predict(u_val=u_i)

            z_list = [measured_x1[i]]
            if actual_measure_x2_for_loop:
                z_list.append(measured_x2[i])
            if actual_measure_x3_for_loop:
                z_list.append(measured_x3[i])
            z = np.array(z_list)

            ukf.update(z=z, hx=self.hx)

            x_est_history[i, :] = ukf.x.copy()

        # x1 je vzdy merany
        x_est_history[:, 0] = measured_x1

        # Prepis x2_est, ak x2 bol merany
        if actual_measure_x2_for_loop or measured_x2 is not None:
            x_est_history[:, 1] = measured_x2

        # Prepis x3_est, ak x3 bol merany
        if actual_measure_x3_for_loop or measured_x3 is not None:
            x_est_history[:, 2] = measured_x3

        # Export results to a Pandas DataFrame and then to a CSV file
        df = pd.DataFrame({
            'x1_est': x_est_history[:, 0],
            'x2_est': x_est_history[:, 1],
            'x3_est': x_est_history[:, 2],
            'u': control_u,
        })

        df.to_csv(self.data_export_path, index=False)
        print(f"Estimation complete. State estimates saved to '{self.data_export_path}'")
        return df
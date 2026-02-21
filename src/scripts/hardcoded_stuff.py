import numpy as np
import pandas as pd

from scipy.signal import savgol_filter

from data_ingestion.data_loader import DataLoader
from utils.config_manager import ConfigManager
from utils.plots import plot_pareto, plot_trajectory
from utils.helpers import compute_time_vector

def Aeroshield_load_and_deriv():
    config_manager = ConfigManager("config")

    with DataLoader(config_manager, "data_raw_dir") as loader:
        X, U, time_step = loader.load_csv_data(
            "Aeroshield",
            [0],
            None,
            0.01,
            [1],
            verbose=False
        )

        X_val, U_val, _ = loader.load_csv_data(
            "Aeroshield_val",
            [0],
            None,
            0.01,
            [1],
            verbose=False
        )

    X_dot = np.gradient(X, time_step, axis=0)
    X_dot_val = np.gradient(X_val, time_step, axis=0)

    X_new = np.vstack([X, X_val])
    X_dot_new = np.vstack([X_dot, X_dot_val])
    U_new = np.vstack([U, U_val])

    data = {
        "x": X_new.flatten(),
        "x_dot": X_dot_new.flatten(),
        "u": U_new.flatten()
    }

    file_path = "data/processed/Aeroshield_with_deriv.csv"
    df = pd.DataFrame(data)  
    df.to_csv(file_path, index=False)

def Aeroshield_plotting_pareto_from_results():
    errs = [0.15727, 0.14345, 0.15363, 0.1488, 0.14959, 0.1469, 0.14586, 0.14959]
    spars = [5, 8, 5, 6, 5, 7, 8, 5]

    plot_pareto(errs, spars)
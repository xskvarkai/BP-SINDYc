import numpy as np
import pandas as pd

from scipy.signal import savgol_filter

from data_ingestion.data_loader import DataLoader
from utils.config_manager import ConfigManager

def load_and_deriv():
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

    X = savgol_filter(X, 21, 2, axis=0) 
    X_val = savgol_filter(X_val, 21, 2, axis=0)

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

    file_path = "data/processed/Aeroshield_with_deriv_filtered.csv"
    df = pd.DataFrame(data)  
    df.to_csv(file_path, index=False)
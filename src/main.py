from main_koopman import koopman_main
from model_recostruction import sindy_model_reconstruction
from main_linear import linearized_model_main

from utils.config_manager import ConfigManager
from data_ingestion.data_loader import DataLoader
from data_processing.data_splitter import TimeSeriesSplitter
from utils.helpers import compute_time_vector
from utils.plots import plot_compared_trajectories
from sklearn.metrics import r2_score

import numpy as np

if __name__ == "__main__":
    config_manager = ConfigManager("config")
    
    koopman_model = koopman_main(config_manager)
    sindy_model = sindy_model_reconstruction(config_manager)
    linear_model = linearized_model_main(np.asarray([0.2, 0.0]), np.asarray([0.9566]))

    with DataLoader(config_manager) as loader:
        X, U, dt = loader.load_csv_data(
            file_name="Aeroshield_with_deriv",
            state_column_indices=[0, 1],
            time=0.01,
            control_input_column_indices=[2],
            verbose=False
        )

    with TimeSeriesSplitter(config_manager, X, dt, U) as splitter:
        X_train, _, X_real, U_train, _, U_real = splitter.split_data(
            train_ratio=0.5,
            val_ratio=0.25,
            apply_savgol_filter=True,
            savgol_window_length=31,
            savgol_polyorder=2,
            verbose=False
        )

    print("\nStarting simulation...")
    x_sim_sindy = sindy_model.simulate(dt, U_real, X_real[0])
    x_sim_koop = koopman_model.koopman_simulate(X_real, dt, U_real ** 2)
    x_sim_linear = linear_model.simulate(X_real[0], U_real, dt, len(X_real))


    min_len = min(len(x_sim_koop), len(x_sim_sindy), len(X_real))
    x_sim_koop = x_sim_koop[:-1]
    x_sim_sindy = x_sim_sindy[:min_len]
    x_sim_linear = x_sim_linear[:-1]
    X_real = X_real[:-1]
    U_real = U_real[:-1]

    r2_score_sindy = r2_score(X_real, x_sim_sindy)
    r2_score_koop = r2_score(X_real, x_sim_koop)
    r2_score_linear = r2_score(X_real, x_sim_linear)

    plot_compared_trajectories(
        compute_time_vector(X_real, dt),
        X_real,
        x_sim_sindy,
        r2_score_sindy,
        koopman_trajectory=x_sim_koop,
        koopman_r2=r2_score_koop,
        linearized_trajectory=x_sim_linear,
        linearized_r2=r2_score_linear,
        input_signal=U_real,
        exportable=True
    )
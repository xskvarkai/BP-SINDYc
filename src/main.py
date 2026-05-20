from main_koopman import dmd_koopman_main, neural_koopman_main
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
    
    #koopman_model = dmd_koopman_main(config_manager)
    #koopmanNeural_model = neural_koopman_main(config_manager)
    sindy_model = sindy_model_reconstruction(config_manager)
    #linear_model = linearized_model_main(np.asarray([0.2, 0.0]), np.asarray([0.9566]))

    with DataLoader(config_manager) as loader:
        X, U, dt = loader.load_csv_data(
            file_name="Floatshield_with_deriv",
            state_column_indices=[0, 1, 2, 3],
            time=0.025,
            control_input_column_indices=[4],
            verbose=False
        )

    with TimeSeriesSplitter(config_manager, X, dt, U) as splitter:
        X_train, _, X_real, U_train, _, U_real = splitter.split_data(
            train_ratio=8800,
            val_ratio=2200,
            apply_savgol_filter=True,
            filtered_set_names=["val", "test"],
            savgol_window_length=51,
            savgol_polyorder=2,
            verbose=False
        )

    print("\nStarting simulation...")
    x_sim_sindy, _, _, _ = sindy_model.simulate(dt, U_real, X_real[0])
    #x_sim_koop = koopman_model.simulate(X_real, dt, U_real ** 2)
    #x_sim_koopNeural = koopmanNeural_model.simulate(X_real, U_real)
    #x_sim_linear = linear_model.simulate(X_real[0], U_real, dt, len(X_real))

    x_sim_sindy = x_sim_sindy[:len(X_real)]

    min_len = min(len(x_sim_sindy), len(X_real))
    #x_sim_koop = x_sim_koop[:min_len]
    #x_sim_koopNeural = x_sim_koopNeural[:min_len]
    x_sim_sindy = x_sim_sindy[:min_len]
    #x_sim_linear = x_sim_linear[:min_len]
    X_real = X_real[:min_len]
    U_real = U_real[:min_len]

    r2_score_sindy = r2_score(X_real, x_sim_sindy)
    #r2_score_koop = r2_score(X_real, x_sim_koop)
    #r2_score_koopNeural = r2_score(X_real, x_sim_koopNeural)
    #r2_score_linear = r2_score(X_real, x_sim_linear)

    plot_compared_trajectories(
        compute_time_vector(X_real, dt),
        X_real,
        x_sim_sindy,
        r2_score_sindy,
        #koopman_trajectory=x_sim_koop,
        #koopman_r2=r2_score_koop,
        #koopmanNeural_trajectory=x_sim_koopNeural,
        #koopmanNeural_r2=r2_score_koopNeural,
        #linearized_trajectory=x_sim_linear,
        #linearized_r2=r2_score_linear,
        input_signal=U_real,
        exportable=True
    )
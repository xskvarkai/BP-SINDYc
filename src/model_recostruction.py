import pysindy as ps
import numpy as np
import pandas as pd

import utils.sindy_helpers as sindy_helpers
from utils.config_manager import ConfigManager
from data_ingestion.data_loader import DataLoader
from data_processing.data_splitter import TimeSeriesSplitter
from utils.custom_libraries import FixedCustomLibrary, FixedWeakPDELibrary
from utils.custom_libraries import x, sin_x, squared_x, quartered_x, \
                                   name_x, name_sin_x, name_squared_x, name_quartered_x
from utils.helpers import compute_time_vector
from utils.plots import plot_trajectory
from data_processing.sindy_preprocessor import generate_trajectories
from simulation.simulator import generate_input_signal

def sindy_model_reconstruction(config_manager: ConfigManager) -> ps.SINDy:

    config_manager.load_config("sindy_params")

    np.random.seed(100)
    random_number_generator = np.random.RandomState(100)

    with DataLoader(config_manager) as loader:
        X, U, dt = loader.load_csv_data(
            file_name="Simulacia",
            state_column_indices=[1, 2, 3],
            time_column_index=0,
            control_input_column_indices=[4],
        )

    with TimeSeriesSplitter(config_manager, X, dt, U) as splitter:
        X_train, _, X_test, U_train, _, U_test = splitter.split_data(
            train_ratio=0.6,
            val_ratio=0.2,
            perturb_input_signal_ratio=None,
            rng=random_number_generator,
            apply_savgol_filter=True,
            filtered_set_names=["val", "test"],
            savgol_window_length=31,
            savgol_polyorder=2
        )
    X_train, U_train = generate_trajectories(X_train, U_train, num_samples_per_trajectory=10000, num_trajectories=5, rng=random_number_generator)
    
    library = ps.PolynomialLibrary(include_bias=False)
    config={
        "feature_library": FixedWeakPDELibrary(H_xt=[0.9999], K=190, p=5, differentiation_method=ps.FiniteDifference(), function_library=library, spatiotemporal_grid=compute_time_vector(X_train, dt)),
        "differentiation_method": None,
        "optimizer": ps.EnsembleOptimizer(bagging=True, n_models=50, n_subset=10000, opt=ps.STLSQ(alpha=1, max_iter=100000, threshold=0.4))
    }

    random_seed=2116986363

    data = {
        "x_train": X_train,
        "x_ref": X_test,
        "u_train": U_train,
        "u_ref": U_test,
        "dt": dt
    }

    model = sindy_helpers.model_reconstruction(config, random_seed, data, False)

    return model


if __name__ == "__main__":

    config_manager = ConfigManager("config")

    sindy_model = sindy_model_reconstruction(config_manager)
    dt = 0.002

    input_signal_params = {
        "pid_kp": 1.0,
        "pid_ki": 0.0,
        "pid_kd": 0.0,
        "tau": 0.5,
        "target_max_change_interval_sec": 10,
        "target_min_change_interval_sec": 5,
        "target_clip_min": 0.2,
        "target_clip_max": 1.0,
    }
    input_signal = generate_input_signal(10000, False, dt, input_signal_params)

    x_sim = sindy_model.simulate(x0=[0, 0, 0], t=compute_time_vector(input_signal, dt), u=input_signal, integrator_kws={"rtol": 1e-6, "atol": 1e-6})
    t_sim = compute_time_vector(x_sim, dt)

    plot_trajectory(t_sim, x_sim, input_signal=input_signal[:len(x_sim)], title="Simulation forward in time")


    print(x_sim.shape)
    data = {
        "x": x_sim[:, 0].reshape(-1, 1).flatten(),
        "y": x_sim[:, 1].reshape(-1, 1).flatten(),
        "z": x_sim[:, 2].reshape(-1, 1).flatten(),
        "u": input_signal[:len(x_sim)].reshape(-1, 1).flatten(),
    }
    df = pd.DataFrame(data)  
    df.to_csv("data/processed/Koopman_Lorenz/Simulation.csv", index=False)
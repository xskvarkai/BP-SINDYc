import pysindy as ps
import numpy as np
import pandas as pd

import utils.sindy_helpers as sindy_helpers
from utils.config_manager import ConfigManager
from data_ingestion.data_loader import DataLoader
from data_processing.data_splitter import TimeSeriesSplitter
from utils.custom_libraries import FixedCustomLibrary, FixedWeakPDELibrary
from utils.custom_libraries import (
    abs_x, x_abs_x, x_y_abs_z, x_squared_abs_y, tanh_x, constant,
    x_fun, x_squared, x_cubed, x_quartered, x_frac, x_squared_frac, x_cubed_frac, x_quartered_frac,
    yx, y_squared_x, y_cubed_x, y_quartered_x,
    yx_frac, y_squared_x_frac, y_cubed_x_frac, y_quartered_x_frac,
    yx_squared_frac, y_squared_x_squared_frac, y_cubed_x_squared_frac, y_quartered_x_squared_frac,
    yx_cubed_frac, y_squared_x_cubed_frac, y_cubed_x_cubed_frac, y_quartered_x_cubed_frac,
    yx_quatered_frac, y_squared_x_quatered_frac, y_cubed_x_quatered_frac, y_quartered_x_quatered_frac,
    yxz, y_squared_xz, y_cubed_xz, y_quartered_xz,
    yx_frac_z, y_squared_x_frac_z, y_cubed_x_frac_z, y_quartered_x_frac_z,
    yx_squared_frac_z, y_squared_x_squared_frac_z, y_cubed_x_squared_frac_z, y_quartered_x_squared_frac_z,
    yx_cubed_frac_z, y_squared_x_cubed_frac_z, y_cubed_x_cubed_frac_z, y_quartered_x_cubed_frac_z,
    yx_quartered_frac_z, y_squared_x_quartered_frac_z, y_cubed_x_quartered_frac_z, y_quartered_x_quartered_frac_z,
    x_sin_2y, x_cos_2y,
    name_abs_x, name_x_abs_x, name_x_y_abs_z, name_x_squared_abs_y, name_tanh_x, name_constant,
    name_x_fun, name_x_squared, name_x_cubed, name_x_quartered, name_x_frac, name_x_squared_frac, name_x_cubed_frac, name_x_quartered_frac,
    name_yx_frac, name_y_squared_x_frac, name_y_cubed_x_frac, name_y_quartered_x_frac,
    name_yx, name_y_squared_x, name_y_cubed_x, name_y_quartered_x,
    name_yx_squared_frac, name_y_squared_x_squared_frac, name_y_cubed_x_squared_frac, name_y_quartered_x_squared_frac,
    name_yx_cubed_frac, name_y_squared_x_cubed_frac, name_y_cubed_x_cubed_frac, name_y_quartered_x_cubed_frac,
    name_yx_quartered_frac, name_y_squared_x_quartered_frac, name_y_cubed_x_quartered_frac, name_y_quartered_x_quartered_frac,
    name_yxz, name_y_squared_xz, name_y_cubed_xz, name_y_quartered_xz,
    name_yx_frac_z, name_y_squared_x_frac_z, name_y_cubed_x_frac_z, name_y_quartered_x_frac_z,
    name_yx_squared_frac_z, name_y_squared_x_squared_frac_z, name_y_cubed_x_squared_frac_z, name_y_quartered_x_squared_frac_z,
    name_yx_cubed_frac_z, name_y_squared_x_cubed_frac_z, name_y_cubed_x_cubed_frac_z, name_y_quartered_x_cubed_frac_z,
    name_yx_quartered_frac_z, name_y_squared_x_quartered_frac_z, name_y_cubed_x_quartered_frac_z, name_y_quartered_x_quartered_frac_z,
    name_x_sin_2y, name_x_cos_2y
)

from utils.helpers import compute_time_vector
from utils.plots import plot_trajectory
from data_processing.sindy_preprocessor import generate_trajectories
from simulation.simulator import generate_input_signal


def sindy_model_reconstruction(config_manager: ConfigManager) -> ps.SINDy:

    config_manager.load_config("sindy_params")

    np.random.seed(42)
    random_number_generator = np.random.RandomState(42)

    with DataLoader(config_manager) as loader:
        X, U, dt = loader.load_csv_data(
            file_name="Floatshield_with_deriv_close-loop",
            state_column_indices=[0, 1],
            time=0.025,
            control_input_column_indices=[3],
            verbose=False,
            plot_data=False
        )

    with TimeSeriesSplitter(config_manager, X, dt, U) as splitter:
        X_train, X_test, _,  U_train, U_test, _ = splitter.split_data(
            train_ratio=0.5,
            val_ratio=0.5,
            perturb_input_signal_ratio=None,
            rng=random_number_generator,
            apply_savgol_filter=True,
            filtered_set_names = ["val", "test"],
            savgol_window_length=51,
            savgol_polyorder=2,
            verbose=False
        )

    X_train, U_train = generate_trajectories(X_train, U_train, num_samples_per_trajectory=3520, num_trajectories=5, rng=random_number_generator)
    
    library = FixedCustomLibrary(
        [constant, x_fun, x_squared, x_cubed,
            yx, y_cubed_x,
            yx_frac, y_squared_x_frac,
            yx_squared_frac, y_squared_x_squared_frac, y_cubed_x_cubed_frac,
            yx_cubed_frac, y_squared_x_cubed_frac,
            yx_quatered_frac, y_squared_x_quatered_frac,
            yxz, y_squared_xz,
            yx_frac_z, y_squared_x_frac_z,
            yx_squared_frac_z, y_squared_x_squared_frac_z,
            yx_cubed_frac_z, y_squared_x_cubed_frac_z,
            yx_quartered_frac_z, y_squared_x_quartered_frac_z,
            x_sin_2y],
        [name_constant, name_x_fun, name_x_squared, name_x_cubed,
            name_yx, name_y_cubed_x,
            name_yx_frac, name_y_squared_x_frac,
            name_yx_squared_frac, name_y_squared_x_squared_frac, name_y_cubed_x_cubed_frac,
            name_yx_cubed_frac, name_y_squared_x_cubed_frac,
            name_yx_quartered_frac, name_y_squared_x_quartered_frac,
            name_yxz, name_y_squared_xz,
            name_yx_frac_z, name_y_squared_x_frac_z,
            name_yx_squared_frac_z, name_y_squared_x_squared_frac_z,
            name_yx_cubed_frac_z, name_y_squared_x_cubed_frac_z,
            name_yx_quartered_frac_z, name_y_squared_x_quartered_frac_z,
            name_x_sin_2y]
    )


    library.fit(np.hstack((X, U)))
    feature_names = library.get_feature_names()

    n_features = len(feature_names)
    n_targets = X.shape[1]

    idx_x1 = 2

    C = np.zeros((n_features, n_features * n_targets))
    d = np.zeros(n_features)

    for i in range(n_features):
        # C nastavuje koeficienty len pre PRVÚ rovnicu (stĺpce 0 až n_features-1)
        C[i, i] = 1 
        if i == idx_x1:
            d[i] = 1.0  # Chceme koeficient 1 pri x1
        else:
            d[i] = 0.0  # Všetko ostatné v prvej rovnici bude 0

    config={
            "differentiation_method": None,
            "optimizer": ps.MIOSR(alpha=0.005, constraint_lhs=C, constraint_rhs=d, group_sparsity=(1, 50), regression_timeout=30, target_sparsity=7),
            "feature_library": FixedWeakPDELibrary(H_xt=[2.5], K=50, differentiation_method=ps.FiniteDifference(), function_library=library, spatiotemporal_grid=compute_time_vector(X_train[0].shape[0], dt))
          }

    random_seed=337831792

    data = {
        "x_train": X_train,
        "x_ref": X_test,
        "u_train": U_train,
        "u_ref": U_test,
        "dt": dt
    }

    model = sindy_helpers.model_reconstruction(config, random_seed, data, True)

    return model

if __name__ == "__main__":
    config_manager = ConfigManager("config")
    sindy_model = sindy_model_reconstruction(config_manager)

    with DataLoader(config_manager) as loader:
        X, U, dt = loader.load_csv_data(
            file_name="Floatshield_3states_iter1",
            state_column_indices=[0, 1, 2],
            time=0.025,
            control_input_column_indices=[3],
            verbose=True,
            plot_data=True
        )

    U = U + U[-1]
    x_sim = sindy_model.simulate(x0=X[0], t=compute_time_vector(U, dt), u=U, integrator_kws={"rtol": 1e-6, "atol": 1e-6})
    t_sim = compute_time_vector(x_sim, dt)

    plot_trajectory(t_sim, x_sim, input_signal=U[:len(x_sim)], title="Simulation forward in time")

    data = {
        "x0": X[:, 0].flatten(),
        "x1": X[:, 1].flatten(),
        "x2": x_sim[:, 2].reshape(-1, 1).flatten(),
        "u": U[:].reshape(-1, 1).flatten(),
    }
    df = pd.DataFrame(data)  
    df.to_csv("data/processed/Koopman_Aeroshield/Simulation.csv", index=False)
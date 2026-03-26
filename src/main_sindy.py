import gc
import re
import pysindy as ps
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.config_manager import ConfigManager
from data_ingestion.data_loader import DataLoader
from data_processing.data_splitter import TimeSeriesSplitter
from data_processing.sindy_preprocessor import find_periodicity, find_noise, generate_trajectories
from models.sindy_estimator import SindyEstimator
from utils.helpers import compute_time_vector
from utils.custom_libraries import FixedCustomLibrary
from utils.custom_libraries import (
    # Základné funkcie a ich názvy
    abs_x, x_abs_x, x_y_abs_z, x_squared_abs_y, tanh_x, constant,
    name_abs_x, name_x_abs_x, name_x_y_abs_z, name_x_squared_abs_y, name_tanh_x, name_constant,

    # Funkcie s 'x' a ich názvy
    x_fun, x_squared, x_cubed, x_quartered,
    x_frac, x_squared_frac, x_cubed_frac, x_quartered_frac,
    name_x_fun, name_x_squared, name_x_cubed, name_x_quartered,
    name_x_frac, name_x_squared_frac, name_x_cubed_frac, name_x_quartered_frac,

    # Funkcie s 'y' a 'x' a ich názvy
    yx, y_squared_x, y_cubed_x, y_quartered_x,
    yx_frac, y_squared_x_frac, y_cubed_x_frac, y_quartered_x_frac,
    yx_squared_frac, y_squared_x_squared_frac, y_cubed_x_squared_frac, y_quartered_x_squared_frac,
    yx_cubed_frac, y_squared_x_cubed_frac, y_cubed_x_cubed_frac, y_quartered_x_cubed_frac,
    yx_quatered_frac, y_squared_x_quatered_frac, y_cubed_x_quatered_frac, y_quartered_x_quatered_frac,
    name_yx, name_y_squared_x, name_y_cubed_x, name_y_quartered_x,
    name_yx_frac, name_y_squared_x_frac, name_y_cubed_x_frac, name_y_quartered_x_frac,
    name_yx_squared_frac, name_y_squared_x_squared_frac, name_y_cubed_x_squared_frac, name_y_quartered_x_squared_frac,
    name_yx_cubed_frac, name_y_squared_x_cubed_frac, name_y_cubed_x_cubed_frac, name_y_quartered_x_cubed_frac,
    name_yx_quartered_frac, name_y_squared_x_quartered_frac, name_y_cubed_x_quartered_frac, name_y_quartered_x_quartered_frac,

    # Funkcie s 'x', 'y' a 'z' a ich názvy
    yxz, yxz_z_squared, yxz_z_cubed, yxz_z_quartered,
    yx_frac_z, yx_frac_z_squared, yx_frac_z_cubed, yx_frac_z_quartered,
    yx_squared_frac_z, yx_squared_frac_z_squared, yx_squared_frac_z_cubed, yx_squared_frac_z_quartered,
    yx_cubed_frac_z, yx_cubed_frac_z_squared, yx_cubed_frac_z_cubed, yx_cubed_frac_z_quartered,
    yx_quartered_frac_z, yx_quartered_frac_z_squared, yx_quartered_frac_z_cubed, yx_quartered_frac_z_quartered,
    name_yxz, name_yxz_z_squared, name_yxz_z_cubed, name_yxz_z_quartered,
    name_yx_frac_z, name_yx_frac_z_squared, name_yx_frac_z_cubed, name_yx_frac_z_quartered,
    name_yx_squared_frac_z, name_yx_squared_frac_z_squared, name_yx_squared_frac_z_cubed, name_yx_squared_frac_z_quartered,
    name_yx_cubed_frac_z, name_yx_cubed_frac_z_squared, name_yx_cubed_frac_z_cubed, name_yx_cubed_frac_z_quartered,
    name_yx_quartered_frac_z, name_yx_quartered_frac_z_squared, name_yx_quartered_frac_z_cubed, name_yx_quartered_frac_z_quartered,

    # Nové funkcie s 'y' a 'mx_drag_term' a ich názvy
    y_mx_drag_term, y_squared_mx_drag_term, y_cubed_mx_drag_term, y_quartered_mx_drag_term,
    name_y_mx_drag_term, name_y_squared_mx_drag_term, name_y_cubed_mx_drag_term, name_y_quartered_mx_drag_term,

    # Nové funkcie s 'z', 'x_frac' a 'my_drag_term' a ich názvy
    zx_frac_my_drag_term, zx_frac_squared_my_drag_term, zx_frac_cubed_my_drag_term, zx_frac_quartered_my_drag_term,
    name_zx_frac_my_drag_term, name_zx_frac_squared_my_drag_term, name_zx_frac_cubed_my_drag_term, name_zx_frac_quartered_my_drag_term,

    # Nové funkcie s 'z_squared', 'x_frac' a 'my_drag_term' a ich názvy
    z_squared_x_frac_my_drag_term, z_squared_x_frac_squared_my_drag_term, z_squared_x_frac_cubed_my_drag_term, z_squared_x_frac_quartered_my_drag_term,
    name_z_squared_x_frac_my_drag_term, name_z_squared_x_frac_squared_my_drag_term, name_z_squared_x_frac_cubed_my_drag_term, name_z_squared_x_frac_quartered_my_drag_term,

    # Nové funkcie s 'z_cubed', 'x_frac' a 'my_drag_term' a ich názvy
    z_cubed_x_frac_my_drag_term, z_cubed_x_frac_squared_my_drag_term, z_cubed_x_frac_cubed_my_drag_term, z_cubed_x_frac_quartered_my_drag_term,
    name_z_cubed_x_frac_my_drag_term, name_z_cubed_x_frac_squared_my_drag_term, name_z_cubed_x_frac_cubed_my_drag_term, name_z_cubed_x_frac_quartered_my_drag_term,

    # Nové funkcie s 'z_quartered', 'x_frac' a 'my_drag_term' a ich názvy
    z_quartered_x_frac_my_drag_term, z_quartered_x_frac_squared_my_drag_term, z_quartered_x_frac_cubed_my_drag_term, z_quartered_x_frac_quartered_my_drag_term,
    name_z_quartered_x_frac_my_drag_term, name_z_quartered_x_frac_squared_my_drag_term, name_z_quartered_x_frac_cubed_my_drag_term, name_z_quartered_x_frac_quartered_my_drag_term,
)

def sindy_main(config_manager: ConfigManager):

    config_manager.load_config("sindy_params")

    np.random.seed(config_manager.get_param("sindy_params.global.random_seed", 42))
    random_number_generator = np.random.RandomState(config_manager.get_param("sindy_params.global.random_seed", 42))

    with DataLoader(config_manager) as loader:
        X, U, dt = loader.load_csv_data(
            **config_manager.get_param("sindy_params.data_loading")
        )

    with TimeSeriesSplitter(config_manager, X, dt, U) as splitter:
        X_train, X_val, X_test, U_train, U_val, U_test = splitter.split_data(
            **config_manager.get_param("sindy_params.data_splitting"), rng=random_number_generator
        )

    with SindyEstimator(config_manager) as estimator:
        noise_level = find_noise(X)
        find_periodicity(X, dt, None, sigma_noise=noise_level)

        config_manager.get_param(
            "sindy_params.data_preprocessing"
        )["num_samples_per_trajectory"] = int(config_manager.get_param("sindy_params.data_preprocessing.num_samples_per_trajectory") * X_train.shape[0])

        # Used for generating sub-trajectories
        X_train, U_train = generate_trajectories(X_train, U_train, **config_manager.get_param("sindy_params.data_preprocessing"), rng=random_number_generator)

        # ===== Sindy model configuration =====
        # All of the configurations for the feature libraries, differentiation methods and optimizers are defined here.
        # You can modify the parameters and add more configurations as needed.
        # The keys of the dictionaries correspond to the names of the methods, and the values are dictionaries of parameters for those methods.
        # Minimum required parameters for method are provided (None takes defaults), but you can add more parameters.

        library = ps.PolynomialLibrary(degree=1, include_bias=True) + FixedCustomLibrary(
            [
             yx_frac, y_squared_x_frac, y_cubed_x_frac, y_quartered_x_frac,
             yx_squared_frac, y_squared_x_squared_frac, y_cubed_x_squared_frac, y_quartered_x_squared_frac,
             yx_cubed_frac, y_squared_x_cubed_frac, y_cubed_x_cubed_frac, y_quartered_x_cubed_frac,
             yx_quatered_frac, y_squared_x_quatered_frac, y_cubed_x_quatered_frac, y_quartered_x_quatered_frac,
             
             yx_frac_z, yx_squared_frac_z, yx_cubed_frac_z, yx_quartered_frac_z,
             yx_frac_z_squared, yx_squared_frac_z_squared, yx_cubed_frac_z_squared, yx_quartered_frac_z_squared,
             yx_frac_z_cubed, yx_squared_frac_z_cubed, yx_cubed_frac_z_cubed, yx_quartered_frac_z_cubed,
             yx_frac_z_quartered, yx_squared_frac_z_quartered, yx_cubed_frac_z_quartered, yx_quartered_frac_z_quartered,
            ],
            [             
             name_yx_frac, name_y_squared_x_frac, name_y_cubed_x_frac, name_y_quartered_x_frac,
             name_yx_squared_frac, name_y_squared_x_squared_frac, name_y_cubed_x_squared_frac, name_y_quartered_x_squared_frac,
             name_yx_cubed_frac, name_y_squared_x_cubed_frac, name_y_cubed_x_cubed_frac, name_y_quartered_x_cubed_frac,
             name_yx_quartered_frac, name_y_squared_x_quartered_frac, name_y_cubed_x_quartered_frac, name_y_quartered_x_quartered_frac,
             
             name_yx_frac_z, name_yx_squared_frac_z, name_yx_cubed_frac_z, name_yx_quartered_frac_z,
             name_yx_frac_z_squared, name_yx_squared_frac_z_squared, name_yx_cubed_frac_z_squared, name_yx_quartered_frac_z_squared,
             name_yx_frac_z_cubed, name_yx_squared_frac_z_cubed, name_yx_cubed_frac_z_cubed, name_yx_quartered_frac_z_cubed,
             name_yx_frac_z_quartered, name_yx_squared_frac_z_quartered, name_yx_cubed_frac_z_quartered, name_yx_quartered_frac_z_quartered,
            ]
        )

        feature_library_kwargs = {
            "WeakPDELibrary": {
                "function_library": library,
                "K": [100],
                "p": [4, 5],
                "spatiotemporal_grid": compute_time_vector(X_train[0].shape[0], dt),
                "H_xt": [[0.5]]
            }
        }

        differentiation_method_kwargs = None

        library.fit(np.hstack((X, U)))
        feature_names = library.get_feature_names()

        #print(feature_names)
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

        optimizer_kwargs = {
            "MIOSR": {
                "target_sparsity": [7, 8],
                "group_sparsity": (1, 50),
                "alpha": [0.01],
                "normalize_columns": False,
                "verbose": False,
                "constraint_lhs": C,
                "constraint_rhs": d,
            }
        }

        # ===== End of Sindy model configuration =====

        estimator.make_grid(feature_library_kwargs, differentiation_method_kwargs, optimizer_kwargs)

        X, U = None, None
        gc.collect()

        estimator.search_configurations(
            X_train, X_val, U_train, U_val, dt,
            config_manager.get_param("sindy_params.params_search.n_processes"),
            config_manager.get_param("sindy_params.params_search.log_file_name"),
            timeout_per_config=config_manager.get_param("sindy_params.params_search.timeout_per_config"),
            **config_manager.get_param("sindy_params.constraints")
        )
        
        #estimator.plot_pareto()
        #estimator.validate_on_test(X_train, X_test, U_train, U_test, dt, **config_manager.get_param("sindy_params.constraints"))

        try:
            raw_libraries = library.libraries
        except:
            raw_libraries = library.library_functions

        payload = {
            "global_random_seed": config_manager.get_param("sindy_params.global.random_seed"),
            "dt": dt,
            "dataset_size_ratio": {
                "train": config_manager.get_param("sindy_params.data_splitting.train_ratio"),
                "test": config_manager.get_param("sindy_params.data_splitting.val_ratio"),
                "val": 1 - config_manager.get_param("sindy_params.data_splitting.train_ratio") - config_manager.get_param("sindy_params.data_splitting.val_ratio") 
            },
            "perturb_input_signal_ratio": config_manager.get_param("sindy_params.data_splitting.perturb_input_signal_ratio"),
            "multiple_trajectories": config_manager.get_param("sindy_params.data_preprocessing"),
            "signal_loading_prefiltering": {
                "savgol_window_length": config_manager.get_param("sindy_params.data_loading.savgol_window_length"),
                "savgol_polyorder": config_manager.get_param("sindy_params.data_loading.savgol_polyorder")
            } if config_manager.get_param("sindy_params.data_loading.apply_savgol_filter") else "non-filtered",
            "signal_splitting_prefiltering": {
                "savgol_window_length": config_manager.get_param("sindy_params.data_splitting.savgol_window_length"),
                "savgol_polyorder": config_manager.get_param("sindy_params.data_splitting.savgol_polyorder"),
                "filtered_set_names": config_manager.get_param("sindy_params.data_splitting.filtered_set_names")
            } if config_manager.get_param("sindy_params.data_splitting.apply_savgol_filter") else "non-filtered",
            "constraints": config_manager.get_param("sindy_params.constraints"),
            "library": raw_libraries
        }

        estimator.export_data(
            payload,
            config_manager.get_param("sindy_params.params_search.export_file_name")
        )

if __name__ == "__main__":
    config_manager = ConfigManager("config")
    sindy_main(config_manager)
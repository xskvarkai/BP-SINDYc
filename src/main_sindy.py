import gc
import pysindy as ps
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter

from utils.config_manager import ConfigManager
from data_ingestion.data_loader import DataLoader
from data_processing.data_splitter import TimeSeriesSplitter
from data_processing.sindy_preprocessor import find_periodicity, find_noise, generate_trajectories, find_optimal_delay
from models.sindy_estimator import SindyEstimator
from utils.helpers import compute_time_vector
from utils.custom_libraries import FixedCustomLibrary
from utils.custom_libraries import (
    
    # Polynom
    x_cubed, x_quartered,
    name_x_cubed, name_x_quartered,
    x_squared_y,
    name_x_squared_y,

    #Racionalne
    yx_frac, y_squared_x_frac,
    name_yx_frac, name_y_squared_x_frac,
 
    yx_squared_frac, y_squared_x_squared_frac,
    name_yx_squared_frac, name_y_squared_x_squared_frac,

    # Absolutna hodnota
    x_abs_x, x_cubed_abs_x, y_abs_x,
    name_x_abs_x, name_x_cubed_abs_x, name_y_abs_x,
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
        find_periodicity(X, dt, 1, sigma_noise=noise_level)

        config_manager.get_param(
            "sindy_params.data_preprocessing"
        )["num_samples_per_trajectory"] = int(config_manager.get_param("sindy_params.data_preprocessing.num_samples_per_trajectory") * X_train.shape[0])

        X_original, U_original = X_train.copy(), U_train.copy()

        # Used for generating sub-trajectories
        X_train, U_train = generate_trajectories(X_train, U_train, **config_manager.get_param("sindy_params.data_preprocessing"), rng=random_number_generator)

        # ===== Sindy model configuration =====
        # All of the configurations for the feature libraries, differentiation methods and optimizers are defined here.
        # You can modify the parameters and add more configurations as needed.
        # The keys of the dictionaries correspond to the names of the methods, and the values are dictionaries of parameters for those methods.
        # Minimum required parameters for method are provided (None takes defaults), but you can add more parameters.

        library = ps.PolynomialLibrary(degree=3, include_bias=True, include_interaction=True) + FixedCustomLibrary(
            [
             #x_cubed, #x_quartered,

             x_squared_y,

             #yx_frac, y_squared_x_frac,
             #yx_squared_frac, y_squared_x_squared_frac,

             x_abs_x, #x_cubed_abs_x,
             #y_mx_drag_term,
             #vanDerPol,
             #some_part,

            ],
            [
             #name_x_cubed, #name_x_quartered,
             
             name_x_squared_y,

             #name_yx_frac, name_y_squared_x_frac,
             #name_yx_squared_frac, name_y_squared_x_squared_frac,

             name_x_abs_x, #name_x_cubed_abs_x,
             #name_y_mx_drag_term, 
             #name_vanDerPol
             #name_some_part,

            ]
        )

        find_optimal_delay(X_original, dt, U_original, library, True, False)

        feature_library_kwargs = {
                "WeakPDELibrary": {
                "function_library": library,
                "derivative_order": 0,
                "K": [5, 10, 20, 30, 40, 50, 70, 100, 150, 200],
                "p": [4, 5, 6],
                "spatiotemporal_grid": compute_time_vector(X_train[0].shape[0], dt),
                "H_xt": [[0.5], [0.75], [1.], [1.25], [1.5], [1.75], [2.0]]
            }
        }

        differentiation_method_kwargs = None

        library.fit(np.hstack((X, U)))
        feature_names = library.get_feature_names()

        #print(feature_names)

        n_features = len(feature_names)
        n_targets = X.shape[1]

        idx_const = 0
        idx_x0 = 1
        idx_x1 = 2
        idx_x2 = 3
        idx_x3 = 4
        idx_x4 = 5

        idx_u0_squared = 20 
        idx_x2_squared_x3 = 28
        idx_x2_squared_u0 = 29
# Máme presne 2 požiadavky (2 obmedzenia), preto 2 riadky
        n_constraints = 2

        C = np.zeros((n_constraints, n_features * n_targets))
        d = np.zeros(n_constraints)  # Dôležité: d musí byť 1D vektor o dĺžke 2!

# 1. obmedzenie: Pre target 0 chcem, aby koeficient na pozícii idx_x1 bol 1.0
        C[0, idx_x1] = 1.0
        d[0] = 1.0

# 2. obmedzenie: Pre target 1 chcem, aby koeficient na pozícii idx_x2 bol 1.0
# Keďže ide o target 1, musíme index posunúť o n_features!
        #C[0, n_features + idx_x2] = 1.0
        #d[0] = 1.0

        C[1, 2 * n_features + idx_x3] = 1.0
        d[1] = 1.0

        initial_guess = np.zeros((n_targets, n_features))
        initial_guess[0, idx_x1] =  1.0
        
        initial_guess[1, idx_const] = -34.0
        
        initial_guess[2, idx_x3] = 1.0

        initial_guess[3, idx_x1] = 15.44
        initial_guess[3, idx_u0_squared] = 0.53
        initial_guess[3, idx_x2_squared_x3] = -16.55
        initial_guess[3, idx_x2_squared_u0] = -0.49

        optimizer_kwargs = {
            "MIOSR": {
                "regression_timeout": 90,
                "target_sparsity": 2 * n_features + 2,
                "group_sparsity": [
                                   (1, 3, 1, 4),
                                   (1, 4, 1, 4), #(1, 8, 1, 3), (1, 9, 1, 3), (1, 10, 1, 3),
                                   (1, 5, 1, 4), #(1, 8, 1, 4), (1, 9, 1, 4), (1, 10, 1, 4),
                                  ],
                "alpha": [5e-6, 5e-7, 5e-8, 5e-9, 5e-10, 5e-11],
                "normalize_columns": False,
                "verbose": False,
                "constraint_lhs": C,
                "constraint_rhs": d,
                "initial_guess": initial_guess,
            }
        }

        # ===== End of Sindy model configuration =====

        estimator.make_grid(feature_library_kwargs, differentiation_method_kwargs, optimizer_kwargs)
        
        X, U = None, None
        gc.collect()

        estimator.search_configurations(
            X_train, X_val, U_train, U_val, dt, 
            config_manager.get_param("sindy_params.params_search.is_discrete"),
            config_manager.get_param("sindy_params.params_search.n_processes"),
            config_manager.get_param("sindy_params.params_search.log_file_name"),
            timeout_per_config=config_manager.get_param("sindy_params.params_search.timeout_per_config"),
            **config_manager.get_param("sindy_params.constraints")
        )
        
        #estimator.plot_pareto()

        estimator.validate_on_test(X_train, X_test, U_train, U_test, dt, config_manager.get_param("sindy_params.params_search.is_discrete"), **config_manager.get_param("sindy_params.constraints"))

        try:
            libraries = library.libraries
            #libraries = library.library_functions
        except:
            libraries = library

        payload = {
            "global_random_seed": config_manager.get_param("sindy_params.global.random_seed", 42),
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
            "library": libraries
        }

        estimator.export_data(
            payload,
            config_manager.get_param("sindy_params.params_search.export_file_name")
        )

if __name__ == "__main__":
    config_manager = ConfigManager("config")
    sindy_main(config_manager)

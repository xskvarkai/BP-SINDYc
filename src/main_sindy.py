import gc
import pysindy as ps
import numpy as np

from utils.config_manager import ConfigManager
from data_ingestion.data_loader import DataLoader
from data_processing.data_splitter import TimeSeriesSplitter
from data_processing.sindy_preprocessor import find_periodicity, find_noise, generate_trajectories, find_optimal_delay
from models.sindy_estimator import SindyEstimator
from utils.helpers import compute_time_vector
from utils.plots import plot_trajectory

ps.PolynomialLibrary()

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
        find_optimal_delay(X_train, dt, U_train)

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

        library = ps.PolynomialLibrary(degree=2, include_bias=True)

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

        # Add constraint
        idx_x1 = 2

        C = np.zeros((n_features, n_features * n_targets))
        d = np.zeros(n_features)

        for i in range(n_features):
            C[i, i] = 1 
            if i == idx_x1:
                d[i] = 1.0
            else:
                d[i] = 0.0

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
        
        estimator.plot_pareto()
        estimator.validate_on_test(X_train, X_test, U_train, U_test, dt, **config_manager.get_param("sindy_params.constraints"))

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
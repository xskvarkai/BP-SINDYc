import gc
import pysindy as ps

from utils.config_manager import ConfigManager
from data_ingestion.data_loader import DataLoader
from data_processing.data_splitter import TimeSeriesSplitter
from data_processing.sindy_preprocessor import find_periodicity, find_noise, estimate_threshold, generate_trajectories
from models.sindy_estimator import SINDYcEstimator
from utils.helpers import compute_time_vector
from utils.custom_libraries import FixedCustomLibrary
from utils.custom_libraries import x, sin_x, squared_x, cubed_x, quartered_x, x_sin_y, \
                                   name_x, name_sin_x, name_squared_x, name_cubed_x, name_quartered_x, name_x_sin_y

if __name__ == "__main__":

    config_manager = ConfigManager("config")
    config_manager.load_config("model_params")
    config_manager.load_config("data_config")

    with DataLoader(config_manager) as loader:
        X, U, dt = loader.load_csv_data(
            **config_manager.get_param("data_config.data_loading")
        )

    with TimeSeriesSplitter(config_manager, X, dt, U) as splitter:
        X_train, X_val, X_test, U_train, U_val, U_test = splitter.split_data(
            **config_manager.get_param("data_config.data_splitting")
        )

    with SINDYcEstimator(config_manager) as estimator:
        noise_level = find_noise(X)
        find_periodicity(X, 1, noise_level)
        
        X_train, U_train = generate_trajectories(X_train, U_train, int(0.6 * X_train.shape[0]))

        library = FixedCustomLibrary(
                [x, sin_x, squared_x, quartered_x],
                [name_x, name_sin_x, name_squared_x, name_quartered_x],
                include_bias=False
            )
        
        feature_library_kwargs = {
            "WeakPDELibrary": {
                "function_library": library,
                "spatiotemporal_grid": compute_time_vector(X, dt),
                "derivative_order": [1, 2, 3],
                "K": [10, 50, 100, 200],
                "H_xt": [[1.0 * dt * 10], [1.5 * dt * 10], [2.0 * dt * 10]],
                "p": [4, 5, 6]
            }
        }

        optimizer_kwargs = {
            "STLSQ": {
                "threshold": estimate_threshold(X, dt, U, library, 6, noise_level)[0: 4],
                "ensemble": True,
                "ensemble_kwargs": {"n_subset": 0.6 * X_train[0].shape[0]},
                "alpha": [1e-4, 1e-3, 1e-2, 1e-1]
            }
        }

        estimator.make_grid(
            "WeakPDELibrary",
            "FiniteDifference",
            "STLSQ",
            feature_library_kwargs,
            None,
            optimizer_kwargs
        )

        X, U = None, None
        gc.collect()

        estimator.search_configurations(X_train, X_val, U_train, U_val, dt, 6, "Aeroshield/worker_results",  **config_manager.get_param("model_params.constraints"))
        estimator.plot_pareto()
        estimator.validate_on_test(X_train, X_test, U_train, U_test, dt, **config_manager.get_param("model_params.constraints"))
        estimator.export_data(config_manager.get_param("model_params.constraints"), "Aeroshield/Aeroshield")

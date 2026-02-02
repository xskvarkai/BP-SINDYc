import pysindy as ps
import gc
import numpy as np

from utils.helpers import generate_trajectories, find_noise, find_periodicity, estimate_threshold, compute_time_vector
from utils.custom_libraries import x, squared_x, cubed_x, tanh_x, sin_x, sign_x, x_abs_x,\
                                   name_x, name_squared_x, name_cubed_x, name_tanh_x, name_sin_x, name_sign_x, name_x_abs_x   
from utils.custom_libraries import FixedCustomLibrary
from utils import constants
from data_processing.data_loader import load_data, load_config
from data_processing.data_splitter import split_data
import models.sindy_model as sindy

if __name__ == "__main__":
    libraries = [
        FixedCustomLibrary([x, sin_x, squared_x, tanh_x, sign_x, x_abs_x], [name_x, name_sin_x, name_squared_x, name_tanh_x, name_sign_x, name_x_abs_x], include_bias=False),
        FixedCustomLibrary([x, sin_x, squared_x, tanh_x, sing_x], [name_x, name_sin_x, name_squared_x, name_tanh_x, name_sing_x], include_bias=False),
        FixedCustomLibrary([x, sin_x, squared_x, tanh_x, x_abs_x], [name_x, name_sin_x, name_squared_x, name_tanh_x, name_x_abs_x], include_bias=False),
        FixedCustomLibrary([x, sin_x, squared_x, tanh_x], [name_x, name_sin_x, name_squared_x, name_tanh_x], include_bias=False),
    ]

    for idx, library in enumerate(libraries):
        sindy_config = load_config("sindy_params")
        val_size, test_size = sindy_config.get("val_size"), sindy_config.get("test_size")
        constraints = sindy_config.get("constraints")
        random_seed = sindy_config.get("random_seed", constants.DEFAULT_RANDOM_SEED)
        num_trajectories = sindy_config.get("num_trajectories", 1)
        processors = sindy_config.get("processors", 4)
        log_file_name = sindy_config.get("log_file_name", "worker_results")
        output_file_name = sindy_config.get("output_file_name", "data")

        X, U, time_step = load_data(
            sindy_config.get("data_file"),
            sindy_config.get("time_step"),
            sindy_config.get("include_control_input", True),
            sindy_config.get("column_indices"),
            perturb_input_signal=sindy_config.get("perturb_input_signal", False),
            plot=False
        )

        X_train, X_val, X_test, U_train, U_val, U_test = split_data(X, U, val_size, test_size, time_step, sindy_config.get("filter_data", False))

        noise_level = find_noise(X_train)
        find_periodicity(X_train)

        function_library = library
        output_file_name += "_lib" + str(idx + 1)

        Ks = [10, 50, 100, 200]
        pses = [4, 5, 6, 7]
        derivative_orders = [1, 2, 3, 4]
        H_xts = [[1.25 * np.sqrt(time_step)], [1.5 * np.sqrt(time_step)], [1.75 * np.sqrt(time_step)], [2 * np.sqrt(time_step)]]
        time_vec = compute_time_vector(X_train, time_step)

        # Tvorba viacerych trajektorii
        num_samples = int(X_val.shape[0]) if X_val.shape[0] <= X_train.shape[0] else int(X_train.shape[0])
        X_train, U_train = generate_trajectories(X_train, U_train, num_samples=num_samples, num_trajectories=num_trajectories, randomseed=random_seed)

        # Vytvorenie gridu
        with sindy.SINDYcEstimator() as estimator:
            for K in Ks:
                for p in pses:
                    for derivative_order in derivative_orders:
                        for H_xt in H_xts:
                            estimator.set_feature_library("WeakPDELibrary", function_library=function_library, spatiotemporal_grid=time_vec, K=K, derivative_order=derivative_order, p=p, H_xt=H_xt)

            estimator.set_differentiation_method("FiniteDifference")

            alphas = [0.001, 0.01, 0.1]
            
            thresholds, _ = estimate_threshold(X, time_step, U, function_library, noise_level)
            for threshold in thresholds:
                for alpha in alphas:
                    estimator.set_optimizer("STLSQ", ensemble=True, ensemble_kwargs={"n_subset": num_samples * constants.ENSEMBLE_N_SUBSET}, threshold=threshold, alpha=alpha)

            X, U = None, None

            # Vytvorenie a hladanie konfiguracie
            sindy_config = None
            gc.collect()

            estimator.generate_configurations()
            estimator.search_configurations(X_train, X_val, U_train, U_val, time_step, processors, log_file_name + str(idx + 1), **constraints)
            estimator.plot_pareto()
            estimator.validate_on_test(X_train, X_test, U_train, U_test, time_step, **constraints)

            data = {
                "np.random.seed": random_seed,
                "dt": time_step,
                "val_size": val_size,
                "test_size": test_size,
                "train_multi_trajectories_num": num_trajectories,
                "train_multi_trajectories_samples": num_samples,
                "constraints": constraints
            }

            estimator.export_data(data, output_file_name)

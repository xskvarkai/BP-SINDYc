from utils.helpers import generate_trajectories, find_noise, find_periodicity, estimate_threshold
from data_processing.data_loader import load_data
from data_processing.data_splitter import split_data

import models.sindy_model as sindy_model

import numpy as np
import pysindy as ps

if __name__ == "__main__":
    X, U, time_step = load_data("Simulacia", column_indices=(1, 2, 3, 4))
    random_seed = 42 
    val_size, test_size = 0.2, 0.2
    noise_level = find_noise(X)
    find_periodicity(X)

    X_train, X_val, X_test, U_train, U_val, U_test = split_data(X, U, val_size=val_size, test_size=test_size)

    # Tvorba viacerych trajektorii
    num_samples = int(X.shape[0] * val_size)
    num_trajectories = 5
    X_train, U_train = generate_trajectories(X_train, U_train, num_samples=num_samples, num_trajectories=num_trajectories, randomseed=random_seed)

    # Obmedzenia kladene na model
    constraints = {
        "sim_steps": 300,
        "coeff_precision": None,
        "max_complexity": 24,
        "max_coeff": 1e2,
        "min_r2": 0.7,
        "max_state": 1e2
    }

    # Vytvorenie gridu
    with sindy_model.SINDYcEstimator() as estimator:

        library = ps.PolynomialLibrary(degree=2, include_bias=False)
        time_vec = (np.arange(X.shape[0]) * time_step)
        Ks = [170, 200, 230]
        pses = [3, 4, 5]
        derivative_orders = [0, 1]

        for K in Ks:
            for p in pses:
                for derivative_order in derivative_orders:
                    estimator.set_feature_library("WeakPDELibrary", function_library=library, spatiotemporal_grid=time_vec, K=K, derivative_order=derivative_order, p=p)

        estimator.set_differentiation_method("FiniteDifference")

        thresholds = estimate_threshold(X, time_step, U, library, noise_level)
        alphas = [0.001, 0.01, 0.1]

        for threshold in thresholds:
            for alpha in alphas:
                estimator.set_optimizer("STLSQ", threshold=threshold, alpha=alpha, normalize_columns=False)

                
        X, U = None, None

        # Vytvorenie a hladanie konfiguracie
        estimator.generate_configurations()
        estimator.search_configurations(X_train, X_val, U_train, U_val, dt=time_step, n_processes=8, **constraints)
        estimator.plot_pareto()

        data = {
            "np.random.seed": random_seed,
            "dt": time_step,
            "val_size": val_size,
            "test_size": test_size,
            "train_multi_trajectories_num": num_trajectories,
            "train_multi_trajectories_samples": num_samples,
            "constraints": constraints
        }

        estimator.validate_on_test(X_train, X_test, U_train, U_test, dt=time_step)
        estimator.export_data(data, "data.json")
import numpy as np
import pysindy as ps

from utils.helpers import generate_trajectories, find_noise, find_periodicity, estimate_threshold
from utils import constants
from data_processing.data_loader import load_data, load_config
from data_processing.data_splitter import split_data
import models.sindy_model as sindy

if __name__ == "__main__":
    sindy_config = load_config("sindy_params")
    val_size, test_size = sindy_config.get("val_size"), sindy_config.get("test_size")
    constraints = sindy_config.get("constraints")
    random_seed = sindy_config.get("random_seed", constants.DEFAULT_RANDOM_SEED)
    num_trajectories = sindy_config.get("num_trajectories", 1)
    
    X, U, time_step = load_data("Simulacia", column_indices=(1, 2, 3, 4))

    library = ps.PolynomialLibrary(degree=2, include_bias=False)
    time_vec = (np.arange(X.shape[0]) * time_step)
    Ks = [170, 200]
    pses = [4, 5]
    derivative_orders = [0, 1]
    alphas = [0.001, 0.01, 0.1, 1]

    sindy_config = None
    noise_level = find_noise(X)
    find_periodicity(X)

    X_train, X_val, X_test, U_train, U_val, U_test = split_data(X, U, val_size, test_size, True)

    # Tvorba viacerych trajektorii
    num_samples = int(X.shape[0] * val_size)
    X_train, U_train = generate_trajectories(X_train, U_train, num_samples=num_samples, num_trajectories=num_trajectories, randomseed=random_seed)

    # Vytvorenie gridu
    with sindy.SINDYcEstimator() as estimator:
        for K in Ks:
            for p in pses:
                for derivative_order in derivative_orders:
                    estimator.set_feature_library("WeakPDELibrary", function_library=library, spatiotemporal_grid=time_vec, K=K, derivative_order=derivative_order, p=p)

        estimator.set_differentiation_method("FiniteDifference")

        thresholds = estimate_threshold(X, time_step, U, library, noise_level)
        for threshold in thresholds:
            for alpha in alphas:
                estimator.set_optimizer("STLSQ", ensemble=True, ensemble_kwargs={"n_subset": num_samples * 0.6}, threshold=threshold, alpha=alpha, unbias=True, normalize_columns=False)

        X, U = None, None

        # Vytvorenie a hladanie konfiguracie
        estimator.generate_configurations()
        estimator.search_configurations(X_train, X_val, U_train, U_val, dt=time_step, n_processes=8, **constraints)
        estimator.plot_pareto()
        estimator.validate_on_test(X_train, X_test, U_train, U_test, dt=time_step, **constraints)

        data = {
            "np.random.seed": random_seed,
            "dt": time_step,
            "val_size": val_size,
            "test_size": test_size,
            "train_multi_trajectories_num": num_trajectories,
            "train_multi_trajectories_samples": num_samples,
            "constraints": constraints
        }

        estimator.export_data(data, "data")
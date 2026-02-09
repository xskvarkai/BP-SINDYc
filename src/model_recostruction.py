import pysindy as ps
import numpy as np

import utils.sindy_helpers as sindy_helpers
from utils.config_manager import ConfigManager
from data_ingestion.data_loader import DataLoader
from data_processing.data_splitter import TimeSeriesSplitter
from utils.custom_libraries import FixedCustomLibrary, FixedWeakPDELibrary
from utils.custom_libraries import x, sin_x, squared_x, \
                                   name_x, name_sin_x, name_squared_x
from utils.helpers import compute_time_vector
from utils.plots import plot_trajectory
from data_processing.sindy_preprocessor import generate_trajectories

if __name__ == "__main__":
    config_manager = ConfigManager("config")
    config_manager.load_config("sindy_params")

    np.random.seed(100)
    random_number_generator = np.random.RandomState(100)

    with DataLoader(config_manager) as loader:
        X, U, dt = loader.load_csv_data(
            **config_manager.get_param("sindy_params.data_loading")
        )

    with TimeSeriesSplitter(config_manager, X, dt, U) as splitter:
        X_train, X_val, X_test, U_train, U_val, U_test = splitter.split_data(
            **config_manager.get_param("sindy_params.data_splitting"), rng=random_number_generator
        )
    X_train, U_train = generate_trajectories(X_train, U_train, num_samples_per_trajectory=2500, num_trajectories=5, rng=random_number_generator)
    
    library = FixedCustomLibrary(function_names=[name_x, name_sin_x , name_squared_x], library_functions=[x, sin_x, squared_x])
    data = {
        "x_train": X_train,
        "x_ref": X_test,
        "u_train": U_train,
        "u_ref": U_test,
        "dt": dt
    }

    model = sindy_helpers.model_costruction(
        config={
            "feature_library": FixedWeakPDELibrary(H_xt=[0.15], K=10, derivative_order=1, differentiation_method=ps.FiniteDifference(), function_library=library, spatiotemporal_grid=compute_time_vector(X_train, dt)),
            "differentiation_method": None,
            "optimizer": ps.EnsembleOptimizer(bagging=True, n_models=50, n_subset=1500, opt=ps.STLSQ(alpha=0.01, max_iter=100000, normalize_columns=True, threshold=0.8)),
            "random_seed": 224
        },
        data=data
    )

    print("\nStarting validation on test data...")
    x_sim, rmse, r2, aic = sindy_helpers.evaluate_model(
        model,
        data,
        start_index=0,
        current_steps=2500,
        integrator_kwargs={"rtol": 1e-6,"atol": 1e-6}
    )
    
    min_len = min(len(X_test), len(x_sim))        

    print(f"Recostructed model R2 score: {r2:.3%}")

    t_test = np.arange(min_len) * dt
    x_ref_cut = X_test[:min_len]
    x_sim_cut = x_sim[:min_len]
    plot_trajectory(t_test, x_ref_cut, x_sim_cut, title="Validation on test data")
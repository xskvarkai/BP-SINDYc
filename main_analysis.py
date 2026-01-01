import numpy as np
import pandas as pd
import pysindy as ps
import data_analysis_lib as lib

def SINDYc():
    # Hlavne data
    data_csv = pd.read_csv("Simulacia.csv")
    data = data_csv.to_numpy()
    time_step = np.round(np.median(np.diff(data[:, 0])), decimals=5)
    print(f"\nEstimated time step (dt): {time_step}")

    # Priprava dat na trenovanie modelu
    X = np.stack((data[:, 1], data[:, 2], data[:, 3]), axis=-1)
    U = np.stack((data[:, 4]), axis=-1)
    X_train, X_val, X_test, U_train, U_val, U_test = lib.split_data(X, U, val_size=0.2, test_size=0.2)

    # Tvorba viacerych trajektorii
    num_samples = int(20 / time_step)
    X_train, U_train = lib.generate_trajectories(X_train, U_train, num_samples=num_samples, num_trajectories=3)

    # Obmedzenia kladene na model
    constraints = {
        "sim_steps": 100,
        "max_sparsity": 24,
        "max_coeff": 1e2,
        "max_rmse": 50,
        "max_state": 1e2,
        "rmse_weight": 0.6
    }

    # Vytvorenie gridu
    window_lengths = []
    for time_interval in [0.01, 0.05, 0.1, 0.5, 1]:
        window_length = time_interval / time_step
        window_lengths.append(int(window_length))

    polyorders = [2, 3]

    relax_coeff_nus = [0.01, 0.1, 1.0, 10.0, 100.0]
    reg_weight_lams = [0.01, 0.1, 1.0, 10.0, 100.0]

    estimator = lib.SINDYcEstimator()
    for window_length in window_lengths:
        for polyorder in polyorders:
            estimator.set_differentiation_method("SmoothedFiniteDifference", window_length=window_length, polyorder=polyorder)
    
    for relax_coeff_nu in relax_coeff_nus:
        for reg_weight_lam in reg_weight_lams:
            estimator.set_optimizer("SR3", reg_weight_lam=reg_weight_lam, relax_coeff_nu=relax_coeff_nu)

    estimator.set_feature_library("PolynomialLibrary")

    # Vytvorenie a hladanie konfiguracie
    estimator.generate_configurations()
    estimator.search_configurations(X_train, X_val, U_train, U_val, dt=time_step, n_processes=8, **constraints)
    estimator.plot_pareto()
    estimator.export_data()

    print("\nBest model:")
    feature_names = ["x", "y", "z"]
    model = estimator.best_config["model"]
    model.fit(x=X_train, u=U_train, t=time_step, feature_names=feature_names)
    model.print()
    print(f"Score: {model.score(x=X_test, u=U_test, t=time_step)}")

    return 0

if __name__ == "__main__":
    SINDYc()
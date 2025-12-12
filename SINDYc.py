import numpy as np
import pandas as pd
import pysindy as ps
import SINDYcLib as lib
import matplotlib.pyplot as plt
if __name__ == "__main__":
    data_csv = pd.read_csv("Simulacia.csv")
    data = data_csv.to_numpy()

    time_step = np.round(np.median(np.diff(data[:, 0])), decimals=5)
    print(f"\nEstimated time step (dt): {time_step}")

    # Priprava dat na trenovanie modelu
    X = np.stack((data[:, 1], data[:, 2], data[:, 3]), axis=-1)
    U = np.stack((data[:, 4]), axis=-1)

    X_train, X_val, X_test, U_train, U_val, U_test = lib.split_data(X, U, val_size=0.2, test_size=0.2)

    constraints = {
        "sim_steps": 350,
        "max_sparsity": 24,
        "max_coeff": 1e2,
        "max_rmse": 50,
        "max_state": 1e2
    }

    reg_weight_lams, relax_coeff_nus, window_lengths = lib.recommended_grid(x=X, dt=time_step, num_lam=10)

    estimator = lib.SINDYcEstimator()
    for window_length in window_lengths:
        for polyorder in [2, 3]:
            estimator.set_differentiation_method("SmoothedFiniteDifference", window_length=window_length, polyorder=polyorder)

    for reg_weight_lam in reg_weight_lams:
        for relax_coeff_nu in relax_coeff_nus:
            estimator.set_optimizer("SR3", reg_weight_lam=reg_weight_lam, relax_coeff_nu=relax_coeff_nu)

    estimator.set_feature_library(degree=2, include_bias=False)

    estimator.generate_configurations()
    estimator.search_configurations(X_train, X_val, U_train, U_val, dt=time_step, n_processes=8, **constraints)

    errs = np.array([r["rmse"] for r in estimator.pareto_front], dtype=float)
    spars = np.array([r["sparsity"] for r in estimator.pareto_front], dtype=float)

    plt.figure(figsize=(6, 4))
    plt.scatter(errs, spars, color="tab:blue", label="Pareto body")
    plt.xlabel("RMSE")
    plt.ylabel("Sparsity (počet nenulových koeficientov)")
    plt.title("Pareto front")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    model = estimator.best_config["model"]
    model.fit(x=X_train, u=U_train, t=time_step)
    model.print()
    print(model.score(x=X_test, u=U_test, t=time_step))


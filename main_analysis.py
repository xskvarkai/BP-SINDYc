import numpy as np
import pandas as pd
import data_analysis_lib as lib
import pysindy as ps
from sklearn.metrics import root_mean_squared_error

# ========== Globalne funkcie pre CustomLibrary ==========
def x(x): return x
def xy(x, y): return x * y
def squared_x(x): return x ** 2
def x_name(x): return x
def xy_name(x, y): return x + "" + y
def squared_x_name(x): return x + "^2"

# ========== Funkcia pre hladanie SINDY modelu ==========
def sindyc_model(data):

    X_train, U_train, X_val, U_val, X_test, U_test, time_step = data

    total_size = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    val_size = np.round(X_val.shape[0]/total_size, decimals=3)
    test_size = np.round(X_test.shape[0]/total_size, decimals=3)

    # Tvorba viacerych trajektorii
    num_samples = int(20 / time_step)
    num_trajectories = 5
    X_train, U_train = lib.generate_trajectories(X_train, U_train, num_samples=num_samples, num_trajectories=num_trajectories)

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
    reg_weight_lams = [0.1, 1.0, 10.0, 100.0]

    estimator = lib.SINDYcEstimator()
    for window_length in window_lengths:
        for polyorder in polyorders:
            estimator.set_differentiation_method("SmoothedFiniteDifference", window_length=window_length, polyorder=polyorder)

    for relax_coeff_nu in relax_coeff_nus:
        for reg_weight_lam in reg_weight_lams:
            estimator.set_optimizer("SR3", reg_weight_lam=reg_weight_lam, relax_coeff_nu=relax_coeff_nu)
    
    estimator.set_feature_library("PolynomialLibrary", include_bias=False)

    # Vytvorenie a hladanie konfiguracie
    estimator.generate_configurations()
    estimator.search_configurations(X_train, X_val, U_train, U_val, dt=time_step, n_processes=8, **constraints)
    estimator.plot_pareto()

    data = {
        "np.random.seed": 42,
        "dt": time_step,
        "val_size": val_size,
        "test_size": test_size,
        "train_multi_trajectories_num": num_trajectories,
        "train_multi_trajectories_samples": num_samples
    }

    estimator.export_data(data, "data.json")
    estimator.delete_tempfiles()

# ========== Funkcia pre hladanie koopmanovho operatora ==========
def koopman_model(data):
    import pykoopman
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    X_train, U_train, X_val, U_val, X_test, U_test, time_step = data

    # Tvorba viacerych trajektorii
    num_samples = int(20 / time_step)
    num_trajectories = 5
    X_train, U_train = lib.generate_trajectories(X_train, U_train, num_samples=num_samples, num_trajectories=num_trajectories)

    # Ziskany model
    sindyc_model = ps.SINDy(
        # optimizer=,
        # differentiation_method=,
        # feature_library=
    )

    sindyc_model.fit(x=X_train, u=U_train, t=time_step)
    X_koopman_train = sindyc_model.simulate(x0=X_val[0], u=U_val, t=time_step)
    X_koopman_val = sindyc_model.simulate(x0=X_test[0], u=U_test, t=time_step)

    library_functions = [x, xy, squared_x]
    functions_names = [x_name, xy_name, squared_x_name]

    observables = pykoopman.observables.CustomObservables(library_functions, functions_names)
    edmdc = pykoopman.regression.EDMDc()
    koopman_model = pykoopman.KoopmanContinuous(observables=observables, regressor=edmdc)
    koopman_model.fit(x=X_koopman_train, u=U_val.reshape(-1, 1), dt=time_step)

    steps = X_koopman_val.shape[0]
    x0 = X_koopman_val[0]
    x_sim = koopman_model.simulate(x0=x0, u=U_val.reshape(-1, 1), n_steps=steps)
    rmse_predict = root_mean_squared_error(X_koopman_val, x_sim)
    print(rmse_predict)

    # time_vector = data[X_train.shape[0]: X_train.shape[0] + steps, 0]
    # import simulation as sim
    # sim.vizualize_trajectory(time_vector=data[:, 0], trajectory=X, input=U)
    # sim.vizualize_trajectory(time_vector=time_vector, trajectory=X_val, comparison_trajectory=x_val_sim, input=U_val)


if __name__ == "__main__":

    # Hlavne data
    np.random.seed(42)
    data_csv = pd.read_csv("Simulacia.csv")
    data = data_csv.to_numpy()
    time_step = np.round(np.median(np.diff(data[:, 0])), decimals=5)
    print(f"\nEstimated time step (dt): {time_step}")
    # Priprava dat na trenovanie modelu
    X = np.stack((data[:, 1], data[:, 2], data[:, 3]), axis=-1)
    U = np.stack((data[:, 4]), axis=-1)
    X_train, X_val, X_test, U_train, U_val, U_test = lib.split_data(X, U, val_size=0.2, test_size=0.2)
    data_csv = None
    data = None
    X = None
    U = None

    data = [X_train, U_train, X_val, U_val, X_test, U_test, time_step]
    sindyc_model(data)
    # koopman_model(data)
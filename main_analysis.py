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
    np.random.seed(42)
    num_samples = int(10 // time_step)
    num_trajectories = 5
    X_train, U_train = lib.generate_trajectories(X_train, U_train, num_samples=num_samples, num_trajectories=num_trajectories)

    # Obmedzenia kladene na model
    constraints = {
        "sim_steps": 100,
        "coeff_precision": None,
        "max_sparsity": 24,
        "max_coeff": 1e2,
        "max_predict_r2": 0.7,
        "max_state": 1e2,
        "rmse_weight": 0.6
    }

    # Vytvorenie gridu
    
    window_lengths = []
    for time_interval in [0.05, 0.1, 0.5]:
        window_length = time_interval / time_step
        window_lengths.append(int(window_length))

    polyorders = [2, 3]
    
    # alphas = [1e-2, 1e-1]

    relax_coeff_nus = [0.01, 0.1, 1.0, 10.0, 100.0]
    reg_weight_lams = [0.1, 1.0, 10.0, 100.0]
    regularizers = ["l1", "l2"]
    unbiases = [False]

    estimator = lib.SINDYcEstimator()
    
    # for polyorder in polyorders:
    #     for alpha in alphas:
    #         estimator.set_differentiation_method("TotalVariationDenoising", order=polyorder, alpha=alpha)

    for window_length in window_lengths:
        for polyorder in polyorders:
            estimator.set_differentiation_method("SmoothedFiniteDifference", window_length=window_length, polyorder=polyorder)

    for relax_coeff_nu in relax_coeff_nus:
        for reg_weight_lam in reg_weight_lams:
            for regularizer in regularizers:
                for unbias in unbiases:
                    estimator.set_optimizer("SR3", reg_weight_lam=reg_weight_lam, relax_coeff_nu=relax_coeff_nu, regularizer=regularizer, unbias=unbias)
    
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
        "train_multi_trajectories_samples": num_samples,
        "constraints": constraints
    }

    estimator.export_data(data, "data.json")
    estimator.delete_tempfiles()

# ========== Funkcia pre hladanie koopmanovho operatora ==========
def koopman_model(data):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    import pykoopman

    X_train, U_train, X_val, U_val, X_test, U_test, time_step = data

    # Tvorba viacerych trajektorii
    np.random.seed(42)
    num_samples = int(20 / time_step)
    num_trajectories = 5
    X_train, U_train = lib.generate_trajectories(X_train, U_train, num_samples=num_samples, num_trajectories=num_trajectories)

    # Ziskany model
    sindy = ps.SINDy(
        optimizer=ps.SR3(max_iter=100000, normalize_columns=True, reg_weight_lam=1.0, regularizer='L1', relax_coeff_nu=100.0, tol=1e-10),
        differentiation_method=ps.SmoothedFiniteDifference(smoother_kws={'axis': 0, 'mode': 'interp', 'polyorder': 3, 'window_length': 51}),
        feature_library=ps.PolynomialLibrary(include_bias=False)
    )

    sindy.fit(x=X_train, u=U_train, t=time_step)
    print()
    sindy.print()
    print(sindy.score(x=X_test, u=U_test, t=time_step), end="\n\n")

    # Udaje potrebne na ziskanie trajektorie zo SINDY
    train_steps = len(X_val)
    val_steps = int((len(X_val)/0.8) * 0.41)
    t_train = np.arange(train_steps) * time_step
    t_val = np.arange(val_steps) * time_step
    u_train = U_val[:train_steps]
    x_train = X_val[:train_steps]
    u_val = U_test[:val_steps]
    x_val = X_test[:val_steps]

    # Vytvaranie trajektorie
    X_koopman_train = sindy.simulate(x0=x_train[0], u=u_train, t=t_train)
    X_koopman_val = sindy.simulate(x0=x_val[0], u=u_val, t=t_val)

    
    print("SINDYc trajectory generation done.", end="\r", flush=True)

    # Kniznice podla SINDYc modelu
    library_functions = [x, xy, squared_x]
    function_names = [x_name, xy_name, squared_x_name]

    min_len = len(X_koopman_train)
    u_train = u_train[:min_len]

    # Koopmanov model
    observables = pykoopman.observables.CustomObservables(library_functions, function_names)
    edmdc = pykoopman.regression.EDMDc()
    koopman_model = pykoopman.KoopmanContinuous(observables=observables, regressor=edmdc)
    koopman_model.fit(x=X_koopman_train, u=u_train.reshape(-1, 1), dt=time_step)

    min_len = len(X_koopman_val)
    u_val = u_val[:min_len]

    x_sim = koopman_model.simulate(x=X_koopman_val, u=u_val.reshape(-1, 1))
    
    print("Koopman trajectory simulation done.", end="\r", flush=True)

    min_len = min(len(X_koopman_val), len(x_sim))
    rmse = root_mean_squared_error(X_koopman_val[:min_len], x_sim[:min_len])
    print(f"RMSE of Koopman model simulation: {rmse:.2f}", flush=True)

    from simulation import vizualize_trajectory
    vizualize_trajectory(t_val[:min_len], X_koopman_val[:min_len], x_sim[:min_len], u_val[:min_len])

if __name__ == "__main__":
    # Hlavne data
    np.random.seed(42)
    data_csv = pd.read_csv("Simulacia.csv")
    data = data_csv.to_numpy()
    time_step = np.round(np.median(np.diff(data[:, 0])), decimals=5)
    print(f"\nEstimated time step (dt): {time_step}")
    X = np.stack((data[:, 1], data[:, 2], data[:, 3]), axis=-1)
    U = np.stack((data[:, 4]), axis=-1)
    X_train, X_val, X_test, U_train, U_val, U_test = lib.split_data(X, U, val_size=0.2, test_size=0.2)
    data_csv, data, X, U = None, None, None, None

    data = [X_train, U_train, X_val, U_val, X_test, U_test, time_step]
    sindyc_model(data)
    # koopman_model(data)
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

    X, U, time_step, random_seed = data
    val_size, test_size = 0.2, 0.2
    noise_level = lib.find_noise(X)
    lib.find_periodicity(X)

    X_train, X_val, _, U_train, U_val, _ = lib.split_data(X, U, val_size=val_size, test_size=test_size)

    # Tvorba viacerych trajektorii
    num_samples = int(X.shape[0] * val_size)
    num_trajectories = 5
    X_train, U_train = lib.generate_trajectories(X_train, U_train, num_samples=num_samples, num_trajectories=num_trajectories, randomseed=random_seed)

    # Obmedzenia kladene na model
    constraints = {
        "sim_steps": 100,
        "coeff_precision": None,
        "max_complexity": 24,
        "max_coeff": 1e2,
        "min_r2": 0.7,
        "max_state": 1e2
    }

    # Vytvorenie gridu
    with lib.SINDYcEstimator() as estimator:

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

        thresholds = lib.estimate_threshold(X, time_step, U, library, noise_level)
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

        estimator.export_data(data, "data.json")
        estimator.delete_tempfiles()

# ========== Funkcia pre hladanie koopmanovho operatora ==========
def koopman_model(data):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    import pykoopman

    # Podla dat z grid search
    X, U, _, _= data
    time_step = 0.002
    val_size, test_size = 0.2, 0.2

    X_train, X_val, X_test, U_train, U_val, U_test = lib.split_data(X, U, val_size=val_size, test_size=test_size)

    # Tvorba viacerych trajektorii podla dat z grid search
    num_samples = 10000
    num_trajectories = 5
    X_train, U_train = lib.generate_trajectories(X_train, U_train, num_samples=num_samples, num_trajectories=num_trajectories, randomseed=42)

    # Ziskany model z grid search
    np.random.seed(175)
    finded_sindy = ps.SINDy(
        optimizer=ps.STLSQ(alpha=0.01, max_iter=100000, threshold=np.float64(0.07339287), unbias=False),
        differentiation_method=None,
        feature_library = ps.WeakPDELibrary(
            K=200,
            derivative_order=1,
            differentiation_method=ps.FiniteDifference(),
            function_library=ps.PolynomialLibrary(include_bias=False),
            p=5,
            spatiotemporal_grid=np.arange(0.0, 1.9998e+01 + time_step, time_step))
    )
    finded_sindy.fit(x=X_train, u=U_train, t=time_step)

    # Pristup k weak formulation a simulacii trajektorii
    sindy = ps.SINDy(
        feature_library=ps.PolynomialLibrary(include_bias=False),
    )
    sindy.fit(x=X_train[:5], u=U_train[:5], t=time_step)
    sindy.optimizer.coef_ = finded_sindy.optimizer.coef_
    print()
    # Vypisanie modelu a jeho skore na testovacej sade (ina ako validacna)
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

    print("SINDYc trajectory generation started.", end="\r")
    # Vytvaranie trajektorie
    X_koopman_train = sindy.simulate(x0=x_train[0], u=u_train, t=t_train, integrator_kws={"rtol": 1e-6,"atol": 1e-6})
    X_koopman_val = sindy.simulate(x0=x_val[0], u=u_val, t=t_val, integrator_kws={"rtol": 1e-6,"atol": 1e-6})
    
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
    koopman_model.fit(x=X_koopman_train, y=X_koopman_train, u=u_train.reshape(-1, 1), dt=time_step)

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
    data_csv = pd.read_csv("Simulacia.csv")
    data = data_csv.to_numpy()
    time_step = np.round(np.median(np.diff(data[:, 0])), decimals=5)
    print(f"\nEstimated time step (dt): {time_step}")
    X = np.stack((data[:, 1], data[:, 2], data[:, 3]), axis=-1)
    U = np.stack((data[:, 4]), axis=-1)
    random_seed = 42
    data_csv, data,= None, None
 
    data = [X, U, time_step, random_seed]
    sindyc_model(data)
    # koopman_model(data)
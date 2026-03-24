import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
from sklearn.preprocessing import Normalizer

from data_ingestion.data_loader import DataLoader
from utils.config_manager import ConfigManager
from utils.plots import plot_pareto, plot_noisy_filtered_trajectory
from utils.helpers import compute_time_vector
from data_processing.state_estimator import SindyUKFEstimator

def ilustate_noise():
    config_manager = ConfigManager("config")

    with DataLoader(config_manager, "data_raw_dir") as loader:
        X, U, time_step = loader.load_csv_data(
            "Aeroshield",
            [0],
            None,
            0.01,
            [1],
            plot_data=True,
            verbose=False
        )
    X = X[0: int(10 / time_step)]

    X_dot = np.gradient(X, time_step, axis=0)
    X_noisy = np.hstack([X, X_dot])

    X_filtered = savgol_filter(X_noisy, 31, 2, axis=0)

    plot_noisy_filtered_trajectory(compute_time_vector(X_noisy, time_step), X_noisy, X_filtered, exportable=True)

def Aeroshield_load_and_deriv():
    config_manager = ConfigManager("config")

    with DataLoader(config_manager, "data_raw_dir") as loader:
        X, U, time_step = loader.load_csv_data(
            "Aeroshield",
            [0],
            None,
            0.01,
            [1],
            plot_data=True,
            verbose=False
        )

        X_val, U_val, _ = loader.load_csv_data(
            "Aeroshield_val",
            [0],
            None,
            0.01,
            [1],
            verbose=False
        )

    X_dot = np.gradient(X, time_step, axis=0)
    X_dot_val = np.gradient(X_val, time_step, axis=0)

    X_new = np.vstack([X, X_val])
    X_dot_new = np.vstack([X_dot, X_dot_val])
    U_new = np.vstack([U, U_val])

    data = {
        "x": X_new.flatten(),
        "x_dot": X_dot_new.flatten(),
        "u": U_new.flatten()
    }

    file_path = "data/processed/Aeroshield_with_deriv.csv"
    df = pd.DataFrame(data)  
    df.to_csv(file_path, index=False)

def Aeroshield_plotting_pareto_from_results():
    errs = [0.15727, 0.14345, 0.15363, 0.1488, 0.14959, 0.1469, 0.14586, 0.14959]
    spars = [5, 8, 5, 6, 5, 7, 8, 5]

    plot_pareto(errs, spars)

def Floatshield_load_and_deriv():
    config_manager = ConfigManager("config")

    with DataLoader(config_manager, "data_raw_dir") as loader:
        X, U, dt = loader.load_csv_data(
            "Floatshield_close-loop2",
            [0, 1],
            None,
            0.025,
            [2],
            apply_savgol_filter=True,
            savgol_polyorder=2,
            savgol_window_length=71,
            plot_data=True,
            verbose=False
        )

        X_val, U_val, dt = loader.load_csv_data(
            "Floatshield_close-loop2_val",
            [0, 1],
            None,
            0.025,
            [2],
            apply_savgol_filter=True,
            savgol_polyorder=2,
            savgol_window_length=71,
            plot_data=True,
            verbose=False
        )
    
    X[:, 0] = X[:, 0] / (320) # Prevod na precenta
    X_dot = np.gradient(X[:, 0], dt, axis=0)
    
    X_val[:, 0] = X_val[:, 0] / (320) # Prevod na precenta
    X_dot_val = np.gradient(X_val[:, 0], dt, axis=0)

    U = U / 100
    U_val = U_val / 100

    X[:, 1] = X[:, 1] / 17000
    X_val[:, 1] = X_val[:, 1] / 17000

    X_new = np.vstack([X, X_val])
    X_dot_new = np.vstack([X_dot.reshape(-1, 1), X_dot_val.reshape(-1, 1)])
    U_new = np.vstack([U, U_val])

    # Ak má byť u stĺpcový vektor a porovnávame ho so stĺpcom z X_new
    u_compare = U_new.flatten() # Prevedieme na 1D pole pre porovnanie s 1D stĺpcom
    x_second_column_compare = X_dot_new.flatten() # Vezmeme druhý stĺpec z X_new

    # Pre správnosť porovnania musia mať u_compare a x_second_column_compare rovnakú dĺžku.
    # Upravím u_compare, aby zodpovedalo dĺžke x_second_column_compare, ak je to potrebné.
    # V tomto kontexte predpokladám, že ich dĺžky sú už kompatibilné vďaka vstack operáciám.
    # Ak by U_new a X_new nemali rovnaký počet riadkov, bolo by potrebné to prepočítať alebo inak prispôsobiť.
    # Pre tento príklad predpokladám, že X_new.shape[0] == U_new.shape[0]

    # Prevedenie u_compare a x_second_column_compare na kompatibilné tvary
    # Tu je predpoklad, že chceme porovnať U_new s X_new[:, 1]
    # a že oboje sú 1D polia rovnakej dĺžky.
    # Ak U_new je stĺpcový vektor, tak ho pre porovnanie s druhým stĺpcom X_new najprv sploštíme.

    result = u_compare**3 - x_second_column_compare < 0
    true_indices = np.where(result)[0]
    print("Indexy, kde je 'result' True:", true_indices)

    true_pairs = np.column_stack((u_compare[result], x_second_column_compare[result]))
    print("Páry (u_compare, x_second_column_compare), kde je 'result' True:\n", true_pairs)

    print(f"Tvar výsledného booleovského poľa: {result.shape}")
    print(f"Prvých 5 hodnôt výsledku: {result[:5]}") # Vypíše len prvých 5, pre prehľadnosť

    # Kontrola, či je aspoň jeden výsledok True
    if result.any():
        print("Aspoň jeden výsledok podmienky (u - x_second_column >= 0) je True.")
    else:
        print("Žiaden výsledok podmienky (u - x_second_column >= 0) nie je True.")

    data = {
        "x": X_new[:, 0].flatten(),
        "x_dot": X_dot_new.flatten(),
        "omega": X_new[:, 1].flatten(),
        "u": U_new.flatten(),
    }

    file_path = "data/processed/Floatshield_with_deriv_close-loop2.csv"
    df = pd.DataFrame(data)  
    df.to_csv(file_path, index=False)


def estimate_state():
    config_manager = ConfigManager("config")

    def sindy_dxdt_init(x: np.ndarray, u0: float) -> np.ndarray:
        """
        Placeholder for SINDy dynamics function.
        This function should represent the actual identified dynamics of your system.
        """
        x0, x1, x2 = x

        dx0 = 1.0 * x1
        dx1 = -9.81 - 0.010343833 * u0**4 * np.abs(u0**4 - x1) + 0.010343833 * x1 * np.abs(u0**4 - x1)
        dx2 = 7.78 * u0 - 7.1428 * x2
        return np.array([dx0, dx1, dx2])

    config_manager = ConfigManager("config")

    with DataLoader(config_manager) as loader:
        X, U, dt = loader.load_csv_data(
            "Floatshield_with_deriv",
            [0, 1],
            None,
            0.025,
            [2],
            apply_savgol_filter=False,
            plot_data=False,
            verbose=False
        )

    time_vec = compute_time_vector(X, dt)
    # --- Instantiate and Run the SindyEKFEstimator Class ---
    ekf_estimator = SindyUKFEstimator(config_manager=config_manager, 
                                      sindy_dxdt_func=sindy_dxdt_init,
                                      dt=dt)

    df_results_class = ekf_estimator.estimate(
        time=time_vec,
        measured_x1=X[:, 0].flatten(),
        measured_x2=X[:, 1].flatten(), # Pass None if x is not measured
        measured_x3=None,
        control_u=U.flatten()
    )

    # --- Visualization of Results ---
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 10))

    plt.subplot(4, 1, 1)
    plt.plot(time_vec, X[:, 0], 'rx', markersize=3, alpha=0.5, label='Measured x1 (noisy)')
    plt.plot(time_vec, df_results_class['x1_est'], 'b-', label='Estimated x1 (EKF)')
    plt.title('EKF State Estimation using SINDy Model')
    plt.ylabel('State x1')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(time_vec, X[:, 1], 'rx', markersize=3, alpha=0.5, label='Measured x2 (noisy)')
    plt.plot(time_vec, df_results_class['x2_est'], 'b-', label='Estimated x2 (EKF)')
    plt.ylabel('State x2')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(time_vec, df_results_class['x3_est'], 'r-', label='Estimated x3 (EKF)')
    plt.ylabel('State x3')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    plt.plot(time_vec, U, 'k-', label='Control Input u')
    plt.xlabel('Time [s]')
    plt.ylabel('Control u')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
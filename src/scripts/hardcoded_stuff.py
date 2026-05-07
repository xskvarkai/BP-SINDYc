import numpy as np
import pandas as pd

from scipy import signal
from scipy.signal import savgol_filter
from sklearn.preprocessing import Normalizer

from data_ingestion.data_loader import DataLoader
from utils.config_manager import ConfigManager
from utils.plots import plot_pareto, plot_noisy_filtered_trajectory, plot_trajectory
from utils.helpers import compute_time_vector, rk4_integrator
from data_processing.state_estimator import SindyUKFEstimator

def ilustate_noise():
    config_manager = ConfigManager("config")

    with DataLoader(config_manager, "data_raw_dir") as loader:
        X, U, dt = loader.load_csv_data(
            "Aeroshield",
            [0],
            None,
            0.01,
            [1],
            plot_data=True,
            verbose=False
        )
    X = X[0: int(10 / dt)]

    X_dot = np.gradient(X, dt, axis=0)
    X_noisy = np.hstack([X, X_dot])

    X_filtered = savgol_filter(X_noisy, 31, 2, axis=0)

    plot_noisy_filtered_trajectory(compute_time_vector(X_noisy, dt), X_noisy, X_filtered, exportable=True)

def Aeroshield_load_and_deriv():
    config_manager = ConfigManager("config")

    with DataLoader(config_manager, "data_raw_dir") as loader:
        X, U, dt = loader.load_csv_data(
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

    X_dot = np.gradient(X, dt, axis=0)
    X_dot_val = np.gradient(X_val, dt, axis=0)

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

    plot_pareto(errs, spars, exportable=True)

def Floatshield_load_and_deriv():
    config_manager = ConfigManager("config")

    with DataLoader(config_manager, "data_raw_dir") as loader:
        X, U, dt = loader.load_csv_data(
            "Floatshield",
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

        X_val, U_val, _ = loader.load_csv_data(
            "Floatshield_val",
            [0, 1],
            None,
            dt,
            [2],
            apply_savgol_filter=True,
            savgol_polyorder=2,
            savgol_window_length=71,
            plot_data=False,
            verbose=False
        )

        X_test, U_test, _ = loader.load_csv_data(
            "Floatshield_test",
            [0, 1],
            None,
            dt,
            [2],
            apply_savgol_filter=True,
            savgol_polyorder=2,
            savgol_window_length=71,
            plot_data=False,
            verbose=False
        )
    # Rozdelenie stavu
    omega = X[:, 1]
    omega_val = X_val[:, 1]
    omega_test = X_test[:, 1]

    X = X[:, 0]
    X_val = X_val[:, 0]
    X_test = X_test[:, 0]

    # Orezenie extremov a chyb snimaca
    X[X < 0] = 0
    X_val[X_val < 0] = 0
    X_test[X_test < 0] = 0

    U[U < 0] = 0
    U_val[U_val < 0] = 0
    U_test[U_test < 0] = 0

    omega[omega < 0] = 0
    omega_val[omega_val < 0] = 0
    omega_test[omega_test < 0] = 0

    X[X > 321.74] = 321.74
    X_val[X_val > 321.74] = 321.74
    X_test[X_test > 321.74] = 321.74

    r = 0.02 * 2 * np.pi # prevod na rad/s
    omega = omega * r / 60
    omega_val = omega_val * r / 60
    omega_test = omega_test * r / 60

    X = X / 1000 # Prevod na meter
    X_val = X_val / 1000
    X_test = X_test / 1000

    U = U * 0.033 # Prevod na volt
    U_val = U_val * 0.033
    U_test = U_test * 0.033

    X_max = np.max(X)
    U_max = np.max(U)
    omega_max = np.max(omega)

    X = X / X_max # Prevod na precenta
    X_dot = np.gradient(X, dt, axis=0)
    X_val = X_val / X_max # Prevod na precenta
    X_dot_val = np.gradient(X_val, dt, axis=0)
    X_test = X_test / X_max # Prevod na precenta
    X_dot_test = np.gradient(X_test, dt, axis=0)

    U = U / U_max # Prevod na precenta
    U_val = U_val / U_max # Prevod na precenta
    U_test = U_test / U_max # Prevod na precenta
    
    omega = omega / omega_max # Prevod na precenta
    omega = savgol_filter(omega, 71, 2, axis=0)
    omega_dot = np.gradient(omega, dt, axis=0)
                            
    omega_val = omega_val / omega_max # Prevod na precenta
    omega_val = savgol_filter(omega_val, 71, 2, axis=0)
    omega_dot_val = np.gradient(omega_val, dt, axis=0)

    omega_test = omega_test / omega_max # Prevod na precenta
    omega_test = savgol_filter(omega_test, 71, 2, axis=0)
    omega_dot_test = np.gradient(omega_test, dt, axis=0)

    df_train = pd.DataFrame({
        "x": X.flatten(),
        "x_dot": X_dot.flatten(),
        "omega": omega.flatten(),
        "omega_dot": omega_dot.flatten(),
        "u": U.flatten(),
        "X_max": X_max,
        "u_max": U_max,
        "omega_max": omega_max
    })

    df_val = pd.DataFrame({
        "x": X_val.flatten(),
        "x_dot": X_dot_val.flatten(),
        "omega": omega_val.flatten(),
        "omega_dot": omega_dot_val.flatten(),
        "u": U_val.flatten(),
        "X_max": X_max,
        "u_max": U_max,
        "omega_max": omega_max
    })

    df_test = pd.DataFrame({
        "x": X_test.flatten(),
        "x_dot": X_dot_test.flatten(),
        "omega": omega_test.flatten(),
        "omega_dot": omega_dot_test.flatten(),
        "u": U_test.flatten(),
        "X_max": X_max,
        "u_max": U_max,
        "omega_max": omega_max
    })

    def add_delays(df: pd.DataFrame, delays: int=3, delay_indices: list=None):
        # Pridáme posunuté hodnoty pre napätie (u)
        for i in range(1, delays + 1):

            if delay_indices is not None and i in delay_indices:
                df[f'omega_k-{i}'] = df['omega'].shift(i)
            elif delay_indices is None:
                df[f'omega_k-{i}'] = df['omega'].shift(i)

            if delay_indices is not None and i in delay_indices:
                df[f'omega_dot_k-{i}'] = df['omega_dot'].shift(i)
            elif delay_indices is None:
                df[f'omega_dot_k-{i}'] = df['omega_dot'].shift(i)

            if delay_indices is not None and i in delay_indices:
                df[f'u_k-{i}'] = df['u'].shift(i)
            elif delay_indices is None:
                df[f'u_k-{i}'] = df['u'].shift(i)
            
        # Funkcia shift() vytvorí na prvých 'delays' riadkoch hodnoty NaN (chýbajúce dáta).
        # dropna() tieto neúplné riadky bezpečne zahodí pre všetky stĺpce naraz.
        return df.dropna()

    #df_train = add_delays(df_train, 150)
    #df_val = add_delays(df_val, 150)
    #df_test = add_delays(df_test, 150)

    df_final = pd.concat([df_train, df_val, df_test], ignore_index=True)

    file_path = "data/processed/Floatshield_with_deriv.csv"
    df_final.to_csv(file_path, index=False)
    
    print(f"Dáta boli úspešne uložené. Tvar výsledného DataFrame: {df_final.shape}")

def estimate_state():
    config_manager = ConfigManager("config")

    def sindy_dxdt_init(x: np.ndarray, u0: float) -> np.ndarray:
        """
        Placeholder for SINDy dynamics function.
        This function should represent the actual identified dynamics of your system.
        """
        x0, x1, x2 = x

        dx0 = 1.0 * x1
        dx1 = - 30.65625 + 0.2161725 * (53125 * x2 - x1) * np.abs(53125 * x2 - x1)
        dx2 = 0.0032689 * u0 - 7.142857 * x2
        return np.array([dx0, dx1, dx2])

    config_manager = ConfigManager("config")

    with DataLoader(config_manager, "data_raw_dir") as loader:
        X, U, dt = loader.load_csv_data(
            "Floatshield",
            [0],
            None,
            0.025,
            [2],
            apply_savgol_filter=True,
            savgol_polyorder=2,
            savgol_window_length=71,
            plot_data=False,
            verbose=False
        )

        X_val, U_val, dt = loader.load_csv_data(
            "Floatshield_val",
            [0],
            None,
            0.025,
            [2],
            apply_savgol_filter=True,
            savgol_polyorder=2,
            savgol_window_length=71,
            plot_data=False,
            verbose=False
        )

    X = X / (320) # Prevod na precenta
    X_dot = np.gradient(X, dt, axis=0)
    X_val = X_val / (320) # Prevod na precenta
    X_dot_val = np.gradient(X_val, dt, axis=0)

    U = U / 100 # Prevod na precenta
    U_val = U_val / 100

    time_vec = compute_time_vector(X, dt)
    # --- Instantiate and Run the SindyEKFEstimator Class ---
    ekf_estimator = SindyUKFEstimator(config_manager=config_manager, 
                                      sindy_dxdt_func=sindy_dxdt_init,
                                      dt=dt)

    df_results_class = ekf_estimator.estimate(
        time=time_vec,
        measured_x1=X.flatten(),
        measured_x2=X_dot.flatten(), # Pass None if x is not measured
        measured_x3=None,
        control_u=U.flatten()
    )

    df_results_class_val = ekf_estimator.estimate(
        time=time_vec,
        measured_x1=X_val.flatten(),
        measured_x2=X_dot_val.flatten(), # Pass None if x is not measured
        measured_x3=None,
        control_u=U_val.flatten()
    )

    # 5. Spojenie dát do jedného finálneho celku
    df_final = pd.concat([df_results_class, df_results_class_val], ignore_index=True)

    # 6. Uloženie
    file_path = "data/processed/Floatshield_with_deriv.csv"
    df_final.to_csv(file_path, index=False)
    
    print(f"Dáta boli úspešne uložené s posunmi. Tvar výsledného DataFrame: {df_final.shape}")

    # --- Visualization of Results ---
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 10))

    plt.subplot(4, 1, 1)
    plt.plot(time_vec, X, 'rx', markersize=3, alpha=0.5, label='Measured x1 (noisy)')
    plt.plot(time_vec, df_results_class['x1_est'], 'b-', label='Estimated x1 (EKF)')
    plt.title('EKF State Estimation using SINDy Model')
    plt.ylabel('State x1')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(time_vec, X_dot, 'rx', markersize=3, alpha=0.5, label='Measured x2 (noisy)')
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
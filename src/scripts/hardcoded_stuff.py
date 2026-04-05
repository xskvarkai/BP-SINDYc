import numpy as np
import pandas as pd

from scipy import signal
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
            plot_data=False,
            verbose=False
        )

        X_val, U_val, dt = loader.load_csv_data(
            "Floatshield_val",
            [0, 1],
            None,
            0.025,
            [2],
            apply_savgol_filter=True,
            savgol_polyorder=2,
            savgol_window_length=71,
            plot_data=False,
            verbose=False
        )
    
    delay_cross_correlation(U, X, 40, 0)
    delay_cross_correlation(U_val, X_val, 40, 0)

    #x_old = X
    #x_old = x_old / 320
    #X = X[24:]
    #X_val = X_val[24:]

    #U = U[:-24]
    #U_val = U_val[:-24]

    X[:, 0] = X[:, 0] / (320) # Prevod na precenta
    X_dot = np.gradient(X[:, 0], dt, axis=0)
    
    X_val[:, 0] = X_val[:, 0] / (320) # Prevod na precenta
    X_dot_val = np.gradient(X_val[:, 0], dt, axis=0)

    U = U / 100
    U_val = U_val / 100

    X[:, 1] = X[:, 1] / 17000
    X_val[:, 1] = X_val[:, 1] / 17000

    #plot_noisy_filtered_trajectory(compute_time_vector(X, dt), X, x_old[:-24], U)

    X_new = np.vstack([X, X_val])
    X_dot_new = np.vstack([X_dot.reshape(-1, 1), X_dot_val.reshape(-1, 1)])
    U_new = np.vstack([U, U_val])

    data = {
        "x": X_new[:, 0].flatten(),
        "x_dot": X_dot_new.flatten(),
        "omega": X_new[:, 1].flatten(),
        "u": U_new.flatten(),
    }

    file_path = "data/processed/Floatshield_with_deriv.csv"
    df = pd.DataFrame(data)  
    df.to_csv(file_path, index=False)

def delay_cross_correlation(input_signal, output_signal, sampling_rate, output_column_index=None):
    """
    Odhadne časové oneskorenie medzi dvoma signálmi pomocou vzájomnej korelácie.
    Ak je 'output_signal' viacerodimenzionálny, musí byť špecifikovaný output_column_index.

    Args:
        input_signal (np.array): Prvý signál (napr. vstup u0). Predpokladá sa, že je 1D.
        output_signal (np.array): Druhý signál (napr. výstup x0), ktorý je oneskorený vzhľadom na input_signal.
                                  Môže byť 1D alebo 2D. Ak je 2D, je potrebný output_column_index.
        sampling_rate (float): Vzorkovacia frekvencia signálov (počet vzoriek za jednotku času, napr. Hz).
        output_column_index (int, optional): Index stĺpca, ktorý sa má použiť z output_signal,
                                             ak je output_signal viacerodimenzionálny. Predvolené None.

    Returns:
        float: Odhadované oneskorenie v časových jednotkách (napr. sekundy).
        int: Odhadované oneskorenie v počte vzoriek.
        np.array: Pole s hodnotami vzájomnej korelácie.
        np.array: Pole s posunmi (lags) v počte vzoriek.
    """
    print(f"--- STARTING delay_cross_correlation CALL ---")
    print(f"Debug: Dĺžka pôvodného input_signal: {len(input_signal)}")
    print(f"Debug: Tvar pôvodného input_signal: {input_signal.shape}")
    print(f"Debug: Dĺžka pôvodného output_signal: {len(output_signal)}")
    print(f"Debug: Tvar pôvodného output_signal: {output_signal.shape}")
    print(f"Debug: Typ pôvodného input_signal: {type(input_signal)}")
    print(f"Debug: Typ pôvodného output_signal: {type(output_signal)}")
    print(f"Debug: output_column_index: {output_column_index}")

    # Ensure input_signal is 1D
    # input_signal (U) by mal byť už 1D, ak je shape (N,) alebo (N,1) a flatten() to správne upraví
    signal1_processed = np.asarray(input_signal).flatten()
    
    # Process output_signal based on its dimensions and output_column_index
    if output_signal.ndim > 1: # Ak je output_signal (N, M) kde M > 1
        if output_column_index is None:
            raise ValueError("Output signal is multi-dimensional (shape "
                             f"{output_signal.shape}). Please specify 'output_column_index' "
                             "to select which column to use for delay estimation (e.g., 0, 1, ...).")
        # Vyberie špecifický stĺpec a sploští ho na 1D
        signal2_processed = np.asarray(output_signal[:, output_column_index]).flatten()
        print(f"Debug: Použitý stĺpec {output_column_index} z output_signal.")
    else: # Ak je output_signal už 1D (shape (N,))
        signal2_processed = np.asarray(output_signal).flatten()

    # Zabezpečenie, že oba spracované signály majú rovnakú dĺžku pre koreláciu.
    # To orezanie je tu dôležité, ak by signály mali naozaj rôzne dĺžky po predchádzajúcom spracovaní
    # (čo by však nemali, ak DataLoader vráti konzistentné dáta).
    min_len = min(len(signal1_processed), len(signal2_processed))
    
    signal1_processed = signal1_processed[:min_len]
    signal2_processed = signal2_processed[:min_len]

    print(f"Debug: Minimálna dĺžka po orezaní (min_len): {min_len}")
    print(f"Debug: Dĺžka orezaného a splošteného input_signal (signal1_processed): {len(signal1_processed)}")
    print(f"Debug: Tvar orezaného a splošteného input_signal (signal1_processed.shape): {signal1_processed.shape}")
    print(f"Debug: Dĺžka orezaného a splošteného output_signal (signal2_processed): {len(signal2_processed)}")
    print(f"Debug: Tvar orezaného a splošteného output_signal (signal2_processed.shape): {signal2_processed.shape}")
    print(f"Debug: Typ signal1_processed: {type(signal1_processed)}")
    print(f"Debug: Typ signal2_processed: {type(signal2_processed)}")

    # Vypočítame vzájomnú koreláciu
    correlation = signal.correlate(signal2_processed, signal1_processed, mode='full')

    # Vygenerujeme pole posunov (lags)
    lags = signal.correlation_lags(len(signal1_processed), len(signal2_processed), mode='full')

    print(f"Debug: Dĺžka poľa korelácie (correlation): {len(correlation)}")
    print(f"Debug: Tvar poľa korelácie (correlation.shape): {correlation.shape}")
    print(f"Debug: Typ dát poľa korelácie (correlation.dtype): {correlation.dtype}")
    print(f"Debug: Obsahuje pole korelácie NaN hodnoty? {np.any(np.isnan(correlation))}")
    print(f"Debug: Obsahuje pole korelácie Inf hodnoty? {np.any(np.isinf(correlation))}")

    # Nájdeme index, kde je vzájomná korelácia maximálna
    delay_samples_idx = np.argmax(correlation)
    
    print(f"Debug: Vypočítaný index pre maximum (delay_samples_idx): {delay_samples_idx}")
    print(f"Debug: Dĺžka poľa posunov (lags): {len(lags)}")
    print(f"Debug: Tvar poľa posunov (lags.shape): {lags.shape}")

    # Získame skutočný počet vzoriek oneskorenia
    delay_samples = lags[delay_samples_idx]

    # Prevedieme oneskorenie zo vzoriek na čas
    delay_time = delay_samples / sampling_rate

    print(f"Odhadované oneskorenie: {delay_time:.2f} sekúnd ({delay_samples} vzoriek)")
    print(f"--- ENDING delay_cross_correlation CALL ---")
    #return delay_time, delay_samples, correlation, lags

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
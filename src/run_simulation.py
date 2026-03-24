import numpy as np

from simulation.dynamic_systems import DynamicSystem
from utils.plots import plot_trajectory
from utils.config_manager import ConfigManager
from utils.helpers import evaluate_simulation

from data_ingestion.data_loader import DataLoader
from data_processing.data_splitter import TimeSeriesSplitter

if __name__ == "__main__":
    config_manager = ConfigManager("config")
    config_manager.load_config("sindy_params")

    np.random.seed(42)
    random_number_generator = np.random.RandomState(42)

    with DataLoader(config_manager) as loader:
        X, U, dt = loader.load_csv_data(
            file_name="Floatshield_with_deriv",
            state_column_indices=[0, 1],
            time=0.025,
            control_input_column_indices=[3],
            verbose=False,
            plot_data=False
        )

    with TimeSeriesSplitter(config_manager, X, dt, U) as splitter:
        X_train, _, X_test,  U_train, _, U_test = splitter.split_data(
            train_ratio=0.5,
            val_ratio=0.25,
            perturb_input_signal_ratio=None,
            rng=random_number_generator,
            apply_savgol_filter=True,
            filtered_set_names = ["val", "test"],
            savgol_window_length=51,
            savgol_polyorder=2,
            plot_data=False,
            verbose=False
        )

    def ode(state_vector: np.ndarray, u0: float) -> np.ndarray:  
        x0, x1 = state_vector  

        dx0 = 1 * x1
        dx1 = -1.76549 * x0 +  5.97614 * x0 * u0**2 + -0.10254 * x0**4 + -0.03613 * (x1/x0**3) * u0**2
        

        return np.array([dx0, dx1])

    # Inicializacia systemu
    dynamic_system = DynamicSystem(config_manager, ode)

    # Simulovanie trajektorie
    trajectory, noisy_trajectory, input, time_vector = dynamic_system.simulate(dt, U_test, X_test[0])

    rmse, r2, _ = evaluate_simulation(X_test, trajectory, dt)
    print(f"Recostructed model state R2 score: {r2:.3%}")
    print(f"Recostructed model state RMSE: {rmse}")

    plot_trajectory(time_vector=time_vector, input_signal=input, trajectory=X_test, comparison_trajectory=trajectory)
    
    data = {"time": time_vector,
            "x": noisy_trajectory[:, 0],
            "y": noisy_trajectory[:, 1],
            "z": noisy_trajectory[:, 2],
            "u": input.flatten()}

    #dynamic_system.export_data(data)
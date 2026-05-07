import numpy as np

from utils.config_manager import ConfigManager
from data_ingestion.data_loader import DataLoader
from data_processing.data_splitter import TimeSeriesSplitter
from simulation.dynamic_systems import DynamicSystem
from utils.helpers import evaluate_simulation
from utils.plots import plot_trajectory

def sindy_model_reconstruction(config_manager: ConfigManager) -> DynamicSystem:

    config_manager.load_config("sindy_params")


    def ode(state_vector: np.ndarray, input_vector: np.ndarray) -> np.ndarray:  
        x0, x1, x2, x3 = state_vector
        u0 = input_vector
        epsilon = 1e-9

        dx0 = 1.00000 * x1
        dx1 = -0.68416 * x1 +  0.20241 * x0**2 + -268.49706 * x3**2 + -3.75676 * x1 * np.abs(x1) +  442.69511 * x3 * np.abs(x3) + -0.34158 * x0**3 * np.abs(x0)
        dx2 = 1.00000 * x3
        dx3 = -0.49493 * x3 +  0.32221 * u0**2 + -0.33940 * x2**4

        return np.array([dx0, dx1, dx2, dx3])

    # Inicializacia systemu
    model = DynamicSystem(config_manager, ode)

    return model

if __name__ == "__main__":
    config_manager = ConfigManager("config")
    sindy_model = sindy_model_reconstruction(config_manager)

    np.random.seed(42)
    random_number_generator = np.random.RandomState(42)

    with DataLoader(config_manager) as loader:
        X, U, dt = loader.load_csv_data(
            file_name="Floatshield_with_deriv",
            state_column_indices=[0, 1, 2, 3],
            time=0.025,
            control_input_column_indices=[4],
            verbose=False,
            plot_data=False
        )

    with TimeSeriesSplitter(config_manager, X, dt, U) as splitter:
        X_train, X_val, X_test, U_train, U_val, U_test = splitter.split_data(
            train_ratio=8750,
            val_ratio=4350,
            perturb_input_signal_ratio=None,
            rng=random_number_generator,
            apply_savgol_filter=True,
            filtered_set_names = ["val", "test"],
            savgol_window_length=51,
            savgol_polyorder=2,
            plot_data=False,
            verbose=False
        )

    X_test = X_val
    U_test = U_val

    ksteps = X_test.shape[0]

    if ksteps != X_test.shape[0]:
        trajectory_list = []
        input_sim_list = []
        time_vector_list = []

        total_simulation_steps = X_test.shape[0]

        current_index = 0
        while current_index < total_simulation_steps:
            x0_segment = X_test[current_index] # Zmenit ak chceme simulovat bez resetu

            segment_len = min(ksteps, total_simulation_steps - current_index)
            u_segment = U_test[current_index : current_index + segment_len]
            
            seg_trajectory, _, seg_input, seg_time_vector = sindy_model.simulate(dt=dt, input_signal=u_segment, initial_conditions=x0_segment)

            trajectory_list.append(seg_trajectory)
            input_sim_list.append(seg_input)
            time_vector_list.append(seg_time_vector + current_index * dt)

            current_index += segment_len

        x_sim = np.concatenate(trajectory_list, axis=0)
        input_sim = np.concatenate(input_sim_list, axis=0)
        t_sim = np.concatenate(time_vector_list, axis=0)
    else:
        x_sim, _, _, t_sim = sindy_model.simulate(dt=dt, input_signal=U_test, initial_conditions=X_test[0])

    rmse, r2 = evaluate_simulation(X_test[:, 1], x_sim[:, 1], dt)
    print(f"Recostructed model state R2 score: {r2:.3%}")
    print(f"Recostructed model state RMSE: {rmse}")

    plot_trajectory(t_sim, X_test, x_sim, input_signal=U_test, title="Validation on test data")

    data = {
        "x0": X[:, 0].flatten(),
        "x1": X[:, 1].flatten(),
        "x2": x_sim[:, 2].reshape(-1, 1).flatten(),
        "u": U[:].reshape(-1, 1).flatten(),
    }

    #sindy_model.export_data(data)
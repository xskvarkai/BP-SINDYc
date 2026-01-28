import numpy as np
from typing import Union

from simulation.dynamic_systems import DynamicSystem
from utils.vizualization import vizualize_trajectory
from data_processing.data_loader import load_config
from simulation.simulator import export_data

if __name__ == "__main__":
    def ode(state_vector: np.ndarray, input_signal: Union[float, np.ndarray]) -> np.ndarray:  
        x, y, z = state_vector  

        dx = -10 * x + 10 * y + 1 * input_signal ** 2
        dy = 28 * x - 1 * y - 1 * x * z
        dz = -1 * z + 1 * x * y

        return np.array([dx, dy, dz])

    simulation_config = load_config("simulation_params")
    # Podmienky pre simulaciu
    time_span = (simulation_config.get("time_span_start"), simulation_config.get("time_span_end")) # casovy interval
    time_step = simulation_config.get("time_step") # krok
    initial_conditions = simulation_config.get("initial_conditions") # pociatocne podmienky
    noise_ratio = simulation_config.get("noise_ratio") # * 100 sum v datach [%]: 0.02 = 2%

    num_samples = int((time_span[1] - time_span[0]) / time_step) + 1 # pocet vzoriek
    time_vector = np.linspace(time_span[0], time_span[1], num_samples) # casovy vektor

    # Inicializacia systemu
    dynamic_system = DynamicSystem(ode)

    # Simulovanie trajektorie
    trajectory, noisy_trajectory, input = dynamic_system.simulate(
        initial_state=initial_conditions,
        dt=time_step,
        num_samples=num_samples,
        is_free_body=False,
        noise_ratio=noise_ratio)

    vizualize_trajectory(time_vector=time_vector, input_signal=input, trajectory=trajectory, comparison_trajectory=noisy_trajectory)
    
    data = {"time": time_vector,
            "x": noisy_trajectory[:, 0],
            "y": noisy_trajectory[:, 1],
            "z": noisy_trajectory[:, 2],
            "u": input}

    export_data(data, "Simulacia")
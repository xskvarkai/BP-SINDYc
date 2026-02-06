import numpy as np
from typing import Union

from simulation.dynamic_systems import DynamicSystem
from utils.plots import plot_trajectory
from utils.config_manager import ConfigManager

if __name__ == "__main__":
    config_manager = ConfigManager("config")

    def ode(state_vector: np.ndarray, input_signal: Union[float, np.ndarray]) -> np.ndarray:  
        x, y, z = state_vector  

        dx = -10 * x + 10 * y + 1 * input_signal ** 2
        dy = 28 * x - 1 * y - 1 * x * z
        dz = -1 * z + 1 * x * y

        return np.array([dx, dy, dz])

    # Inicializacia systemu
    dynamic_system = DynamicSystem(ode, config_manager)

    # Simulovanie trajektorie
    trajectory, noisy_trajectory, input, time_vector = dynamic_system.simulate()

    plot_trajectory(time_vector=time_vector, input_signal=input, trajectory=trajectory, comparison_trajectory=noisy_trajectory, title=f"Data simulation")
    
    data = {"time": time_vector,
            "x": noisy_trajectory[:, 0],
            "y": noisy_trajectory[:, 1],
            "z": noisy_trajectory[:, 2],
            "u": input}

    dynamic_system.export_data(data, "Simulacia")
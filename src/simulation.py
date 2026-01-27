import numpy as np
from typing import Union

from simulation.dynamic_systems import DynamicSystem
from utils.vizualization import vizualize_trajectory
from simulation.simulator import export_data

if __name__ == "__main__":
    # Podmienky pre simulaciu
    time_step = 0.002 # krok
    time_span = (0, 100) # casovy interval
    num_samples = int((time_span[1] - time_span[0]) / time_step) + 1 # pocet vzoriek
    time_vector = np.linspace(time_span[0], time_span[1], num_samples) # casovy vektor
    initial_conditions = [-8.0, 8.0, 27.0] # pociatocne podmienky
    noise_ratio = 0.1 # * 100 sum v datach [%]: 0.02 = 2%

    def ode(state_vector: np.ndarray, input_signal: Union[float, np.ndarray]) -> np.ndarray:  
        x, y, z = state_vector  

        dx = -10 * x + 10 * y + 1 * input_signal ** 2
        dy = 28 * x - 1 * y - 1 * x * z
        dz = -1 * z + 1 * x * y

        return np.array([dx, dy, dz])

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
            "x": trajectory[:, 0],
            "y": trajectory[:, 1],
            "z": trajectory[:, 2],
            "u": input}

    export_data(data, "refaktoring/data/raw/Simulacia")
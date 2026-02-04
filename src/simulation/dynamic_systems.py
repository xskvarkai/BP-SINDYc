import numpy as np
from typing import Callable, Union, Dict, Any

from utils.config_manager import ConfigManager
from simulation.simulator import generate_input_signal, rk4_step
from utils.helpers import compute_time_vector

class DynamicSystem:  
    def __init__(self, dynamics_func: Callable[[np.ndarray, Union[float, np.ndarray]], np.ndarray], config_manager: ConfigManager):
        if not callable(dynamics_func):
            raise TypeError("The 'dynamics_func' parameter must be a callable object (function).")
        
        self._dynamics_func = dynamics_func
        self.config_manager = config_manager

        self.config_manager.load_config("simulation_params")
        self._simulation: Dict[str, Any] = self.config_manager.get_param("simulation_params.simulation")
        self._input_signal_params: Dict[str, Any] = self.config_manager.get_param("simulation_params.input_signal")

    def dynamics(self, state_vector: np.ndarray, input_signal: Union[float, np.ndarray]) -> np.ndarray:
        return self._dynamics_func(state_vector, input_signal)

    def simulate(self):
        np.random.seed(self._simulation.get("random_seed", 100))

        dt = self._simulation.get("time_step")
        num_samples = (self._simulation.get("time_span_end") - self._simulation.get("time_span_start")) // dt + 1
        num_samples = int(num_samples)
        is_free_body = self._simulation.get("is_free_body")
        noise_ratio = self._simulation.get("noise_ratio", 1e-3)
        initial_conditions = self._simulation.get("initial_conditions")

        input_signal = generate_input_signal(num_samples, is_free_body, dt, self._input_signal_params)
        state_trajectory = np.zeros((num_samples, len(initial_conditions)))

        current_state = initial_conditions.copy()

        if noise_ratio is not None:
            noise_level = max(noise_ratio * np.std(input_signal), 1e-3)
            noise = np.random.normal(0, noise_level, input_signal.shape)
            input_signal += noise

        for k in range(num_samples):
            current_input = input_signal[k]
            current_state = rk4_step(self, x_k=current_state, u_k=current_input, dt=dt)
            state_trajectory[k, :] = current_state

        if noise_ratio is not None:
            noise_level = max(noise_ratio * np.std(state_trajectory), 1e-3)
            noise = np.random.normal(0, noise_level, state_trajectory.shape)
            noisy_state_trajectory = state_trajectory + noise
            
            print(f"Pridaný šum na úrovni {noise_ratio:.1%} voči dátam")

        return state_trajectory, noisy_state_trajectory, input_signal, compute_time_vector(state_trajectory, dt)
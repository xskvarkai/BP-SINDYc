import numpy as np
import pandas as pd
from typing import Callable, Union, Dict, Any, Tuple

from utils.config_manager import ConfigManager
from simulation.simulator import generate_input_signal, rk4_step
from utils.helpers import compute_time_vector

class DynamicSystem:
    """
    Class representing a general dynamic system defined by a set of ordinary differential equations (ODEs).
    The system is defined by a dynamics function that takes the current state and input and returns
    the time derivative of the state. The class also includes a method for simulating the system's trajectory over
    """
    def __init__(self, dynamics_func: Callable[[np.ndarray, Union[float, np.ndarray]], np.ndarray], config_manager: ConfigManager):
        """
        Initializes the DynamicSystem with a given dynamics function and configuration manager.
        """
        if not callable(dynamics_func):
            raise TypeError("The 'dynamics_func' parameter must be a callable object (function).")
        
        self._dynamics_func = dynamics_func
        self.config_manager = config_manager

        self.config_manager.load_config("simulation_params")
        self._simulation: Dict[str, Any] = self.config_manager.get_param("simulation_params.simulation")
        self._input_signal_params: Dict[str, Any] = self.config_manager.get_param("simulation_params.input_signal")

    def dynamics(self, state_vector: np.ndarray, input_signal: Union[float, np.ndarray]) -> np.ndarray:
        """ Returns the time derivative of the state vector given the current state and input signal. """
        return self._dynamics_func(state_vector, input_signal)

    def simulate(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulates the trajectory of the dynamic system over a specified time span 
        with given initial conditions and input signal parameters.
        Returns the state trajectory, noisy state trajectory, input signal, and time vector.
        """
        np.random.seed(self._simulation.get("random_seed", 100))
        random_number_generator = np.random.RandomState(self._simulation.get("random_seed", 100))

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
            noise = random_number_generator.normal(0, noise_level, input_signal.shape)
            input_signal += noise

        for k in range(num_samples):
            current_input = input_signal[k]
            current_state = rk4_step(self, x_k=current_state, u_k=current_input, dt=dt)
            state_trajectory[k, :] = current_state

        if noise_ratio is not None:
            noise_level = max(noise_ratio * np.std(state_trajectory), 1e-3)
            noise = random_number_generator.normal(0, noise_level, state_trajectory.shape)
            noisy_state_trajectory = state_trajectory + noise
            
            if verbose:
                print(f"Added noise in level of the standard deviation of the state trajectory (noise level: {noise_level:.1%}).")

        return state_trajectory, noisy_state_trajectory, input_signal, compute_time_vector(state_trajectory, dt)
    
    def export_data(self, data: Dict[str, Any] = {}, file_path="raw/simulation.csv"):
        """ Exports the user defined data to a CSV file. """
        df = pd.DataFrame(data)  
        df.to_csv(file_path, index=False)
        return None
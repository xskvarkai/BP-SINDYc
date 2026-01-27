import numpy as np
from simulation.simulator import generate_input_signal, rk4_step
from typing import Optional, Callable, Union

class DynamicSystem:  
    def __init__(self, dynamics_func: Callable[[np.ndarray, Union[float, np.ndarray]], np.ndarray]):
        if not callable(dynamics_func):
            raise TypeError("The \"dynamics_func\" parameter must be a callable object (function).")
        self._dynamics_func = dynamics_func

    def dynamics(self, state_vector: np.ndarray, input_signal: Union[float, np.ndarray]) -> np.ndarray:
        return self._dynamics_func(state_vector, input_signal)

    def simulate(self, initial_state, dt, num_samples, is_free_body=True, noise_ratio=None, random_seed=None):
        np.random.seed(100 if random_seed is None else random_seed)
        input_signal = generate_input_signal(num_samples=num_samples, is_free_body=is_free_body, dt=dt)
        state_trajectory = np.zeros((num_samples, len(initial_state)))
        current_state = initial_state.copy()

        if noise_ratio is not None:
            noise_level = noise_ratio * np.std(input_signal)
            noise = np.random.normal(0, noise_level, input_signal.shape)
            input_signal += noise

        for k in range(num_samples):
            current_input = input_signal[k]
            current_state = rk4_step(self, x_k=current_state, u_k=current_input, dt=dt)
            state_trajectory[k, :] = current_state

        if noise_ratio is not None:
            noise_level = noise_ratio * np.std(state_trajectory)
            noise = np.random.normal(0, noise_level, state_trajectory.shape)
            noisy_state_trajectory = state_trajectory + noise
            
            print(f"Pridaný šum na úrovni {noise_ratio * 100} % voči dátam")

        return state_trajectory, noisy_state_trajectory, input_signal

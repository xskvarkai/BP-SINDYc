import numpy as np
import pandas as pd

# Eulerov diskretny krok:
# x_{k+1} = x_k + dt * f(x_k, u_k)
def discrete_step(dynamic_system, state_vector, input_signal, time_step):
    return state_vector + time_step * dynamic_system.dynamics(state_vector, input_signal)

# Runge-Kutta 4-teho radu pre diskretny system:
# k1 = f(x_k, u_k)
# k2 = f(x_k + 0.5 * dt * k1, u_k)
# k3 = f(x_k + 0.5 * dt * k2, u_k)
# k4 = f(x_k + dt * k3, u_k)
# x_{k+1} = x_k + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
def rk4_step(dynamic_system, state_vector, input_signal, time_step):
    k1 = dynamic_system.dynamics(state_vector, input_signal)
    k2 = dynamic_system.dynamics(state_vector + 0.5 * time_step * k1, input_signal)
    k3 = dynamic_system.dynamics(state_vector + 0.5 * time_step * k2, input_signal)
    k4 = dynamic_system.dynamics(state_vector + time_step * k3, input_signal)

    return state_vector + (time_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Generovaie AMPRBS
# Nadhodna amplituda
# PRBS - pseudo nahodny binarny signal
def generate_AMPRBS(num_samples, time_step, amplitude_range, carrier_freq, noise_level):
    time_vector = time_step * np.arange(num_samples)
    amplitude_sequence = np.random.uniform(*amplitude_range, size=num_samples)
    prbs_sequence = np.random.choice([0, 1], size=num_samples)
    carrier_wave = amplitude_sequence * np.cos(2 * np.pi * carrier_freq * time_vector)
    amprbs_signal = prbs_sequence * carrier_wave

    if noise_level > 0:
        amprbs_signal += generate_noise(0, noise_level, num_samples)

    return amprbs_signal

# Generovanie vstupneho signalu
# Moznost volnej odozvy
def generate_input_signal(num_samples, time_step, amplitude_range=(0.0, 1.0), carrier_freq=10, noise_level=0.0, is_free_body=False):
    if is_free_body:  
        return np.zeros(num_samples)
    else:   
        return generate_AMPRBS(num_samples, time_step, amplitude_range=amplitude_range, carrier_freq=carrier_freq, noise_level=noise_level)

# Generovanie sumu 
def generate_noise(level=0.1, size=1):
    return np.random.normal(0, level, size)

# Simulovanie trajektorie
def simulate(dynamic_system, input_sequence, initial_state, dt, integrator_method="RK4"):
    num_samples = len(input_sequence)
    state_trajectory = np.zeros((num_samples, len(initial_state)))
    current_state = initial_state.copy()

    for k in range(num_samples):
        current_input = input_sequence[k]
        if integrator_method == "RK4":
            current_state = rk4_step(dynamic_system, current_state, current_input, dt)
        elif integrator_method == "E":
            current_state = discrete_step(dynamic_system, current_state, current_input, dt)
        state_trajectory[k, :] = current_state

    return state_trajectory

def export_data(data={}, export_name="data"):
    df = pd.DataFrame(data)
    df.to_csv(export_name + ".csv", index=False)
    return 0
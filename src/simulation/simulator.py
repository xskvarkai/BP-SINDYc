import numpy as np

from typing import Dict, Any

# Runge-Kutta 4-teho radu pre diskretny system:   
def rk4_step(dynamic_system, x_k, u_k, dt) -> np.ndarray:
    """
    Estimate the next state x_{k+1} using the RK4 method for a given dynamic system, current state x_k, input u_k, and time step dt.
    Returns the estimated next state x_{k+1}.
    """
    k1 = dynamic_system.dynamics(x_k, u_k)
    k2 = dynamic_system.dynamics(x_k + 0.5 * dt * k1, u_k)
    k3 = dynamic_system.dynamics(x_k + 0.5 * dt * k2, u_k)
    k4 = dynamic_system.dynamics(x_k + dt * k3, u_k)

    # x_{k+1} = x_k + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_k + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Generovanie vstupneho signalu 
def generate_input_signal(num_samples, is_free_body, dt, input_signal_params: Dict[str, Any]) -> np.ndarray:
    """
    Generates an input signal for the dynamic system. If 'is_free_body' is True, the input signal will be zero (free body).
    Otherwise, it generates a signal based on a simple PID control strategy to create a more complex input.
    Returns the generated input signal.
    """
    if is_free_body:
        input_signal = np.zeros(num_samples, dtype=float)
            
    else:
        input_signal = np.zeros(num_samples, dtype=float)

        # Parametre PID simulacie
        kp, ki, kd = input_signal_params.get("kp", 2.0), input_signal_params.get("ki", 0.5), input_signal_params.get("kd", 0.1)  # konstanty regulatora
        integral = 0.0
        prev_error = 0.0
        
        # Stav systemu - fiktivny
        system_val = 0.0
        tau = 2.0
        target = 0.0

        target_change_interval_sec = input_signal_params.get("target_change_interval_sec", 10)
        target_clip_min = input_signal_params.get("target_clip_min", -15.0)
        target_clip_max = input_signal_params.get("target_clip_max", 15.0)

        for i in range(num_samples):
            # Kazdych 10 sekund zmeni pozadovanu hodnotu (nahodny skok)
            if i % int(target_change_interval_sec/dt) == 0:
                target = np.random.uniform(-10.0, 10.0)

            # Ulozenie pozadovanej hodnoty
            target = np.clip(target, target_clip_min, target_clip_max) 
            input_signal[i] = target

            # Vypocet chyby
            error = target - system_val
            
            # PID regulacia
            integral += error * dt
            derivative = (error - prev_error) / dt
            u = (kp * error) + (ki * integral) + (kd * derivative)
            
            # Aktualizacia systému
            system_val += (u - system_val) / tau * dt
            prev_error = error

    return input_signal
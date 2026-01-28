import numpy as np
import pandas as pd
from pathlib import Path

from utils import constants

# Runge-Kutta 4-teho radu pre diskretny system:   
def rk4_step(dynamic_system, x_k, u_k, dt):
    k1 = dynamic_system.dynamics(x_k, u_k)
    k2 = dynamic_system.dynamics(x_k + 0.5 * dt * k1, u_k)
    k3 = dynamic_system.dynamics(x_k + 0.5 * dt * k2, u_k)
    k4 = dynamic_system.dynamics(x_k + dt * k3, u_k)

    # x_{k+1} = x_k + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_k + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Generovanie vstupneho signalu 
def generate_input_signal(num_samples, is_free_body, dt):
    if is_free_body:
        input_signal = np.zeros(num_samples, dtype=float)
            
    else:
        input_signal = np.zeros(num_samples, dtype=float)

        # Parametre PID simulacie
        kp, ki, kd = 2.0, 0.5, 0.1  # konstanty regulatora
        integral = 0.0
        prev_error = 0.0
        
        # Stav systemu - fiktivny
        system_val = 0.0
        tau = 2.0
        target = 0.0

        for i in range(num_samples):
            # Kazdych 10 sekund zmeni pozadovanu hodnotu (nahodny skok)
            if i % int(10/dt) == 0:
                target = np.random.uniform(-10.0, 10.0)
    
            # Ulozenie pozadovanej hodnoty
            target = np.clip(target, -15.0, 15.0) 
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

def export_data(data={}, file_name="data"):
    data_dir = Path(constants.SIM_DATA_EXPORT_PATH)  
    file_path = data_dir / f"{file_name}.csv"
    df = pd.DataFrame(data)  
    df.to_csv(file_path, index=False)
    return None

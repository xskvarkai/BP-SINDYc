import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Runge-Kutta 4-teho radu pre diskretny system:   
def rk4_step(dynamic_system, x_k, u_k, dt):
    k1 = dynamic_system.dynamics(x_k, u_k)
    k2 = dynamic_system.dynamics(x_k + 0.5 * dt * k1, u_k)
    k3 = dynamic_system.dynamics(x_k + 0.5 * dt * k2, u_k)
    k4 = dynamic_system.dynamics(x_k + dt * k3, u_k)

    # x_{k+1} = x_k + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_k + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# def aprbs_generate(N, random_seed=1, minu=-1.0, maxu=1.0):   
#     rng = np.random.default_rng(seed=int(random_seed) & 0xFFFFFFFF)  
#     prbs = rng.integers(0, 2, size=N, dtype=int)  
#     aprbs = np.zeros(N, dtype=float)  
#     range_amp = abs(maxu - minu)  
#     seed = int(random_seed) if random_seed is not None else 1  

#     for i in range(N - 1):  
#         if prbs[i] == prbs[i + 1]:  
#             rng_seed = seed  
#             rng = np.random.default_rng(rng_seed)  
#             r = range_amp * rng.random()  
#             if prbs[i] == maxu:  
#                 aprbs[i] = prbs[i] - r  
#                 aprbs[i + 1] = prbs[i + 1] - r  
#             else:  
#                 aprbs[i] = prbs[i] + r  
#                 aprbs[i + 1] = prbs[i + 1] + r  
#         else:  
#             seed = (seed + 10) if seed <= (2**31) else int(random_seed)  
#             rng = np.random.default_rng(seed)  
#             r = range_amp * rng.random()  
#             if prbs[i + 1] == maxu:  
#                 aprbs[i + 1] = prbs[i + 1] - r  
#             else:  
#                 aprbs[i + 1] = prbs[i + 1] + r  

#     if N > 0:  
#         if aprbs[0] == 0.0:  
#             aprbs[0] = prbs[0] * (minu + (maxu - minu) * 0.5)  

#     return aprbs

# Generovanie vstupneho signalu 
def generate_input_signal(num_samples, is_free_body, dt, noise_ratio=None):
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

        if noise_ratio is not None:
            noise_level = noise_ratio * np.std(input_signal)
            noise = np.random.normal(0, noise_level, input_signal.shape)
            input_signal += noise

    return input_signal

def simulate(dynamic_system, initial_state, dt, num_samples, is_free_body=True, noise_ratio=None):
    input_signal = generate_input_signal(num_samples=num_samples, is_free_body=is_free_body, dt=dt, noise_ratio=noise_ratio)
    state_trajectory = np.zeros((num_samples, len(initial_state)))
    current_state = initial_state.copy()

    for k in range(num_samples):
        current_input = input_signal[k]
        current_state = rk4_step(dynamic_system=dynamic_system, x_k=current_state, u_k=current_input, dt=dt)
        state_trajectory[k, :] = current_state

    if noise_ratio is not None:
        noise_level = noise_ratio * np.std(state_trajectory)
        noise = np.random.normal(0, noise_level, state_trajectory.shape)
        state_trajectory += noise

        print(f"Pridaný šum na úrovni {noise_ratio * 100} % voči dátam")

    return state_trajectory, input_signal

def export_data(data={}, export_name="data"):
    df = pd.DataFrame(data)
    df.to_csv(export_name + ".csv", index=False)
    return 0

def vizualize_trajectory(time_vector, trajectory, comparison_trajectory=None, input=None, feature_name=["x", "y", "z"]):
    num_axes = trajectory.shape[1]
    if input is not None:
        num_axes += 1

    fig, axes = plt.subplots(num_axes, sharex=True, figsize=(15, 10))
    fig.suptitle("Simulation vizualization")

    if comparison_trajectory is not None:
        for j in range(0, trajectory.shape[1]):
            axes[j].plot(time_vector, comparison_trajectory[:, j], color="gold", label="Comparing trajectory")
            axes[j].legend()

    for i in range(0, trajectory.shape[1]):
        axes[i].plot(time_vector, trajectory[:, i], color="silver", linestyle="--", label="Base trajectory")
        axes[i].set_title("Trajectory of " + feature_name[i])
        axes[i].set_ylabel(feature_name[i])
        axes[i].legend()
        axes[i].grid(True)
    
    if input is None:
        pass
    else:
        axes[num_axes - 1].plot(time_vector, input, color="blue")
        axes[num_axes - 1].set_title("Input signal")
        axes[num_axes - 1].set_ylabel("u")
        axes[num_axes - 1].set_xlabel("Time (s)")
        axes[num_axes - 1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()

    return 0


if __name__ == "__main__":
    class DynamicSystem:  
        def __init__(self, a11=0.0, a12=0.0, a13=0.0, a21=0.0, a22=0.0, a23=0.0, a31=0.0, a32=0.0, a33=0.0, b1=0.0, b2=0.0, b3=0.0):  
            # Stavova matica systemu  
            self.a11 = a11
            self.a12 = a12
            self.a13 = a13
            self.a21 = a21
            self.a22 = a22
            self.a23 = a23
            self.a31 = a31
            self.a32 = a32
            self.a33 = a33

            # Matica vstupov systemu
            self.b1 = b1
            self.b2 = b2
            self.b3 = b3

        def dynamics(self, state_vector, input_signal):  
            x, y, z = state_vector  

            dx = self.a11 * x + self.a12 * y + self.b1 * input_signal ** 2
            dy = self.a21 * x + self.a22 * y + self.a23 * x * z
            dz = self.a31 * z + self.a32 * x * y

            return np.array([dx, dy, dz])

    # Podmienky pre simulaciu
    time_step = 0.002 # krok
    time_span = (0, 100) # casovy interval
    num_samples = int((time_span[1] - time_span[0]) / time_step) + 1 # pocet vzoriek
    time_vector = np.linspace(time_span[0], time_span[1], num_samples) # casovy vektor
    initial_conditions = [-8.0, 8.0, 27.0] # pociatocne podmienky
    noise_ratio = 0.02 # * 100 sum v datach [%]: 0.02 = 2%

    # Inicializacia systemu
    dynamic_system = DynamicSystem(
        a11=-10, a12=10, a13=0,
        a21=28, a22=-1, a23=-1,
        a31=-1, a32=1, a33=0,
        b1=1, b2=0, b3=0
    )

    # Simulovanie trajektorie
    trajectory, input = simulate(
        dynamic_system=dynamic_system,
        initial_state=initial_conditions,
        dt=time_step,
        num_samples=num_samples,
        is_free_body=False,
        noise_ratio=noise_ratio)

    vizualize_trajectory(time_vector=time_vector, input=input, trajectory=trajectory)
    
    data = {"time": time_vector,
            "x": trajectory[:, 0],
            "y": trajectory[:, 1],
            "z": trajectory[:, 2],
            "u": input}

    export_data(data, "Simulacia")
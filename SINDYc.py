import numpy as np
import matplotlib.pyplot as plt
from simulation import generate_input_signal, simulate, generate_noise, export_data

class DynamicSystem:
    def __init__(self, l=1.0, m=1.0, I = 1.0, k=1.0, b=0.0, g=9.81):
        self.l = l
        
        # Konstanty typicke pre vacsinu systemov
        self.m = m # hmotnost
        self.I = I # moment zotrvacnosti
        self.g = g # gravitacne zrychlenie
        self.b = b # tlmenie
        self.k = k # tuhost
        self.omega_n = np.sqrt(k / m) # prirodzena frekvencia

    def dynamics(self, state_vector, input_signal):
        angle, angular_velocity = state_vector
        d_angle = angular_velocity
        d_angular_velocity = (-self.g / self.l) * np.sin(angle) - (self.b / self.m * self.l**2) * angular_velocity + (self.k / self.m * self.l) * input_signal
        return np.array([d_angle, d_angular_velocity])

# Podmienky pre simulaciu
time_step = 0.1 # krok
time_span = (0, 100) # casovy interval
num_samples = int((time_span[1] - time_span[0]) / time_step) + 1 # pocet vzoriek
time_vector = np.linspace(time_span[0], time_span[1], num_samples) # casovy vektor
initial_conditions = [np.pi / 4, 0.0] # pociatocne podmienky
frequency_amprbs = 1 / (time_step * 2) # frekvencia pre AMPRBS

# Inicializacia systemu
dynamic_system = DynamicSystem(m=0.005, l=0.1535, b=4.26e-4, k=0.0293)

# Vytvorenie vstupneho signalu
input_free_body = generate_input_signal(num_samples, time_step, carrier_freq=frequency_amprbs, is_free_body=True)
input_amprbs = generate_input_signal(num_samples, time_step, carrier_freq=frequency_amprbs, is_free_body=False)

# Simulovanie trajektorie
trajectory_free_body_RK4 = simulate(dynamic_system, input_free_body, initial_conditions, time_step, integrator_method="RK4")
trajectory_free_body_Euler = simulate(dynamic_system, input_free_body, initial_conditions, time_step, integrator_method="E")
trajectory_amprbs_RK4 = simulate(dynamic_system, input_amprbs, initial_conditions, time_step, integrator_method="RK4")
trajectory_amprbs_Euler = simulate(dynamic_system, input_amprbs, initial_conditions, time_step, integrator_method="E")

# Pridanie sumu
trajectory_amprbs_RK4_noisy = trajectory_amprbs_RK4.copy()
trajectory_amprbs_RK4_noisy += generate_noise(level=0.5, size=num_samples).reshape(-1, 1)

# Priprava dat na export
data = {
    "time": time_vector,
    "input": input_amprbs,
    "angle": trajectory_amprbs_RK4_noisy[:, 0],
    "angular_speed": trajectory_amprbs_RK4_noisy[:, 0]
}

export_data(data=data, export_name="AeroShield")

# Odtialto je to od AI
# Visualization of simulation results
plt.figure(figsize=(15, 10))

# Visualization of the AMPRBS signal
plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=2)
plt.plot(np.arange(num_samples) * time_step, input_amprbs)
plt.title("AMPRBS Signal")
plt.xlabel("Time (s)")
plt.ylabel("AMPRBS Signal")
plt.grid()

# Free motion subplot
plt.subplot(3, 2, 3)
plt.plot(time_vector, trajectory_free_body_RK4[:, 0], label="RK4 - Free Conditions", color="blue")
plt.plot(time_vector, trajectory_free_body_Euler[:, 0], label="Euler - Free Conditions", linestyle="dashed", color="lightblue")
plt.title("Angle (theta) - Free Conditions")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.grid()
plt.legend()

# AMPRBS subplot
plt.subplot(3, 2, 4)
plt.plot(time_vector, trajectory_amprbs_RK4[:, 0], label="RK4 - AMPRBS", color="orange")
plt.plot(time_vector, trajectory_amprbs_Euler[:, 0], label="Euler - AMPRBS", linestyle="dashed", color="gold")
plt.title("Angle (theta) - AMPRBS")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.grid()
plt.legend()

# Velocity for free motion subplot
plt.subplot(3, 2, 5)
plt.plot(time_vector, trajectory_free_body_RK4[:, 1], label="RK4 - Free Conditions", color="blue")
plt.plot(time_vector, trajectory_free_body_Euler[:, 1], label="Euler - Free Conditions", linestyle="dashed", color="lightblue")
plt.title("Velocity (omega) - Free Conditions")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.grid()
plt.legend()

# Velocity for AMPRBS subplot
plt.subplot(3, 2, 6)
plt.plot(time_vector, trajectory_amprbs_RK4[:, 1], label="RK4 - AMPRBS", color="orange")
plt.plot(time_vector, trajectory_amprbs_Euler[:, 1], label="Euler - AMPRBS", linestyle="dashed", color="gold")
plt.title("Velocity (omega) - AMPRBS")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# Visualization of simulation results
plt.figure(figsize=(15, 10))

# Visualization of the AMPRBS signal
plt.subplot(3, 1, 1)
plt.plot(np.arange(num_samples) * time_step, input_amprbs)
plt.title("AMPRBS Signal")
plt.xlabel("Time (s)")
plt.ylabel("AMPRBS Signal")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time_vector, trajectory_amprbs_RK4[:, 0], label="RK4 - AMPRBS", linestyle="dashed", color="orange")
plt.plot(time_vector, trajectory_amprbs_RK4_noisy[:, 0], label="RK4 noisy - AMPRBS", color="gold")
plt.title("Angle (theta)")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.grid()
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time_vector, trajectory_amprbs_RK4[:, 1], label="RK4 - AMPRBS", linestyle="dashed", color="orange")
plt.plot(time_vector, trajectory_amprbs_RK4_noisy[:, 1], label="RK4 noisy - AMPRBS", color="gold")
plt.title("Velocity (omega)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
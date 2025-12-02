import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parametre simulacie
dt = 0.1 # krok
t_span = (0, 20) # casovy inteval
ns = int((t_span[1] - t_span[0]) / dt) + 1 # pocet vzoriek
t = np.linspace(t_span[0], t_span[1], ns) # casovy vektor
inits = [0.0, 0.0] # pociatocne podmienky

freq_amprbs = 1 / (dt * 2)

# Parametre modelu
m = 1.0
k = 2.0

# Dynamika modelu
def dynamics(x, u):
    x1, x2 = x
    dx1 = x2
    dx2 = (u - k * x1**3) / m
    return np.array([dx1, dx2])

# Funkcie potrebne k simulacii

# Eulerova diskretna integracia:
# x_{k+1} = x_k + dt * f(x_k, u_k)
def discrete_step(x, u, dt):
    return x + dt * dynamics(x, u)

# Runge-Kutta 4-teho radu pre diskretny system:
# k1 = f(x_k, u_k)
# k2 = f(x_k + 0.5 * dt * k1, u_k)
# k3 = f(x_k + 0.5 * dt * k2, u_k)
# k4 = f(x_k + dt * k3, u_k)
# x_{k+1} = x_k + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
def rk4_step(x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5*dt*k1, u)
    k3 = dynamics(x + 0.5*dt*k2, u)
    k4 = dynamics(x + dt*k3, u)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# AMPRBS
# Nadhodna amplituda a frekvencia nosnej vlny
# PRBS - pseudo nahodny binarny signal
def generateAMPRBS(ns, dt, amplitude, freq, noise):
    t_k = dt * np.arange(ns)
    amplitude = np.random.uniform(*amplitude, size=ns)

    prbs = np.random.choice([0, 1], size=ns)

    carrier_wave = amplitude * np.cos(2 * np.pi * freq * t_k)
    amprbs = prbs * carrier_wave

    if(noise > 0):
        amprbs += noiseSignal(0, noise, ns)

    return amprbs

# Vstupny signal
# Volna odozva alebo vstup AMPRBS
def inputSignal(ns, dt, amplitude=(0.0, 1.0), freq=10, noise=0.0, freebody=False):
    if(freebody):  
        return np.zeros(ns)
    else:   
        return generateAMPRBS(ns, dt, amplitude=amplitude, freq=freq, noise=noise)

# Sum
def noiseSignal(level=0.1, size=1):
    return np.random.normal(0, level, size)

# Simulacia s pouzitim numerickych metod
def simulate(u_seq, x, method="RK4"):
    x_traj = np.zeros((ns, 2))
    for k in range(ns):
        u = u_seq[k]
        if(method=="RK4"):
            x = rk4_step(x, u, dt)
        elif(method=="E"):
            x = discrete_step(x, u, dt)
        x_traj[k, :] = x
    return x_traj




u_free = inputSignal(ns, dt, freq=freq_amprbs, freebody=True)
u_amprbs = inputSignal(ns, dt, freq=freq_amprbs, freebody=False)

traj_free_RK4 = simulate(u_free, inits, method="RK4")
traj_free_E = simulate(u_free, inits, method="E")
traj_amprbs_RK4 = simulate(u_amprbs, inits, method="RK4")
traj_amprbs_E = simulate(u_amprbs, inits, method="E")



# Odtialto je to cisto od AI

plt.figure()
plt.plot(np.arange(ns) * dt, u_amprbs)
plt.title('AMPRBS Signal')
plt.xlabel('Time (s)')
plt.ylabel('AMPRBS Signal')
plt.grid()
plt.show()

# Vizualizácia výsledkov
plt.figure(figsize=(15, 10))

# Subgraf pre voľný pohyb
plt.subplot(2, 1, 1)
plt.plot(t, traj_free_RK4[:, 0], label='RK4 - Voľné podmienky', color='blue')
plt.plot(t, traj_free_E[:, 0], label='Euler - Voľné podmienky', linestyle='dashed', color='lightblue')
plt.title('Uhol (theta) - Voľné podmienky')
plt.xlabel('Čas (s)')
plt.ylabel('Uhol (rad)')
plt.grid()
plt.legend()

# Subgraf pre AMPRBS
plt.subplot(2, 1, 1)
plt.plot(t, traj_amprbs_RK4[:, 0], label='RK4 - AMPRBS', color='orange')
plt.plot(t, traj_amprbs_E[:, 0], label='Euler - AMPRBS', linestyle='dashed', color='gold')
plt.title('Uhol (theta) - AMPRBS')
plt.xlabel('Čas (s)')
plt.ylabel('Uhol (rad)')
plt.grid()
plt.legend()

# Subgraf pre rýchlosť pre voľný pohyb
plt.subplot(2, 1, 2)
plt.plot(t, traj_free_RK4[:, 1], label='RK4 - Voľné podmienky', color='blue')
plt.plot(t, traj_free_E[:, 1], label='Euler - Voľné podmienky', linestyle='dashed', color='lightblue')
plt.title('Rýchlosť (omega) - Voľné podmienky')
plt.xlabel('Čas (s)')
plt.ylabel('Rýchlosť (rad/s)')
plt.grid()
plt.legend()

# Subgraf pre rýchlosť pre AMPRBS
plt.subplot(2, 1, 2)
plt.plot(t, traj_amprbs_RK4[:, 1], label='RK4 - AMPRBS', color='orange')
plt.plot(t, traj_amprbs_E[:, 1], label='Euler - AMPRBS', linestyle='dashed', color='gold')
plt.title('Rýchlosť (omega) - AMPRBS')
plt.xlabel('Čas (s)')
plt.ylabel('Rýchlosť (rad/s)')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
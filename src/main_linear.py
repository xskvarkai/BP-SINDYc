from models.linearized_model import LinearizedSystem
import numpy as np
import sympy as sp

def linearized_model_main(x_bar_equilibrium: np.ndarray, u_bar_equilibrium: np.ndarray) -> LinearizedSystem:
    x0, x1 = sp.symbols('x0 x1')
    u0 = sp.symbols('u0')
    c1, c2, c3, c4, c5 = sp.symbols('c1 c2 c3 c4 c5')

    x_syms = [x0, x1]
    u_syms = [u0]

    f_sym = [c1 * x1, c2 * x1 + c3 * sp.sin(x0) + c4 * sp.sin(x1) +  c5 * u0**2]
    g_sym = [x0, x1]

    params = {c1: 1.001, c2: 1.887, c3: -80.797, c4: -3.719, c5: 16.584}

    simulator = LinearizedSystem(f_sym, g_sym, x_syms, u_syms, params)

    simulator.linearize(x_bar_equilibrium, u_bar_equilibrium)
    print(simulator.A_sym)
    print(simulator.B_sym)

    print(simulator.A)
    print(simulator.B)

    return simulator

if __name__ == "__main__":
    linear_model = linearized_model_main(np.asarray([0.2, 0.0]), np.asarray([0.9566]))
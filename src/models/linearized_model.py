import sympy as sp
import numpy as np
from typing import Callable, List, Union
from utils.helpers import rk4_integrator

class LinearizedSystem:
    def __init__(self, f_sym: List[sp.Expr], g_sym: List[sp.Expr], x_syms: List[sp.Symbol], u_syms: List[sp.Symbol], params: dict = None):
        """
        Inicializuje simulátor pre linearizovaný systém.

        Args:
            f_sym (list): Zoznam symbolických výrazov pre dynamiku systému (dx/dt = f(x, u)).
            g_sym (list): Zoznam symbolických výrazov pre výstup systému (y = g(x, u)).
            x_syms (list): Zoznam symbolických premenných stavu (e.g., [x1, x2]).
            u_syms (list): Zoznam symbolických premenných vstupu (e.g., [u1]).
            params (dict): Slovník s hodnotami systémových parametrov (napr. {g: 9.81, L: 1.0}).
        """
        self.f_sym_orig = f_sym
        self.g_sym_orig = g_sym
        self.x_syms = x_syms
        self.u_syms = u_syms
        self.params = params if params is not None else {}

        if self.params: # Add parameter values to the symbolic expressions
            self.f_sym = [expr.subs(self.params) for expr in f_sym]
            self.g_sym = [expr.subs(self.params) for expr in g_sym]
        else:
            self.f_sym = f_sym
            self.g_sym = g_sym

        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.A_sym = None
        self.B_sym = None
        self.C_sym = None
        self.D_sym = None
        self.x_bar = None
        self.u_bar = None

        self._f_vec = sp.Matrix(self.f_sym)
        self._g_vec = sp.Matrix(self.g_sym)
        self._x_vec = sp.Matrix(self.x_syms)
        self._u_vec = sp.Matrix(self.u_syms)

    def linearize(self, x_bar: np.ndarray, u_bar: np.ndarray):
        """ 
        Linearize system around the equilibrium point (x_bar, u_bar) and compute the Jacobian matrices A, B, C, D.
        Args:
            x_bar (np.ndarray): State vector at the equilibrium point.
            u_bar (np.ndarray): Input vector at the equilibrium point.   
        """
        self.x_bar = x_bar
        self.u_bar = u_bar

        # Calculate Jacobians symbolically
        self.A_sym = self._f_vec.jacobian(self._x_vec)
        self.B_sym = self._f_vec.jacobian(self._u_vec)
        self.C_sym = self._g_vec.jacobian(self._x_vec)
        self.D_sym = self._g_vec.jacobian(self._u_vec)

        subs_dict = dict(zip(self.x_syms + self.u_syms, list(x_bar) + list(u_bar))) # Add values for x_bar and u_bar to the substitution dictionary

        # Substitute the equilibrium point values into the Jacobians to get numerical matrices
        self.A = np.array(self.A_sym.subs(subs_dict)).astype(float)
        self.B = np.array(self.B_sym.subs(subs_dict)).astype(float)
        self.C = np.array(self.C_sym.subs(subs_dict)).astype(float)
        self.D = np.array(self.D_sym.subs(subs_dict)).astype(float)

    def _dxdt_linearized(self, delta_x: np.ndarray, delta_u: np.ndarray) -> np.ndarray:
        """
        Calculates the time derivative of the state deviation (delta_x) given the state deviation and input deviation using the linearized system matrices A and B.
        Args:
            delta_x (np.ndarray): Deviation of the state from the equilibrium point (x - x_bar).
            delta_u (np.ndarray): Deviation of the input from the equilibrium point (u - u_bar).
        Returns:
            np.ndarray: Time derivative of the state deviation (d(delta_x)/dt).
        """
        if self.A is None or self.B is None:
            raise ValueError("System must be linearized first. Use `linearize()`.")

        delta_x_col = delta_x.reshape(-1, 1) # Transform matrix to column vector 
        delta_u_col = delta_u.reshape(-1, 1) # Transform matrix to column vector
        
        return (self.A @ delta_x_col + self.B @ delta_u_col).flatten()

    def simulate(self, initial_x_real: np.ndarray,   
                 u_real_input: np.ndarray,   
                 dt: float, num_steps: int) -> np.ndarray:  
 
        if self.A is None or self.B is None or self.x_bar is None or self.u_bar is None:  
            raise ValueError("System must be linearized first. Use `linearize()`.")  

        x_real_trajectory = [initial_x_real]  
        current_x_real = initial_x_real  

        for i in range(num_steps - 1):   
            current_u_real = u_real_input[i]
 
            current_delta_x = current_x_real - self.x_bar  
            current_delta_u = current_u_real - self.u_bar  

            next_delta_x = rk4_integrator(current_delta_x, current_delta_u, dt, self._dxdt_linearized) # Simulate the next state deviation using RK4 integrator
             
            next_x_real = next_delta_x + self.x_bar # Transform the next state deviation back to the actual state by adding the equilibrium point
            
            x_real_trajectory.append(next_x_real)  
            current_x_real = next_x_real  

        return np.array(x_real_trajectory)
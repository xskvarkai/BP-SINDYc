from pysindy import WeakPDELibrary, CustomLibrary, FiniteDifference
from pysindy.feature_library.base import BaseFeatureLibrary
import numpy as np
from typing import Optional, Any
import inspect

# Vlastna upravena verzia WeakPDELibrary na obidenie chyby
class FixedWeakPDELibrary(WeakPDELibrary):
    def __init__(
        self,
        function_library: Optional[BaseFeatureLibrary] = None,
        derivative_order: int = 0,
        spatiotemporal_grid=None,
        include_bias: bool = False,
        include_interaction: bool = True,
        K: int = 100,
        H_xt=None,
        p: int = 4,
        num_pts_per_domain=None,
        implicit_terms: bool = False,
        multiindices=None,
        differentiation_method=FiniteDifference,
        diff_kwargs: dict = {},
        is_uniform=None,
        periodic=None,
    ):
        # Zavolajte povodny konstruktor rodicovskej triedy
        super().__init__(
            function_library=function_library,
            derivative_order=derivative_order,
            spatiotemporal_grid=spatiotemporal_grid,
            include_bias=include_bias,
            include_interaction=include_interaction,
            K=K,
            H_xt=H_xt,
            p=p,
            num_pts_per_domain=num_pts_per_domain,
            implicit_terms=implicit_terms,
            multiindices=multiindices,
            differentiation_method=differentiation_method,
            diff_kwargs=diff_kwargs,
            is_uniform=is_uniform,
            periodic=periodic,
        )
        self.is_uniform = is_uniform
        self.periodic = periodic
        self.num_pts_per_domain = num_pts_per_domain

class FixedCustomLibrary(CustomLibrary):
    def __init__(
        self,
        library_functions: Any,
        function_names: Any | None = None,
        interaction_only: bool = True,
        include_bias: bool = False
    ):
        # Zavolajte povodny konstruktor rodicovskej triedy
        super().__init__(
            library_functions=library_functions,
            function_names=function_names,
            interaction_only=interaction_only,
            include_bias=include_bias
        )
        self.library_functions = library_functions

# Funkcie
# Polynomicke
def x(x): return x
def xy(x, y): return x * y
def squared_x(x): return x ** 2
def cubed_x(x): return x ** 3
def abs_x(x): return np.abs(x)
def x_abs_x(x): return x * np.abs(x)
# Goniometricke a ine
def sin_x(x): return np.sin(x)
def cos_x(x): return np.cos(x)
def tanh_x(x): return np.tanh(x)
def signum_x(x): return np.sign(x)
# Kombinacie
def x_sin_y(x, y): return x * np.sin(y)
def x_cos_y(x, y): return x * np.cos(y)
def sin_x_cos_x(x): return np.sin(x) * np.cos(x)
def sin_x_cos_y(x, y): return np.sin(x) * np.cos(y)
def x_sin_x_cos_y(x, y): return x * np.sin(x) * np.cos(y)
def x_sin_y_cos_z(x, y, z): return x * np.sin(y) * np.cos(z)

# Mena
# Polynomicke
def name_x(x): return x
def name_xy(x, y): return x + " " + y
def name_squared_x(x): return x + "^2"
def name_cubed_x(x): return x + "^3"
def name_abs_x(x): return "|" + x + "|"
def name_x_abs_x(x): return x + " |" + x + "|"
# Goniometricke a ine
def name_sin_x(x): return "sin(" + x + ")"
def name_cos_x(x): return "cos(" + x + ")"
def name_tanh_x(x): return "tanh(" + x + ")"
def name_signum_x(x): return "sign(" + x + ")"
# Kombinacie
def name_x_sin_y(x, y): return x + " sin(" + y + ")"
def name_x_cos_y(x, y): return x + " cos(" + y + ")"
def name_sin_x_cos_x(x): return "sin(" + x + ") cos(" + x + ")"
def name_sin_x_cos_y(x, y): return "sin(" + x + ") cos(" + y + ")"
def name_x_sin_x_cos_y(x, y): return x + " sin(" + x + ") cos(" + y + ")"
def name_x_sin_y_cos_z(x, y, z): return x + " sin(" + y + ") cos(" + z + ")"
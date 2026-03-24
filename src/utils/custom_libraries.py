from pysindy import WeakPDELibrary, CustomLibrary, FiniteDifference
from pysindy.feature_library.base import BaseFeatureLibrary
from typing import Optional, Any
import numpy as np

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
        super().__init__(
            library_functions=library_functions,
            function_names=function_names,
            interaction_only=interaction_only,
            include_bias=include_bias
        )
        self.library_functions = library_functions

# ========== Callable functions ========== 

# ========== Funkcie ==========
def constant(): return 1
# Polynomicke
def x_fun(x): return x
def x_squared(x): return x**2
def x_cubed(x): return x**3
def x_quartered(x): return x**4

def x_squared_y(x, y): return x_squared(x) * x_fun(y)
def x_cubed_y(x, y): return x_cubed(x) * x_fun(y)

def yx(x, y): return x_fun(y) * x_fun(x)
def y_squared_x(x, y): return x_squared(y) * x_fun(x)
def y_cubed_x(x, y): return x_cubed(y) * x_fun(x)
def y_quartered_x(x, y): return x_quartered(y) * x_fun(x)
# Racionalne
def x_frac(x): return x**(-1)
def x_squared_frac(x): return x**(-2)
def x_cubed_frac(x): return x**(-3)
def x_quartered_frac(x): return x**(-4)

def sin_x(x): return np.sin(x)
def cos_x(x): return np.cos(x)
def x_sin_y(x, y): return x * sin_x(y)
def x_cos_y(x, y): return x * cos_x(y)
def x_sin_2y(x, y): return x * sin_x(2 * y)
def x_cos_2y(x, y): return x * cos_x(2 * y)
def x_sin_5y(x, y): return x * sin_x(5 * y)
def x_cos_5y(x, y): return x * cos_x(5 * y)
def x_sin_10y(x, y): return x * sin_x(10 * y)
def x_cos_10y(x, y): return x * cos_x(10 * y)
def x_sin_20y(x, y): return x * sin_x(20 * y)
def x_cos_20y(x, y): return x * cos_x(20 * y)

# Absolutna hodnota pre odpor vzduchu
def abs_x(x): return np.abs(x)
def x_abs_x(x): return x * np.abs(x) # x1 * |x1|
def x_y_abs_z(x, y, z): return x * y * np.abs(z) # u0 * x1 * |x1|
def x_squared_abs_y(x, y): return (x ** 2) * np.abs(y) # x1^2 * |x1| alebo u0^2 * |x1|
def exp(x): return np.exp(x)
def tanh_x(x): return np.tanh(x)

# ========== Mena funkcii ==========
def name_constant(): return "1"
# Polynomicke
def name_x_fun(x_str): return f"{x_str}"
def name_x_squared(x_str): return f"{x_str}^2"
def name_x_cubed(x_str): return f"{x_str}^3"
def name_x_quartered(x_str): return f"{x_str}^4"

def name_x_squared_y(x_str, y_str): return f"{x_str}^2 {y_str}"
def name_x_cubed_y(x_str, y_str): return f"{x_str}^3 {y_str}"

def name_yx(x_str, y_str): return f"{y_str} {x_str}"
def name_y_squared_x(x_str, y_str): return f"{y_str}^2 {x_str}"
def name_y_cubed_x(x_str, y_str): return f"{y_str}^3 {x_str}"
def name_y_quartered_x(x_str, y_str): return f"{y_str}^4 {x_str}"
#Racionalne
def name_x_frac(x_str): return f"1/{x_str}"
def name_x_squared_frac(x_str): return f"1/{x_str}^2"
def name_x_cubed_frac(x_str): return f"1/{x_str}^3"
def name_x_quartered_frac(x_str): return f"1/{x_str}^4"

def name_sin_x(x_str): return f"sin({x_str})"
def name_cos_x(x_str): return f"cos({x_str})"
def name_x_sin_y(x_str, y_str): return f"{x_str} sin({y_str})"
def name_x_cos_y(x_str, y_str): return f"{x_str} cos({y_str})"
def name_x_sin_2y(x_str, y_str): return f"{x_str} sin(2 {y_str})"
def name_x_cos_2y(x_str, y_str): return f"{x_str} cos(2 {y_str})"
def name_x_sin_5y(x_str, y_str): return f"{x_str} sin(5 {y_str})"
def name_x_cos_5y(x_str, y_str): return f"{x_str} cos(5 {y_str})"
def name_x_sin_10y(x_str, y_str): return f"{x_str} sin(10 {y_str})"
def name_x_cos_10y(x_str, y_str): return f"{x_str} cos(10 {y_str})"
def name_x_sin_20y(x_str, y_str): return f"{x_str} sin(20 {y_str})"
def name_x_cos_20y(x_str, y_str): return f"{x_str} cos(20 {y_str})"

# Absolutna hodnota
def name_abs_x(x_str): return f"|{x_str}|"
def name_x_abs_x(x_str): return f"{x_str} |{x_str}|"
def name_x_y_abs_z(x_str, y_str, z_str): return f"{x_str} {y_str} |{z_str}|"
def name_x_squared_abs_y(x_str, y_str): return f"{x_str}^2 |{y_str}|"

def name_tanh_x(x_str): return f"tanh({x_str})"

# ========== Kombinacie funkcii ==========
# Racionalne KOMBINACIE x, y
def yx_frac(x, y): return x_fun(y) * x_frac(x)
def y_squared_x_frac(x, y): return x_squared(y) * x_frac(x)
def y_cubed_x_frac(x, y): return x_cubed(y) * x_frac(x)
def y_quartered_x_frac(x, y): return x_quartered(y) * x_frac(x)

def yx_squared_frac(x, y): return x_fun(y) * x_squared_frac(x)
def y_squared_x_squared_frac(x, y): return x_squared(y) * x_squared_frac(x)
def y_cubed_x_squared_frac(x, y): return x_cubed(y) * x_squared_frac(x)
def y_quartered_x_squared_frac(x, y): return x_quartered(y) * x_squared_frac(x)

def yx_cubed_frac(x, y): return x_fun(y) * x_cubed_frac(x)
def y_squared_x_cubed_frac(x, y): return x_squared(y) * x_cubed_frac(x)
def y_cubed_x_cubed_frac(x, y): return x_cubed(y) * x_cubed_frac(x)
def y_quartered_x_cubed_frac(x, y): return x_quartered(y) * x_cubed_frac(x)

def yx_quatered_frac(x, y): return x_fun(y) * x_quartered_frac(x)
def y_squared_x_quatered_frac(x, y): return x_squared(y) * x_quartered_frac(x)
def y_cubed_x_quatered_frac(x, y): return x_cubed(y) * x_quartered_frac(x)
def y_quartered_x_quatered_frac(x, y): return x_quartered(y) * x_quartered_frac(x)

def name_yx_frac(x_str, y_str): return f"{y_str} 1/{x_str}"
def name_y_squared_x_frac(x_str, y_str): return f"{y_str}^2 1/{x_str}"
def name_y_cubed_x_frac(x_str, y_str): return f"{y_str}^3 1/{x_str}"
def name_y_quartered_x_frac(x_str, y_str): return f"{y_str}^4 1/{x_str}"

def name_yx_squared_frac(x_str, y_str): return f"{y_str} 1/{x_str}^2"
def name_y_squared_x_squared_frac(x_str, y_str): return f"{y_str}^2 1/{x_str}^2"
def name_y_cubed_x_squared_frac(x_str, y_str): return f"{y_str}^3 1/{x_str}^2"
def name_y_quartered_x_squared_frac(x_str, y_str): return f"{y_str}^4 1/{x_str}^2"

def name_yx_cubed_frac(x_str, y_str): return f"{y_str} 1/{x_str}^3"
def name_y_squared_x_cubed_frac(x_str, y_str): return f"{y_str}^2 1/{x_str}^3"
def name_y_cubed_x_cubed_frac(x_str, y_str): return f"{y_str}^3 1/{x_str}^3"
def name_y_quartered_x_cubed_frac(x_str, y_str): return f"{y_str}^4 1/{x_str}^3"

def name_yx_quartered_frac(x_str, y_str): return f"{y_str} 1/{x_str}^4"
def name_y_squared_x_quartered_frac(x_str, y_str): return f"{y_str}^2 1/{x_str}^4"
def name_y_cubed_x_quartered_frac(x_str, y_str): return f"{y_str}^3 1/{x_str}^4"
def name_y_quartered_x_quartered_frac(x_str, y_str): return f"{y_str}^4 1/{x_str}^4"

# Racionalne KOMBINACIE x, y, z
# Skupina 1: x_fun(x)
def yxz(x, y, z): return x_fun(y) * x_fun(x) * x_fun(z)
def yxz_z_squared(x, y, z): return x_fun(y) * x_fun(x) * x_squared(z)
def yxz_z_cubed(x, y, z): return x_fun(y) * x_fun(x) * x_cubed(z)
def yxz_z_quartered(x, y, z): return x_fun(y) * x_fun(x) * x_quartered(z)

def name_yxz(x_str, y_str, z_str): return f"{y_str} {x_str} {z_str}"
def name_yxz_z_squared(x_str, y_str, z_str): return f"{y_str} {x_str} {z_str}^2"
def name_yxz_z_cubed(x_str, y_str, z_str): return f"{y_str} {x_str} {z_str}^3"
def name_yxz_z_quartered(x_str, y_str, z_str): return f"{y_str} {x_str} {z_str}^4"

# Skupina 2: x_frac(x)
def yx_frac_z(x, y, z): return x_fun(y) * x_frac(x) * x_fun(z)
def yx_frac_z_squared(x, y, z): return x_fun(y) * x_frac(x) * x_squared(z)
def yx_frac_z_cubed(x, y, z): return x_fun(y) * x_frac(x) * x_cubed(z)
def yx_frac_z_quartered(x, y, z): return x_fun(y) * x_frac(x) * x_quartered(z)

def name_yx_frac_z(x_str, y_str, z_str): return f"{y_str} 1/{x_str} {z_str}"
def name_yx_frac_z_squared(x_str, y_str, z_str): return f"{y_str} 1/{x_str} {z_str}^2"
def name_yx_frac_z_cubed(x_str, y_str, z_str): return f"{y_str} 1/{x_str} {z_str}^3"
def name_yx_frac_z_quartered(x_str, y_str, z_str): return f"{y_str} 1/{x_str} {z_str}^4"

# Skupina 3: x_squared_frac(x)
def yx_squared_frac_z(x, y, z): return x_fun(y) * x_squared_frac(x) * x_fun(z)
def yx_squared_frac_z_squared(x, y, z): return x_fun(y) * x_squared_frac(x) * x_squared(z)
def yx_squared_frac_z_cubed(x, y, z): return x_fun(y) * x_squared_frac(x) * x_cubed(z)
def yx_squared_frac_z_quartered(x, y, z): return x_fun(y) * x_squared_frac(x) * x_quartered(z)

def name_yx_squared_frac_z(x_str, y_str, z_str): return f"{y_str} 1/{x_str}^2 {z_str}"
def name_yx_squared_frac_z_squared(x_str, y_str, z_str): return f"{y_str} 1/{x_str}^2 {z_str}^2"
def name_yx_squared_frac_z_cubed(x_str, y_str, z_str): return f"{y_str} 1/{x_str}^2 {z_str}^3"
def name_yx_squared_frac_z_quartered(x_str, y_str, z_str): return f"{y_str} 1/{x_str}^2 {z_str}^4"

# Skupina 4: x_cubed_frac(x)
def yx_cubed_frac_z(x, y, z): return x_fun(y) * x_cubed_frac(x) * x_fun(z)
def yx_cubed_frac_z_squared(x, y, z): return x_fun(y) * x_cubed_frac(x) * x_squared(z)
def yx_cubed_frac_z_cubed(x, y, z): return x_fun(y) * x_cubed_frac(x) * x_cubed(z)
def yx_cubed_frac_z_quartered(x, y, z): return x_fun(y) * x_cubed_frac(x) * x_quartered(z)

def name_yx_cubed_frac_z(x_str, y_str, z_str): return f"{y_str} 1/{x_str}^3 {z_str}"
def name_yx_cubed_frac_z_squared(x_str, y_str, z_str): return f"{y_str} 1/{x_str}^3 {z_str}^2"
def name_yx_cubed_frac_z_cubed(x_str, y_str, z_str): return f"{y_str} 1/{x_str}^3 {z_str}^3"
def name_yx_cubed_frac_z_quartered(x_str, y_str, z_str): return f"{y_str} 1/{x_str}^3 {z_str}^4"

# Skupina 5: x_quartered_frac(x)
def yx_quartered_frac_z(x, y, z): return x_fun(y) * x_quartered_frac(x) * x_fun(z)
def yx_quartered_frac_z_squared(x, y, z): return x_fun(y) * x_quartered_frac(x) * x_squared(z)
def yx_quartered_frac_z_cubed(x, y, z): return x_fun(y) * x_quartered_frac(x) * x_cubed(z)
def yx_quartered_frac_z_quartered(x, y, z): return x_fun(y) * x_quartered_frac(x) * x_quartered(z)

def name_yx_quartered_frac_z(x_str, y_str, z_str): return f"{y_str} 1/{x_str}^4 {z_str}"
def name_yx_quartered_frac_z_squared(x_str, y_str, z_str): return f"{y_str} 1/{x_str}^4 {z_str}^2"
def name_yx_quartered_frac_z_cubed(x_str, y_str, z_str): return f"{y_str} 1/{x_str}^4 {z_str}^3"
def name_yx_quartered_frac_z_quartered(x_str, y_str, z_str): return f"{y_str} 1/{x_str}^4 {z_str}^4"
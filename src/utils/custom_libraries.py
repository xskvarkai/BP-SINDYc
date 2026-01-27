from pysindy import WeakPDELibrary
import pysindy
from typing import Optional


# Vlastna upravena verzia WeakPDELibrary na obidenie chyby
class FixedWeakPDELibrary(WeakPDELibrary):
    def __init__(
        self,
        function_library: Optional[pysindy.feature_library.base.BaseFeatureLibrary] = None,
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
        differentiation_method=pysindy.FiniteDifference,
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

def x(x): return x
def xy(x, y): return x * y
def squared_x(x): return x ** 2
def x_name(x): return x
def xy_name(x, y): return x + "" + y
def squared_x_name(x): return x + "^2"
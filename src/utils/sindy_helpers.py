import numpy as np
import pysindy as ps
from sklearn.metrics import r2_score, root_mean_squared_error
from typing import Dict, Any, Optional, Union, Tuple, List

import warnings

from utils.custom_libraries import FixedWeakPDELibrary

def sanitize_WeakPDELibrary(config: Dict[str, Any]):
    """
    If the feature library in the config is a FixedWeakPDELibrary, 
    we need to create a new instance of it with the same parameters, 
    since the original instance is not directly usable for fitting the model.
    This function checks if the feature library is a FixedWeakPDELibrary and if so,
    creates a new instance with the same parameters.
    Returns the modified config with the sanitized feature library.
    """
    if isinstance(config.get("feature_library"), FixedWeakPDELibrary):
        params = config["feature_library"].get_params()
        config["feature_library"] = FixedWeakPDELibrary(
            function_library=params.get("function_library", None),
            derivative_order=params.get("derivative_order", 0),
            spatiotemporal_grid=np.asarray(params.get("spatiotemporal_grid")),
            K=params.get("K", 100),
            p=params.get("p", 4),
            H_xt=np.asarray(params.get("H_xt")),
            differentiation_method=config["differentiation_method"]
        )
        config["differentiation_method"] = None
        
    return config

def make_model_callable(model: ps.SINDy, data: Dict[str, Any]) -> ps.SINDy:
    """
    Create a callable SINDy model for simulation. If the model uses a FixedWeakPDELibrary, 
    we need to create a new SINDy instance with the same feature library and copy the coefficients
    from the original model, since the original model is not directly callable.
    Returns the callable SINDy model.
    """

    x_train = data.get("x_train")
    u_train = data.get("u_train")
    dt = data.get("dt")

    if isinstance(model.feature_library, FixedWeakPDELibrary):
        model_sim = ps.SINDy(
                feature_library=model.feature_library.get_params().get("function_library"),
            )
        dummy_x = x_train[0] if isinstance(x_train, list) else x_train
        dummy_u = u_train[0] if isinstance(u_train, list) else u_train
        dummy_u = dummy_u[:10] if dummy_u is not None else None
        model_sim.fit(dummy_x[:10], t=dt, u=dummy_u)
        model_sim.optimizer.coef_ = model.optimizer.coef_
    else:
        model_sim = model
        
    return model_sim

def make_model(config: Dict[str, Any], data: Dict[str, Any]) -> ps.SINDy:
    """
    Create and fit a SINDy model based on the provided configuration and data.
    The configuration should include the optimizer, feature library, and differentiation method to be used for fitting the model.
    Returns the fitted SINDy model.
    """

    model = ps.SINDy(
        optimizer=config.get("optimizer"),
        feature_library=config.get("feature_library"),
        differentiation_method=config.get("differentiation_method")
    )

    model.fit(
        x=data.get("x_train"),
        u=data.get("u_train"),
        x_dot=data.get("x_dot_train"),
        t=data.get("dt")
    )

    return model

def compute_derivative(config: Dict[str, Any], data: Dict[str, Any]) -> np.ndarray:
    """
    Copute the time derivative of the training data using the specified differentiation method in the config.
    """
    x_train = data.get("x_train")
    dt = data.get("dt")
    if isinstance(x_train, list):
        x_dot_train = [config["differentiation_method"](traj, dt) for traj in x_train]
    else:
        x_dot_train = config["differentiation_method"](x_train, dt)

    return x_dot_train

def model_simulate(
        model: ps.SINDy,
        data: Dict[str, Any],
        start_index: int,
        current_steps: int,
        integrator_kwargs: Dict[str, Any] = {"method": "LSODA","rtol": 1e-12,"atol": 1e-12}
    ) -> Union[np.ndarray, str]:
    """
    Simulate the SINDy model starting from the initial condition at start_index for current_steps time steps.
    Returns the simulated trajectory or an error message if simulation fails.
    """

    x0 = data.get("x_ref")[start_index]
    t = np.arange(current_steps) * data.get("dt")
    u = data.get("u_ref")[start_index : start_index + current_steps] if data.get("u_ref") is not None else None

    # try:
    x_sim = model.simulate(x0=x0, t=t, u=u, integrator="solve_ivp", integrator_kws=integrator_kwargs)
    return x_sim
    
    # except Exception as e:
    #     return str(e)

def filter_model(
        model: ps.SINDy,
        constraints: Dict[str, Any],
    ) -> Union[np.ndarray, str]:
    """
    Apply a series of filters to the model based on the specified constraints.
    Returns a string message if the model fails any of the filters, or None if the model passes all filters.
    """

    model_coeffs = model.coefficients()
    model_complexity = np.count_nonzero(model_coeffs)

    # FILTER 1: Zlozitost
    # Ak je poziadavka na pocet koeficientov nesplnena alebo je model trivialny
    if model_complexity == 0 or model_complexity > constraints.get("max_complexity"):
        return f"Model is trivial or exceed max complexity. Early stopped with complexity: {model_complexity}."

    # FILTER 2: Velkost koeficientov
    # Ak je poziadavka na maximalnu velkost koefficientov nesplnena alebo su Nan/Inf
    if np.max(np.abs(model_coeffs)) > constraints.get("max_coeff") or not np.all(np.isfinite(model_coeffs)):
        return "Model coeff exceed max coeff or is Inf/Nan. Early stopped due this message."

    return None # Model presiel vsetkymi filtrami

def evaluate_model(
        model: ps.SINDy,
        data: Dict[str, Any],
        start_index: int,
        current_steps: int,
        integrator_kwargs: Dict[str, Any] = {"method": "LSODA","rtol": 1e-12,"atol": 1e-12}
    ) -> Tuple[np.ndarray, float, float, float]:
    """
    Evaluate the model by simulating it and computing performance metrics (RMSE, R2 score, AIC).
    Returns the simulated trajectory, RMSE, R2 score, and AIC.
    """
    
    model_coeffs = model.coefficients()
    model_complexity = np.count_nonzero(model_coeffs)

    warnings.filterwarnings("ignore", module="pysindy")

    x_sim = model_simulate(model, data, start_index, current_steps, integrator_kwargs)
    if isinstance(x_sim, str):
        return x_sim, np.inf, -np.inf, np.inf  # Ak simulacia zlyha, vratime extremne hodnoty metrik
    
    warnings.filterwarnings("default", category=UserWarning)

    x_ref = data.get("x_ref")[start_index : start_index + current_steps]

    min_len = min(len(x_ref), len(x_sim))
    rmse = root_mean_squared_error(x_ref[:min_len], x_sim[:min_len])
    r2 = r2_score(x_ref[:min_len], x_sim[:min_len])
    
    # AIC (Akaike Information Criterion)
    aic_denominator = min_len - model_complexity - 1
    if aic_denominator <= 0:
        aic = np.inf  # Ak je vzorka prilis mala, AIC je nekonecno (model je nevhodny)
    else:
        aic = min_len * np.log(rmse ** 2) + 2 * model_complexity + 2 * model_complexity * (model_complexity + 1) /  aic_denominator # Korigovane AIC
            
    return (x_sim, rmse, r2, aic)

def model_costruction(
        config: Dict[str, Any],
        data: Dict[str, Union[np.ndarray, List[np.ndarray]]],
        random_seed: Optional[int] = 42,
        coeff_precision: Optional[int] = None
    ) -> ps.SINDy:
    
    np.random.seed(random_seed)

    if config.get("differentiation_method") is None:
        config["differentiation_method"] = config["feature_library"].get_params().get("differentiation_method")

    warnings.filterwarnings("ignore", module="pysindy")
    config = sanitize_WeakPDELibrary(config)
    model = make_model(config, data)

    if coeff_precision is not None: # Ak je definovana poziadavka na presnost koeficientov, aplikujeme ju na koeficienty modelu.
        if coeff_precision == 0:
            model.optimizer.coef_ = np.rint(model.optimizer.coef_)
        else:
            model.optimizer.coef_ = np.round(model.optimizer.coef_, decimals=coeff_precision)

    model = make_model_callable(model, data)

    return model
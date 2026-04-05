import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pykoopman

import traceback
from sklearn.metrics import r2_score, root_mean_squared_error
from typing import Dict, Any, Union, Tuple
import numpy as np

def make_model(config: Dict[str, Any], data: Dict[str, Any]) -> pykoopman.Koopman:
    """
    """

    model = pykoopman.Koopman(
        observables=config.get("observables"),
        regressor=config.get("regressor")
    )

    model.fit(
        x=data.get("x_train"),
        u=data.get("u_train"),
        y=data.get("y_train"),
        dt=data.get("dt")
    )

    return model

def model_simulate(
        model: pykoopman.Koopman,
        data: Dict[str, Any],
        start_index: int,
        current_steps: int
    ) -> Union[np.ndarray, str]:
    """
    """

    x0 = data.get("x_ref")[start_index, :]
    u = data.get("u_ref")[start_index : start_index + current_steps] if data.get("u_ref") is not None else None

    try:
        x_sim = model.simulate(x0=x0, u=u, n_steps=current_steps)
        return x_sim
    
    except Exception as e:
        print(traceback.format_exc())
        return str(e)

def evaluate_model(
        model: pykoopman.Koopman,
        data: Dict[str, Any],
        start_index: int,
        current_steps: int
    ) -> Tuple[np.ndarray, float, float]:
    """
    Evaluate the model by simulating it and computing performance metrics (RMSE, R2 score).
    Returns the simulated trajectory, RMSE, R2 score.
    """
    
    x_sim = model_simulate(model, data, start_index, current_steps)
    if isinstance(x_sim, str):
        return x_sim, np.inf, -np.inf  # Ak simulacia zlyha, vratime extremne hodnoty metrik

    x_ref = data.get("x_ref")[start_index : start_index + current_steps]

    min_len = min(len(x_ref), len(x_sim))
    rmse = root_mean_squared_error(x_ref[:min_len], x_sim[:min_len])
    r2 = r2_score(x_ref[:min_len], x_sim[:min_len])
    
    return (x_sim, rmse, r2)
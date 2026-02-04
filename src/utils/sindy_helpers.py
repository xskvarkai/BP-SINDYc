from pysindy import SINDy
import numpy as np 
from typing import Dict, Any, List, Optional

from utils.custom_libraries import FixedWeakPDELibrary

def sanitize_WeakPDELibrary(config: Dict[str, Any], time_vec: List):
    if isinstance(config.get("feature_library"), FixedWeakPDELibrary):
        params = config["feature_library"].get_params()
        
        config["feature_library"] = FixedWeakPDELibrary(
            function_library=params.get("function_library", None),
            derivative_order=params.get("derivative_order", 0),
            spatiotemporal_grid=time_vec,
            K=params.get("K", 100),
            p=params.get("p", 4),
            H_xt=np.asarray(params.get("H_xt")),
            differentiation_method=config["differentiation_method"]
        )
        config["differentiation_method"] = None
        
    return config

def make_model_callable(model: SINDy, x_train: np.ndarray|List[np.ndarray], u_train: Optional[np.ndarray|List[np.ndarray]], dt: float):
    if isinstance(model.feature_library, FixedWeakPDELibrary):
        model_sim = SINDy(
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
import json
import pickle
import hashlib
import numpy as np
from typing import List, Dict, Any

import warnings
import traceback

import utils.sindy_helpers as sindy_helpers

def run_config(configuration_and_data: List[Any]) -> Dict[str, Any]:
    try:
        
        _, config, x_train, x_val, u_train, u_val, dt, constraints = configuration_and_data

        try:
           config_bytes = json.dumps(config, sort_keys=True).encode("utf-8")
        except:
           config_bytes = pickle.dumps(config)
        random_seed = int(hashlib.sha256(config_bytes).hexdigest(), 16) % (2**32 - 1)

        # Ignorovanie warningov pocas hladania
        warnings.filterwarnings("ignore", module="pysindy")
        warnings.filterwarnings("ignore", category=UserWarning)
        numpy_old_settings = np.seterr(over="raise")

        data = {
            "x_train": x_train,
            "x_ref": x_val,
            "u_train": u_train,
            "u_ref": u_val,
            "dt": dt
        }

        config = sindy_helpers.sanitize_WeakPDELibrary(config)

        if config.get("differentiation_method") is not None:
            x_dot_train = sindy_helpers.compute_derivative(config, data)
        else:
            x_dot_train = None

        data["x_dot_train"] = x_dot_train
        model = sindy_helpers.model_costruction(config, data, random_seed, constraints.get("coeff_precision"))

        total_val_samples = x_val.shape[0]

        filter_results = sindy_helpers.filter_model(model, constraints)
        if filter_results is not None: # Ak model neprejde filtrami, vratime informaciu o zlyhani a zahodime model bez dalsich simulacii a hodnoteni.
            return {"configuration": config, "error": filter_results}
        
        # Dlhsia simulacia pre vypocet finalnych metrik.
        # Simuluje sa bud do konca validacnych dat alebo do maximalneho poctu krokov definovaneho v constraints.
        current_steps = min(total_val_samples, constraints.get("sim_steps"))
        start_index = max(0, total_val_samples - current_steps)

        x_sim, rmse, r2, aic = sindy_helpers.evaluate_model(model, data, start_index, current_steps, {"rtol": 1e-4,"atol": 1e-4})

        if isinstance(x_sim, str):
            return {"configuration": config, "error": f"Model simulation failed with error: {x_sim}"}
        
        if np.max(np.abs(x_sim)) > constraints.get("max_state") or not np.all(np.isfinite(x_sim)):
            return {"configuration": config, "error": "Model diverg too much (exceed max state) or is not stable. Stopped after long simulation."}
   
        if r2 < constraints.get("min_r2"):
            return {"configuration": config, "error": f"Model have low R2 score. Stopped after long simulation with R2 score: {r2:.3f}."}
    
        result = {
            "configuration": config,
            "random_seed": random_seed,
            "equations": model.equations(precision=constraints.get("coeff_precision") if constraints.get("coeff_precision", 3) is not None else 3),
            "r2_score": np.round(r2, 5),
            "rmse": np.round(rmse, 5),
            "complexity": np.count_nonzero(model.coefficients()),
            "aic": aic,
            "coefficients": model.optimizer.coef_
        }

        np.seterr(**numpy_old_settings)

        return result

    # Zlyhanie este pre trenovanim modelu
    except Exception as e:
        print(e)
        return {"configuration": config, "error": str(e), "traceback": traceback.format_exc()}


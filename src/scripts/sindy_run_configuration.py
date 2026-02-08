import pysindy as ps
import numpy as np
from typing import List, Dict, Any

import warnings
import traceback

import utils.sindy_helpers as sindy_helpers

def run_config(configuration_and_data: List[Any]) -> Dict[str, Any]:
    try:
        index, config, x_train, x_val, u_train, u_val, dt, constraints, cache_dict, lock = configuration_and_data
        np.random.seed(index + constraints.get("random_seed", 42))

        # Ignorovanie warningov pocas hladania
        warnings.filterwarnings("ignore", module="pysindy")
        warnings.filterwarnings("ignore", category=UserWarning)

        config = sindy_helpers.sanitize_WeakPDELibrary(config)

        # Caching derivacii (x_dot), pre rovnake parametre, pouzijeme derivacie zo zdielanej pamate.
        if config.get("differentiation_method") is not None:
            key = str(config["differentiation_method"])
            if key in cache_dict.keys():
                x_dot_train = cache_dict[key]
            else:
                with lock:
                    # Double-check locking pattern
                    if key not in cache_dict.keys():
                        x_dot_train = sindy_helpers.compute_derivative(config, data)
                        cache_dict[key] = x_dot_train
                    else:
                        x_dot_train = cache_dict[key]
        else:
            x_dot_train = None

        data = {
            "x_train": x_train,
            "x_ref": x_val,
            "x_dot_train": x_dot_train,
            "u_train": u_train,
            "u_ref": u_val,
            "dt": dt
        }

        model = sindy_helpers.make_model(config, data)

        precision: int = 3
        if constraints.get("coeff_precision") is not None: # Ak je definovana poziadavka na presnost koeficientov, aplikujeme ju na koeficienty modelu.
            precision = constraints["coeff_precision"]
            model.optimizer.coef_ = np.round(model.optimizer.coef_, decimals=precision)

        model_sim = sindy_helpers.make_model_callable(model, data)

        total_val_samples = x_val.shape[0]
        val_steps = constraints.get("min_validation_sim_steps", 20)

        # Kratka simulacia pre rychle zhodnotenie modelu a aplikovanie filterov. 
        # Ak model neprejde filtrami, nezacina sa dlha simulacia a model sa zahadzuje hned.
        current_steps = min(val_steps, total_val_samples)
        start_index = max(0, total_val_samples - current_steps)

        filter_results = sindy_helpers.filter_model(model_sim, constraints, data, start_index, current_steps, {"rtol": 1e-3,"atol": 1e-3})
        if filter_results is not None: # Ak model neprejde filtrami, vratime informaciu o zlyhani a zahodime model bez dalsich simulacii a hodnoteni.
            return {"configuration": config, "error": filter_results}
        
        # Dlhsia simulacia pre vypocet finalnych metrik.
        # Simuluje sa bud do konca validacnych dat alebo do maximalneho poctu krokov definovaneho v constraints.
        current_steps = min(total_val_samples, constraints.get("sim_steps"))
        start_index = max(0, total_val_samples - current_steps)

        _, rmse, r2, aic = sindy_helpers.evaluate_model(model_sim, data, start_index, current_steps, {"rtol": 1e-6,"atol": 1e-6})
        result = {
            "configuration": config,
            "equations": model.equations(precision=precision),
            "r2_score": np.round(r2, 5),
            "rmse": np.round(rmse, 5),
            "complexity": np.count_nonzero(model.coefficients()),
            "aic": aic,
            "random_seed": index + constraints.get("random_seed", 42),
        }
 
        return result

    # Zlyhanie este pre trenovanim modelu
    except Exception as e:
        print(e)
        return {"configuration": config, "error": str(e), "traceback": traceback.format_exc()}


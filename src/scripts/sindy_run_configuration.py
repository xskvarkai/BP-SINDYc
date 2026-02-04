import pysindy as ps
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
from typing import List, Dict, Any, Optional, Tuple
import warnings

from utils.sindy_helpers import sanitize_WeakPDELibrary, make_model_callable
from utils.helpers import compute_time_vector

def run_config(configuration_and_data: List[Any]) -> Dict[str, Any]:
    try:
        # Rozbalenie dat
        index, config, x_train, x_val, u_train, u_val, dt, constraints, cache_dict, lock = configuration_and_data

        # Fix seedu pre reprodukovatelnost v kazdom procese
        np.random.seed(index + 42)

        # Ignorovanie warningov pocas hladania
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", module="pysindy.utils")

        time_vec = compute_time_vector(x_train, dt)
        config = sanitize_WeakPDELibrary(config, time_vec)

        # Zostavenie SINDy modelu
        model = ps.SINDy(
            optimizer=config["optimizer"],
            feature_library=config["feature_library"],
            differentiation_method=config["differentiation_method"]
        )

        # Caching derivacii (x_dot), pre rovnake parametre, pouzijeme derivacie zo zdielanej pamate.
        if config["differentiation_method"] is not None:
            key = str(config["differentiation_method"])
            if key in cache_dict.keys():
                x_dot_train = cache_dict[key]
            else:
                with lock:
                    # Double-check locking pattern
                    if key not in cache_dict.keys():
                        if isinstance(x_train, list):
                            x_dot_train = [model.differentiation_method(traj, dt) for traj in x_train]
                        else:
                            x_dot_train = model.differentiation_method(x_train, dt)
                        cache_dict[key] = x_dot_train
                    else:
                        x_dot_train = cache_dict[key]
        else:
            x_dot_train = None

        # Fitting dat do modelu
        model.fit(x=x_train, u=u_train, x_dot=x_dot_train, t=dt)

        # Nastavenie presnoti koeficientov v modeli
        precision: int = 3
        if constraints.get("coeff_precision") is not None:
            precision = constraints["coeff_precision"]
            model.optimizer.coef_ = np.round(model.optimizer.coef_, decimals=precision)

        # Zistenie celkoveho poctu validacnych dat
        total_val_samples = x_val.shape[0]
        val_steps = constraints.get("min_validation_sim_steps", 20)

        model_coeffs = model.coefficients()
        model_complexity = np.count_nonzero(model_coeffs)

        # FILTER 1: Zlozitost
        # Ak je poziadavka na pocet koeficientov nesplnena alebo je model trivialny
        if model_complexity == 0 or model_complexity > constraints.get("max_complexity"):
            return {"configuration": config, "error": f"Model is trivial or exceed max complexity. Early stopped with complexity: {model_complexity}"}

        # FILTER 2: Velkost koeficientov
        # Ak je poziadavka na maximalnu velkost koefficientov nesplnena alebo su Nan/Inf
        if np.max(np.abs(model_coeffs)) > constraints.get("max_coeff") or not np.all(np.isfinite(model_coeffs)):
            return {"configuration": config, "error": "Model coeff exceed max coeff or is Inf/Nan. Early stopped due this message."}
        
        # Vsetky predchadzajuce kontroli boli splnene, takze mozeme kontrolovat, ci je model stabilny
        # Specialna vetva pre osetrenie modelu pred simulaciou (WeakPDE vyzaduje iny pristup k simulatoru)
        model_sim = make_model_callable(model, x_train, u_train, dt)

        # SHORT SIMULATION
        # Robime kratku simulaciu (val_steps).
        current_steps = min(val_steps, total_val_samples)
        start_index = max(0, total_val_samples - current_steps)

        x0 = x_val[start_index]
        t = np.arange(current_steps) * dt
        u = u_val[start_index : start_index + current_steps] if u_val is not None else None
        x_ref = x_val[start_index : start_index + current_steps]
        try:
            x_sim = model_sim.simulate(x0=x0, t=t, u=u, integrator="solve_ivp", integrator_kws={"rtol": 1e-3, "atol": 1e-3})
            min_len = min(len(x_ref), len(x_sim))

            # Ak model nepredikuje trajektoriu dostatocne presne (podla R2), zahodime ho.
            model_r2_score = r2_score(x_ref[:min_len], x_sim[:min_len])
            if model_r2_score < constraints["min_r2"]:
                return {"configuration": config, "error": f"Model have low R2 score. Early stopped with R2 score: {model_r2_score:.3f}"}
                
        # Vznikla ina neocakavana chyba
        except Exception as e:
            return {"configuration": config, "error": str(e)}

        # FINAL SIMULATION
        # Dlhsia simulacia pre vypocet finalnych metrik.
        current_steps = min(total_val_samples, constraints.get("sim_steps"))
        start_index = max(0, total_val_samples - current_steps)

        x0 = x_val[start_index]
        t = np.arange(current_steps) * dt
        u = u_val[start_index : start_index + current_steps] if u_val is not None else None
        x_ref = x_val[start_index : start_index + current_steps]
        try:
            x_sim = model_sim.simulate(x0=x0, t=t, u=u, integrator="solve_ivp", integrator_kws={"rtol": 1e-6,"atol": 1e-6})
            min_len = min(len(x_ref), len(x_sim))

            # Ak model simuluje prilis nespravne (diverguje do nekonecna)
            if np.max(np.abs(x_sim)) > constraints.get("max_state") or not np.all(np.isfinite(x_sim)):
                return {"configuration": config, "error": "Model diverg too much (exceed max state) or is not stable. Early stopped due this message."}

            # Vypocet metrik
            rmse = root_mean_squared_error(x_ref[:min_len], x_sim[:min_len])
            r2 = r2_score(x_ref[:min_len], x_sim[:min_len])
            
            # AIC (Akaike Information Criterion)
            # Aproximacia pre Gaussian noise: AIC = N * ln(MSE) + 2k
            # Pouzita korekcia AICc pre male vzorky: + (2k(k+1)) / (n - k - 1)
            aic = min_len * np.log(rmse ** 2) + 2 * model_complexity + 2 * model_complexity / (min_len - model_complexity - 1) # Korigovane AIC
            
            result = {
                "configuration": config,
                "equations": model.equations(precision=precision),
                "r2_score": np.round(r2, 5),
                "rmse": np.round(rmse, 5),
                "complexity": model_complexity,
                "aic": aic,
                "random_seed": index + 42,
            }
            return result

        except Exception as e:
            return {"configuration": config, "error": str(e)}

    # Zlyhanie este pre trenovanim modelu
    except Exception as e:
        print(e)
        return {"configuration": config, "error": str(e)}

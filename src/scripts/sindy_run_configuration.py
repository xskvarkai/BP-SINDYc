import json
import pickle
import hashlib
import numpy as np
from typing import List, Dict, Any

import warnings
import traceback

import utils.sindy_helpers as sindy_helpers

def run_config(configuration_and_data: List[Any]) -> Dict[str, Any]:
    """  
    Executes a single SINDy model configuration, including data preprocessing,  
    model construction, filtering, simulation, and evaluation.  
    This function is designed to be run in parallel processes during a  
    parameter search.  

    Args:  
        configuration_and_data (List[Any]): A list containing:  
            - _ (Any): Placeholder for an index (not used internally by run_config).  
            - config (Dict[str, Any]): The SINDy model configuration (differentiation method,  
                                        optimizer, feature library).  
            - x_train (np.ndarray): Training state variables.  
            - x_val (np.ndarray): Validation state variables.  
            - u_train (Optional[np.ndarray]): Training control inputs.  
            - u_val (Optional[np.ndarray]): Validation control inputs.  
            - dt (float): The time step of the data.  
            - constraints (Dict[str, Any]): Constraints for model filtering and evaluation,  
                                           e.g., "sim_steps", "coeff_precision", "max_state", "min_r2".  

    Returns:  
        Dict[str, Any]: A dictionary containing the evaluation results for the configuration,  
                        including the configuration itself, random seed, equations, RMSE,  
                        R2 score, complexity, AIC, or an error message if the evaluation fails.  
    """ 
        
    try:
        _, config, x_train, x_val, u_train, u_val, dt, constraints = configuration_and_data # Unpack the input arguments
        try: # Generate a consistent random seed based on the configuration for reproducibility
           config_bytes = json.dumps(config, sort_keys=True).encode("utf-8")
        except:
           config_bytes = pickle.dumps(config)
        random_seed = int(hashlib.sha256(config_bytes).hexdigest(), 16) % (2**32 - 1)

        warnings.filterwarnings("ignore", module="pysindy") # Suppress PySINDy warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        numpy_old_settings = np.seterr(over="raise") # Set NumPy error handling for numerical stability

        data = { # Prepare data dictionary for SINDy helper functions
            "x_train": x_train,
            "x_ref": x_val,
            "u_train": u_train,
            "u_ref": u_val,
            "dt": dt
        }

        config = sindy_helpers.sanitize_WeakPDELibrary(config) # Sanitize WeakPDELibrary if present in the configuration

        if config.get("differentiation_method") is not None: # Compute derivative if a differentiation method is specified
            x_dot_train = sindy_helpers.compute_derivative(config, data)
        else:
            x_dot_train = None

        data["x_dot_train"] = x_dot_train
        model = sindy_helpers.model_costruction(config, data, random_seed, constraints.get("coeff_precision")) # Construct the SINDy model

        total_val_samples = x_val.shape[0]

        filter_results = sindy_helpers.filter_model(model, constraints) # Filter the model based on predefined constraints (e.g., complexity, stability)
        if filter_results is not None: #  If the model fails filtering, return early with an error message
            return {"configuration": config, "error": filter_results}
        
        current_steps = min(total_val_samples, constraints.get("sim_steps")) # Simulate either to the end of validation data or up to the maximum steps defined in constraints 
        start_index = max(0, total_val_samples - current_steps)

        x_sim, rmse, r2, aic = sindy_helpers.evaluate_model(model, data, start_index, current_steps, {"rtol": 1e-4,"atol": 1e-4}) # Perform a longer simulation for final metric calculation and evaluate the model

        if isinstance(x_sim, str): # Check for simulation failures (e.g., numerical instability)
            return {"configuration": config, "error": f"Model simulation failed with error: {x_sim}"}
        
        if np.max(np.abs(x_sim)) > constraints.get("max_state") or not np.all(np.isfinite(x_sim)): # Check for model divergence or instability during simulation
            return {"configuration": config, "error": "Model diverg too much (exceed max state) or is not stable. Stopped after long simulation."}
   
        if r2 < constraints.get("min_r2"): # Check if the R2 score meets the minimum requirement
            return {"configuration": config, "error": f"Model have low R2 score. Stopped after long simulation with R2 score: {r2:.3f}."}
    
        result = { # If all checks pass, package the results
            "configuration": config,
            "random_seed": random_seed,
            "equations": model.equations(precision=constraints.get("coeff_precision") if constraints.get("coeff_precision", 3) is not None else 3),
            "r2_score": np.round(r2, 5),
            "rmse": np.round(rmse, 5),
            "complexity": np.count_nonzero(model.coefficients()),
            "aic": aic,
            "coefficients": model.optimizer.coef_
        }

        np.seterr(**numpy_old_settings) # Restore original NumPy error handling settings

        return result

    except Exception as e: # Catch any exceptions that occur during the configuration run (e.g., before model training)
        print(e)  # Print the error for debugging and return an error dictionary
        return {"configuration": config, "error": str(e), "traceback": traceback.format_exc()}


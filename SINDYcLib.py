import numpy as np
from sklearn.metrics import root_mean_squared_error
import pysindy as ps
import warnings

def split_data(data, val_size=0.2, test_size=0.2):

    num_samples = data.shape[0]  

    if val_size > 0:  
        val_count = int(np.floor(num_samples * val_size))  
    else:  
        val_count = 0  

    if test_size > 0:  
        test_count = int(np.floor(num_samples * test_size))  
    else:  
        test_count = 0  

    train_count = num_samples - val_count - test_count  

    data_train = data[:train_count]  
    data_val   = data[train_count:train_count + val_count] if val_count > 0 else None  
    data_test  = data[train_count + val_count:] if test_count > 0 else None  


    return data_train, data_val, data_test

def normalize_data(data_train, data_valid, data_test):
    mean = np.mean(data_train, axis=0)
    stddev = np.std(data_train, axis=0, ddof=0)

    data_train_norm = (data_train - mean) / stddev
    data_valid_norm = (data_valid - mean) / stddev
    data_test_norm = (data_test - mean) / stddev

    return data_train_norm, data_valid_norm, data_test_norm

def build_optimizers(
        regularizers=None,
        relax_coeff_nu_vals=None,
        reg_weight_lam_vals=None
):
    if reg_weight_lam_vals is None:
        reg_weight_lam_vals = np.linspace(0, 1, 11) 

    if relax_coeff_nu_vals is None: 
        relax_coeff_nu_vals = np.logspace(-1, 2, 4)

    if regularizers is None: 
        regularizers = ["L0", "L1", "L2"]
    
    optimizers = []  
    for lam in reg_weight_lam_vals:  
        for nu in relax_coeff_nu_vals:  
            for reg in regularizers:
                optimizers.append(  
                    ps.SR3(  
                        reg_weight_lam=lam,  
                        regularizer=reg,  
                        relax_coeff_nu=nu,  
                        max_iter=1000,  
                        tol=1e-10,  
                        normalize_columns=True  
                    )
                )

    return optimizers 

def build_differentiation_methods(
        n_samples=None,
        base_windows=None,
        polyorders=None
):
    def _odd_below(n):  
        n = int(n)  
        if n % 2 == 0:  
            n -= 1  
        return max(n, 5)
    if base_windows is None:
        base_windows = [5, 9]

    windows = [w for w in (_odd_below(min(w, n_samples - 1)) for w in base_windows)  
               if w < n_samples]
    
    if polyorders is None:
        polyorders = [2, 3]  
    
    diffs = []  
    for wl in windows:  
        for po in polyorders:  
            diffs.append(  
                ps.SmoothedFiniteDifference(  
                    smoother_kws={"window_length": wl, "polyorder": po}  
                )  
            )  
    #diffs.append(ps.SpectralDerivative())  

    return diffs

def _diff_method_repr(diff_method):  
    class_name = diff_method.__class__.__name__  
    
    if class_name == "SmoothedFiniteDifference":  
        smoother_kws = getattr(diff_method, "smoother_kws", {})  
        wl = smoother_kws.get("window_length", "?")  
        po = smoother_kws.get("polyorder", "?") 
        return f"SmoothedFiniteDiff(wl={wl}, po={po})"  
    
    elif class_name == "SpectralDerivative":  
        return "SpectralDerivative()"  
    
    elif class_name == "FiniteDifference":  
        order = getattr(diff_method, "order", "?")  
        return f"FiniteDifference(order={order})"  
    
    else:   
        return class_name  

def find_optimal_parameters(
        x_train, x_valid, u_train, u_valid, dt,
        optimizers=None,
        feature_libraries=None,
        differentiation_methods=None,
): 
    if differentiation_methods is None:
        differentiation_methods = build_differentiation_methods(n_samples=x_train.shape[0])

    if feature_libraries is None:
        feature_libraries = [ps.PolynomialLibrary(degree=2, include_bias=False)]

    if optimizers is None:
        optimizers = build_optimizers()

    def _library_name(lib):  
        try:  
            return lib.name  
        except AttributeError:  
            return lib.__class__.__name__

    print("Začiatok hľadania optimálnych parametrov...")  
    warnings.filterwarnings("ignore", category=UserWarning)  

    pareto_records = []

    header = f"{"Knižnica":<30} | {"Diferenciácia":<35} | {"Regularizér":<8} | {"nu_coeff":<8} | {"lambda":<8}"  
    print(header)  
    print("-" * len(header)) 

    for feature_library in feature_libraries:  
        method_pareto_records = []  

        for differentiation_method in differentiation_methods:  
            diff_name = _diff_method_repr(differentiation_method)  

            for optimizer in optimizers:  
                
                print(  
                    f"{_library_name(feature_library):<30} | {diff_name:<35} | "  
                    f"{optimizer.regularizer:<8} | {optimizer.relax_coeff_nu:<8.2f} | {optimizer.reg_weight_lam:<8.2f}",  
                    end=" : "  
                )  

                model = ps.SINDy(  
                    optimizer=optimizer,  
                    feature_library=feature_library,  
                    differentiation_method=differentiation_method  
                )  

                model.fit(  
                    x=x_train,
                    u=u_train,
                    t=dt
                )  

                x_dot_valid = model.differentiation_method(x=x_valid, t=dt)  

                rmse = root_mean_squared_error(x_dot_valid, model.predict(x=x_valid, u=u_valid))
                sparsity = np.count_nonzero(model.coefficients())  
                score = model.score(x=x_valid, u=u_valid, t=dt)  

                print(f"RMSE: {rmse:.4f}, Sparsity: {sparsity}, Score: {score:.4f}")  

                method_pareto_records.append({
                    "model": model,
                    "library": feature_library,
                    "differentiation_method": differentiation_method,
                    "optimizer": optimizer,
                    "rmse": rmse,
                    "sparsity": sparsity,
                    "score": score,
                })  

        method_pareto_front = compute_pareto_front(method_pareto_records)  
        pareto_records.extend(method_pareto_front)  

    print("\nHľadanie optimálnych parametrov dokončené.\n")  

    pareto_front = compute_pareto_front(pareto_records)  
    best_config = select_best_pareto_config(pareto_front)  

    print_config = best_config.copy()  
    print_config["library"] = _library_name(best_config["library"])  
    print_config["differentiation_method"] = _diff_method_repr(differentiation_method) 
    print(f"\nNajlepšia konfigurácia: {print_config}\n")  

    warnings.filterwarnings("always", category=UserWarning)  
    return best_config, pareto_front  

def compute_pareto_front(results):
    pareto_front = []
    for candidate in results:
        is_pareto_optimal = True
        for other in results:
            if (
                other["rmse"] <= candidate["rmse"]
                and other["sparsity"] <= candidate["sparsity"]
                and (
                    other["rmse"] < candidate["rmse"]
                    or other["sparsity"] < candidate["sparsity"]
                )
            ):
                is_pareto_optimal = False
                break
        if is_pareto_optimal:
            pareto_front.append(candidate)
    return pareto_front

def select_best_pareto_config(pareto_records):
    rmse_values = np.array([record["rmse"] for record in pareto_records], dtype=float)
    sparsity_values = np.array([record["sparsity"] for record in pareto_records], dtype=float)
    
    rmse_range = np.ptp(rmse_values) if np.ptp(rmse_values) > 0 else 1.0  
    sparsity_range = np.ptp(sparsity_values) if np.ptp(sparsity_values) > 0 else 1.0  

    normalized_rmse = (rmse_values - np.min(rmse_values)) / rmse_range  
    normalized_sparsity = (sparsity_values - np.min(sparsity_values)) / sparsity_range 

    scores = [0.6 * (1 - rmse) + 0.4 * (1 - sparsity) 
            for rmse, sparsity in zip(normalized_rmse, normalized_sparsity)]
    
    best_index = int(np.argmax(scores))
    return pareto_records[best_index]
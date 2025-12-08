import numpy as np
from sklearn.metrics import root_mean_squared_error
import pysindy as ps
import warnings
from typing import List, Tuple, Any, Dict, Optional
from scipy.signal import savgol_filter

def split_data(
        data: np.ndarray, 
        val_size: float = 0.2, 
        test_size: float = 0.0
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Splits data into training, validation, and testing sets sequentially.
    
    This function preserves the temporal order of the data. It does not shuffle the data.

    Args:
        data (np.ndarray): Input data array of shape (n_samples, n_features).
        val_size (float, optional): Proportion of the dataset to include in the 
            validation split (0.0 to 1.0). Defaults to 0.0.
        test_size (float, optional): Proportion of the dataset to include in the 
            test split (0.0 to 1.0). Defaults to 0.2.

    Returns:
        Tuple(np.ndarray, Optional[np.ndarray], Optional[np.ndarray]): 
            A tuple containing (data_train, data_val, data_test).
            Validation or test sets may be None if their sizes are 0.
    """

    # Zistenie rozmeru dat
    num_samples = data.shape[0]  

    # Pocet vzoriek pre validacnu cast, ak sa pozaduju
    if val_size > 0:  
        val_count = int(np.floor(num_samples * val_size))  
    else:  
        val_count = 0  

    # Pocet vzoriek pre testovaciu cast, ak sa pozaduju
    if test_size > 0:  
        test_count = int(np.floor(num_samples * test_size))  
    else:  
        test_count = 0  

    # Vypocet poctu vzoriek pre trenovaciu cast
    train_count = num_samples - val_count - test_count  

    # Rozdelenie podla predchazdajucich vypoctov
    data_train = data[:train_count]  
    data_val   = data[train_count:train_count + val_count] if val_count > 0 else None  
    data_test  = data[train_count + val_count:] if test_count > 0 else None  

    return data_train, data_val, data_test

def normalize_data(
        data_train: np.ndarray, 
        data_valid: np.ndarray, 
        data_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Zistenie priemeru a standardne odchylky
    mean = np.mean(data_train, axis=0)
    stddev = np.std(data_train, axis=0, ddof=0)
    
    # Ochrana pred delením 0.0  
    std = np.where(std == 0, 1.0, std) 

    # Normalizacia dat podla Z-score
    data_train_norm = (data_train - mean) / stddev
    data_valid_norm = (data_valid - mean) / stddev
    data_test_norm = (data_test - mean) / stddev

    return data_train_norm, data_valid_norm, data_test_norm,  mean, stddev

def denormalize_data(
    data_norm: np.ndarray, 
    mean: np.ndarray, 
    std: np.ndarray, 
    is_derivative: bool = False
) -> np.ndarray:
    
    if is_derivative:
        # Pre derivacie: dx/dt = std * dx_norm/dt => dx = std * dx_norm
        return data_norm * std
    else:
        # Pre stavy: x = x_norm * std + mean
        return data_norm * std + mean

def build_SR3_optimizers(
        regularizers: Optional[List[str]] = None,
        relax_coeff_nu_vals: Optional[List[float]] = None,
        reg_weight_lam_vals: Optional[List[float]] = None
) -> List[ps.SR3]:
    """
    Generates a list of SR3 optimizers with varying hyperparameters.

    It performs a grid search creation over the provided (or default) lists of
    regularizers, relaxation coefficients, and regularization weights.

    Args:
        regularizers (Optional[List[str]]): List of regularizers (e.g., ["L0", "L1"]).
            Defaults to ["L0", "L1", "L2"].
        relax_coeff_nu_vals (Optional[List[float]]): List of relaxation coefficients (nu).
            Defaults to np.logspace(-1, 2, 4).
        reg_weight_lam_vals (Optional[List[float]]): List of regularization weights (lambda).
            Defaults to [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10].

    Returns:
        List[ps.SR3]: A list of instantiated SR3 optimizer objects.
    """
    
    # Osetrenie v pripade nezadnia hodnot
    if reg_weight_lam_vals is None:
        reg_weight_lam_vals = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50] 

    if relax_coeff_nu_vals is None: 
        relax_coeff_nu_vals = np.logspace(-1, 2, 4)

    if regularizers is None: 
        regularizers = ["L0", "L1", "L2"]
    

    # Iterovanie cez vsetky a vytvorenie optimizer-u pre kazdy jeden
    optimizers = [] 
    for reg in regularizers:   
        for nu in relax_coeff_nu_vals:  
             for lam in reg_weight_lam_vals:
                optimizers.append(  
                    ps.SR3(  
                        reg_weight_lam=lam,  
                        regularizer=reg,  
                        relax_coeff_nu=nu,  
                        max_iter=10000,  
                        tol=1e-10,  
                        normalize_columns=True  
                    )
                )

    return optimizers 

def build_differentiation_methods(
        n_samples: int,
        base_windows: Optional[List[int]] = [5, 41],
        polyorders: Optional[List[int]] = [2, 3] 
) -> List[ps.SmoothedFiniteDifference]:
    """
    Constructs differentiation methods suitable for the dataset size.

    Creates SmoothedFiniteDifference objects. Ensures window lengths are odd
    and fit within the number of samples.

    Args:
        n_samples (int): Total number of samples in the training data.
        base_windows (Optional[List[int]]): Desired window lengths. Defaults to [5, 41].
        polyorders (Optional[List[int]]): Polynomial orders for smoothing. Defaults to [2, 3].

    Returns:
        List[ps.SmoothedFiniteDifference]: A list of differentiation method objects.
    """

    # Funkcia na osetrenie proti parnym cislam
    def _odd_below(n):  
        n = int(n)  
        if n % 2 == 0:  
            n -= 1  
        return max(n, 5)
    
    # Filterovanie pre okna, ktore su prilis velke
    windows = [w for w in (_odd_below(min(w, n_samples - 1)) for w in base_windows)  
               if w < n_samples]
    
    # Iterovanie cez vsetky a vytvorenie diff metody pre kazdy jeden
    diffs = []  
    for wl in windows:  
        for po in polyorders:  
            diffs.append(  
                ps.SmoothedFiniteDifference(  
                    smoother_kws={"window_length": wl, "polyorder": po, "mode": "interp"}  
                )  
            )  

    return diffs

def _diff_method_repr(diff_method: Any) -> str:
    """
    Helper function to create a readable string representation of a differentiation method.
    """
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
        x_train: np.ndarray, 
        x_valid: np.ndarray, 
        u_train: Optional[np.ndarray], 
        u_valid: Optional[np.ndarray], 
        dt: float,
        optimizers: Optional[List[Any]] = None,
        feature_libraries: Optional[List[Any]] = None,
        differentiation_methods: Optional[List[Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Performs a grid search to identify optimal SINDy model parameters (Pareto optimization).

    It iterates through combinations of feature libraries, differentiation methods,
    and optimizers, training a SINDy model for each. It evaluates models on the 
    validation set using RMSE and coefficient sparsity.

    Args:
        x_train (np.ndarray): Training state data.
        x_valid (np.ndarray): Validation state data.
        u_train (Optional[np.ndarray]): Training control inputs (if any).
        u_valid (Optional[np.ndarray]): Validation control inputs (if any).
        dt (float): Time step between samples.
        optimizers (Optional[List]): List of optimizer instances. Defaults to SR3 grid.
        feature_libraries (Optional[List]): List of feature libraries. Defaults to Poly degree 2.
        differentiation_methods (Optional[List]): List of diff methods. Defaults to auto-built.

    Returns:
        Tuple(Dict, List):
            - best_config: Dictionary containing the best model and its metadata.
            - pareto_front: List of all Pareto-optimal configurations found.
    """

    # Osetrenie v pripade nezadnia hodnot
    if differentiation_methods is None:
        differentiation_methods = build_differentiation_methods(n_samples=x_train.shape[0])

    if feature_libraries is None:
        feature_libraries = [ps.PolynomialLibrary(degree=2)]

    if optimizers is None:
        optimizers = build_SR3_optimizers()

    # Helper pre vypisovanie mena kniznice
    def _library_name(lib):  
        try:  
            return lib.name  
        except AttributeError:  
            return lib.__class__.__name__

    print("Starting parameter search...")  
    warnings.filterwarnings("ignore", category=UserWarning)  

    pareto_records = []

    header = f"{'Library':<35} | {'Differentiation method':<35} | {'Regularizer':<12} | {'nu':<8} | {'Lambda':<8}"
    print(header)  
    print("-" * len(header)) 

    # Iterovanie cez vsetky
    for feature_library in feature_libraries:  
        method_pareto_records = []  

        for differentiation_method in differentiation_methods:  
            diff_name = _diff_method_repr(differentiation_method)  

            for optimizer in optimizers:  
                
                # Vypisanie aktulanych hodnot pri aktualnom search-y
                print(  
                    f"{_library_name(feature_library):<35} | {diff_name:<35} | "  
                    f"{optimizer.regularizer:<12} | {optimizer.relax_coeff_nu:<8.3f} | {optimizer.reg_weight_lam:<8.3f}",  
                    end=" : "  
                )  

                # Vyber trenovaneho modelu
                model = ps.SINDy(  
                    optimizer=optimizer,  
                    feature_library=feature_library,  
                    differentiation_method=differentiation_method  
                )  

                # Fittovanie dat do modelu
                model.fit(  
                    x=x_train,
                    u=u_train,
                    t=dt
                )  

                # Vypocet skutocnej derivacie trajektorie za ucelom porovnia
                # Pouziva tu istu metodu, na ktorej bol trenovany
                x_dot_valid = model.differentiation_method(x=x_valid, t=dt)  

                # Porovnavacie metriky
                rmse = root_mean_squared_error(x_dot_valid, model.predict(x=x_valid, u=u_valid))
                sparsity = np.count_nonzero(model.coefficients())  
                score = model.score(x=x_valid, u=u_valid, t=dt)  

                # Vypis vysledkov (do riadku) pre kazdu iteraciu
                print(f"RMSE: {rmse:.4f}, Sparsity: {sparsity}, Score: {score:.4f}")  

                # Pridanie do zaznamu budovaneho za ucelom zostavenia Pareto fronty
                method_pareto_records.append({
                    "model": model,
                    "library": feature_library,
                    "differentiation_method": differentiation_method,
                    "optimizer": optimizer,
                    "rmse": rmse,
                    "sparsity": sparsity,
                    "score": score,
                })  

        # Vypocet lokalnej Pareto fronty pre aktualnu kniznicu/metodu
        method_pareto_front = compute_pareto_front(method_pareto_records)  
        pareto_records.extend(method_pareto_front)  

    print("\nParameter search complete.\n")  

    # Vypocet globalnej Pareto fronty cez vsetky iteracie
    # Tento vypocet sa prevadza uz na vyfiltrovanej vzorke
    # z prechadzajuceho Pareto frontu
    pareto_front = compute_pareto_front(pareto_records)  
    best_config = select_best_pareto_config(pareto_front)  

    # Ukoncenie haldanie a vypis najlepsieho modelu
    print(f"Best configuration:\n {best_config["model"]}\n")  

    warnings.filterwarnings("always", category=UserWarning)  
    return best_config, pareto_front  

def compute_pareto_front(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identifies the Pareto frontier from a set of model results.

    A model is on the Pareto front if no other model has BOTH lower RMSE 
    and lower (or equal) sparsity.

    Args:
        results (List[Dict]): List of result dictionaries containing 'rmse' and 'sparsity'.

    Returns:
        List[Dict]: Subset of results that are Pareto optimal.
    """

    # Iterovanie cez vsetky vysledky a zostavovanie kandidatov
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

def select_best_pareto_config(pareto_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Selects the single best configuration from the Pareto front using a weighted score.

    The selection is based on a weighted sum of normalized RMSE (60%) and 
    normalized Sparsity (40%).

    Args:
        pareto_records (List[Dict]): List of Pareto optimal result dictionaries.

    Returns:
        Dict: The dictionary corresponding to the best selected model configuration.
    """

    # Nacitanie vsetkych potrebnych parametrov
    rmse_values = np.array([record["rmse"] for record in pareto_records], dtype=float)
    sparsity_values = np.array([record["sparsity"] for record in pareto_records], dtype=float)
    
    # Vyhnutie sa deleniu 0.0 ak su vsetky hodnoty rovnake
    rmse_range = np.ptp(rmse_values) if np.ptp(rmse_values) > 0 else 1.0  
    sparsity_range = np.ptp(sparsity_values) if np.ptp(sparsity_values) > 0 else 1.0  

    # Normalizacia
    normalized_rmse = (rmse_values - np.min(rmse_values)) / rmse_range  
    normalized_sparsity = (sparsity_values - np.min(sparsity_values)) / sparsity_range 

    # Vazene skore: prednost (60%) ma presnost pre riedkostou (40%)
    scores = [0.6 * (1 - rmse) + 0.4 * (1 - sparsity) 
            for rmse, sparsity in zip(normalized_rmse, normalized_sparsity)]
    
    best_index = int(np.argmax(scores))
    return pareto_records[best_index]
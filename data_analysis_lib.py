import numpy as np
import pysindy as ps
import multiprocessing
import warnings
import json
import tempfile
import pickle
import gc
import os
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

from sklearn.metrics import root_mean_squared_error, r2_score
from scipy.stats import median_abs_deviation

import pywt

# ========== Pomocne funkcie pre spracovanie dat ==========

def split_data(
    x: np.ndarray,
    u: Optional[np.ndarray] = None,
    val_size: float = 0.0,
    test_size: float = 0.2
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], 
           Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:

    # Ziskanie poctu vzoriek a poctu pre validacnu a testovaciu sadu
    num_samples = x.shape[0]
    val_count = int(np.floor(num_samples * val_size)) if val_size > 0 else 0
    test_count = int(np.floor(num_samples * test_size)) if test_size > 0 else 0

    # Vypocet indexov na rozdelenie dat
    train_index = num_samples - val_count - test_count
    val_index = train_index + val_count
    test_index = num_samples

    # Rozdelenie dat na casti
    x_train = x[:train_index]
    x_val = x[train_index:val_index] if val_count > 0 else None
    x_test = x[val_index:test_index] if test_count > 0 else None

    # Rozdelenie vstupneho signalu ak existuje
    if not np.any(u) or u is None:
        u_train = u_val = u_test = None
    else:
        if u.ndim == 1:
            u = u.reshape(-1, 1)
        u_train = u[:train_index]
        u_val = u[train_index:val_index] if val_count > 0 else None
        u_test = u[val_index:test_index] if test_count > 0 else None
        
    return x_train, x_val, x_test, u_train, u_val, u_test

def generate_trajectories(
    x_train: np.ndarray,
    u_train: Optional[np.ndarray] = None,
    num_samples: int = 10000,
    num_trajectories: int = 5,
    randomseed: int = 42,
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:

    np.random.seed(randomseed)
    # Ziskanie poctu vzoriek tranovacej sady
    total_train_samples = x_train.shape[0]

    x_multi = []
    u_multi = []

    for trajectory in range(0, num_trajectories):
        if total_train_samples < num_samples:
            start_index = 0
            warnings.warn("Insufficient samples for diverse trajectories. Consider adding more data or reducing the sample size.")
        else:
            start_index = np.random.randint(0, total_train_samples - num_samples)
        
        end_index = start_index + num_samples
        trajectory = x_train[start_index:end_index]
        x_multi.append(trajectory)

        if not np.any(u_train) or u_train is None:
            u_multi = None
        else:
            input_signal = u_train[start_index:end_index]
            u_multi.append(input_signal)

    return x_multi, u_multi

def find_noise(x: np.ndarray, detail_level: int = 1) -> float:
    coeffs = pywt.wavedec(x, "haar", axis=0)
    details = coeffs[-detail_level] 
    sigma_noise = median_abs_deviation(details, scale="normal", axis=None) / np.sqrt(2)
    
    print(f"\nNoise Analysis -> Sigma: {sigma_noise:.4f}")
    return sigma_noise

def find_periodicity(x:np.ndarray, sigma_noise: float = 0.0) -> bool:
    N = len(x)
    window = np.hanning(N)
    window = window[:, np.newaxis]
    signal_centred = x - np.mean(x, axis=0) * window

    fft_spectrum = np.fft.rfft(signal_centred, axis=0)

    amplitudes = np.abs(fft_spectrum) / N * 4
    amplitudes[0, :] = 0
    
    if sigma_noise > 0:
        noise_threshold = 3.0 * sigma_noise
        mask = amplitudes > noise_threshold
        amplitudes_clean = amplitudes * mask
    else:
        amplitudes_clean = amplitudes


    power_spectrum = amplitudes_clean ** 2
    power_spectrum[0, :] = 0
    total_energy = np.sum(power_spectrum, axis=0)

    if total_energy.all() == 0:
        warnings.warn("Signal have zero energy!")
        return False

    total_energy[total_energy == 0] = 1.0
    max_peak = np.max(power_spectrum, axis=0)
    concentration = np.mean(max_peak / total_energy)
    is_periodic = True if concentration > 0.45 else False

    status = "Periodic" if is_periodic else "Aperiodic"
    print(f"\nPeriodicity Check -> Status: {status} (Concentration: {concentration:.3f})")
    return is_periodic

def estimate_threshold(
    x: np.ndarray,
    dt: float,
    u: Optional[np.ndarray] = None,
    feature_library: ps.feature_library = None,
    noise_level: Optional[float] = None,
    normalized_columns: bool = False
) -> np.ndarray:

    if feature_library is None or x is None or dt is None:
        raise ValueError(f"Data (x), time_step (dt) and feature_library are required.")

    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=0.0, alpha=1e-5, normalize_columns=True),
        feature_library=feature_library
    )

    model.fit(x=x, t=dt, u=u)
    coeffs = np.abs(model.coefficients())

    if normalized_columns:
        if u is not None:
            u_reshaped = u.reshape(-1, 1) if u.ndim == 1 else u
            x_for_lib = np.hstack((x, u_reshaped))
        else:
            x_for_lib = x

        Theta = model.feature_library.transform(x_for_lib)
        norms = np.linalg.norm(Theta, axis=0)
        norms[norms == 0] = 1.0
        coeffs = coeffs * norms
    
    coeffs_threshold = noise_level if noise_level is not None else 1e-10
    non_zero_coeffs = coeffs[coeffs > coeffs_threshold].flatten()

    if len(non_zero_coeffs) == 0:
        warnings.warn(f"All coefficients for feature_library {str(feature_library)} are nearly to zero. Returning default grid.")
        return np.logspace(-3, 1, 4)

    lower_bound = np.percentile(non_zero_coeffs, 5)
    upper_bound = np.percentile(non_zero_coeffs, 95)
    trimmed_coeffs = non_zero_coeffs[(non_zero_coeffs >= lower_bound) & (non_zero_coeffs <= upper_bound)]
    if len(trimmed_coeffs) < 2:
        trimmed_coeffs = non_zero_coeffs
    
    min_val = np.min(trimmed_coeffs)
    max_val = np.max(trimmed_coeffs)

    thresholds_non_rounded = np.logspace(np.log10(min_val), np.log10(max_val), 4)
    thresholds = [np.round(threshold, decimals=8) for threshold in thresholds_non_rounded]

    return thresholds

# ========== Trieda pre hladanie modelu ========== 
class SINDYcEstimator:
    def __init__(self):
        self.differentiation_methods = []
        self.optimizers = []
        self.feature_libraries = []
        self.configurations = []
        self.results = []
        self.pareto_front = []
        self.best_config = None
        self.results_file_name = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.delete_tempfiles()

    # Nastavenie differentiation method
    def set_differentiation_method(self, method: str, **kwargs: Any) -> "SINDYcEstimator":
        # Povolene metody
        valid_methods = {
            "FiniteDifference": ps.FiniteDifference,
            "SmoothedFiniteDifference": ps.SmoothedFiniteDifference
        }

        # Raise error-u, ak pozadovana metoda nie je povolena
        if method not in valid_methods:
            raise ValueError(f"Invalid method. Choose from {list(valid_methods.keys())}")

        # Predvolene key word argumenty
        default_kwargs = {
            "FiniteDifference": {},
            "SmoothedFiniteDifference": {
                "polyorder": 2,
                "window_length": 11,
                "mode": "interp"
            }
        }

        # Nastavenie argumentov pre pozadovanu metodu. Musia byt zadane iba tie co su v predovelnej (pripadne menej, nie viac), inac warning.
        method_kwargs = default_kwargs[method].copy()
        for key, value in kwargs.items():
            if key not in method_kwargs:
                warnings.warn(f"Unexpected parameter {key} for {method} method. Ignoring.")
            else:
                method_kwargs[key] = value

        # Ak je metoda SmoothedFiniteDifference, overit ci je window_length neparne, ak nie je + 1 (malo by byt vacsie ako polyorder)
        if method == "SmoothedFiniteDifference":
            method_kwargs["window_length"] = max(11, method_kwargs["window_length"] if method_kwargs["window_length"] % 2 != 0 else method_kwargs["window_length"] + 1)
            # Zapisanie do self
            self.differentiation_methods.append(valid_methods[method](smoother_kws=method_kwargs))
        else:
            # Zapisanie do self
            self.differentiation_methods.append(valid_methods[method](**method_kwargs))
        return self

    # Nastavenie optimalizera
    def set_optimizer(self, method: str, ensemble: bool = False, ensemble_kwargs: Optional[Dict] = None, **kwargs: Any) -> "SINDYcEstimator":
        # Povolene metody
        valid_methods = {
            "STLSQ": ps.STLSQ,
            "SR3": ps.SR3
        }

        # Raise error-u, ak pozadovana metoda nie je povolena
        if method not in valid_methods:
            raise ValueError(f"Invalid method. Choose from {list(valid_methods.keys())}")

        # Predvolene key word argumenty
        default_kwargs = {
            "STLSQ": {
                "threshold": 0.1,
                "alpha": 0.05,
                "max_iter": 1e5,
                "normalize_columns": True,
                "unbias": False
            },
            "SR3": {
                "regularizer": "L1",
                "reg_weight_lam": 0.1,
                "relax_coeff_nu": 1.0,
                "trimming_fraction": 0.0,
                "trimming_step_size": 1.0,
                "max_iter": 1e5,
                "tol": 1e-8,
                "normalize_columns": True,
                "unbias": False
            }
        }

        # Nastavenie argumentov pre pozadovanu metodu. Musia byt zadane iba tie co su v predovelnej (pripadne menej, nie viac), inac warning.
        method_kwargs = default_kwargs[method].copy()
        for key, value in kwargs.items():
            if key not in method_kwargs:
                warnings.warn(f"Unexpected parameter {key} for {method} optimizer. Ignoring.")
            else:
                method_kwargs[key] = value

        # Vzdy prepisat na integer
        method_kwargs["max_iter"] = int(method_kwargs["max_iter"])

        base_optimizer = valid_methods[method](**method_kwargs)

        if ensemble:
            default_ensemble_kwargs = {
                "bagging": True,
                "n_models": 10,
                "n_subset": None
            }

            if ensemble_kwargs:
                default_ensemble_kwargs.update(ensemble_kwargs)

            final_optimizer = ps.EnsembleOptimizer(opt=base_optimizer, **default_ensemble_kwargs)
            # Zapisanie do self
            self.optimizers.append(final_optimizer)

        else:
            # Zapisanie do self
            self.optimizers.append(base_optimizer)
        
        return self

    # Nastavenie kniznice
    def set_feature_library(self, method: str, **kwargs: Any) -> "SINDYcEstimator":

        # Povolene metody
        valid_methods = {
            "PolynomialLibrary": ps.PolynomialLibrary,
            "FourierLibrary": ps.FourierLibrary,
            "CustomLibrary": ps.CustomLibrary,
            "ConcatLibrary": ps.ConcatLibrary,
            "TensoredLibrary": ps.TensoredLibrary,
            "WeakPDELibrary": ps.WeakPDELibrary
        }

        # Raise error-u, ak pozadovana metoda nie je povolena
        if method not in valid_methods:
            raise ValueError(f"Invalid method. Choose from {list(valid_methods.keys())}")

        # Predvolene key word argumenty
        default_kwargs = {
            "PolynomialLibrary": {
                "degree": 2,
                "include_interaction": True,
                "interaction_only": False,
                "include_bias": True 
            },
            "FourierLibrary": {
                "n_frequencies": 1,
                "include_sin": True,
                "include_cos": True
            },
            "CustomLibrary": {
                "library_functions": [],
                "function_names": [],
                "interaction_only": False,
                "include_bias": True 
            },
            "ConcatLibrary": {
                "libraries": []
            },
            "TensoredLibrary": {
                "libraries": []
            },
            "WeakPDELibrary": {
                "function_library": None,
                "derivative_order": 0,
                "spatiotemporal_grid": None,
                "K": 100,
                "p": 4
            }
        }
        
        # Nastavenie argumentov pre pozadovanu metodu. Musia byt zadane iba tie co su v predovelnej (pripadne menej, nie viac), inac warning.
        method_kwargs = default_kwargs[method].copy()
        for key, value in kwargs.items():
            if key not in method_kwargs:
                warnings.warn(f"Unexpected parameter {key} for feature library. Ignoring.")
            else:
                method_kwargs[key] = value

        # Zapisanie do self
        self.feature_libraries.append(valid_methods[method](**method_kwargs))
        return self

    # Generovanie konfiguracii
    def generate_configurations(self) -> "SINDYcEstimator":
        # Pripad, ked nastavene poziadavky
        if not self.differentiation_methods:
            raise ValueError("No differentiation methods defined. Use set_differentiation_method() first.")
        if not self.optimizers:
            raise ValueError("No optimizers defined. Use set_optimizers() first.")
        if not self.feature_libraries:
            raise ValueError("No feature libraries defined. Use set_feature_library() first.")

        # Iterovanie cez vsetky moznosti a vytvorenie konfiguracie pre kazdu jednu
        configurations = []
        for feature_library in self.feature_libraries:
            for optimizer in self.optimizers:
                for differentiation_method in self.differentiation_methods:
                    configurations.append({
                        "differentiation_method": differentiation_method,
                        "optimizer": optimizer,
                        "feature_library": feature_library
                    })

        # Zapisanie do self
        self.configurations = configurations
        self.differentiation_methods.clear()
        self.optimizers.clear()
        self.feature_libraries.clear()

        return self

    # Paralelne hladanie najlepsej konfiguracie
    def search_configurations(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        u_train: Optional[np.ndarray] = None,
        u_val: Optional[np.ndarray] = None,
        dt: float = 0.01,
        n_processes: int = 4,
        **constraints: Any
    ) -> "SINDYcEstimator":

        # Predvolene obmedzenie (poziadavky na model)
        default_constraints = {
            "sim_steps": 350,
            "coeff_precision": None,
            "max_complexity": 50,
            "max_coeff": 1e2,
            "min_r2": 0.7,
            "max_state": 1e3
        }
        # Ak su poziadavky ine, update poziadaviek
        default_constraints.update(constraints)

        # Zistenie celkoveho poctu validacnych dat
        total_val_samples = x_val.shape[0]

        # Raise error-u, ak neboli vygenerovane konfiguracie
        if not self.configurations:
            raise ValueError("No configurations defined. Use generate_configurations() first.")

        # Raise warning-u, ak je malo validacnych krokov a zmena na 11
        if default_constraints["sim_steps"] <= 20:
            default_constraints["sim_steps"] = 21
            warnings.warn(f"Minimum required simulation steps are 20, validation steps increased automatically to 21")

        # Raise error-u, ak nie je dostatocne vela validacnych dat
        if total_val_samples < default_constraints["sim_steps"]:
            raise ValueError(f"Not enough validation samples. Increase validation steps to {default_constraints["sim_steps"]} or validation size")

        with multiprocessing.Manager() as manager:
            cache_dict = manager.dict()
            lock = manager.Lock()

            # Zbalenie dat a konfiguracie, kvoli multiprocessingu
            configurations_and_data = [
                    (index, config, x_train, x_val, u_train, u_val, dt, default_constraints, cache_dict, lock)
                    for index, config in enumerate(self.configurations)
                ]

            self.configurations.clear()

            # Zistenie celkoveho poctu konfiguracii
            total_configurations = len(configurations_and_data)

            # UI/UX
            start_time = datetime.now()
            print(f"\nParameter search started...")
            print(f"Total configurations to explore: {total_configurations}")
            print(f"Using {n_processes} parallel processes")
            print(f"Start time: {start_time.strftime("%H:%M:%S")}")

            # Ak je definovana CustomLibrary, pepisat na pri exporte na string
            # Bol problem pri exporte kniznice a tym aj s reprodukovanim modelu
            def sanitize_input(record):
                try:
                    if isinstance(record.get("configuration")["feature_library"], ps.CustomLibrary):
                        record.get("configuration")["feature_library"] = "CustomLibrary"        
                except Exception:
                    pass

            # Otvorenie docasnych suborov na zapis konfiguracii
            with tempfile.NamedTemporaryFile(delete=False, mode="wb", prefix="sindyR", suffix=".pkl") as results_file:

                # Ziskanie ciest k docasnym suborom - umoznuje pracu s nimi
                self.results_file_name = results_file.name

                # Multiprocessing
                with multiprocessing.Pool(processes=n_processes) as pool:
                    for index, result in enumerate(pool.imap(run_config, configurations_and_data), 1):
                        if result is not None:
                            sanitize_input(result)
                            result["index"] = index
                            pickle.dump(result, results_file)
                        gc.collect()
                        # UI/UX
                        print(f"Processing configuration {index}/{total_configurations} ({(index/total_configurations)*100:.2f}%)", end="\r", flush=True)
                    print()

        # Znovuzapnutie warningov a vycistenie nepotrebnych dat
        warnings.filterwarnings("default", category=UserWarning)
        configurations_and_data.clear()

        # Nacitanie vysledkov do zaznamov
        with open(self.results_file_name, "rb") as f:
            try:
                while True:
                    self.results.append(pickle.load(f))
            except EOFError:
                pass

        # UI/UX
        print("\nParameter search complete.")
        duraction = datetime.now() - start_time
        seconds = duraction.total_seconds()
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"The process took {int(hours):02}:{int(minutes):02}:{int(seconds):02} hours")
        valid_configs = sum(1 for result in self.results if result is not None)
        print(f"Valid configurations found: {valid_configs} out of {total_configurations}")

        # Ak existuje aspon jeden zaznam
        if self.results:
            # Najdi najlepsi vysledok
            self.best_config = self._select_best_config(self.results)

        return self

    # Zostavenie pareto fronty
    def _compute_pareto_front(self, results: List[Dict]) -> List[Dict]:
        # Nacitanie iba validnych (not None) zaznamov
        valid_results = [result for result in results if result is not None]

        # Warning, ak neexistuje ani jeden validny zaznam
        if not valid_results:
            warnings.warn("No valid configurations found. All configurations were filtered out.")
            return None

        # Zoradenie od najnizsieho po najvyssie podla rmse
        sorted_results = sorted(valid_results, key=lambda x: x["rmse"])

        # Definicia a priradienie najlespieho rmse do pareto frontu
        pareto_front = [sorted_results[0]]
        # Kazdeho kandidata, ktory je jednoduchsi ako ten z najlepsim rmse prirad do pareto fronty
        for candidate in sorted_results[1:]:
            if candidate["complexity"] < pareto_front[-1]["complexity"]:
                    pareto_front.append(candidate)

        return pareto_front

    # Vyber najlepsieho modelu z pareto zaznamov (fronty)
    def _select_best_config(self, records: List[Dict]) -> Dict[str, Any]:
        sorted_results = sorted(records, key=lambda x: x["aic"])
        best_model = sorted_results[0]
        
        return best_model

    # Zobrazenie pareto fronty v grafe
    def plot_pareto(self)  -> "SINDYcEstimator":
        self.pareto_front = self._compute_pareto_front(self.results)
        self.results.clear()
        if self.pareto_front is None:
            return self

        # Nacitanie rmse a riedkosti
        errs = np.array([r["rmse"] for r in self.pareto_front], dtype=float)
        spars = np.array([r["complexity"] for r in self.pareto_front], dtype=float)

        # Vykreslenie
        plt.figure(figsize=(8, 5))
        plt.scatter(errs, spars, color="tab:blue", label="Pareto body")
        plt.xlabel("RMSE")
        plt.ylabel("Complexity (počet nenulových koeficientov)")
        plt.title("Pareto front")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return self

    # Export dat
    def export_data(self, data: dict = None, path: str = "data.json") -> "SINDYcEstimator":
        # Nacitanie docasneho suboru a dat pre export
        def read_temp_file(path):
            with open(path, "rb") as f:
                data = []
                try:
                    while True:
                        data.append(pickle.load(f))
                except EOFError:
                    pass
            return data
        
        # Data pre export
        all_results = read_temp_file(self.results_file_name)
        
        if self.pareto_front is None:
            warnings.warn("Pareto front is None")
        if self.best_config is None:
            warnings.warn("Best configuration is None")

        payload = {
            "best_result": self.best_config,
            "pareto_front": self.pareto_front,
            "all_results": all_results,
            "user_data": data
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=5, default=str)
        except Exception as e:
            warnings.warn(e)

        return self

    # Odstranenie docasnych suborov
    def delete_tempfiles(self):
        if self.results_file_name and os.path.exists(self.results_file_name):
            os.remove(self.results_file_name)
            
        self.results_file_name = None
        return self

# ========== Spustanie konfiguracie na kazdom procesore ========== 
def run_config(configuration_and_data: List[Dict[str, Any], np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], Optional[str]]:  
    try:
        # Rozbalenie dat
        index, config, x_train, x_val, u_train, u_val, dt, constraints, cache_dict, lock = configuration_and_data

        np.random.seed(index + 42)

        # Ignorovanie warningov pocas hladania
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", module="pysindy.utils")

        if isinstance(config["feature_library"], ps.WeakPDELibrary):
            params = config["feature_library"].get_params()

            if isinstance(x_train, list):
                time_vec = (np.arange(x_train[0].shape[0]) * dt)
            else:
                time_vec = (np.arange(x_train.shape[0]) * dt)
            
            config["feature_library"] = ps.WeakPDELibrary(
                function_library=params.get('function_library', None),
                derivative_order=params.get('derivative_order', 0),
                spatiotemporal_grid=time_vec,
                K=params.get('K', 100),
                p=params.get('p', 4),
                differentiation_method=config["differentiation_method"]
            )

            config["differentiation_method"] = None

        # Zostavenie modelu
        model = ps.SINDy(
            optimizer=config["optimizer"],
            feature_library=config["feature_library"],
            differentiation_method=config["differentiation_method"]
        )

        if config["differentiation_method"] is not None:
            key = str(config["differentiation_method"])
            if key in cache_dict.keys():
                x_dot_train, x_dot_val = cache_dict[key]
            else:
                with lock:
                    if key not in cache_dict.keys():
                        if isinstance(x_train, list):
                            x_dot_train = [model.differentiation_method(traj, dt) for traj in x_train]
                        else:
                            x_dot_train = model.differentiation_method(x_train, dt)
                        x_dot_val = model.differentiation_method(x_val, dt)
                        cache_dict[key] = (x_dot_train, x_dot_val)
                    else:
                        x_dot_train, x_dot_val = cache_dict[key]
        else:
            x_dot_train, x_dot_val = None, None

        # Fitting dat do modelu
        model.fit(
            x=x_train,
            u=u_train,
            x_dot=x_dot_train,
            t=dt
        )

        # Nastavenie presnoti koeficientov v modeli
        precision = 3
        if constraints.get("coeff_precision") is not None:
            precision = constraints["coeff_precision"]
            model.optimizer.coef_ = np.round(model.optimizer.coef_, decimals=constraints["coeff_precision"])

        # Zistenie celkoveho poctu validacnych dat
        total_val_samples = x_val.shape[0]
        # Pocet krokov na rychlu validaciu
        val_steps = 20
        # Poznamka: bola by vhodna kontrola ci je total_val_samples > val_steps

        model_coeffs = model.coefficients()
        model_complexity = np.count_nonzero(model_coeffs)

        # Ak je poziadavka na pocet koeficientov nesplnena alebo je model trivialny
        if model_complexity == 0 or model_complexity > constraints["max_complexity"]:
            return None

        # Ak je poziadavka na maximalnu velkost koefficientov nesplnena alebo su Nan/Inf
        if np.max(np.abs(model_coeffs)) > constraints["max_coeff"] or not np.all(np.isfinite(model_coeffs)):
            return None
        
        if isinstance(config["feature_library"], ps.WeakPDELibrary):
            model_sim = ps.SINDy(
                    feature_library=model.feature_library.get_params().get("function_library"),
                )
            dummy_x = x_train[0] if isinstance(x_train, list) else x_train
            dummy_u = u_train[0] if isinstance(u_train, list) else u_train
            dummy_u = dummy_u[:10] if dummy_u is not None else None
            model_sim.fit(dummy_x[:10], t=dt, u=dummy_u)
            model_sim.optimizer.coef_ = model.optimizer.coef_

            # Vsetky predchadzajuce kontroli boli splnene, takze mozeme kontrolovat, ci je model stabilny pri simulaciach
            current_steps = min(val_steps, total_val_samples)
            start_index = max(0, min(2 * total_val_samples // 3, total_val_samples - current_steps))

            # Data pre simulaciu
            x0 = x_val[start_index]
            t = np.arange(current_steps) * dt
            u = u_val[start_index : start_index + current_steps] if u_val is not None else None
            x_ref = x_val[start_index : start_index + current_steps]
            try:
                x_sim = model_sim.simulate(x0=x0, t=t, u=u, integrator="solve_ivp", integrator_kws={"rtol": 1e-3, "atol": 1e-3})
                min_len = min(len(x_ref), len(x_sim))

                if r2_score(x_ref[:min_len], x_sim[:min_len]) < constraints["min_r2"]:
                    return None

            # Vznikla ina neocakavana chyba
            except Exception as e:
                return None

        else:
            model_sim = model
            # Ak je r2 pre predikciu vacsie ako poziadavka na maximalne r2      
            if model_sim.score(x_val, dt, x_dot_val, u_val) < constraints["min_r2"]:
                return None

        current_steps = min(total_val_samples, constraints["sim_steps"])
        start_index = max(0, min(total_val_samples // 2, total_val_samples - current_steps))

        # Vytvorenie dat pre finalnu simulacie kvoli RMSE stavov
        # Model.predict a Model.score pracuju s derivaciami stavov, 
        # je lespie kontrolovat stavy a derivacie budu spravne
        x0 = x_val[start_index]
        t_segment = np.arange(current_steps) * dt
        u_segment = u_val[start_index : start_index + current_steps] if u_val is not None else None
        x_ref = x_val[start_index : start_index + current_steps]
        try:
            x_sim = model_sim.simulate(
                    x0=x0,
                    t=t_segment,
                    u=u_segment,
                    integrator="solve_ivp",
                    integrator_kws={"rtol": 1e-6,"atol": 1e-6}
            )
            min_len = min(len(x_ref), len(x_sim))

            # Ak model simuluje prilis nespravne
            if np.max(np.abs(x_sim)) > constraints["max_state"] or not np.all(np.isfinite(x_sim)):
                return None

            rmse = root_mean_squared_error(x_ref[:min_len], x_sim[:min_len])
            r2 = r2_score(x_ref[:min_len], x_sim[:min_len])
            aic = min_len * np.log(rmse ** 2) + 2 * model_complexity + 2 * model_complexity / (min_len - model_complexity - 1) # Korigovane AIC
            
            result = {
                "configuration": config,
                "equations": model.equations(precision=precision),
                "r2_score": r2,
                "rmse": rmse,
                "complexity": model_complexity,
                "aic": aic,
                "random_seed": index + 42,
            }
            return result

        except Exception as e:
            return None

    # Zlyhanie este pre trenovanim modelu
    except Exception as e:
        print(e)
        return None
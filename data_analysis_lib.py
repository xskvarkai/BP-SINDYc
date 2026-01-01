import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pysindy as ps
from sklearn.metrics import root_mean_squared_error
import multiprocessing
from multiprocessing import shared_memory
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Union, Tuple
import json

def load_data(
        filepaths: List[str] = None,
        save_dt = False) -> List[dict]:
        
    datasets = []

    for filepath in filepaths:
        data_csv = pd.read_csv(filepath)
        data = data_csv.to_numpy()  
        X = np.stack((data[:, 1], data[:, 2], data[:, 3]), axis=-1)  
        U = np.stack((data[:, 4]), axis=-1)

        datasets.append({
            "X": X,
            "U": U
        })

    return datasets

def generate_trajectories(
        x_train: np.ndarray,
        u_train: np.ndarray = None,
        num_samples: int = 10000,
        num_trajectories: int = 50) -> List[np.ndarray]:

    total_val_samples = x_train.shape[0]
    x_train_multi = []
    u_train_multi = []
    for trajectory in range(0, num_trajectories):
        if total_val_samples < num_samples:
            start_index = 0
            warnings.warn("Insufficient samples for diverse trajectories. Consider adding more data or reducing the sample size.")
        else:
            start_index = np.random.randint(0, total_val_samples - num_samples)
        
        end_index = start_index + num_samples
        trajectory = x_train[start_index:end_index]
        x_train_multi.append(trajectory)

        if u_train is not None:
            input_signal = u_train[start_index:end_index]
            u_train_multi.append(input_signal)

    return x_train_multi, u_train_multi

def split_data(
        x: np.ndarray,
        u: Optional[np.ndarray] = None,
        val_size: float = 0.0,
        test_size: float = 0.2
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:

    # Ziskanie poctu vzoriek
    num_samples = x.shape[0]

    # Ziskanie poctu pre validacnu a testovaciu sadu
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
    if u is not None:
        u_train = u[:train_index]
        u_val = u[train_index:val_index] if val_count > 0 else None
        u_test = u[val_index:test_index] if test_count > 0 else None
    else:
        u_train = u_val = u_test = None

    return x_train, x_val, x_test, u_train, u_val, u_test

class SINDYcEstimator:
    def __init__(self):
        self.differentiation_methods = []
        self.optimizers = []
        self.feature_libraries = []
        self.configurations = []
        self.pareto_records = []
        self.pareto_front = []
        self.errors = []
        self.best_config = None

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
                "window_length": 5,
                "polyorder": 2,
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
            method_kwargs["window_length"] = max(5, method_kwargs["window_length"] if method_kwargs["window_length"] % 2 != 0 else method_kwargs["window_length"] + 1)

        # Zapisanie do self
        self.differentiation_methods.append(valid_methods[method](smoother_kws=method_kwargs))
        return self

    # Nastavenie optimalizera
    def set_optimizer(self, method: str, **kwargs: Any) -> "SINDYcEstimator":

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
                "normalize_columns": True
            },
            "SR3": {
                "regularizer": "L1",
                "reg_weight_lam": 0.1,
                "relax_coeff_nu": 1.0,
                "max_iter": 1e5,
                "tol": 1e-10,
                "normalize_columns": True
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

        # Zapisanie do self
        self.optimizers.append(valid_methods[method](**method_kwargs))
        return self

    # Nastavenie kniznice
    def set_feature_library(self, method: str, **kwargs: Any) -> "SINDYcEstimator":

        # Povolene metody
        valid_methods = {
            "PolynomialLibrary": ps.PolynomialLibrary,
            "FourierLibrary": ps.FourierLibrary,
            "CustomLibrary": ps.CustomLibrary,
            "ConcatLibrary": ps.ConcatLibrary,
            "TensoredLibrary": ps.TensoredLibrary
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
                "library_functions": [lambda x: x, lambda x: x ** 2, lambda x, y: x * y, lambda x: np.sin(x), lambda x: np.cos(x)],
                "function_names": [lambda x: x, lambda x: x + "^2", lambda x, y: x + "" + y, lambda x: "sin(" + x + ")", lambda x: "cos(" + x + ")"],
                "interaction_only": False,
                "include_bias": True 
            },
            "ConcatLibrary": {
                "libraries": [ps.PolynomialLibrary(degree=2, include_bias=False), ps.FourierLibrary()]
            },
            "TensoredLibrary": {
                "libraries": [ps.PolynomialLibrary(degree=2, include_bias=False), ps.FourierLibrary()]
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
    def generate_configurations(self, configurations: Optional[Union[List[Dict[str, Any]], Set[Dict[str, Any]]]] = None) -> "SINDYcEstimator":
        # Pripad, ked nastavene poziadavky
        if not self.differentiation_methods:
            self.set_differentiation_method("SmoothedFiniteDifference")
        if not self.optimizers:
            self.set_optimizer("SR3")
        if not self.feature_libraries:
            self.set_feature_library()

        # Iterovanie cez vsetky moznosti a vytvorenie konfiguracie pre kazdu jednu
        configurations = []
        for differentiation_method in self.differentiation_methods:
            for optimizer in self.optimizers:
                for feature_library in self.feature_libraries:
                    configurations.append({
                        "differentiation_method": differentiation_method,
                        "optimizer": optimizer,
                        "feature_library": feature_library
                    })

        # Zapisanie do self
        self.configurations = configurations
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
            **constraints: Any) -> "SINDYcEstimator":

        # Predvolene obmedzenie (poziadavky na model)
        default_constraints = {
            "sim_steps": 350,
            "max_sparsity": 50,
            "max_coeff": 1e2,
            "max_rmse": 1e3,
            "max_state": 1e3,
            "rmse_weight": 0.6
        }
        # Ak su poziadavky ine, update poziadaviek
        default_constraints.update(constraints)

        # Zistenie celkoveho poctu validacnych dat
        total_val_samples = x_val.shape[0]

        # Raise error-u, ak neboli vygenerovane konfiguracie
        if not self.configurations:
            raise ValueError("No configurations defined. Use generate_configurations() first.")

        # Raise warning-u, ak je poziadavka na rmse_weight nezmysel a zmena na 0.6
        if default_constraints["rmse_weight"] >= 1 or default_constraints["rmse_weight"] <= 0:
            default_constraints["rmse_weight"] = 0.6
            warnings.warn(f"RMSE weight out of bounds: {default_constraints["rmse_weight"]}. Must satisfy 0 < rmse_weight < 1. Automatically changed to 0.6")

        # Raise warning-u, ak je malo validacnych krokov a zmena na 11
        if default_constraints["sim_steps"] <= 10:
            default_constraints["sim_steps"] = 11
            warnings.warn(f"Minimum required simulation steps are 10, validation steps increased automatically to 11")

        # Raise error-u, ak nie je dostatocne vela validacnych dat
        if total_val_samples < default_constraints["sim_steps"]:
            raise ValueError(f"Not enough validation samples. Increase validation steps to {default_constraints["sim_steps"]} or validation size")



        # Zbalenie dat a konfiguracie, kvoli multiprocessingu
        configurations_and_data = [
            (config, x_train, x_val, u_train, u_val, dt, default_constraints)
            for config in self.configurations
        ]

        # Zistenie celkoveho poctu konfiguracii
        total_configurations = len(configurations_and_data)

        # UI/UX
        start_time = datetime.now()
        print(f"\nParameter search started...")
        print(f"Total configurations to explore: {total_configurations}")
        print(f"Using {n_processes} parallel processes")
        print(f"Start time: {start_time.strftime("%H:%M:%S")}")

        # Multiprocessing
        with multiprocessing.Pool(processes=n_processes) as pool:
            # UI/UX
            results = []
            for index, (result, error) in enumerate(pool.imap(run_config, configurations_and_data), 1):
                results.append(result)
                self.errors.append(error)
                progress_msg = f"Processing configuration {index}/{total_configurations} ({(index/total_configurations)*100:.2f}%)"
                print(progress_msg, end="\r", flush=True)
            print()
        
        # UI/UX
        print("\nParameter search complete.")
        duraction = datetime.now() - start_time
        print(f"The process took {duraction} hours")
        valid_configs = sum(1 for result in results if result is not None)
        print(f"Valid configurations found: {valid_configs} out of {total_configurations}")

        # Vyskladanie pareto zaznamov
        self.pareto_records = [result for result in results if result is not None] 
        # Ak existuje aspon jeden zaznam
        if self.pareto_records:
            # Zostav pareto frontier
            self.pareto_front = self._compute_pareto_front(self.pareto_records)
            # Najdi najlepsi vysledok
            self.best_config = self._select_best_pareto_config(self.pareto_front, default_constraints["rmse_weight"])

        return self

    def _compute_pareto_front(self, results: List[Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        # Nacitanie iba velidnych (not None) zaznamov
        valid_results = [result for result in results if result is not None]

        # Raise error-u, ak neexistuje ani jeden validny zaznam
        if not valid_results:
            raise ValueError("No valid configurations found. All configurations were filtered out.")

        # Zoradenie od najnizsieho po najvyssie podla rmse
        sorted_results = sorted(valid_results, key=lambda x: x["rmse"])

        # Definicia a priradienie najlespieho rmse do pareto frontu
        pareto_front = [sorted_results[0]]
        # Kazdeho kandidata, ktory je jednoduchsi ako ten z najlepsim rmse rprirad do pareto fronty
        for candidate in sorted_results[1:]:
            if candidate["sparsity"] < pareto_front[-1]["sparsity"]:
                    pareto_front.append(candidate)

        return pareto_front

    def _select_best_pareto_config(self, pareto_records: List[Dict[str, Any]], rmse_weight) -> Dict[str, Any]:
        # Nacitanie data potrebnych pre porovnavanie
        rmse_values = np.array([record["rmse"] for record in pareto_records], dtype=float)
        sparsity_values = np.array([record["sparsity"] for record in pareto_records], dtype=float)

        # Zistenie rozsahu dat (min-max), potrebnych pre skalovanie
        rmse_range = np.ptp(rmse_values) if np.ptp(rmse_values) > 0 else 1.0
        sparsity_range = np.ptp(sparsity_values) if np.ptp(sparsity_values) > 0 else 1.0

        # Skalovanie dat pre jednoduchi computing
        normalized_rmse = (rmse_values - np.min(rmse_values)) / rmse_range
        normalized_sparsity = (sparsity_values - np.min(sparsity_values)) / sparsity_range

        # Kazdej kombinacii v pareto fronte priradi score
        spars_weight = 1 - rmse_weight
        scores = [rmse_weight * (1 - rmse) + spars_weight * (1 - sparsity)
                for rmse, sparsity in zip(normalized_rmse, normalized_sparsity)]

        # Najdenie najlepsieho skore (najvyssieho cisla)
        best_index = int(np.argmax(scores))

        return self.pareto_front[best_index]

    def plot_pareto(self):
        errs = np.array([r["rmse"] for r in self.pareto_front], dtype=float)
        spars = np.array([r["sparsity"] for r in self.pareto_front], dtype=float)

        plt.figure(figsize=(6, 4))
        plt.scatter(errs, spars, color="tab:blue", label="Pareto body")
        plt.xlabel("RMSE")
        plt.ylabel("Sparsity (počet nenulových koeficientov)")
        plt.title("Pareto front")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return self

    def export_data(self, path: str = "data.json") -> "SINDYcEstimator":
        if self.pareto_records is None:
            warnings.warn("Pareto records are None")
        if self.pareto_front is None:
            warnings.warn("Pareto front is None")
        if self.best_config is None:
            warnings.warn("Best configuration is None")
        
        payload = {}

        records = []
        pareto_front = self.pareto_front.copy()
        for record in self.pareto_front:
            record["best"] = (self.best_config is not None and ("model" in self.best_config and record.get("model") == self.best_config["model"]))
            records.append(record)
        
        errors = []
        for error in self.errors:
            if error is not None:
                errors.append(error)

        payload["pareto_front"] = records
        payload["pareto_records"] = self.pareto_records
        payload["errors"] = errors

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=5, default=str)

        return self

# ========== Spustanie konfiguracie na kazdom procesore ========== 
def run_config(configuration_and_data: Tuple[Dict[str, Any], np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], Optional[str]]:  
    error = {
        "model": None,
        "error": None,
    }
    try:
        # Rozbalenie dat
        config, x_train, x_val, u_train, u_val, dt, constraints = configuration_and_data
        # Ignorovanie warningov pocas hladnia (exceptions volane na konci)
        warnings.filterwarnings("ignore", category=UserWarning)

        # Zostavenie modelu
        model = ps.SINDy(
            optimizer=config["optimizer"],
            feature_library=config["feature_library"],
            differentiation_method=config["differentiation_method"]
        )

        # Fitting dat do modelu
        model.fit(
            x=x_train,
            u=u_train,
            t=dt
        )

        # Zistenie celkoveho poctu validacnych dat
        total_val_samples = x_val.shape[0]
        # Pocet krokov na rychlu validaciu
        val_steps = 10
        # Poznamka: bola by vhodna kontrola ci je total_val_samples > val_steps

        model_coeffs = model.coefficients()
        model_sparsity = np.count_nonzero(model_coeffs)

        if model_sparsity == 0 or model_sparsity > constraints["max_sparsity"]:
            error["model"] = model
            error["error"] = "Model is trivial exceed max sparsity"
            return None, error

        # Ak je poziadavka na maximalnu velkost koefficientov nesplnena
        if np.max(np.abs(model_coeffs)) > constraints["max_coeff"] or not np.all(np.isfinite(model_coeffs)):
            error["model"] = model
            error["error"] = "Model coeff exceed max coeff or is Inf/Nan"
            return None, error

        # Ak model presiel predoslimi kontrolami, kontrola ci spravne predikuje derivacie
        rmse_predict = root_mean_squared_error(model.differentiation_method(x_val), model.predict(x_val, u_val))

        # Ak je rmse pre predikciu vacsie ako poziadavka na maximalne rmse
        if rmse_predict > constraints["max_rmse"]:
            error["model"] = model
            error["error"] = "Model can't predict"
            return None, error

        # Vsetky predchadzajuce kontroli boli splnene, takze mozeme kontrolovat, ci je model stabilny pre simulaciach
        # Starty simulacii (jeden na zaciatku, druhy v strede valicanych dat, ak sa da, ak nie total_val_samples - val_steps)
        starts = [0, min(total_val_samples // 2, total_val_samples - val_steps)]
        for start in starts:
            x0 = x_val[start]
            t = np.arange(val_steps) * dt
            u = u_val[start : start + val_steps] if u_val is not None else None
            try:
                x_sim = model.simulate(x0=x0, t=t, u=u, integrator="solve_ivp", integrator_kws={"rtol": 0.1, "atol": 0.1})

                if np.max(np.abs(x_sim)) > constraints["max_state"]:
                    error["model"] = model
                    error["error"] = "Model diverg too much (exceed max state)"
                    return None, error

                if not np.all(np.isfinite(x_sim)):
                    error["model"] = model
                    error["error"] = "Model is not stable"
                    return None, error

            except Exception as e:
                error["model"] = model
                error["error"] = e
                return None, error

        if total_val_samples < constraints["sim_steps"]:
            start_index = 0
            current_steps = total_val_samples
        else:
            start_index = np.random.randint(0, total_val_samples - constraints["sim_steps"])
            current_steps = constraints["sim_steps"]

        x0 = x_val[start_index]
        t_segment = np.arange(current_steps) * dt
        u_segment = u_val[start_index : start_index + current_steps] if u_val is not None else None
        x_ref = x_val[start_index : start_index + current_steps]

        x_sim = model.simulate(
                x0=x0,
                t=t_segment,
                u=u_segment,
                integrator="solve_ivp",
                integrator_kws={"rtol": 1e-6,"atol": 1e-6}
        )
        min_len = min(len(x_ref), len(x_sim))

        rmse = root_mean_squared_error(x_ref[:min_len], x_sim[:min_len])
        sparsity = np.count_nonzero(model_coeffs)

        result = {
            "model": model,
            "rmse": rmse,
            "sparsity": sparsity
        }

        return result, None

    except Exception as e:
        error["error"] = e
        return None, error
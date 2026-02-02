import numpy as np
import pysindy as ps
import multiprocessing
import warnings
import json
import tempfile
import pickle
import gc
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import root_mean_squared_error, r2_score
from pathlib import Path

from utils.custom_libraries import FixedWeakPDELibrary, FixedCustomLibrary
from utils import constants
from utils.helpers import sanitize_WeakPDELibrary, compute_time_vector, make_model_callable

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

    # Implementacia Context Managera pre automaticke cistenie docasnych suborov
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._delete_tempfiles()

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
                "polyorder": constants.SAVGOL_POLYORDER,
                "window_length": constants.SAVGOL_WINDOW_LENGTH,
                "mode": "interp",
                "delta": 1,
                "axis": 0
            }
        }

        # Nastavenie argumentov pre pozadovanu metodu. Musia byt zadane iba tie co su v predovelnej (pripadne menej, nie viac), 
        # inac warning.
        method_kwargs = default_kwargs[method].copy()
        for key, value in kwargs.items():
            if key not in method_kwargs:
                warnings.warn(f"Unexpected parameter {key} for {method} method. Ignoring.")
            else:
                method_kwargs[key] = value

        # Ak je metoda SmoothedFiniteDifference, overit ci je window_length neparne, ak nie je + 1 (malo by byt vacsie ako polyorder)
        # Savitzky-Golay vyzaduje neparnu dlzku okna.
        if method == "SmoothedFiniteDifference":
            method_kwargs["window_length"] = max(11, method_kwargs["window_length"] if method_kwargs["window_length"] % 2 != 0 else method_kwargs["window_length"] + 1)
            # Zapisanie do self
            self.differentiation_methods.append(valid_methods[method](smoother_kws=method_kwargs))
        else:
            # Zapisanie do self
            self.differentiation_methods.append(valid_methods[method](**method_kwargs))
        return self

    # Nastavenie optimalizera (riesitela riedkej regresie)
    # Riesi problem: min ||dx - Theta * Xi|| + lambda * ||Xi||
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
                "unbias": True
            },
            "SR3": {
                "regularizer": "L0",
                "reg_weight_lam": 0.1,
                "relax_coeff_nu": 1.0,
                "trimming_fraction": 0.0,
                "trimming_step_size": 1.0,
                "max_iter": 1e5,
                "tol": 1e-8,
                "normalize_columns": True,
                "unbias": True
            }
        }

        # Nastavenie argumentov pre pozadovanu metodu. Musia byt zadane iba tie co su v predovelnej (pripadne menej, nie viac),
        # inac warning.
        method_kwargs = default_kwargs[method].copy()
        for key, value in kwargs.items():
            if key not in method_kwargs:
                warnings.warn(f"Unexpected parameter {key} for {method} optimizer. Ignoring.")
            else:
                method_kwargs[key] = value

        # Vzdy prepisat na integer
        method_kwargs["max_iter"] = int(method_kwargs["max_iter"])

        base_optimizer = valid_methods[method](**method_kwargs)

        # Konfigurácia Ensemble metedy (Bagging / Bootstrapping)
        # Vytvori viacero modelov na podmnozinach dat a spriemeruje/vyberie najcastejsie koeficienty.
        # Zvysuje robustnost voci sumu.
        if ensemble:
            default_ensemble_kwargs = {
                "bagging": True,
                "n_models": 50,
                "n_subset": None
            }

            if ensemble_kwargs:
                default_ensemble_kwargs.update(ensemble_kwargs)

            default_ensemble_kwargs["n_subset"] = int(default_ensemble_kwargs["n_subset"]) if default_ensemble_kwargs.get("n_subset") is not None else None
            final_optimizer = ps.EnsembleOptimizer(opt=base_optimizer, **default_ensemble_kwargs)
            # Zapisanie do self
            self.optimizers.append(final_optimizer)

        else:
            # Zapisanie do self
            self.optimizers.append(base_optimizer)
        
        return self

    # Nastavenie kniznice kandidatskych funkcii Theta(X)
    def set_feature_library(self, method: str, **kwargs: Any) -> "SINDYcEstimator":

        # Povolene metody
        valid_methods = {
            "PolynomialLibrary": ps.PolynomialLibrary,
            "FourierLibrary": ps.FourierLibrary,
            "CustomLibrary": FixedCustomLibrary,
            "ConcatLibrary": ps.ConcatLibrary,
            "TensoredLibrary": ps.TensoredLibrary,
            "WeakPDELibrary": FixedWeakPDELibrary # Pre slabu formulaciu (integralna forma), vhodne pre vysoky sum
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
                "include_bias": False 
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
                "p": 4,
                "H_xt": None
            }
        }
        
        # Nastavenie argumentov pre pozadovanu metodu. Musia byt zadane iba tie co su v predovelnej (pripadne menej, nie viac),
        # inac warning.
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
        # Pripad, ked nie su nastavene poziadavky
        if not self.differentiation_methods:
            raise ValueError("No differentiation methods defined. Use set_differentiation_method() first.")
        if not self.optimizers:
            raise ValueError("No optimizers defined. Use set_optimizers() first.")
        if not self.feature_libraries:
            raise ValueError("No feature libraries defined. Use set_feature_library() first.")

        # Iterovanie cez vsetky moznosti a vytvorenie zoznamu konfiguracii
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
        log_file_name: str = "worker_results",
        **constraints: Any
    ) -> "SINDYcEstimator":

        # Predvolene obmedzenie (poziadavky na model)
        default_constraints = {
            "sim_steps": 350, # Kolko krokov simulovat pre validaciu
            "coeff_precision": None, # Zaokruhlovanie koeficientov
            "max_complexity": 50, # Max pocet nenulovych členov
            "max_coeff": 1e2, # Max hodnota koeficientu (ochrana proti explozii)
            "min_r2": 0.7, # Minimalna presnost predikcie alebo simulacie (kratkej)
            "max_state": 1e3 # Prah pre detekciu nestability pri integracii
        }
        # Ak su poziadavky ine, update poziadaviek
        default_constraints.update(constraints)

        # Zistenie celkoveho poctu validacnych dat
        total_val_samples = x_val.shape[0]

        # Raise error-u, ak neboli vygenerovane konfiguracie
        if not self.configurations:
            raise ValueError("No configurations defined. Use generate_configurations() first.")

        # Raise warning-u, ak je malo validacnych krokov a zmena na 21
        if default_constraints["sim_steps"] <= constants.MIN_VALIDATION_SIM_STEPS:
            default_constraints["sim_steps"] = constants.MIN_VALIDATION_SIM_STEPS
            warnings.warn(f"Minimum required simulation steps are {constants.MIN_VALIDATION_SIM_STEPS},"
                          f"validation steps increased automatically to {constants.MIN_VALIDATION_SIM_STEPS}")

        # Raise error-u, ak nie je dostatocne vela validacnych dat
        if total_val_samples < default_constraints["sim_steps"]:
            raise ValueError(f"Not enough validation samples. Increase validation steps to {default_constraints["sim_steps"]} or validation size")

        # Pouzitie multiprocessing Manager-a pre zdielanu pamat
        with multiprocessing.Manager() as manager:
            # Cache pre vypocitane derivacie (x_dot).
            # Rozne konfiguracie mozu pouzivat rovnaku metodu derivacie,
            # takze je zbytocne to pocitat znova v kazdom procese.
            cache_dict = manager.dict()
            lock = manager.Lock()

            # Zbalenie dat a konfiguracie, kvoli multiprocessingu (argument pre map funkciu)
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

            # Otvorenie docasnych suborov na zapis konfiguracii
            # Pouzivame subor namiesto drzania v pamati, pretoze vysledky mozu byt velke
            # a pri velkom pocte konfiguracii by doslo k MemoryError.
            data_dir = Path(constants.DATA_EXPORT_PATH)
            log_filepath = data_dir / f"{log_file_name}.log"
            if os.path.exists(log_filepath):
                os.remove(log_filepath)

            with tempfile.NamedTemporaryFile(delete=False, mode="wb", prefix="sindyR", suffix=".pkl") as results_file, \
                 open(log_filepath, "a", encoding="utf-8") as f_log:

                # Ziskanie ciest k docasnym suborom - umoznuje pracu s nimi
                self.results_file_name = results_file.name

                # Multiprocessing
                with multiprocessing.Pool(processes=n_processes) as pool:
                    # imap vracia vysledky postupne ako su hotove
                    for index, result in enumerate(pool.imap(run_config, configurations_and_data), 1):
                        try:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            f_log.write(f"[{timestamp}] Result {index}: {str(result)}\n{"-"*180}\n\n")
                            f_log.flush()
                            if not result.get("error"):
                                result["index"] = index
                                pickle.dump(result, results_file)
                            gc.collect()
                        except Exception as e:
                            warnings.warn(e)
                        # UI/UX - progress bar
                        print(f"Processing configuration {index}/{total_configurations}" 
                              f"({(index/total_configurations)*100:.2f}%)", end="\r", flush=True)
                    print()

        # Znovuzapnutie warningov a vycistenie nepotrebnych dat
        warnings.filterwarnings("default", category=UserWarning)
        configurations_and_data.clear()

        # Nacitanie vysledkov zo suboru do pamate
        with open(self.results_file_name, "rb") as f:
            try:
                while True:
                    self.results.append(pickle.load(f))
            except EOFError:
                pass

        # UI/UX - štatistika trvania
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
            # Najdi najlepsi vysledok (podla AIC - Akaike information criterion)
            self.best_config = self._select_best_config(self.results)
            self.pareto_front = self._compute_pareto_front(self.results)
            self.results.clear()

        return self

    def plot_pareto(self):
        from utils.vizualization import plot_pareto
        plot_pareto(self.pareto_front)
        return None

    # Export dat do JSON
    def export_data(self, data: dict = None, file_name: str = "data") -> "SINDYcEstimator":
        if self.pareto_front is None:
            warnings.warn("Pareto front is None")
        if self.best_config is None:
            warnings.warn("Best configuration is None")

        payload = {
            "best_result": self.best_config,
            "pareto_front": self.pareto_front,
            "user_data": data
        }

        try: 
            data_dir = Path(constants.DATA_EXPORT_PATH)  
            filepath = data_dir / f"{file_name}.json" 
            with open(filepath, "w", encoding="utf-8") as f:
                # default=str zabezpeci serializaciu objektov ako NumPy polia
                json.dump(payload, f, indent=5, default=str)
        except Exception as e:
            warnings.warn(e)

        return None

    def validate_on_test(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        u_train: Optional[np.ndarray] = None,
        u_test: Optional[np.ndarray] = None,
        dt: float = 0.01,
        plot: bool = True,
        **constraints
    ):
        if self.best_config is None:
            warnings.warn("No best configuration found. Run search_configurations() first.")
            return None

        config = self.best_config["configuration"]
        config["differentiation_method"] = config["feature_library"].get_params().get("differentiation_method")
        np.random.seed(self.best_config.get("random_seed", constants.DEFAULT_RANDOM_SEED))

        # Ignorovanie warningov pocas testovania
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", module="pysindy.utils")

        # Osetrenie pre WeakPDE (rovnaka logika ako v run_config)
        time_vec = compute_time_vector(x_train, dt)
        config = sanitize_WeakPDELibrary(config, time_vec)
        
        model = ps.SINDy(
            optimizer=config["optimizer"],
            feature_library=config["feature_library"],
            differentiation_method=config["differentiation_method"]
        )
        model.fit(x=x_train, u=u_train, t=dt)

        if constraints.get("coeff_precision") is not None:
            precision = constraints["coeff_precision"]
            if constraints["coeff_precision"] == 0:
                model.optimizer.coef_ = np.rint(model.optimizer.coef_)
                precision = 1
            else:
                model.optimizer.coef_ = np.round(model.optimizer.coef_, decimals=precision)

        model_sim = make_model_callable(model, x_train, u_train, dt)

        print("\nStarting validation on test data...")
        x0 = x_test[0]
        t_test = np.arange(x_test.shape[0]) * dt
        try:
            x_sim = model_sim.simulate(x0=x0, t=t_test, u=u_test)
            min_len = min(len(x_test), len(x_sim))
            
            x_ref_cut = x_test[:min_len]
            x_sim_cut = x_sim[:min_len]
            t_test_cut = t_test[:min_len]

            test_rmse = root_mean_squared_error(x_ref_cut, x_sim_cut)
            test_r2 = r2_score(x_ref_cut, x_sim_cut)

            print(f"Best model R2 score: {test_r2:.3%}")

            self.best_config["test_metrics"] = {
                "rmse": np.round(test_rmse, 5),
                "r2": np.round(test_r2, 5),
                "simulation_length": min_len
            }

            if plot:
                from utils.vizualization import vizualize_trajectory
                vizualize_trajectory(t_test_cut, x_ref_cut, x_sim_cut)

            warnings.filterwarnings("default", category=UserWarning)
            return None

        except Exception as e:
            print(f"Test simulation failed: {e}")
            warnings.filterwarnings("default", category=UserWarning)
            return None

    # Zostavenie pareto fronty (kompromis medzi chybou a zlozitostou)
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
        # Algoritmus: Kazdeho kandidata, ktory je jednoduchsi (nizsia complexity) 
        # ako ten s najlepsim rmse, prirad do pareto fronty.
        # Hladáme dominantne riesenia
        for candidate in sorted_results[1:]:
            if candidate["complexity"] < pareto_front[-1]["complexity"]:
                    pareto_front.append(candidate)

        return pareto_front

    # Vyber najlepsieho modelu z pareto zaznamov (fronty)
    def _select_best_config(self, results: List[Dict]) -> Dict[str, Any]:
        # Nacitanie iba validnych (not None) zaznamov
        valid_results = [result for result in results if result is not None]

        # Warning, ak neexistuje ani jeden validny zaznam
        if not valid_results:
            warnings.warn("No valid configurations with AIC found. Cannot select best configuration.")
            return None
        
        # Vyber na zaklade AIC (Akaike Information Criterion), 
        # ktore penalizuje zlozitost modelu
        sorted_results = sorted(results, key=lambda x: x["aic"])
        best_model = sorted_results[0]
        
        return best_model

    # Odstranenie docasnych suborov
    def _delete_tempfiles(self):
        if self.results_file_name and os.path.exists(self.results_file_name):
            os.remove(self.results_file_name)
            
        self.results_file_name = None
        return self

# ========== Spustanie konfiguracie na kazdom procesore ========== 
def run_config(configuration_and_data: List[Dict[str, Any], np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], Optional[str]]:  
    try:
        # Rozbalenie dat
        index, config, x_train, x_val, u_train, u_val, dt, constraints, cache_dict, lock = configuration_and_data

        # Fix seedu pre reprodukovatelnost v kazdom procese
        np.random.seed(index + constants.DEFAULT_RANDOM_SEED)

        # Ignorovanie warningov pocas hladania
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", module="pysindy.utils")

        # Specialne osetrenie pre WeakPDELibrary (integralna formulacia)
        # Tento typ kniznice interne pocita derivacie inak, preto differentiation_method = None
        # Taktiez problem s AxesArray
        time_vec = compute_time_vector(x_train, dt)
        config = sanitize_WeakPDELibrary(config, time_vec)

        # Zostavenie SINDy modelu
        model = ps.SINDy(
            optimizer=config["optimizer"],
            feature_library=config["feature_library"],
            differentiation_method=config["differentiation_method"]
        )

        # Caching derivacii (x_dot)
        # Vypocet derivacie je drahy. Ak uz iny proces vypocital deriváciu
        # pre rovnake parametre, pouzijeme ju zo zdielanej pamate.
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
            model.optimizer.coef_ = np.round(model.optimizer.coef_, decimals=precision)

        # Zistenie celkoveho poctu validacnych dat
        total_val_samples = x_val.shape[0]
        # Pocet krokov na rychlu validaciu (stabilita)
        val_steps = constants.MIN_VALIDATION_SIM_STEPS

        model_coeffs = model.coefficients()
        model_complexity = np.count_nonzero(model_coeffs)

        # FILTER 1: Zlozitost
        # Ak je poziadavka na pocet koeficientov nesplnena alebo je model trivialny
        if model_complexity == 0 or model_complexity > constraints["max_complexity"]:
            return {"configuration": config, "error": f"Model is trivial or exceed max complexity. Early stopped with complexity: {model_complexity}"}

        # FILTER 2: Velkost koeficientov
        # Ak je poziadavka na maximalnu velkost koefficientov nesplnena alebo su Nan/Inf
        if np.max(np.abs(model_coeffs)) > constraints["max_coeff"] or not np.all(np.isfinite(model_coeffs)):
            return {"configuration": config, "error": "Model coeff exceed max coeff or is Inf/Nan. Early stopped due this message."}
        
        # Vsetky predchadzajuce kontroli boli splnene, takze mozeme kontrolovat, ci je model stabilny
        # Specialna vetva pre simulaciu WeakPDE (vyzaduje iny pristup k simulatoru)
        model_sim = make_model_callable(model, x_train, u_train, dt)

        # Robime kratku simulaciu (val_steps).
        current_steps = min(val_steps, total_val_samples)
        start_index = max(0, total_val_samples - current_steps)

        # Data pre simulaciu
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
        current_steps = min(total_val_samples, constraints["sim_steps"])
        start_index = max(0, total_val_samples - current_steps)

        x0 = x_val[start_index]
        t_segment = np.arange(current_steps) * dt
        u_segment = u_val[start_index : start_index + current_steps] if u_val is not None else None
        x_ref = x_val[start_index : start_index + current_steps]
        try:
            x_sim = model_sim.simulate(x0=x0, t=t_segment, u=u_segment, integrator="solve_ivp", integrator_kws={"rtol": 1e-6,"atol": 1e-6})
            min_len = min(len(x_ref), len(x_sim))

            # Ak model simuluje prilis nespravne (diverguje do nekonecna)
            if np.max(np.abs(x_sim)) > constraints["max_state"] or not np.all(np.isfinite(x_sim)):
                return {"configuration": config, "error": "Model diverg too much (exceed max state) or is not stable. Early stopped due this message."}

            # Vypocet metrik
            rmse = root_mean_squared_error(x_ref[:min_len], x_sim[:min_len])
            r2 = r2_score(x_ref[:min_len], x_sim[:min_len])
            
            # AIC (Akaike Information Criterion)
            # Aproximacia pre Gaussian noise: AIC = N * ln(MSE) + 2k
            # Pouzita korekcia AICc pre male vzorky: + (2k(k+1)) / (n - k - 1)
            # Penalizuje modely, ktore su prilis komplexne (vela koeficientov k).
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
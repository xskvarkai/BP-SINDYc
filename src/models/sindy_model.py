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
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import root_mean_squared_error, r2_score

from utils.custom_libraries import FixedWeakPDELibrary

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
                "unbias": False
            },
            "SR3": {
                "regularizer": "L1",
                "reg_weight_lam": 0.1,
                "relax_coeff_nu": 1.0,
                "max_iter": 1e5,
                "tol": 1e-8,
                "normalize_columns": True,
                "unbias": False
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

    # Nastavenie kniznice kandidatskych funkcii Theta(X)
    def set_feature_library(self, method: str, **kwargs: Any) -> "SINDYcEstimator":

        # Povolene metody
        valid_methods = {
            "PolynomialLibrary": ps.PolynomialLibrary,
            "FourierLibrary": ps.FourierLibrary,
            "CustomLibrary": ps.CustomLibrary,
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
        if default_constraints["sim_steps"] <= 20:
            default_constraints["sim_steps"] = 21
            warnings.warn(f"Minimum required simulation steps are 20, validation steps increased automatically to 21")

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

            # Ak je definovana CustomLibrary, pepisat na pri exporte na string
            # Bol problem pri exporte kniznice a tym aj s reprodukovanim modelu
            def sanitize_input(record):
                try:
                    if isinstance(record.get("configuration")["feature_library"], ps.CustomLibrary):
                        record.get("configuration")["feature_library"] = "CustomLibrary"        
                except Exception:
                    pass

            # Otvorenie docasnych suborov na zapis konfiguracii
            # Pouzivame subor namiesto drzania v pamati, pretoze vysledky mozu byt velke
            # a pri velkom pocte konfiguracii by doslo k MemoryError.
            log_file_path = "data/processed/worker_results.log"
            if os.path.exists(log_file_path):
                os.remove(log_file_path)

            with tempfile.NamedTemporaryFile(delete=False, mode="wb", prefix="sindyR", suffix=".pkl") as results_file, \
                 open(log_file_path, "a", encoding="utf-8") as f_log:

                # Ziskanie ciest k docasnym suborom - umoznuje pracu s nimi
                self.results_file_name = results_file.name

                # Multiprocessing
                with multiprocessing.Pool(processes=n_processes) as pool:
                    # imap vracia vysledky postupne ako su hotove
                    for index, result in enumerate(pool.imap(run_config, configurations_and_data), 1):
                        try:
                            sanitize_input(result)
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            f_log.write(f"[{timestamp}] Result {index}: {str(result)}\n{"-"*180}\n\n")
                            f_log.flush()
                            if not result.get("error"):
                                result["index"] = index
                                pickle.dump(result, results_file)
                        except Exception as e:
                            warnings.warn(e)
                        gc.collect() # Vynutenie Garbage Collection
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

        return self

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

    # Zobrazenie pareto fronty v grafe
    def plot_pareto(self)  -> "SINDYcEstimator":
        self.pareto_front = self._compute_pareto_front(self.results)
        self.results.clear() # Uvolnenie pamate
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

    # Export dat do JSON
    def export_data(self, data: dict = None, path: str = "data.json") -> "SINDYcEstimator":
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
            with open("data/processed/" + path, "w", encoding="utf-8") as f:
                # default=str zabezpeci serializaciu objektov ako NumPy polia
                json.dump(payload, f, indent=5, default=str)
        except Exception as e:
            warnings.warn(e)

        return self

    def validate_on_test(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        u_train: Optional[np.ndarray] = None,
        u_test: Optional[np.ndarray] = None,
        dt: float = 0.01,
        plot: bool = True
    ):
        # Ignorovanie warningov pocas testovania
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", module="pysindy.utils")

        if self.best_config is None:
            raise ValueError("No best configuration found. Run search_configurations() first.")

        config = self.best_config["configuration"]
        np.random.seed(self.best_config.get("random_seed", 42))

        # Osetrenie pre WeakPDE (rovnaka logika ako v run_config)
        if isinstance(config["feature_library"], FixedWeakPDELibrary):
            params = config["feature_library"].get_params()
            if isinstance(x_train, list):
                time_vec = (np.arange(x_train[0].shape[0]) * dt)
            else:
                time_vec = (np.arange(x_train.shape[0]) * dt)
            
            config["feature_library"] = FixedWeakPDELibrary(
                function_library=params.get("function_library", None),
                derivative_order=params.get("derivative_order", 0),
                spatiotemporal_grid=time_vec,
                K=params.get("K", 100),
                p=params.get("p", 4),
                differentiation_method=config["differentiation_method"]
            )
            config["differentiation_method"] = None

        model = ps.SINDy(
            optimizer=config["optimizer"],
            feature_library=config["feature_library"],
            differentiation_method=config["differentiation_method"]
        )
        model.fit(x=x_train, u=u_train, t=dt)

        if config["differentiation_method"] is None:
            model_sim = ps.SINDy(
                    feature_library=model.feature_library.get_params().get("function_library"),
                )
            dummy_x = x_train[0] if isinstance(x_train, list) else x_train
            dummy_u = u_train[0] if isinstance(u_train, list) else u_train
            dummy_u = dummy_u[:10] if dummy_u is not None else None
            model_sim.fit(dummy_x[:10], t=dt, u=dummy_u)
            model_sim.optimizer.coef_ = model.optimizer.coef_

        else:
            model_sim = model

        model = None

        print("\nStarting validation on test data...")
        x0 = x_test[0]
        t_test = np.arange(x_test.shape[0]) * dt
        u_test = u_test[0 : 0 + x_test.shape[0]] if u_test is not None else None
        try:
            x_sim = model_sim.simulate(
                x0=x0,
                t=t_test,
                u=u_test,
                integrator="solve_ivp",
                integrator_kws={"rtol": 1e-6, "atol": 1e-6}
            )

            min_len = min(len(x_test), len(x_sim))
            x_test_cut = x_test[:min_len]
            x_sim_cut = x_sim[:min_len]
            t_test_cut = t_test[:min_len]

            test_rmse = root_mean_squared_error(x_test_cut, x_sim_cut)
            test_r2 = r2_score(x_test_cut, x_sim_cut)

            print(f"Best model R2 score: {test_r2:5%}")

            self.best_config["test_metrics"] = {
                "rmse": test_rmse,
                "r2": test_r2,
                "simulation_length": min_len
            }

            if plot:
                from utils.vizualization import vizualize_trajectory

                vizualize_trajectory(t_test_cut, x_sim_cut, x_test_cut)

            warnings.filterwarnings("default", category=UserWarning)
            return self

        except Exception as e:
            print(f"Test simulation failed: {e}")
            warnings.filterwarnings("default", category=UserWarning)
            return None

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

        # Fix seedu pre reprodukovatelnost v kazdom procese
        np.random.seed(index + 42)

        # Ignorovanie warningov pocas hladania
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", module="pysindy.utils")

        # Specialne osetrenie pre WeakPDELibrary (integralna formulacia)
        # Tento typ kniznice interne pocita derivacie inak, preto differentiation_method = None
        # Taktiez problem s AxesArray
        if isinstance(config["feature_library"], FixedWeakPDELibrary):
            params = config["feature_library"].get_params()

            if isinstance(x_train, list):
                time_vec = (np.arange(x_train[0].shape[0]) * dt)
            else:
                time_vec = (np.arange(x_train.shape[0]) * dt)
            
            config["feature_library"] = FixedWeakPDELibrary(
                function_library=params.get("function_library", None),
                derivative_order=params.get("derivative_order", 0),
                spatiotemporal_grid=time_vec,
                K=params.get("K", 100),
                p=params.get("p", 4),
                differentiation_method=config["differentiation_method"]
            )

            config["differentiation_method"] = None

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
                x_dot_train, x_dot_val = cache_dict[key]
            else:
                with lock:
                    # Double-check locking pattern
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
        # Pocet krokov na rychlu validaciu (stabilita)
        val_steps = 20

        model_coeffs = model.coefficients()
        model_complexity = np.count_nonzero(model_coeffs)

        # FILTER 1: Zlozitost
        # Ak je poziadavka na pocet koeficientov nesplnena alebo je model trivialny
        if model_complexity == 0 or model_complexity > constraints["max_complexity"]:
            return {"configuration": config, "error": "Model is trivial or exceed max complexity"}

        # FILTER 2: Velkost koeficientov
        # Ak je poziadavka na maximalnu velkost koefficientov nesplnena alebo su Nan/Inf
        if np.max(np.abs(model_coeffs)) > constraints["max_coeff"] or not np.all(np.isfinite(model_coeffs)):
            return {"configuration": config, "error": "Model coeff exceed max coeff or is Inf/Nan"}
        
        # Vsetky predchadzajuce kontroli boli splnene, takze mozeme kontrolovat, ci je model stabilny
        # Specialna vetva pre simulaciu WeakPDE (vyzaduje iny pristup k simulatoru)
        if isinstance(config["feature_library"], FixedWeakPDELibrary):
            model_sim = ps.SINDy(
                    feature_library=model.feature_library.get_params().get("function_library"),
                )
            dummy_x = x_train[0] if isinstance(x_train, list) else x_train
            dummy_u = u_train[0] if isinstance(u_train, list) else u_train
            dummy_u = dummy_u[:10] if dummy_u is not None else None
            model_sim.fit(dummy_x[:10], t=dt, u=dummy_u)
            model_sim.optimizer.coef_ = model.optimizer.coef_

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
                if r2_score(x_ref[:min_len], x_sim[:min_len]) < constraints["min_r2"]:
                    return {"configuration": config, "error": "Model have low R2 score"}

            # Vznikla ina neocakavana chyba
            except Exception as e:
                return {"configuration": config, "error": str(e)}

        else:
            model_sim = model
            # Rychly check pre standardne modely:
            # model.score() pocita R2 derivacie x_dot = f(x), nie trajektorie x(t).
            # Je to rychlejsie ako simulacia.
            if model_sim.score(x_val, dt, x_dot_val, u_val) < constraints["min_r2"]:
                return {"configuration": config, "error": "Model have low R2 score"}

        # FINAL SIMULATION
        # Dlhsia simulacia pre vypocet finalnych metrik.
        current_steps = min(total_val_samples, constraints["sim_steps"])
        start_index = max(0, total_val_samples - current_steps)

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

            # Ak model simuluje prilis nespravne (diverguje do nekonecna)
            if np.max(np.abs(x_sim)) > constraints["max_state"] or not np.all(np.isfinite(x_sim)):
                return {"configuration": config, "error": "Model diverg too much (exceed max state) or is not stable"}

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
                "r2_score": r2,
                "rmse": rmse,
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
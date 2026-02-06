import pysindy as ps
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import json
import pickle
import multiprocessing
import tempfile
import os
import gc
import warnings

from models.base import BaseSindyEstimator
from utils.config_manager import ConfigManager
from utils.plots import plot_trajectory, plot_pareto
import utils.sindy_helpers as sindy_helpers
from scripts.sindy_run_configuration import run_config

PysindyConfigObject = Union[ps.optimizers.BaseOptimizer, ps.feature_library.base.BaseFeatureLibrary, ps.differentiation.BaseDifferentiation]

class SindyEstimator(BaseSindyEstimator):
    """
    A comprehensive class for estimating SINDy models, managing configurations,
    performing parallel searches for optimal models, and evaluating results.

    Emphasizes modularity, configuration management, and memory efficiency.
    """

    def __init__(self, config_manager: ConfigManager):
        """
        Initializes the SINDYcEstimator with a ConfigManager instance.
        """
        self.pareto_front: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []
        self.best_config: Optional[Dict[str, Any]] = None
        self.results_file_name: Optional[str] = None

        self.config_manager = config_manager
        config_manager.load_config("settings")
        self.data_export_path = Path(self.config_manager.get_path("settings.paths.data_export_dir"))
        self._default_constraints: Dict[str, Any] = self.config_manager.get_param('settings.valid_methods.search_constraints', default={})
        
        super().__init__(config_manager)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._delete_tempfiles()
        self.differentiation_methods.clear()
        self.optimizers.clear()
        self.feature_libraries.clear()
        self.configurations.clear()
        self.results.clear()
        
    # Paralelne hladanie najlepsej konfiguracie
    def search_configurations(
        self,
        x_train: np.ndarray|List[np.ndarray],
        x_val: np.ndarray|List[np.ndarray],
        u_train: Optional[np.ndarray|List[np.ndarray]] = None,
        u_val: Optional[np.ndarray|List[np.ndarray]] = None,
        dt: float = None,
        n_processes: int = 4,
        log_file_name: str = "worker_results",
        verbose: bool = True,
        **constraints: Any
    ):

        if not self.configurations:
            raise ValueError("No configurations defined. Use generate_configurations() first.")

        if constraints:
            self._default_constraints.update(constraints)

        self._default_constraints.update(self.config_manager.get_param('settings.constants.SINDYEstimator', default={}))
        
        total_val_samples = x_val.shape[0]

        if self._default_constraints.get("sim_steps") <= self._default_constraints.get("min_validation_sim_steps"):
            self._default_constraints["sim_steps"] = self._default_constraints["min_validation_sim_steps"]
            warnings.warn(f"Minimum required simulation steps are {self._default_constraints["min_validation_sim_steps"]},"
                          f"validation steps increased/decreased automatically to match this requirement.")

        if total_val_samples < self._default_constraints.get("sim_steps"):
            raise ValueError(f"Not enough validation samples. Decrease validation steps to {total_val_samples} or increase validation size.")

        # Pouzitie multiprocessing Manager-a pre zdielanu pamat
        with multiprocessing.Manager() as manager:
            # Cache pre vypocitane derivacie (x_dot).
            cache_dict = manager.dict()
            lock = manager.Lock()

            # Zbalenie dat a konfiguracie, kvoli multiprocessingu (argument pre map funkciu)
            configurations_and_data = [
                    (index, config, x_train, x_val, u_train, u_val, dt, self._default_constraints, cache_dict, lock)
                    for index, config in enumerate(self.configurations)
                ]
            total_configurations = len(configurations_and_data)

            self.configurations.clear()

            if verbose:
                start_time = datetime.now()
                print(f"\nParameter search started...")
                print(f"Total configurations to explore: {total_configurations}")
                print(f"Using {n_processes} parallel processes")
                print(f"Start time: {start_time.strftime("%H:%M:%S")}")

            # Otvorenie docasnych suborov na zapis konfiguracii
            # Subor namiesto drzania v pamati, pretoze vysledky mozu byt velke
            log_filepath = self.data_export_path / f"{log_file_name}.log"
            if os.path.exists(log_filepath):
                os.remove(log_filepath)

            with tempfile.NamedTemporaryFile(delete=False, mode="wb", prefix="sindyR", suffix=".pkl") as results_file, \
                 open(log_filepath, "a", encoding="utf-8") as f_log:
                
                self.results_file_name = results_file.name

                # Multiprocessing
                with multiprocessing.Pool(processes=n_processes) as pool:
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
                            warnings.warn(str(e))
                        # UI/UX - progress bar
                        if verbose:
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
        if verbose:
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
        
        if constraints:
            self._default_constraints.update(constraints)

        self._default_constraints.update(self.config_manager.get_param('settings.defaults.constants.SINDYEstimator', default={}))

        data = {
            "x_train": x_train,
            "x_ref": x_test,
            "u_train": u_train,
            "u_ref": u_test,
            "dt": dt
        }

        config = self.best_config["configuration"]
        if "differentiation_method" not in config:
            # Pre pripad, ze by sa v configu nenachadzala kluc "differentiation_method", ale je potrebna pre validaciu
            config["differentiation_method"] = config["feature_library"].get_params().get("differentiation_method")
            
        np.random.seed(self.best_config.get("random_seed"))
        # Ignorovanie warningov pocas testovania
        warnings.filterwarnings("ignore", module="pysindy")

        config = sindy_helpers.sanitize_WeakPDELibrary(config)
        model = sindy_helpers.make_model(config, data)

        if constraints.get("coeff_precision") is not None:
            if constraints["coeff_precision"] == 0:
                model.optimizer.coef_ = np.rint(model.optimizer.coef_)
            else:
                model.optimizer.coef_ = np.round(model.optimizer.coef_, decimals=constraints["coeff_precision"])

        model_sim = sindy_helpers.make_model_callable(model, data)

        print("\nStarting validation on test data...")
        x_sim, rmse, r2, _ = sindy_helpers.evaluate_model(
            model_sim,
            data,
            start_index=0,
            current_steps=x_test.shape[0],
            integrator_kwargs={"rtol": 1e-6,"atol": 1e-6}
        )
        warnings.filterwarnings("default", category=UserWarning)
        
        min_len = min(len(x_test), len(x_sim))        

        print(f"Best model R2 score: {r2:.3%}")
        self.best_config["test_metrics"] = {
            "rmse": np.round(rmse, 5),
            "r2": np.round(r2, 5),
            "simulation_length": min_len
        }

        if plot:
            t_test = np.arange(min_len) * dt
            x_ref_cut = x_test[:min_len]
            x_sim_cut = x_sim[:min_len]
            plot_trajectory(t_test, x_ref_cut, x_sim_cut, title="Validation on test data")
        
        return None

    # Export dat do JSON
    def export_data(self, data: dict = None, export_file_name: str = "data"):
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
            filepath = self.data_export_path / f"{export_file_name}.json" 
            with open(filepath, "w", encoding="utf-8") as f:
                # default=str zabezpeci serializaciu objektov ako NumPy polia
                json.dump(payload, f, indent=5, default=str)
        except Exception as e:
            warnings.warn(str(e))

        return None


    def plot_pareto(self):
        plot_pareto(self.pareto_front)
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
        # Hladame dominantne riesenia
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
        
        # Vyber na zaklade AIC (Akaike Information Criterion)
        sorted_results = sorted(results, key=lambda x: x["aic"])
        best_model = sorted_results[0]
        
        return best_model

    def _delete_tempfiles(self):
        if self.results_file_name and os.path.exists(self.results_file_name):
            os.remove(self.results_file_name)
            
        return None
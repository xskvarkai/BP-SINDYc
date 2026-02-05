import pysindy as ps
import warnings
import itertools
from typing import List, Dict, Any, Optional, Union

from utils.custom_libraries import FixedCustomLibrary, FixedWeakPDELibrary
from utils.config_manager import ConfigManager

PysindyConfigObject = Union[ps.optimizers.BaseOptimizer, ps.feature_library.base.BaseFeatureLibrary, ps.differentiation.BaseDifferentiation]

class BaseSindyEstimator():
    def __init__(self, config_manager: ConfigManager):
        self.differentiation_methods: List[PysindyConfigObject] = []
        self.optimizers: List[PysindyConfigObject] = []
        self.feature_libraries: List[PysindyConfigObject] = []
        self.configurations: List[Dict[str, Any]] = []

        self.config_manager = config_manager

        self._default_diff_params: Dict[str, Any] = self.config_manager.get_param('settings.valid_methods.differentiation_methods', default={})
        self._default_opt_params: Dict[str, Any] = self.config_manager.get_param('settings.valid_methods.optimizers', default={})
        self._default_ensemble_opt_params: Dict[str, Any] = self.config_manager.get_param('settings.valid_methods.ensemble_optimizer', default={})
        self._default_feat_lib_params: Dict[str, Any] = self.config_manager.get_param('settings.valid_methods.feature_libraries', default={})

    def set_differentiation_method(self, method: str, **kwargs: Any):
        """
        Sets a differentiation method for SINDy.
        """

        valid_methods = {
            "FiniteDifference": ps.FiniteDifference,
            "SmoothedFiniteDifference": ps.SmoothedFiniteDifference
        }

        if method not in valid_methods:
            raise ValueError(f"Invalid method {method}. Choose from {list(valid_methods.keys())}")

        method_kwargs = self._validate_and_merge_kwargs(method, "differentiation method", kwargs, self._default_diff_params)
        
        if method == "SmoothedFiniteDifference":
            method_kwargs["window_length"] = max(11, method_kwargs["window_length"] if method_kwargs["window_length"] % 2 != 0 else method_kwargs["window_length"] + 1)
            self.differentiation_methods.append(valid_methods[method](smoother_kws=method_kwargs))
            
            return None
        
        self.differentiation_methods.append(valid_methods[method](**method_kwargs))

        return None
    
    def set_optimizer(self, method: str, ensemble: bool = False, ensemble_kwargs: Optional[Dict] = None, **kwargs: Any):
        """
        Sets an optimizer for sparse regression in SINDy.
        """

        valid_methods = {
            "STLSQ": ps.STLSQ,
            "SR3": ps.SR3
        }

        if method not in valid_methods:
            raise ValueError(f"Invalid method {method}. Choose from {list(valid_methods.keys())}")

        method_kwargs = self._validate_and_merge_kwargs(method, "optimizer", kwargs, self._default_opt_params)
        method_kwargs["max_iter"] = int(method_kwargs.get("max_iter", 1e5))

        base_optimizer = valid_methods[method](**method_kwargs)

        if ensemble:
            default_ensemble_kwargs = self._default_ensemble_opt_params.copy()
            if ensemble_kwargs:
                default_ensemble_kwargs.update(ensemble_kwargs)

            default_ensemble_kwargs["n_subset"] = int(default_ensemble_kwargs["n_subset"]) if default_ensemble_kwargs.get("n_subset") is not None else None
            final_optimizer = ps.EnsembleOptimizer(opt=base_optimizer, **default_ensemble_kwargs)
            self.optimizers.append(final_optimizer)

        else:
            self.optimizers.append(base_optimizer)
        
        return None

    def set_feature_library(self, method: str, **kwargs: Any):
        """
        Sets a feature library for SINDy.
        """

        valid_methods = {
            "PolynomialLibrary": ps.PolynomialLibrary,
            "FourierLibrary": ps.FourierLibrary,
            "CustomLibrary": FixedCustomLibrary,
            "ConcatLibrary": ps.ConcatLibrary,
            "TensoredLibrary": ps.TensoredLibrary,
            "WeakPDELibrary": FixedWeakPDELibrary
        }

        if method not in valid_methods:
            raise ValueError(f"Invalid method. Choose from {list(valid_methods.keys())}")

        method_kwargs = self._validate_and_merge_kwargs(method, "feature library", kwargs, self._default_feat_lib_params)
        self.feature_libraries.append(valid_methods[method](**method_kwargs))

        return None

    def generate_configurations(self):
        """
        Generates all possible combinations of differentiation methods, optimizers,
        and feature libraries defined by the user.
        """

        if not self.differentiation_methods:
            raise ValueError("No differentiation methods defined. Use set_differentiation_method() first.")
        if not self.optimizers:
            raise ValueError("No optimizers defined. Use set_optimizers() first.")
        if not self.feature_libraries:
            raise ValueError("No feature libraries defined. Use set_feature_library() first.")

        configurations = []
        for feature_library in self.feature_libraries:
            for differentiation_method in self.differentiation_methods:
                for optimizer in self.optimizers:
                    configurations.append({
                        "differentiation_method": differentiation_method,
                        "optimizer": optimizer,
                        "feature_library": feature_library
                    })

        self.configurations = configurations
        
        # Vycistenie bufferu
        self.differentiation_methods.clear()
        self.optimizers.clear()
        self.feature_libraries.clear()

        return None
    
    def _validate_and_merge_kwargs(
            self,
            method_name: str,
            method_type: str,
            user_kwargs: Dict[str, Any],
            default_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validates and merges user-provided keyword arguments with default configuration.
        """

        final_kwargs = default_kwargs.get(method_name, {}).copy()
        
        for key, value in user_kwargs.items():
            if key not in final_kwargs:
                warnings.warn(f"Unexpected parameter '{key}' for {method_name} {method_type}. Ignoring.")
            else:
                final_kwargs[key] = value

        return final_kwargs
    
    def _expand_kwargs(self, kwargs: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not kwargs:
            return [{}]

        keys = list(kwargs.keys())
        processed_values = []
        for key in keys:
            value = kwargs[key]
            if isinstance(value, list):
                processed_values.append(value)
            else:
                processed_values.append([value])

        expanded_dicts = []
        for product_values in itertools.product(*processed_values):
            expanded_dicts.append(dict(zip(keys, product_values)))
        return expanded_dicts

    def make_grid(
            self,
            feature_library_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
            differentiation_method_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
            optimizer_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        
        _feature_libraries_names = feature_library_kwargs.keys() if feature_library_kwargs else ["PolynomialLibrary"]
        _differentiation_methods_names = differentiation_method_kwargs.keys() if differentiation_method_kwargs else ["FiniteDifference"]
        _optimizers_names = optimizer_kwargs.keys() if optimizer_kwargs else ["STLSQ"]

        self.feature_libraries.clear()
        self.differentiation_methods.clear()
        self.optimizers.clear()
        self.configurations.clear()

        for fl_name in _feature_libraries_names:
            specific_fl_kwargs = feature_library_kwargs.get(fl_name, {}) if feature_library_kwargs else {}
            expanded_kwargs_list = self._expand_kwargs(specific_fl_kwargs)
            for kwargs_dict in expanded_kwargs_list:
                self.set_feature_library(fl_name, **kwargs_dict)

        for dm_name in _differentiation_methods_names:
            specific_dm_kwargs = differentiation_method_kwargs.get(dm_name, {}) if differentiation_method_kwargs else {}
            expanded_kwargs_list = self._expand_kwargs(specific_dm_kwargs)
            for kwargs_dict in expanded_kwargs_list:
                self.set_differentiation_method(dm_name, **kwargs_dict)

        for opt_name in _optimizers_names:
            specific_opt_kwargs = optimizer_kwargs.get(opt_name, {}) if optimizer_kwargs else {}
            expanded_kwargs_list = self._expand_kwargs(specific_opt_kwargs)
            for kwargs_dict in expanded_kwargs_list:
                self.set_optimizer(opt_name, **kwargs_dict)

        self.generate_configurations()

        return None
import pysindy as ps
import warnings
import itertools
from typing import List, Dict, Any, Optional, Union

from utils.custom_libraries import FixedCustomLibrary, FixedWeakPDELibrary
from utils.config_manager import ConfigManager

PysindyConfigObject = Union[ps.optimizers.BaseOptimizer, ps.feature_library.base.BaseFeatureLibrary, ps.differentiation.BaseDifferentiation] # Define a type alias for PySINDy configuration objects to improve readability

class BaseSindyEstimator():
    """
    A base class providing a framework for configuring SINDy models.
    It allows users to dynamically set differentiation methods, optimizers, and feature libraries,
    and generates all possible combinations of these configurations.
    """

    def __init__(self, config_manager: ConfigManager):
        """
        Initializes the BaseSindyEstimator with a ConfigManager instance.

        Args:
            config_manager (ConfigManager): An instance of ConfigManager to access configuration settings.
        """

        # Lists to store configured PySINDy objects
        self.differentiation_methods: List[PysindyConfigObject] = []
        self.optimizers: List[PysindyConfigObject] = []
        self.feature_libraries: List[PysindyConfigObject] = []
        self.configurations: List[Dict[str, Any]] = [] # List to store all generated configurations

        self.config_manager = config_manager

        # Load default parameters for various SINDy components from configuration
        self._default_diff_params: Dict[str, Any] = self.config_manager.get_param('settings.valid_methods.differentiation_methods', default={})
        self._default_opt_params: Dict[str, Any] = self.config_manager.get_param('settings.valid_methods.optimizers', default={})
        self._default_ensemble_opt_params: Dict[str, Any] = self.config_manager.get_param('settings.valid_methods.ensemble_optimizer', default={})
        self._default_feat_lib_params: Dict[str, Any] = self.config_manager.get_param('settings.valid_methods.feature_libraries', default={})

    def set_differentiation_method(self, method: str, **kwargs: Any):
        """
        Sets a differentiation method for SINDy.
        Validates the method against predefined options and merges user-specified
        parameters with default configuration settings.

        Args:
            method (str): The name of the differentiation method (e.g., "FiniteDifference", "SmoothedFiniteDifference").
            **kwargs (Any): Additional keyword arguments to configure the differentiation method.

        Raises:
            ValueError: If an invalid differentiation method is specified.
        """

        valid_methods = { # Dictionary of valid PySINDy differentiation methods
            "FiniteDifference": ps.FiniteDifference,
            "SmoothedFiniteDifference": ps.SmoothedFiniteDifference
        }

        if method not in valid_methods:
            raise ValueError(f"Invalid method {method}. Choose from {list(valid_methods.keys())}")

        method_kwargs = self._validate_and_merge_kwargs(method, "differentiation method", kwargs, self._default_diff_params) # Validate and merge user-provided kwargs with defaults

        if method == "SmoothedFiniteDifference": # Special handling for SmoothedFiniteDifference's window_length to ensure it's odd and at least 11
            method_kwargs["window_length"] = max(11, method_kwargs["window_length"] if method_kwargs["window_length"] % 2 != 0 else method_kwargs["window_length"] + 1)
            self.differentiation_methods.append(valid_methods[method](smoother_kws=method_kwargs))

            return None

        self.differentiation_methods.append(valid_methods[method](**method_kwargs))

        return None

    def set_optimizer(self, method: str, ensemble: bool = False, ensemble_kwargs: Optional[Dict] = None, **kwargs: Any):
        """
        Sets an optimizer for sparse regression in SINDy.
        Supports standard optimizers and an optional ensemble mode for robust estimation.

        Args:
            method (str): The name of the optimizer (e.g., "STLSQ", "MIOSR", "SR3", "ConstrainedSR3").
            ensemble (bool): If True, the optimizer will be wrapped in an EnsembleOptimizer.
            ensemble_kwargs (Optional[Dict]): Keyword arguments specific to the EnsembleOptimizer.
            **kwargs (Any): Additional keyword arguments to configure the base optimizer.

        Raises:
            ValueError: If an invalid optimizer method is specified.
        """

        valid_methods = { # Dictionary of valid PySINDy optimizers
            "STLSQ": ps.STLSQ,
            "MIOSR": ps.MIOSR,
            "SR3": ps.SR3,
            "ConstrainedSR3": ps.ConstrainedSR3
        }

        if method not in valid_methods:
            raise ValueError(f"Invalid method {method}. Choose from {list(valid_methods.keys())}")

        method_kwargs = self._validate_and_merge_kwargs(method, "optimizer", kwargs, self._default_opt_params) # Dictionary of valid PySINDy optimizers
        if method != "MIOSR": # Ensure max_iter is an integer, especially for non-MIOSR optimizers
            method_kwargs["max_iter"] = int(method_kwargs.get("max_iter", 1e5))

        base_optimizer = valid_methods[method](**method_kwargs)

        if ensemble: # If ensemble mode is enabled, wrap the base optimizer with an EnsembleOptimizer
            default_ensemble_kwargs = self._default_ensemble_opt_params.copy()
            if ensemble_kwargs:
                default_ensemble_kwargs.update(ensemble_kwargs)

            default_ensemble_kwargs["n_subset"] = int(default_ensemble_kwargs["n_subset"]) if default_ensemble_kwargs.get("n_subset") is not None else None # Ensure n_subset is an integer if specified
            final_optimizer = ps.EnsembleOptimizer(opt=base_optimizer, **default_ensemble_kwargs)
            self.optimizers.append(final_optimizer)

        else:
            self.optimizers.append(base_optimizer)

        return None

    def set_feature_library(self, method: str, **kwargs: Any):
        """
        Sets a feature library for SINDy.
        Validates the method against predefined options and merges user-specified
        parameters with default configuration settings.

        Args:
            method (str): The name of the feature library (e.g., "PolynomialLibrary", "FourierLibrary", "CustomLibrary").
            **kwargs (Any): Additional keyword arguments to configure the feature library.

        Raises:
            ValueError: If an invalid feature library method is specified.
        """

        valid_methods = { # Dictionary of valid PySINDy feature libraries, including custom ones
            "PolynomialLibrary": ps.PolynomialLibrary,
            "FourierLibrary": ps.FourierLibrary,
            "CustomLibrary": FixedCustomLibrary, # Custom library from local module
            "ConcatLibrary": ps.ConcatLibrary,
            "TensoredLibrary": ps.TensoredLibrary,
            "WeakPDELibrary": FixedWeakPDELibrary # Custom Weak PDE library from local module
        }

        if method not in valid_methods:
            raise ValueError(f"Invalid method. Choose from {list(valid_methods.keys())}")

        method_kwargs = self._validate_and_merge_kwargs(method, "feature library", kwargs, self._default_feat_lib_params) # Validate and merge user-provided kwargs with defaults

        self.feature_libraries.append(valid_methods[method](**method_kwargs))

        return None

    def generate_configurations(self):
        """
        Generates all possible combinations of differentiation methods, optimizers,
        and feature libraries that have been defined by the user.
        The generated configurations are stored in the `configurations` attribute.
        After generation, the individual method lists are cleared to prevent reuse.

        Raises:
            ValueError: If any of the method lists (differentiation, optimizers, feature libraries) are empty.
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

        # Clear buffers to prevent accidental reuse of the same objects in subsequent calls
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
        Internal method to validate and merge user-provided keyword arguments with default configuration.
        It issues warnings for unexpected parameters and prioritizes user-defined values.

        Args:
            method_name (str): The name of the method being configured.
            method_type (str): The type of method (e.g., "differentiation method", "optimizer").
            user_kwargs (Dict[str, Any]): Keyword arguments provided by the user.
            default_kwargs (Dict[str, Any]): Default keyword arguments for the method.

        Returns:
            Dict[str, Any]: A dictionary containing the final merged keyword arguments.
        """

        final_kwargs = default_kwargs.get(method_name, {}).copy()

        for key, value in user_kwargs.items():
            if key not in final_kwargs:
                warnings.warn(f"Unexpected parameter '{key}' for {method_name} {method_type}. Ignoring.")
            else:
                final_kwargs[key] = value

        return final_kwargs

    def _expand_kwargs(self, kwargs: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Internal method to expand a dictionary of keyword arguments where some values might be lists.
        It generates all combinations of these list values into separate dictionaries.

        Args:
            kwargs (Optional[Dict[str, Any]]): A dictionary of keyword arguments, possibly with list values.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a unique combination of keyword arguments.
        """
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
        for product_values in itertools.product(*processed_values): # Generate Cartesian product of all value lists
            expanded_dicts.append(dict(zip(keys, product_values)))
        return expanded_dicts

    def make_grid(
            self,
            feature_library_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
            differentiation_method_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
            optimizer_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Prepares the estimator for a grid search by setting up multiple configurations
        for feature libraries, differentiation methods, and optimizers.
        This method clears any previously set configurations and then generates new ones
        based on the provided keyword arguments.

        Args:
            feature_library_kwargs (Optional[Dict[str, Dict[str, Any]]]): A dictionary where keys are feature
                                                                           library names and values are dictionaries
                                                                           of parameters. Parameter values can be lists
                                                                           to explore multiple options.
            differentiation_method_kwargs (Optional[Dict[str, Dict[str, Any]]]): Similar to feature_library_kwargs,
                                                                               but for differentiation methods.
            optimizer_kwargs (Optional[Dict[str, Dict[str, Any]]]): Similar to feature_library_kwargs,
                                                                  but for optimizers.
        """

        # Determine names of methods to configure, defaulting to basic options if none provided
        _feature_libraries_names = feature_library_kwargs.keys() if feature_library_kwargs else ["PolynomialLibrary"]
        _differentiation_methods_names = differentiation_method_kwargs.keys() if differentiation_method_kwargs else ["FiniteDifference"]
        _optimizers_names = optimizer_kwargs.keys() if optimizer_kwargs else ["STLSQ"]

        # Clear all existing configurations and method lists to prepare for new grid generation
        self.feature_libraries.clear()
        self.differentiation_methods.clear()
        self.optimizers.clear()
        self.configurations.clear()

        for fl_name in _feature_libraries_names: # Iterate through specified feature libraries and their expanded kwargs to set them
            specific_fl_kwargs = feature_library_kwargs.get(fl_name, {}) if feature_library_kwargs else {}
            expanded_kwargs_list = self._expand_kwargs(specific_fl_kwargs)
            for kwargs_dict in expanded_kwargs_list:
                self.set_feature_library(fl_name, **kwargs_dict)

        for dm_name in _differentiation_methods_names: # Iterate through specified differentiation methods and their expanded kwargs to set them
            specific_dm_kwargs = differentiation_method_kwargs.get(dm_name, {}) if differentiation_method_kwargs else {}
            expanded_kwargs_list = self._expand_kwargs(specific_dm_kwargs)
            for kwargs_dict in expanded_kwargs_list:
                self.set_differentiation_method(dm_name, **kwargs_dict)

        for opt_name in _optimizers_names: # Iterate through specified optimizers and their expanded kwargs to set them
            specific_opt_kwargs = optimizer_kwargs.get(opt_name, {}) if optimizer_kwargs else {}
            expanded_kwargs_list = self._expand_kwargs(specific_opt_kwargs)
            for kwargs_dict in expanded_kwargs_list:
                self.set_optimizer(opt_name, **kwargs_dict)

        self.generate_configurations() # Finally, generate all possible combinations of the set methods

        return None
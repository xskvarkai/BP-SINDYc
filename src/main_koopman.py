import numpy as np

from utils.config_manager import ConfigManager
from data_ingestion.data_loader import DataLoader
from data_processing.data_splitter import TimeSeriesSplitter
from models.koopman_model import KoopmanModel
import warnings
warnings.filterwarnings("ignore", module="pykoopman")
import pykoopman

from utils.custom_libraries import x, sin_x, squared_x, \
                                   name_x, name_sin_x, name_squared_x

if __name__ == "__main__":
    config_manager = ConfigManager("config")
    config_manager.load_config("koopman_params")

    np.random.seed(config_manager.get_param("koopman_params.global.random_seed", 42))
    random_number_generator = np.random.RandomState(config_manager.get_param("koopman_params.global.random_seed", 42))

    with DataLoader(config_manager) as loader:
        X, U, dt = loader.load_csv_data(
            **config_manager.get_param("koopman_params.data_loading")
        )

    with TimeSeriesSplitter(config_manager, X, dt, U) as splitter:
        X_train, X_test, _, U_train, U_test, _ = splitter.split_data(
            **config_manager.get_param("koopman_params.data_splitting"), rng=random_number_generator
        )

    function_names=[name_x, name_sin_x, name_squared_x]
    library_functions=[x, sin_x, squared_x]
    
    config = {
        "observables": pykoopman.observables.CustomObservables(library_functions, function_names),
        "regressor": pykoopman.regression.EDMDc()
    }

    with KoopmanModel(config_manager, config, X_train, None, U_train, X_test, U_test, dt) as koopman_model:
        koopman_model.evaluateModel(print_metrics=True)
        koopman_model.plot_koopman_spectrum()
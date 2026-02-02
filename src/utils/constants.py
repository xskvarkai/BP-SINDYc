# Konstanty pre funkciu find_periodicity v helpers.py
NOISE_3SIGMA_FACTOR = 3.0
PERIODICITY_CONCENTRATION_THRESHOLD = 0.45

# Konstanty pre funkciu estimate_threshold v helpers.py
LOWER_BOUND_PERCENTILE = 5
UPPER_BOUND_PERCENTILE = 75
N_THRESHOLDS = 4

# Konstanty pre triedu SINDYcEstimator v sindy_model.py
MIN_VALIDATION_SIM_STEPS = 100
DEFAULT_RANDOM_SEED = 42
ENSEMBLE_N_SUBSET = 0.6

# Konstanty pre funkciu noise_level v dynamics_systems.py
MINIMAL_NOISE_VALUE = 1e-3
DEFAULT_SIMULATION_RANDOM_SEED = 100

# Konstanty pre cesty
SIM_DATA_EXPORT_PATH = "data/raw"
DATA_EXPORT_PATH = "data/processed"
DATA_LOAD_PATH = "data/raw"
CONFIGURATION_PATH = "config"

# Konstanty pre vyhladzovanie dat
SAVGOL_WINDOW_LENGTH = 31
SAVGOL_POLYORDER = 2

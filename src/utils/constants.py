# Konstanty pre funkciu find_periodicity v helpers.py
NOISE_3SIGMA_FACTOR = 3.0
PERIODICITY_CONCENTRATION_THRESHOLD = 0.45

# Konstanty pre triedu SINDYcEstimator v sindy_model.py
MIN_VALIDATION_SIM_STEPS = 500
DEFAULT_RANDOM_SEED = 42

# Konstanty pre funkciu noise_level v dynamics_systems.py
MINIMAL_NOISE_VALUE = 1e-6
DEFAULT_SIMULATION_RANDOM_SEED = 100

# Konstanty pre cesty
SIM_DATA_EXPORT_PATH = "data/raw"
DATA_EXPORT_PATH = "data/processed"
DATA_LOAD_PATH = "data/raw"
CONFIGURATION_PATH = "config"

# Konstanty pre vyhladzovanie dat
SAVGOL_WINDOW_LENGTH = 11
SAVGOL_POLYORDER = 2
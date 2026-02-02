import numpy as np
import pysindy as ps
import warnings
from typing import Optional, List, Tuple
from scipy.stats import median_abs_deviation
import pywt

from utils.custom_libraries import FixedWeakPDELibrary
from utils import constants

def find_noise(x: np.ndarray, detail_level: int = 1) -> float:
    
    # Odhad sumu pomocou vlnkovej transformacie (Wavelet Denoising princip).
    # Predpoklad: sum je obsiahnuty v najvyssich frekvenciach (detail coefficients).
    # Pouziva sa 'Haar' wavelet, ktory je citlivy na nahle zmeny.
    
    coeffs = pywt.wavedec(x, "haar", axis=0)
    details = coeffs[-detail_level]

    # Vzorec: sigma = MAD / 0.6745 (pre normalne rozdelenie sumu)
    sigma_noise = median_abs_deviation(details, scale="normal", axis=None) / np.sqrt(2)
    
    print(f"\nNoise Analysis -> Sigma: {sigma_noise:.4f}")
    return sigma_noise

def find_periodicity(x:np.ndarray, sigma_noise: float = 0.0) -> bool:
    
    # Spektralna analyza na urcenie, ci je signal periodicky alebo chaoticky/aperiodicky.
    
    N = len(x)

    # Aplikacia Hanningovho okna na potlacenie spectral leakage
    window = np.hanning(N)
    window = window[:, np.newaxis]

    # Centrovanie signalu (odstranenie DC zlozky) vazene oknom
    signal_centred = x - np.mean(x, axis=0) * window

    # Rychla Fourierova Transformacia - FFT realna cast
    fft_spectrum = np.fft.rfft(signal_centred, axis=0)

    # Vypocet amplitud a normalizacia
    amplitudes = np.abs(fft_spectrum) / N * 4
    amplitudes[0, :] = 0 # Nulovanie DC zlozky natvrdo
    
    # Prahovanie amplitud na zaklade odhadnuteho sumu (Hard Thresholding)
    if sigma_noise > 0:
        noise_threshold = constants.NOISE_3SIGMA_FACTOR * sigma_noise # 3-sigma pravidlo
        mask = amplitudes > noise_threshold
        amplitudes_clean = amplitudes * mask
    else:
        amplitudes_clean = amplitudes

    # Vypocet vykonoveho spektra (Power Spectrum Density - PSD)
    power_spectrum = amplitudes_clean ** 2
    power_spectrum[0, :] = 0
    total_energy = np.sum(power_spectrum, axis=0)

    # Ochrana proti deleniu nulou
    if total_energy.all() == 0:
        warnings.warn("Signal have zero energy!")
        return False
    total_energy[total_energy == 0] = 1.0

    # Vypocet koncentracie energie.
    # Ak je "vacsina" energie sustredena v maxime (dominantna frekvencia), signal je periodicky.
    max_peak = np.max(power_spectrum, axis=0)
    concentration = np.mean(max_peak / total_energy)

    # Heuristicky prah 0.45
    is_periodic = True if concentration > constants.PERIODICITY_CONCENTRATION_THRESHOLD else False

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
) -> Tuple[np.ndarray, float]:

    # Heuristicka funkcia na odhad optimalnej mriezky prahov (thresholds) pre algoritmus.
    # Namiesto slepeho hadania lambda parametra, SINDy s nulovym prahom,
    # pozrieme sa na vsetky koeficienty a vygenerujeme logaritmicku skalu
    # medzi najmensim a najvacsim relevantnym koeficientom.
    warnings.filterwarnings("ignore", module="pysindy.utils")
    if feature_library is None or x is None or dt is None:
        raise ValueError(f"Data (x), time_step (dt) and feature_library are required.")

    # Spustenie predbezneho modelu bez prahovania (metoda najmensich stvorcov)
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=0.0, alpha=1e-5, normalize_columns=True),
        feature_library=feature_library
    )

    model.fit(x=x, t=dt, u=u)
    coeffs = np.abs(model.coefficients())
    max_coeff = np.max(coeffs)

    # Ak budeme pouzivat normalizaciu stlpcov, musime koeficienty "denormalizovat" pre odhad,
    # alebo pracovat s vazenymi hodnotami, aby prahy davali zmysel v realnom meritku pre SR3
    if normalized_columns:
        if u is not None:
            u_reshaped = u.reshape(-1, 1) if u.ndim == 1 else u
            x_for_lib = np.hstack((x, u_reshaped))
        else:
            x_for_lib = x

        Theta = model.feature_library.transform(x_for_lib)
        norms = np.linalg.norm(Theta, axis=0)
        norms[norms == 0] = 1.0 # Ochrana proti deleniu nulou
        coeffs = coeffs * norms
    
    # Nastavenie minimalneho prahu sumu
    coeffs_threshold = noise_level / constants.NOISE_3SIGMA_FACTOR if noise_level is not None else 1e-10
    non_zero_coeffs = coeffs[coeffs > coeffs_threshold].flatten()

    # Ak su vsetky koeficienty nulove (model nenasiel nic), vrati default grid
    if len(non_zero_coeffs) == 0:
        warnings.warn(f"All coefficients for feature_library {str(feature_library)} are nearly to zero. Returning default grid.")
        return np.logspace(-3, 1, 4)

    # Odrezanie extremov (percentily)
    lower_bound = np.percentile(non_zero_coeffs, constants.LOWER_BOUND_PERCENTILE)
    upper_bound = np.percentile(non_zero_coeffs, constants.UPPER_BOUND_PERCENTILE)
    trimmed_coeffs = non_zero_coeffs[(non_zero_coeffs >= lower_bound) & (non_zero_coeffs <= upper_bound)]
    if len(trimmed_coeffs) < 2:
        trimmed_coeffs = non_zero_coeffs
    
    min_val = np.min(trimmed_coeffs)
    max_val = np.max(trimmed_coeffs)

    # Generovanie 4 bodov na logaritmickej skale pre grid search
    thresholds_non_rounded = np.logspace(np.log10(min_val), np.log10(max_val), constants.N_THRESHOLDS)
    thresholds = [np.round(threshold, decimals=8) for threshold in thresholds_non_rounded]

    return thresholds, max_coeff + coeffs_threshold

def generate_trajectories(
    x_train: np.ndarray,
    u_train: Optional[np.ndarray] = None,
    num_samples: int = 10000,
    num_trajectories: int = 5,
    randomseed: int = constants.DEFAULT_RANDOM_SEED,
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:

    # Funkcia nahodne "vysekne" pod-trajektorie z treningovych dat.

    np.random.seed(randomseed)
    # Ziskanie poctu vzoriek tranovacej sady
    total_train_samples = x_train.shape[0]

    x_multi = [] # Zoznam pre stavove premenne
    u_multi = [] # Zoznam pre riadece signaly

    if num_trajectories == 1:
        return x_train, u_train

    for trajectory in range(0, num_trajectories):
        # Kontrola poctu dat na vytvorenie trajektorie pozadovanej dlzky
        if total_train_samples < num_samples:
            start_index = 0
            warnings.warn("Insufficient samples for diverse trajectories. Consider adding more data or reducing the sample size.")
        else:
            # Nahodny start, tak aby sa okno zmestilo do dat
            start_index = np.random.randint(0, total_train_samples - num_samples)
        
        end_index = start_index + num_samples
        trajectory = x_train[start_index:end_index]
        x_multi.append(trajectory)

        # Spracovanie riadiaceho signalu
        if not np.any(u_train) or u_train is None:
            u_multi = None
        else:
            input_signal = u_train[start_index:end_index]
            u_multi.append(input_signal)

    return x_multi, u_multi

def sanitize_WeakPDELibrary(config, time_vec):
    if isinstance(config.get("feature_library"), FixedWeakPDELibrary):
        params = config["feature_library"].get_params()
        
        config["feature_library"] = FixedWeakPDELibrary(
            function_library=params.get("function_library", None),
            derivative_order=params.get("derivative_order", 0),
            spatiotemporal_grid=time_vec,
            K=params.get("K", 100),
            p=params.get("p", 4),
            H_xt=np.asarray(params.get("H_xt")),
            differentiation_method=config["differentiation_method"]
        )
        config["differentiation_method"] = None
        
    return config

def compute_time_vector(x, dt):
    if isinstance(x, list):
        time_vec = (np.arange(x[0].shape[0]) * dt)
    else:
        time_vec = (np.arange(x.shape[0]) * dt)

    return time_vec

def make_model_callable(model, x_train, u_train, dt):
    if isinstance(model.feature_library, FixedWeakPDELibrary):
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
        
    return model_sim
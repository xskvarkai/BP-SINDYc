import numpy as np
import pysindy as ps
import warnings
from typing import Optional, List, Tuple, Union
from scipy.stats import median_abs_deviation
import pywt

def find_noise(x: np.ndarray, detail_level: int = 1, verbose: bool = True) -> float:
    """
    Estimates the noise level in a signal based on the median absolute deviation (MAD)
    of its Wavelet Denoising detail coefficients.
    This method assumes that noise primarily resides in the highest frequency components
    (detail coefficients) of the signal.

    Args:
        x (np.ndarray): The input signal (time series data) as a NumPy array.
        detail_level (int): The level of wavelet decomposition to consider for detail coefficients.
                            Higher values look at finer details (higher frequencies).
                            Defaults to 1, meaning the highest frequency detail.
        verbose (bool): If True, prints the estimated noise sigma.

    Returns:
        float: The estimated standard deviation (sigma) of the noise in the signal.
    """

    # Estimate noise using Wavelet Transform principles.
    # Assumption: noise is contained in the highest frequencies (detail coefficients).
    # 'Haar' wavelet is used due to its sensitivity to abrupt changes.

    coeffs = pywt.wavedec(x, "haar", axis=0)
    details = coeffs[-detail_level]

    # Formula: sigma = MAD / 0.6745 (for normally distributed noise)
    # The factor 0.6745 is for normal distribution. We use `scale="normal"` in median_abs_deviation
    # which already accounts for this. The division by sqrt(2) is specific to certain noise models.
    sigma_noise = median_abs_deviation(details, scale="normal", axis=None) / np.sqrt(2)

    if verbose:
        print(f"\nNoise Analysis -> Estimated noise sigma: {sigma_noise:.4f}.")

    return sigma_noise

def find_periodicity(
        x:np.ndarray,
        dt: Union[int, float],
        x_dim: Optional[int] = None,
        sigma_noise: float = 0.0,
        verbose: bool = True
    ) -> bool:
    """
    Checks for periodicity in a signal using its Fourier Transform.
    It applies a Hanning window to reduce spectral leakage, centers the signal,
    and then computes the power spectrum. Periodicity is determined by the
    concentration of energy in dominant frequency peaks, relative to a noise threshold.

    Args:
        x (np.ndarray): The input signal (time series data) as a NumPy array.
        dt (Union[int, float]): The time step of the data.
        x_dim (Optional[int]): If x is multi-dimensional, specify the column index (1-based)
                                to analyze for periodicity. If None, the first dimension is used.
        sigma_noise (float): Estimated noise level. Used for hard thresholding the Fourier amplitudes.
                             Defaults to 0.0 (no noise consideration).
        verbose (bool): If True, prints the periodicity status and dominant frequency.

    Returns:
        bool: True if the signal is considered periodic, False otherwise.
    """

    N = len(x)
    if x.ndim > 1 and x_dim: # Adjust x_dim to be 0-based for NumPy indexing
        x_to_analyze = x[:, x_dim - 1]
    elif x.ndim > 1: # Default to the first column if x_dim is not specified for multi-dimensional array
        # warnings.warn("x_dim not specified for multi-dimensional array, analyzing the first column.")
        x_to_analyze = x[:, 0]
    else:
        x_to_analyze = x

    window = np.hanning(N) # Apply Hanning window to suppress spectral leakage
    if x_to_analyze.ndim == 1: # Ensure window can be broadcast correctly if x_to_analyze is 1D
        signal_centred = x_to_analyze - np.mean(x_to_analyze) * window
    else: # Should not happen with current logic, but as a safeguard
        signal_centred = x_to_analyze - np.mean(x_to_analyze, axis=0) * window[:, np.newaxis]

    fft_spectrum = np.fft.rfft(signal_centred) # Fast Fourier Transform - real part

    amplitudes = np.abs(fft_spectrum) / N * 4 # Calculate amplitudes and normalize
    if amplitudes.ndim > 1:
        amplitudes[0, :] = 0 # Zero out DC component (mean)
    else:
        amplitudes[0] = 0

    if sigma_noise > 0: # Hard Thresholding of amplitudes based on estimated noise (3-sigma rule)
        noise_threshold = 3 * sigma_noise
        amplitudes_clean = np.where(amplitudes > noise_threshold, amplitudes, 0)
    else:
        amplitudes_clean = amplitudes

    power_spectrum = amplitudes_clean ** 2 # Calculate Power Spectrum Density (PSD)
    if power_spectrum.ndim > 1:
        power_spectrum[0, :] = 0
    else:
        power_spectrum[0] = 0

    total_energy = np.sum(power_spectrum, axis=0)

    if np.all(total_energy == 0): # Protection against division by zero
        warnings.warn("Signal has zero energy after noise filtering!")
        return False
    total_energy = np.where(total_energy == 0, 1.0, total_energy) # Replace zero total_energy with 1.0 to avoid NaNs in division if only some columns are zero

    max_peak = np.max(power_spectrum, axis=0) # If "most" of the energy is concentrated in the maximum (dominant frequency), the signal is periodic
    concentration = np.mean(max_peak / total_energy) # Calculate energy concentration

    is_periodic = concentration > 0.45 # Heuristic threshold 0.45 for periodicity

    status = "Periodic" if is_periodic else "Aperiodic"

    if verbose:
        print(f"\nPeriodicity Check -> Status: {status} (Concentration: {concentration:.3f}).")

    if is_periodic:
        max_peak_idx = np.argmax(power_spectrum, axis=0)
        freqs = np.fft.rfftfreq(N, d=dt)
        if power_spectrum.ndim > 1: # If multi-dimensional, take the dominant frequency of the analyzed column
            dominant_freq_hz = freqs[max_peak_idx]
        else:
            dominant_freq_hz = freqs[max_peak_idx]
        dominant_omega = 2 * np.pi * dominant_freq_hz
        if verbose:
            print(f"Dominant frequency (omega): {dominant_omega:.4f} rad/s")

    return is_periodic


def find_optimal_delay(x: np.ndarray, dt: float, u: np.ndarray=None) -> None:
    def evaluate_delay(tau_candidate, x, dt, u=None):
        delay_steps = int(tau_candidate / dt)
        x_delayed = np.roll(x, delay_steps)[delay_steps:]
        x_current = x[delay_steps:]
        u_current = u[delay_steps:] if u is not None else None
        
        X = np.hstack([x_current, x_delayed])
        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=0.0, normalize_columns=True), 
            differentiation_method=ps.SmoothedFiniteDifference(smoother_kws={"window_length": 31, "polyorder": 3}),
            feature_library=ps.PolynomialLibrary(degree=4, include_bias=True) + ps.FourierLibrary(n_frequencies=2)
        )
        model.fit(X, t=dt, u=u_current)
        
        return model.score(X, t=dt, u=u_current)
    
    best_tau = 0
    best_score = -np.inf
    for tau in np.linspace(0.1, 2.0, 20):
        score = evaluate_delay(tau, x, dt, u)
        if score > best_score:
            best_score = score
            best_tau = tau

    print(f"Optimálne oneskorenie: {best_tau}")
    print(f"Skóre modelu s optimálnym oneskorením: {best_score}")


def estimate_threshold(
        x: np.ndarray,
        dt: float,
        u: Optional[np.ndarray] = None,
        feature_library: ps.feature_library = None,
        n_threshold: Optional[int] = 4,
        noise_level: Optional[float] = None,
        verbose: bool = True
    ) -> List:
    """
    Estimates a range of candidate thresholds for sparse regression based on data characteristics.
    This function heuristically determines optimal thresholds for the SINDy algorithm's `lambda` parameter.
    Instead of blindly guessing, it runs a preliminary SINDy model with zero threshold (least squares),
    inspects all coefficients, and generates a logarithmic scale of `n_threshold` values
    between the smallest and largest relevant coefficients.

    Args:
        x (np.ndarray): The state variables (features) as a NumPy array.
        dt (float): The time step of the data.
        u (Optional[np.ndarray]): The control inputs as a NumPy array, if available.
        feature_library (ps.feature_library): The PySINDy feature library to be used.
        n_threshold (Optional[int]): The number of threshold values to generate. Defaults to 4.
        noise_level (Optional[float]): Estimated noise level. Used to define a minimal threshold
                                       to filter out very small coefficients likely due to noise.
        verbose (bool): If True, prints the estimated threshold values.

    Returns:
        List[float]: A list of logarithmically spaced candidate thresholds, from lowest to highest.

    Raises:
        ValueError: If data (x), time_step (dt), or feature_library are not provided.
    """

    warnings.filterwarnings("ignore", module="pysindy.utils")
    if feature_library is None or x is None or dt is None:
        raise ValueError(f"Data (x), time_step (dt) and feature_library are required.")

    model = ps.SINDy( # Run a preliminary model without thresholding (least squares method)
        optimizer=ps.STLSQ(threshold=0.0, alpha=1e-5, normalize_columns=True),
        feature_library=feature_library
    )

    model.fit(x=x, t=dt, u=u)
    coeffs = np.abs(model.coefficients())

    coeffs_threshold = noise_level / 3 if noise_level is not None else 1e-10 # Set a minimal noise threshold to consider coefficients as non-zero
    non_zero_coeffs = coeffs[coeffs > coeffs_threshold].flatten()

    if len(non_zero_coeffs) == 0: # If all coefficients are near zero (model found nothing), return a default grid
        warnings.warn(f"All coefficients for feature_library {str(feature_library)} are nearly to zero. Returning default grid.")
        return np.logspace(-3, 1, 4)

    lower_bound = np.percentile(non_zero_coeffs, 5)
    upper_bound = np.percentile(non_zero_coeffs, 95)
    trimmed_coeffs = non_zero_coeffs[(non_zero_coeffs >= lower_bound) & (non_zero_coeffs <= upper_bound)] # Trim extreme values (percentiles) to make the range more robust to outliers
    if len(trimmed_coeffs) < 2: # Ensure there are at least two points for logspace if trimming removes too many
        trimmed_coeffs = non_zero_coeffs

    min_val = np.min(trimmed_coeffs)
    max_val = np.max(trimmed_coeffs)

    thresholds_non_rounded = np.logspace(np.log10(min_val), np.log10(max_val), n_threshold) # Generate n_threshold points on a logarithmic scale for grid search
    thresholds = np.unique(np.round(thresholds_non_rounded, 8)) # np.unique is used to remove potential duplicate values if min_val and max_val are very close

    if verbose:
        print(f"\nEstimated threshold values -> {thresholds}.")

    return list(thresholds.flatten())

def generate_trajectories(
    x_train: np.ndarray,
    u_train: Optional[np.ndarray] = None,
    num_samples_per_trajectory: int = 10000,
    num_trajectories: int = 5,
    rng: Optional[np.random.RandomState] = np.random.RandomState(42),
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    """
    Generates random sub-trajectories from a single long trajectory.
    This is useful for creating multiple, shorter training sequences for SINDy,
    especially for Ordinary Differential Equation (ODE) systems.

    Warning: This method is not suitable for all types of systems (e.g., PDEs)
    as it may break temporal continuity and data correlations critical for some models.
    Use with caution and only when appropriate (e.g., for ODE systems with sufficiently long trajectories).

    Args:
        x_train (np.ndarray): The full training state variables as a NumPy array.
        u_train (Optional[np.ndarray]): The full training control inputs as a NumPy array, if available.
        num_samples_per_trajectory (int): The desired number of samples in each generated sub-trajectory.
        num_trajectories (int): The number of sub-trajectories to generate.
        rng (Optional[np.random.RandomState]): Random number generator for reproducibility.

    Returns:
        Tuple[List[np.ndarray], Optional[List[np.ndarray]]]: A tuple containing:
            - A list of NumPy arrays, where each array is a sub-trajectory of state variables.
            - An optional list of NumPy arrays, where each array is a sub-trajectory of control inputs.
              Returns None if u_train was None.
    """

    total_train_samples = x_train.shape[0] # Get total number of samples in the training set

    x_multi = [] # List for state variable sub-trajectories
    u_multi = [] # List for control input sub-trajectories

    if num_trajectories == 1: # If only one trajectory is requested, return the original data as lists
        return [x_train], [u_train]

    for trajectory in range(0, num_trajectories):
        if total_train_samples < num_samples_per_trajectory: # If there is not enough samples
            raise ValueError(f"Not enough training samples ({total_train_samples}) to create a trajectory of length {num_samples_per_trajectory}.")
        else:
            start_index = rng.randint(0, total_train_samples - num_samples_per_trajectory) # Ensure the sub-trajectory does not go out of bounds

        end_index = start_index + num_samples_per_trajectory
        trajectory = x_train[start_index:end_index]
        x_multi.append(trajectory)

        if u_train is None:
            u_multi = None
        else:
            input_signal = u_train[start_index:end_index]
            u_multi.append(input_signal)

    return x_multi, u_multi
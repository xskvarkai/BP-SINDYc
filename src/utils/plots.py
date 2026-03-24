import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Union, Optional, List

def _set_size(width_pt=390.0, fraction=1, subplots=(1, 1)):
    fig_width_in = width_pt * fraction / 72.27
    
    golden_ratio = (5**.5 - 1) / 2
    
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    
    return (fig_width_in, fig_height_in)

def _prepare_export():
    plt.rcParams.update({
        "pgf.preamble": "\n".join([
            r"\usepackage{mathptmx}",
        ]),
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "xtick.labelsize": 12 * 0.8,
        "ytick.labelsize": 12 * 0.8,
        "legend.fontsize": 12 * 0.6,
        "pgf.rcfonts": False,
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
    })

    return _set_size()

def plot_trajectory(
        time_vector: np.ndarray,
        trajectory: Union[np.ndarray, List[np.ndarray]],
        comparison_trajectory: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        input_signal: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        title: str = None,
        exportable: bool = False
    ):
    """
    Plots the trajectory of the system states and optionally the input signal and comparison trajectory.
    """


    num_state_vars = trajectory.shape[1]

    total_plots = num_state_vars
    if input_signal is not None:
        if input_signal.ndim == 1:
            input_signal = input_signal.reshape(-1, 1)
        num_input_vars = input_signal.shape[1]
        total_plots += num_input_vars

    fig_width_in, fig_height_in = 12, 3 * total_plots
    if exportable:
        fig_width_in, fig_height_in = _prepare_export()

    fig = plt.figure(figsize=(fig_width_in, fig_height_in))

    current_plot_idx = 0

    for i in range(num_state_vars):
        current_plot_idx += 1
        ax = plt.subplot(total_plots, 1, current_plot_idx)
        plt.plot(time_vector, trajectory[:, i], "k-", label=f"Real data ($x_{i}$)")
        if comparison_trajectory is not None:
            plt.plot(time_vector, comparison_trajectory[:, i], "r--", label=f"Simulated data ($x_{i}$)")
        plt.ylabel(f"$x_{i}$")
        plt.legend()
        if i == 0:
            plt.title(title) if title is not None else None

        if current_plot_idx != total_plots:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            plt.xlabel("Time (s)")
        plt.grid(True)

    if input_signal is not None:
        for i in range(num_input_vars):
            current_plot_idx += 1
            ax = plt.subplot(total_plots, 1, current_plot_idx)
            plt.plot(time_vector, input_signal[:, i], "b-", label=f"Input signal ($u_{i}$)")
            plt.ylabel(f"$u_{i}$")
            plt.legend()
            if num_state_vars == 0 and total_plots == 1:
                plt.title("Input signal")

            if current_plot_idx != total_plots:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                plt.xlabel("Time (s)")
            plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()
    plt.close(fig)

    if exportable:
        plt.rcParams.update(plt.rcParamsDefault)

    return None

def plot_pareto(rmses: List[float], complexities: List[int], print_title: bool = False ,exportable: bool = False):
    """
    Plots the Pareto front of model configurations based on RMSE and complexity.
    """

    fig_width_in, fig_height_in = 6, 6
    if exportable:
        fig_width_in, fig_height_in = _prepare_export()

    # Vykreslenie
    fig = plt.figure(figsize=(fig_width_in, fig_height_in))
    plt.scatter(rmses, complexities, color="tab:blue", label="Pareto points")
    plt.xlabel("RMSE")
    plt.ylabel("Complexity (count of nonzero coefficients)")
    plt.title("Pareto front") if print_title else None
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    if exportable:
        plt.rcParams.update(plt.rcParamsDefault)

    return None

def plot_koopman_spectrum(eigenvalues: np.ndarray, print_title: bool = False, exportable: bool = False):
    """ """

    fig_width_in, fig_height_in = 8, 8 
    if exportable:
        fig_width_in, fig_height_in = _prepare_export()
    
    fig = plt.figure(figsize=(fig_width_in, fig_height_in))
    plt.scatter(eigenvalues.real, eigenvalues.imag, marker="o")
    circle = plt.Circle((0,0), 1, color="blue", fill=False, linestyle="--", label="Unit Circle")
    plt.gca().add_artist(circle)

    for i, lambda_val in enumerate(eigenvalues):  
        offset_x = 5 if lambda_val.real >= 0 else -30  
        offset_y = 5 if lambda_val.imag >= 0 else -20  
        plt.annotate(f"$\\lambda_{i+1}$", (lambda_val.real, lambda_val.imag), textcoords="offset points", xytext=(offset_x, offset_y), ha='center', fontsize=9)  

    plt.xlabel("Real part $Re$")
    plt.ylabel("Imaginarny part $Im$")
    plt.title("Koopman spectrum") if print_title else None
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()
    plt.close(fig)

    if exportable:
        plt.rcParams.update(plt.rcParamsDefault)

def plot_compared_trajectories(
        time_vector: np.ndarray,
        real_trajectory: Union[np.ndarray, List[np.ndarray]],
        sindy_trajectory: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        sindy_r2: Optional[float] = None,
        koopman_trajectory: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        koopman_r2: Optional[float] = None,
        input_signal: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        print_title: bool = False,
        exportable: bool = False
    ):
    """
    Plots the trajectory of the system states and optionally the input signal and comparison trajectory.
    """

    num_state_vars = real_trajectory.shape[1]

    total_plots = num_state_vars
    if input_signal is not None:
        num_input_vars = input_signal.shape[1]
        total_plots += num_input_vars

    fig_width_in, fig_height_in = 12, 3 * total_plots
    if exportable:
        fig_width_in, fig_height_in = _prepare_export()

    fig = plt.figure(figsize=(fig_width_in, fig_height_in))

    current_plot_idx = 0

    for i in range(num_state_vars):
        current_plot_idx += 1
        ax = plt.subplot(total_plots, 1, current_plot_idx)
        plt.plot(time_vector, real_trajectory[:, i], "k-", label=f"Real data")
        if sindy_trajectory is not None:
            plt.plot(time_vector, sindy_trajectory[:, i], "r--", label=f"SINDyC simulated data ({sindy_r2:.3%})")
        if koopman_trajectory is not None:
            plt.plot(time_vector, koopman_trajectory[:, i], "g--", label=f"Koopman simulated data ({koopman_r2:.3%})")
        plt.ylabel(f"$x_{i}$")
        plt.legend()
        if i == 0:
            plt.title("Comparison with real data") if print_title else None

        if current_plot_idx != total_plots:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            plt.xlabel("Time (s)")
        plt.grid(True)

    if input_signal is not None:
        for i in range(num_input_vars):
            current_plot_idx += 1
            ax = plt.subplot(total_plots, 1, current_plot_idx)
            plt.plot(time_vector, input_signal[:, i], "b-", label=f"Input signal ($u_{i}$)")
            plt.ylabel(f"$u_{i}$")
            plt.legend()
            if num_state_vars == 0 and total_plots == 1:
                plt.title("Input signal")

            if current_plot_idx != total_plots:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                plt.xlabel("Time (s)")
            plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()
    plt.close(fig)

    if exportable:
        plt.rcParams.update(plt.rcParamsDefault)

    return None

def plot_noisy_filtered_trajectory(
        time_vector: np.ndarray,
        trajectory: Union[np.ndarray, List[np.ndarray]],
        comparison_trajectory: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        input_signal: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        title: str = None,
        exportable: bool = False
    ):
    """
    Plots the trajectory of the system states and optionally the input signal and comparison trajectory.
    """
    num_state_vars = trajectory.shape[1]

    total_plots = num_state_vars
    if input_signal is not None:
        num_input_vars = input_signal.shape[1]
        total_plots += num_input_vars

    fig_width_in, fig_height_in = 12, 3 * total_plots
    if exportable:
        fig_width_in, fig_height_in = _prepare_export()

    fig = plt.figure(figsize=(fig_width_in, fig_height_in))

    current_plot_idx = 0

    for i in range(num_state_vars):
        current_plot_idx += 1
        ax = plt.subplot(total_plots, 1, current_plot_idx)
        plt.plot(time_vector, trajectory[:, i], "k-", label=f"Noisy data ($x_{i}$)")
        if comparison_trajectory is not None:
            plt.plot(time_vector, comparison_trajectory[:, i], "r--", label=f"Filtered data ($x_{i}$)")
        plt.ylabel(f"$x_{i}$")
        plt.legend()
        if i == 0:
            plt.title(title) if title is not None else None

        if current_plot_idx != total_plots:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            plt.xlabel("Time (s)")
        plt.grid(True)

    if input_signal is not None:
        for i in range(num_input_vars):
            current_plot_idx += 1
            ax = plt.subplot(total_plots, 1, current_plot_idx)
            plt.plot(time_vector, input_signal[:, i], "b-", label=f"Input signal ($u_{i}$)")
            plt.ylabel(f"$u_{i}$")
            plt.legend()
            if num_state_vars == 0 and total_plots == 1:
                plt.title("Input signal")

            if current_plot_idx != total_plots:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                plt.xlabel("Time (s)")
            plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()
    plt.close(fig)

    if exportable:
        plt.rcParams.update(plt.rcParamsDefault)

    return None
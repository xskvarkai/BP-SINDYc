import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Union, Optional, List

def plot_trajectory(
        time_vector: np.ndarray,
        trajectory: Union[np.ndarray, List[np.ndarray]],
        comparison_trajectory: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        input_signal: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        title: str = "Plot"
    ):
    """
    Plots the trajectory of the system states and optionally the input signal and comparison trajectory.
    """
    num_state_vars = trajectory.shape[1]

    total_plots = num_state_vars
    if input_signal is not None:
        total_plots += 1

    fig = plt.figure(figsize=(12, 3 * total_plots))

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
            plt.title(title)

        if current_plot_idx != total_plots:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            plt.xlabel("Time (s)")
        plt.grid(True)

    if input_signal is not None:
        current_plot_idx += 1
        ax = plt.subplot(total_plots, 1, current_plot_idx)
        plt.plot(time_vector, input_signal, "b-", label="Input signal (u)")
        plt.ylabel("u")
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

    return None

def plot_pareto(pareto_front: Dict):
    """
    Plots the Pareto front of model configurations based on RMSE and complexity.
    
    """

    if pareto_front is None:
        return None

    errs = np.array([r.get("rmse") for r in pareto_front], dtype=float)
    spars = np.array([r.get("complexity") for r in pareto_front], dtype=float)

    # Vykreslenie
    fig = plt.figure(figsize=(8, 5))
    plt.scatter(errs, spars, color="tab:blue", label="Pareto points")
    plt.xlabel("RMSE")
    plt.ylabel("Complexity (count of nonzero coefficients)")
    plt.title("Pareto front")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    return None
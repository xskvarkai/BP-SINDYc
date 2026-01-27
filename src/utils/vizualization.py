import matplotlib.pyplot as plt

def vizualize_trajectory(time_vector, trajectory, comparison_trajectory=None, input_signal=None):
    num_state_vars = trajectory.shape[1]

    total_plots = num_state_vars
    if input_signal is not None:
        total_plots += 1

    plt.figure(figsize=(12, 3 * total_plots))

    current_plot_idx = 0

    for i in range(num_state_vars):
        current_plot_idx += 1
        ax = plt.subplot(total_plots, 1, current_plot_idx)
        plt.plot(time_vector, trajectory[:, i], "k-", label=f"Real data ($x_{i+1}$)")
        if comparison_trajectory is not None:
            plt.plot(time_vector, comparison_trajectory[:, i], "r--", label=f"Comparison data ($x_{i+1}$)")
        plt.ylabel(f"$x_{i+1}$")
        plt.legend()
        if i == 0:
            plt.title("Data comparison")

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

    return 0
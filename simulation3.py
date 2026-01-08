import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool

# Parameters
alpha_values = [.1, .05, .02]  # You can modify this list for different α values
num_steps = 1000
num_runs = 100000  # More samples for higher resolution
num_bins = 100


def function_1(x):
    return x**5


def g(alpha):
    return alpha**(1/6)


# Vectorized simulation function (runs all num_runs simulations in parallel)
def simulate_Y_alpha(alpha):
    np.random.seed(int(alpha * 1000))  # Different seed per alpha for reproducibility
    X = np.zeros(num_runs)
    for _ in range(num_steps):
        noise = np.random.normal(0, 1, num_runs)
        X = X + alpha * (-function_1(X) + noise)
    Y = X / g(alpha)
    return alpha, Y


if __name__ == '__main__':
    # Run simulations in parallel across alpha values
    with Pool(processes=len(alpha_values)) as pool:
        results = pool.map(simulate_Y_alpha, alpha_values)

    # Save data to file
    data_dict = {f'alpha_{alpha}': Y_data for alpha, Y_data in results}
    data_dict['alpha_values'] = np.array(alpha_values)
    data_dict['num_steps'] = num_steps
    data_dict['num_runs'] = num_runs
    np.savez('simulation_results.npz', **data_dict)
    print("Data saved to simulation_results.npz")

    # Create 4 subplots in a row: 3 individual histograms + 1 overlaid KDE
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    
    colors = ['#e63946', '#457b9d', '#2a9d8f']

    # First 3 plots: individual histograms
    for ax, (alpha, Y_data), color in zip(axes[:3], results, colors):
        sns.histplot(Y_data, bins=num_bins, stat="density", color=color, 
                     label=f"$\\alpha$ = {alpha}", ax=ax)
        ax.set_title(f"$\\alpha$ = {alpha}")
        ax.set_xlabel(r"$Y_k^{(\alpha)}$")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4th plot: all curves overlaid
    for (alpha, Y_data), color in zip(results, colors):
        sns.kdeplot(Y_data, label=f"$\\alpha$ = {alpha}", color=color, linewidth=2, ax=axes[3])
    
    axes[3].set_xlabel(r"$Y_k^{(\alpha)}$")
    axes[3].set_ylabel("Density")
    axes[3].set_title("All $\\alpha$ Overlaid")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("Convergence of Scaled SGD", fontsize=14)
    plt.tight_layout()
    
    # Save figure
    fig.savefig('simulation_figure.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('simulation_figure.png', dpi=300, bbox_inches='tight')
    print("Figure saved to simulation_figure.pdf and simulation_figure.png")
    
    plt.show()
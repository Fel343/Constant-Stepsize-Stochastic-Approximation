import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set seed for reproducibility
np.random.seed(42)

# Parameters
alpha_values = [.01]  # You can modify this list for different α values
num_steps = 500
num_runs = 5000  # More samples for higher resolution

# CDF for N(0, 1/2) - slightly wider range for zoom out
x = np.linspace(-3.2, -1.8, 500)
cdf_half_var = norm.cdf(x, loc=0, scale=np.sqrt(0.5))

# Simulation function
def simulate_Y_alpha(alpha, num_steps, num_runs):
    Ys = []
    for _ in range(num_runs):
        X = 0.0
        for _ in range(num_steps):
            noise = np.random.normal(0, 1)
            X = (1 - alpha) * X + alpha * noise
        Y = X / np.sqrt(alpha)
        Ys.append(Y)
    return np.array(Ys)

# Run simulations and plot
for alpha in alpha_values:
    Y_data = simulate_Y_alpha(alpha, num_steps, num_runs)
    
    # Compute empirical CDF at 10 evenly spaced points in [-3, -2]
    bin_edges = np.linspace(-3, -2, 11)  # 11 edges = 10 sections
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # 10 center points
    
    # Calculate empirical CDF at each bin edge (proportion of data <= edge)
    empirical_cdf_values = np.array([np.mean(Y_data <= edge) for edge in bin_edges[1:]])
    
    plt.figure(figsize=(8, 5))
    
    # SA Iterations - red hollow circles with line
    plt.plot(bin_centers, empirical_cdf_values, 'o-', markersize=8, color='red', 
             markerfacecolor='white', markeredgecolor='red', markeredgewidth=2, 
             lw=2, label="SA Iterations")
    
    # Gaussian - solid black line
    plt.plot(x, cdf_half_var, 'k-', lw=3, label="Gaussian")
    
    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel("cdf", fontsize=14)
    plt.xlim(-3.2, -1.8)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set seed for reproducibility
np.random.seed(42)

# Parameters
alpha_values = [.9, .5, .1]  # You can modify this list for different α values
num_steps = 1000
num_runs = 10000  # More samples for higher resolution
C = .01

# CDF for N(0, 1/2) - slightly wider range for zoom out
x = np.linspace(.45, 1.05, 500)
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

# (Calpha^1/2log1/alpha)^1/2)/a
def bound(C, alpha, x):
    return ((C* alpha**(1/2)*np.log(1/alpha)) **(1/2)) / x

# Run simulations and plot - combined figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, alpha in enumerate(alpha_values):
    Y_data = simulate_Y_alpha(alpha, num_steps, num_runs)
    
    # Compute empirical CDF at 10 evenly spaced points in [-3, -2]
    bin_edges = np.linspace(.5, 1, 21)  # 11 edges = 10 sections
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # 10 center points
    
    # Calculate empirical CDF at each bin edge (proportion of data <= edge)
    empirical_cdf_values = np.array([np.mean(Y_data <= edge) for edge in bin_edges[1:]])
    
    ax = axes[idx]
    
    # SA Iterations - red hollow circles with line
    ax.plot(bin_centers, empirical_cdf_values, 'o-', markersize=8, color='red', 
             markerfacecolor='white', markeredgecolor='red', markeredgewidth=2, 
             lw=2, label="SA Iterations")
    
    # Gaussian - solid black line
    ax.plot(x, cdf_half_var, 'k-', lw=3, label="Gaussian")
    ax.plot(x, cdf_half_var + bound(C, alpha, x), 'b-', lw=3, label="Upper Bound")
    ax.plot(x, cdf_half_var - bound(C, alpha, x), 'g-', lw=3, label="Lower Bound")
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel("cdf", fontsize=14)
    ax.set_xlim(.45, 1.05)
    ax.set_ylim(.7, .95)
    ax.set_title(rf"$\alpha = {alpha}$", fontsize=14)
    ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()

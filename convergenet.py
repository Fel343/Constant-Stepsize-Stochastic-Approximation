import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Set seed for reproducibility
np.random.seed(42)

# Parameters
alpha_values = [.9,.5,.1]  # You can modify this list for different α values
num_steps = 500
num_runs = 5000  # More samples for higher resolution
num_bins = 80

# PDF for N(0, 1/2)
x = np.linspace(-4, 4, 1000)
pdf_half_var = norm.pdf(x, loc=0, scale=np.sqrt(0.5))

# Simulation function
def simulate_Y_alpha(alpha, num_steps, num_runs):
    Ys = []
    for _ in range(num_runs):
        X = 0.0
        for _ in range(num_steps):
            noise = np.random.normal(0, 1)
            X = (1-alpha) * X + alpha * noise
        Y = X / np.sqrt(alpha)
        Ys.append(Y)
    return np.array(Ys)

# Run simulations and plot all together
fig, axes = plt.subplots(1, len(alpha_values), figsize=(5 * len(alpha_values), 4))

for i, alpha in enumerate(alpha_values):
    Y_data = simulate_Y_alpha(alpha, num_steps, num_runs)
    ax = axes[i]
    sns.histplot(Y_data, bins=num_bins, stat="density", color='slateblue', label=f"$\\alpha$ = {alpha}", ax=ax)
    ax.plot(x, pdf_half_var, 'b-', lw=2, label=r"$\mathcal{N}(0, \frac{1}{2})$ PDF")
    ax.set_title(f"$\\alpha$ = {alpha}")
    ax.set_xlabel(r"$Y_k^{(\alpha)}$")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)

fig.suptitle(r"Convergence of Scaled SGD Iterate to $\mathcal{N}(0, \frac{1}{2})$", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("convergence_plots.pdf", format="pdf", bbox_inches="tight")
print("Saved to convergence_plots.pdf")

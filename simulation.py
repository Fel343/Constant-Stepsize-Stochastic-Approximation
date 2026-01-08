import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Set seed for reproducibility
np.random.seed(42)

# Parameters
alpha_values = [.1,.05,.01]  # You can modify this list for different α values
num_steps = 500
num_runs = 5000  # More samples for higher resolution
num_bins = 80


def function_1(x):
    return x**5

def function_2(x):
    return x**7

functions = [function_1, function_2]
# PDF for N(0, 1/2)
x = np.linspace(-4, 4, 1000)
pdf_half_var = norm.pdf(x, loc=0, scale=np.sqrt(0.5))

def g(alpha):
    return alpha**(1/6)
# Simulation function
def simulate_Y_alpha(alpha, num_steps, num_runs):
    Ys = []
    for _ in range(num_runs):
        X = 0.0
        for _ in range(num_steps):
            noise = np.random.normal(0, 1)
            X = X + alpha * (-function_1(X) + noise)
        Y = X / g(alpha)
        Ys.append(Y)
    return np.array(Ys)

# Run simulations and plot
for alpha in alpha_values:
    Y_data = simulate_Y_alpha(alpha, num_steps, num_runs)
    plt.figure(figsize=(6, 4))
    sns.histplot(Y_data, bins=num_bins, stat="density", color='slateblue', label=f"$\\alpha$ = {alpha}")
    plt.plot(x, pdf_half_var, 'b-', lw=2, label=r"$\mathcal{N}(0, \frac{1}{2})$ PDF")
    plt.title("Convergence of Scaled SGD Iterate to $\mathcal{N}(0, \tfrac{1}{2})$")
    plt.xlabel(r"$Y_k^{(\alpha)}$")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

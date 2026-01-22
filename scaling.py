import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import os

from functions import Functions
from noise import Noise

import warnings
warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")

# Parameters
alpha_values = [.1, .05, .01]  # You can modify this list for different α values
num_steps = 1000
num_runs = 100000  # More samples for higher resolution
num_bins = 500
k_values = [2,3,4]
noise_variance = 2.0  # Variance for noise distribution
kde_bw_adjust = 2.2  # Bandwidth adjustment for KDE smoothness (higher = smoother)

# Function and noise types to iterate over
function_types = ['polynomial','trigonometric']
noise_types = ['normal', 'signed_pareto']

# Create output directories for results
output_dir = 'results'
pdf_dir = os.path.join(output_dir, 'pdf')
png_dir = os.path.join(output_dir, 'png')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(pdf_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)


def g(alpha, k):
    """Scaling function dependent on k"""
    return alpha**(1 / (2 * k))

def g_baseline(alpha):
    return alpha**(1/2)


def format_noise_name(noise_type):
    """Format noise type for display in plots"""
    noise_display = {
        'normal': 'Gaussian',
        'signed_pareto': 'Signed Pareto',
    }
    return noise_display.get(noise_type, noise_type.replace('_', ' ').title())


def get_function(func_obj, func_type):
    """Get the function method based on function type string"""
    if func_type == 'polynomial':
        return func_obj.polynomial
    elif func_type == 'trigonometric':
        return func_obj.trigonometric
    else:
        raise ValueError(f"Unknown function type: {func_type}")


def get_noise(noise_obj, noise_type):
    """Get the noise method based on noise type string"""
    if noise_type == 'normal':
        return noise_obj.normal
    elif noise_type == 'uniform':
        return noise_obj.uniform
    elif noise_type == 'laplace':
        return noise_obj.laplace
    elif noise_type == 'student_t':
        return noise_obj.student_t
    elif noise_type == 'signed_pareto':
        return noise_obj.signed_pareto
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def simulate_Y_alpha(alpha, k, func_type, noise_type, num_runs, num_steps, noise_variance):
    """
    Vectorized simulation function (runs all num_runs simulations in parallel)
    
    Parameters:
        alpha: Step size
        k: Power parameter for the function
        func_type: Type of function ('polynomial' or 'trigonometric')
        noise_type: Type of noise ('normal', 'uniform', 'laplace', 'student_t')
        num_runs: Number of parallel simulation runs
        num_steps: Number of steps per simulation
        noise_variance: Variance of the noise distribution
    """
    # Create unique seed based on all parameters
    seed = int(alpha * 1000 + k * 100 + hash(func_type) % 100 + hash(noise_type) % 100) % (2**31)
    np.random.seed(abs(seed))
    
    # Initialize function and noise objects
    func_obj = Functions(k)
    noise_obj = Noise(noise_variance, num_runs)
    
    # Get the specific function and noise methods
    func = get_function(func_obj, func_type)
    noise_method = get_noise(noise_obj, noise_type)
    
    X = np.zeros(num_runs)
    divergence_threshold = 100  # Threshold to detect diverged runs
        
    for step in range(num_steps):
        X = X + alpha * (-func(X) + noise_method())

    Y = X / g(alpha, k)
    Y_baseline = X / g_baseline(alpha)
    
    # Filter out diverged runs (Option 5)
    valid_mask = (np.abs(Y) < divergence_threshold) & (np.abs(Y_baseline) < divergence_threshold) & np.isfinite(Y) & np.isfinite(Y_baseline)
    num_valid = np.sum(valid_mask)
    num_diverged = num_runs - num_valid
    
    if num_diverged > 0:
        print(f"    Filtered {num_diverged}/{num_runs} diverged runs ({100*num_diverged/num_runs:.1f}%)")
    
    Y_filtered = Y[valid_mask]
    Y_baseline_filtered = Y_baseline[valid_mask]
    
    return alpha, Y_filtered, Y_baseline_filtered


def run_simulation(k, func_type, noise_type, alpha_values, num_runs, num_steps, noise_variance):
    """Run simulation for a specific combination of k, function type, and noise type"""
    results = []
    for alpha in alpha_values:
        alpha_val, Y, Y_baseline = simulate_Y_alpha(
            alpha, k, func_type, noise_type, num_runs, num_steps, noise_variance
        )
        results.append((alpha_val, Y, Y_baseline))
    return results

# def create_individual_histogram(alpha, Y_data, k, func_type, noise_type, color, num_bins):
#     """Create and save individual histogram for a specific configuration"""
#     fig, ax = plt.subplots(figsize=(6, 4))
    
#     sns.histplot(Y_data, bins=num_bins, stat="density", color=color, 
#                  label=f"$\\alpha$ = {alpha}", ax=ax)
#     ax.set_title(f"{func_type.capitalize()}, {format_noise_name(noise_type)} Noise\n$\\ell$ = {k}, $\\alpha$ = {alpha}")
#     ax.set_xlabel(r"$Y_\ell^{(\alpha)}$")
#     ax.set_ylabel("Density")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     filename = f'hist_{func_type}_{noise_type}_k{k}_alpha{alpha}'
#     fig.savefig(os.path.join(pdf_dir, f'{filename}.pdf'), dpi=300, bbox_inches='tight')
#     fig.savefig(os.path.join(png_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
#     plt.close(fig)


def create_kde_overlay(results, k, func_type, noise_type, colors):
    """Create and save KDE overlay plot for all alpha values, comparing k-scaling vs baseline"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: baseline scaling (α^(1/2))
    for (alpha, _, Y_baseline), color in zip(results, colors):
        sns.kdeplot(Y_baseline, label=f"$\\alpha$ = {alpha}", color=color, linewidth=2, 
                    bw_adjust=kde_bw_adjust, ax=axes[0])
    
    axes[0].set_xlabel(r"$Y_{baseline}^{(\alpha)}$")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Baseline Scaling: $\\alpha^{{1/2}}$")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-10, 10)
    
    # Right plot: k-dependent scaling (α^(1/2k))
    for (alpha, Y_data, _), color in zip(results, colors):
        sns.kdeplot(Y_data, label=f"$\\alpha$ = {alpha}", color=color, linewidth=2, 
                    bw_adjust=kde_bw_adjust, ax=axes[1])
    
    axes[1].set_xlabel(r"$Y_\ell^{(\alpha)}$")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"Scaling: $\\alpha^{{1/(2\\ell)}}$ ($\\ell$ = {k})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-4, 4)
    plt.tight_layout()
    
    filename = f'kde_{func_type}_{noise_type}_k{k}'
    fig.savefig(os.path.join(pdf_dir, f'{filename}.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(png_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def create_combined_plot(results, k, func_type, noise_type, colors, num_bins):
    """Create combined plot with individual histograms + KDE overlay"""
    num_alphas = len(results)
    fig, axes = plt.subplots(1, num_alphas + 1, figsize=(5 * (num_alphas + 1), 4))
    
    # Individual histograms
    for ax, (alpha, Y_data, _), color in zip(axes[:num_alphas], results, colors):
        sns.histplot(Y_data, bins=num_bins, stat="density", color=color, 
                     label=f"$\\alpha$ = {alpha}", ax=ax)
        ax.set_xlabel(r"$Y_\ell^{(\alpha)}$")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # KDE overlay plot
    for (alpha, Y_data, _, _), color in zip(results, colors):
        sns.kdeplot(Y_data, label=f"$\\alpha$ = {alpha}", color=color, linewidth=2, ax=axes[-1])
    
    axes[-1].set_xlabel(r"$Y_k^{(\alpha)}$")
    axes[-1].set_ylabel("Density")
    axes[-1].legend()
    axes[-1].grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'combined_{func_type}_{noise_type}_k{k}'
    fig.savefig(os.path.join(pdf_dir, f'{filename}.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(png_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)



def create_combined_by_noise(all_results, noise_type, function_types, k_values, colors):
    """Create a 3x2 combined plot: rows = k values, columns = function types, for a specific noise type"""
    fig, axes = plt.subplots(len(k_values), len(function_types), 
                              figsize=(6 * len(function_types), 4 * len(k_values)))
    
    for row_idx, k in enumerate(k_values):
        for col_idx, func_type in enumerate(function_types):
            ax = axes[row_idx, col_idx] if len(k_values) > 1 else axes[col_idx]
            results = all_results[(func_type, noise_type, k)]
            
            for (alpha, Y_data, _), color in zip(results, colors):
                sns.kdeplot(Y_data, label=f"$\\alpha$={alpha}", color=color, linewidth=2, 
                            bw_adjust=kde_bw_adjust, ax=ax)
            
            ax.set_title(f"{func_type.capitalize()}, $\\ell$={k}", fontsize=12)
            ax.set_xlabel(r"$Y_\ell^{(\alpha)}$", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-3, 3)
            
            # Only show legend in first subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=9)
    plt.tight_layout()
    
    filename = f'combined_grid_{noise_type}'
    fig.savefig(os.path.join(pdf_dir, f'{filename}.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(png_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}.pdf and .png")


if __name__ == '__main__':
    colors = ['#e63946', '#457b9d', '#2a9d8f', '#f4a261', '#264653']
    
    # Store all results: key = (func_type, noise_type, k)
    all_results = {}
    
    # Calculate total number of simulations
    total_sims = len(function_types) * len(noise_types) * len(k_values)
    current_sim = 0
    
    print("=" * 60)
    print("CONSTANT-STEPSIZE STOCHASTIC APPROXIMATION SIMULATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Function types: {function_types}")
    print(f"  Noise types: {noise_types}")
    print(f"  k values: {k_values}")
    print(f"  Alpha values: {alpha_values}")
    print(f"  Num runs: {num_runs:,}")
    print(f"  Base num steps: {num_steps:,} (scaled by ell)")
    print(f"  Total simulations to run: {total_sims}")
    print("-" * 60)
    
    # Run simulations for all combinations of function type, noise type, and k
    for func_type in function_types:
        for noise_type in noise_types:
            for k in k_values:
                current_sim += 1
                print(f"\n[{current_sim}/{total_sims}] Running: {func_type}, {noise_type} noise, $\\ell$={k}, steps={num_steps * k}...")
                
                # Run simulation with num_steps * ell
                results = run_simulation(
                    k, func_type, noise_type, alpha_values, 
                    num_runs, num_steps * k, noise_variance
                )
                all_results[(func_type, noise_type, k)] = results

                # Create KDE overlay plot
                create_kde_overlay(results, k, func_type, noise_type, colors)
    
    # Create combined 3x2 grid for each noise type
    print("\n" + "-" * 60)
    print("Creating combined grid plots...")
    for noise_type in noise_types:
        create_combined_by_noise(all_results, noise_type, function_types, k_values, colors)
    
    print("\n" + "=" * 60)
    print("ALL SIMULATIONS COMPLETE!")
    print(f"Results saved in '{output_dir}/' directory")
    print("=" * 60)
    
    # Show final grand summary
    plt.show()
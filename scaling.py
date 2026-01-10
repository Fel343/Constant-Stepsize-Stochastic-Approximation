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
num_steps = 3000
num_runs = 10000  # More samples for higher resolution
num_bins = 100
k_values = [2, 3, 4, 5]
noise_variance = 2.0  # Variance for noise distribution

# Function and noise types to iterate over
function_types = ['polynomial', 'trigonometric']
noise_types = ['normal', 'uniform']

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
        
    for step in range(num_steps):

        X = X + alpha * (-func(X) + noise_method())

    Y = X / g(alpha, k)
    Y_baseline = X / g_baseline(alpha)
    return alpha, Y, Y_baseline


def run_simulation(k, func_type, noise_type, alpha_values, num_runs, num_steps, noise_variance):
    """Run simulation for a specific combination of k, function type, and noise type"""
    results = []
    for alpha in alpha_values:
        alpha_val, Y, Y_baseline = simulate_Y_alpha(
            alpha, k, func_type, noise_type, num_runs, num_steps, noise_variance
        )
        results.append((alpha_val, Y, Y_baseline))
    return results


def save_results(k, func_type, noise_type, results, num_runs, num_steps, output_dir):
    """Save simulation results to file"""
    data_dict = {}
    for alpha, Y_data, Y_baseline in results:
        data_dict[f'alpha_{alpha}_Y'] = Y_data
        data_dict[f'alpha_{alpha}_Y_baseline'] = Y_baseline
    
    data_dict['alpha_values'] = np.array([r[0] for r in results])
    data_dict['num_steps'] = num_steps
    data_dict['num_runs'] = num_runs
    data_dict['k'] = k
    data_dict['function_type'] = func_type
    data_dict['noise_type'] = noise_type
    data_dict['total_iterations'] = num_runs * num_steps
    
    filename = os.path.join(output_dir, f'results_{func_type}_{noise_type}_k{k}.npz')
    np.savez(filename, **data_dict)
    print(f"  Data saved to {filename}")
    return filename


def create_individual_histogram(alpha, Y_data, k, func_type, noise_type, color, num_bins):
    """Create and save individual histogram for a specific configuration"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    sns.histplot(Y_data, bins=num_bins, stat="density", color=color, 
                 label=f"$\\alpha$ = {alpha}", ax=ax)
    ax.set_title(f"{func_type.capitalize()}, {noise_type.capitalize()} Noise\nk = {k}, $\\alpha$ = {alpha}")
    ax.set_xlabel(r"$Y_k^{(\alpha)}$")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'hist_{func_type}_{noise_type}_k{k}_alpha{alpha}'
    fig.savefig(os.path.join(pdf_dir, f'{filename}.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(png_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def create_kde_overlay(results, k, func_type, noise_type, colors):
    """Create and save KDE overlay plot for all alpha values, comparing k-scaling vs baseline"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: k-dependent scaling (α^(1/2k))
    for (alpha, Y_data, _), color in zip(results, colors):
        sns.kdeplot(Y_data, label=f"$\\alpha$ = {alpha}", color=color, linewidth=2, ax=axes[0])
    
    axes[0].set_xlabel(r"$Y_k^{(\alpha)}$")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Scaling: $\\alpha^{{1/(2k)}}$ (k = {k})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right plot: baseline scaling (α^(1/2))
    for (alpha, _, Y_baseline), color in zip(results, colors):
        sns.kdeplot(Y_baseline, label=f"$\\alpha$ = {alpha}", color=color, linewidth=2, ax=axes[1])
    
    axes[1].set_xlabel(r"$Y_{baseline}^{(\alpha)}$")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"Baseline Scaling: $\\alpha^{{1/2}}$")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(f"{func_type.capitalize()}, {noise_type.capitalize()} Noise, k = {k}", fontsize=14)
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
        ax.set_title(f"$\\alpha$ = {alpha}")
        ax.set_xlabel(r"$Y_k^{(\alpha)}$")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # KDE overlay plot
    for (alpha, Y_data, _, _), color in zip(results, colors):
        sns.kdeplot(Y_data, label=f"$\\alpha$ = {alpha}", color=color, linewidth=2, ax=axes[-1])
    
    axes[-1].set_xlabel(r"$Y_k^{(\alpha)}$")
    axes[-1].set_ylabel("Density")
    axes[-1].set_title("All $\\alpha$ Overlaid")
    axes[-1].legend()
    axes[-1].grid(True, alpha=0.3)
    
    fig.suptitle(f"Scaled SGD: {func_type.capitalize()}, {noise_type.capitalize()} Noise, k = {k}", fontsize=14)
    plt.tight_layout()
    
    filename = f'combined_{func_type}_{noise_type}_k{k}'
    fig.savefig(os.path.join(pdf_dir, f'{filename}.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(png_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# def create_comparison_by_noise(all_results, func_type, k, noise_types, colors, num_bins, output_dir):
#     """Create a comparison plot across different noise types for a fixed function and k"""
#     fig, axes = plt.subplots(1, len(noise_types), figsize=(5 * len(noise_types), 4))
    
#     noise_colors = ['#e63946', '#457b9d', '#2a9d8f', '#f4a261']
    
#     for ax, noise_type, nc in zip(axes, noise_types, noise_colors):
#         results = all_results[(func_type, noise_type, k)]
#         # Plot KDE for each alpha
#         for (alpha, Y_data, _), color in zip(results, colors):
#             sns.kdeplot(Y_data, label=f"$\\alpha$={alpha}", color=color, linewidth=2, ax=ax)
#         ax.set_title(f"{noise_type.capitalize()} Noise")
#         ax.set_xlabel(r"$Y_k^{(\alpha)}$")
#         ax.set_ylabel("Density")
#         ax.legend()
#         ax.grid(True, alpha=0.3)
    
#     fig.suptitle(f"{func_type.capitalize()} Function, k = {k}: Noise Comparison", fontsize=14)
#     plt.tight_layout()
    
#     filename_base = os.path.join(output_dir, f'comparison_noise_{func_type}_k{k}')
#     fig.savefig(f'{filename_base}.pdf', dpi=300, bbox_inches='tight')
#     fig.savefig(f'{filename_base}.png', dpi=300, bbox_inches='tight')
#     plt.close(fig)


# def create_comparison_by_function(all_results, noise_type, k, function_types, colors, num_bins, output_dir):
#     """Create a comparison plot across different function types for a fixed noise and k"""
#     fig, axes = plt.subplots(1, len(function_types), figsize=(5 * len(function_types), 4))
    
#     if len(function_types) == 1:
#         axes = [axes]
    
#     for ax, func_type in zip(axes, function_types):
#         results = all_results[(func_type, noise_type, k)]
#         # Plot KDE for each alpha
#         for (alpha, Y_data, _), color in zip(results, colors):
#             sns.kdeplot(Y_data, label=f"$\\alpha$={alpha}", color=color, linewidth=2, ax=ax)
#         ax.set_title(f"{func_type.capitalize()}")
#         ax.set_xlabel(r"$Y_k^{(\alpha)}$")
#         ax.set_ylabel("Density")
#         ax.legend()
#         ax.grid(True, alpha=0.3)
    
#     fig.suptitle(f"{noise_type.capitalize()} Noise, k = {k}: Function Comparison", fontsize=14)
#     plt.tight_layout()
    
#     filename_base = os.path.join(output_dir, f'comparison_func_{noise_type}_k{k}')
#     fig.savefig(f'{filename_base}.pdf', dpi=300, bbox_inches='tight')
#     fig.savefig(f'{filename_base}.png', dpi=300, bbox_inches='tight')
#     plt.close(fig)


# def create_grand_summary(all_results, function_types, noise_types, k_values, colors, output_dir):
#     """Create a grand summary plot for each function type: k values vs noise types"""
#     for func_type in function_types:
#         fig, axes = plt.subplots(len(k_values), len(noise_types), 
#                                   figsize=(4 * len(noise_types), 3 * len(k_values)))
        
#         for row_idx, k in enumerate(k_values):
#             for col_idx, noise_type in enumerate(noise_types):
#                 ax = axes[row_idx, col_idx] if len(k_values) > 1 else axes[col_idx]
#                 results = all_results[(func_type, noise_type, k)]
                
#                 for (alpha, Y_data, _), color in zip(results, colors):
#                     sns.kdeplot(Y_data, label=f"$\\alpha$={alpha}", color=color, linewidth=1.5, ax=ax)
                
#                 ax.set_title(f"k={k}, {noise_type}", fontsize=10)
#                 ax.set_xlabel(r"$Y_k^{(\alpha)}$", fontsize=8)
#                 ax.set_ylabel("Density", fontsize=8)
#                 ax.tick_params(labelsize=7)
#                 ax.grid(True, alpha=0.3)
                
#                 # Only show legend in first subplot
#                 if row_idx == 0 and col_idx == 0:
#                     ax.legend(fontsize=7)
        
#         fig.suptitle(f"Grand Summary: {func_type.capitalize()} Function", fontsize=16)
#         plt.tight_layout()
        
#         filename_base = os.path.join(output_dir, f'grand_summary_{func_type}')
#         fig.savefig(f'{filename_base}.pdf', dpi=300, bbox_inches='tight')
#         fig.savefig(f'{filename_base}.png', dpi=300, bbox_inches='tight')
#         plt.close(fig)
#         print(f"Saved {filename_base}.pdf and .png")


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
    print(f"  Num steps: {num_steps:,}")
    print(f"  Total iterations per simulation: {num_runs * num_steps:,}")
    print(f"  Total simulations to run: {total_sims}")
    print("-" * 60)
    
    # Run simulations for all combinations of function type, noise type, and k
    for func_type in function_types:
        for noise_type in noise_types:
            for k in k_values:
                current_sim += 1
                print(f"\n[{current_sim}/{total_sims}] Running: {func_type}, {noise_type} noise, k={k}...")
                
                # Run simulation
                results = run_simulation(
                    k, func_type, noise_type, alpha_values, 
                    num_runs, num_steps, noise_variance
                )
                all_results[(func_type, noise_type, k)] = results
                
                # Save results to file
                save_results(k, func_type, noise_type, results, num_runs, num_steps, output_dir)
                
                # # Create individual histogram plots for each alpha
                # for (alpha, Y_data, _), color in zip(results, colors):
                #     create_individual_histogram(
                #         alpha, Y_data, k, func_type, noise_type, 
                #         color, num_bins
                #     )
                
                # Create KDE overlay plot
                create_kde_overlay(results, k, func_type, noise_type, colors)
                
                # # Create combined plot
                # create_combined_plot(results, k, func_type, noise_type, colors, num_bins)
    
    # # Create comparison plots
    # print("\n" + "-" * 60)
    # print("Creating comparison plots...")
    
    # # Comparison by noise type (for each function and k)
    # for func_type in function_types:
    #     for k in k_values:
    #         create_comparison_by_noise(all_results, func_type, k, noise_types, colors, num_bins, output_dir)
    #         print(f"  Saved noise comparison: {func_type}, k={k}")
    
    # # Comparison by function type (for each noise and k)
    # for noise_type in noise_types:
    #     for k in k_values:
    #         create_comparison_by_function(all_results, noise_type, k, function_types, colors, num_bins, output_dir)
    #         print(f"  Saved function comparison: {noise_type}, k={k}")
    
    # # Grand summary plots
    # print("\nCreating grand summary plots...")
    # create_grand_summary(all_results, function_types, noise_types, k_values, colors, output_dir)
    
    print("\n" + "=" * 60)
    print("ALL SIMULATIONS COMPLETE!")
    print(f"Results saved in '{output_dir}/' directory")
    print("=" * 60)
    
    # Show final grand summary
    plt.show()
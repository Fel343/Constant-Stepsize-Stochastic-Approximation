# Constant-Stepsize Stochastic Approximation

This repository contains the code to reproduce the numerical experiments in our paper.

## Overview

We study the stationary distribution of constant-stepsize stochastic approximation (SA) algorithms. Our analysis reveals that for objective functions with polynomial growth of order $2\ell$, the scaled iterate $X_n / \alpha^{1/(2\ell)}$ converges to a non-degenerate stationary distribution as $\alpha \to 0$.

## Repository Structure

```
├── scaling.py              # Main simulation for scaling analysis
├── convergenet.py          # Convergence to Gaussian distribution
├── concentration_bound.py  # Concentration bound verification
├── functions.py            # Objective function derivatives (polynomial, trigonometric)
├── noise.py                # Noise distributions (Gaussian, Signed Pareto)
├── figures/                # Output directory for convergence/concentration plots
└── results/                # Output directory for scaling experiment results
    ├── pdf/                # PDF figures
    └── png/                # PNG figures
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Seaborn
- SciPy

Install dependencies:
```bash
pip install numpy matplotlib seaborn scipy
```

## Experiments

### 1. Scaling Analysis (`scaling.py`)

Compares baseline scaling ($\alpha^{1/2}$) with the optimal scaling ($\alpha^{1/(2\ell)}$) for different function types and noise distributions.

**Configuration:**
- Function types: `polynomial`, `trigonometric`
- Noise types: `normal` (Gaussian), `signed_pareto` (heavy-tailed)
- $\ell$ values: 2, 3, 4
- Step sizes $\alpha$: 0.1, 0.05, 0.01

```bash
python scaling.py
```

**Output:** KDE plots comparing scaled iterate distributions in `results/pdf/` and `results/png/`.

### 2. Convergence to Gaussian (`convergenet.py`)

Demonstrates convergence of $Y_n^{(\alpha)} = X_n / \sqrt{\alpha}$ to $\mathcal{N}(0, 1/2)$ for the linear case.

```bash
python convergenet.py
```

**Output:** `figures/convergence_plots.pdf`

### 3. Concentration Bounds (`concentration_bound.py`)

Verifies the theoretical concentration bounds on the CDF of the scaled iterates.

```bash
python concentration_bound.py
```

**Output:** `figures/concentration_bound.pdf`

## Key Functions

### Objective Functions (`functions.py`)

- **Polynomial:** $f(x) = x^{2\ell} / (2\ell)$ with derivative $f'(x) = x^{2\ell-1}$
- **Trigonometric:** $f(x) = x^{2\ell}/(2\ell) + \sin(x)^{2\ell}/(2\ell)$

### Noise Distributions (`noise.py`)

- **Gaussian:** $\mathcal{N}(0, \sigma^2)$
- **Signed Pareto:** Heavy-tailed distribution with finite variance

## Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{author2026constant,
  title={Constant-Stepsize Stochastic Approximation},
  author={Author Names},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

## License

This project is released for academic use.
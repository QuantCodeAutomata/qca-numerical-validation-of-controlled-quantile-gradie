# Quantile Gradient Validation Project

## Overview
This project validates analytical gradient and Hessian formulae for the portfolio
quantile function q_alpha(t) = VaR_alpha(t*X1 + (1-t)*X2) under bivariate lognormal
distributions, using Monte Carlo finite differences and quadrature-based surface
measures on level sets S_1 = {x > 0 : u^T x = 1}.

## Module Structure

| File | Purpose |
|------|---------|
| `src/__init__.py` | Empty package marker |
| `src/distributions.py` | Lognormal scenario definitions, simulation, phi_X, a_func |
| `src/quantiles.py` | Empirical quantile estimation, FD gradient/Hessian, narrow-band conditioning |
| `src/surface_measure.py` | Gauss-Legendre quadrature for moments on S_1, eq (3.27) |
| `src/diagnostics.py` | FD vs formula comparison functions |
| `data/__init__.py` | Empty package marker |
| `data/generate_data.py` | Generate & save lognormal samples (npz) for all scenarios |

## Key Conventions

### Matrix / vector conventions
- **B** = precision-like matrix (B = A^{-1})
- **A** = covariance matrix of log X (A = B^{-1})
- **b_i** = sum_j B_{ji} = column sums of B → `B.sum(axis=0)`
- **beta** = sum_i b_i

### Scenarios
| Name | B | regime | b | beta | t*_left |
|------|---|--------|---|------|---------|
| S1 | [[0.5,0.1],[0.1,0.25]] | interior | [0.6, 0.35] | 0.95 | 0.6315789474 |
| S2 | [[5,0.1],[0.1,0.25]] | interior | [5.1, 0.35] | 5.45 | 0.9357798165 |
| S3 | [[0.5,-0.3],[-0.3,0.25]] | corner | [0.2, -0.05] | 0.15 | 1.0 |

- Interior regime: all b_i > 0. t* = b[0]/beta.
- Corner regime: some b_i ≤ 0. For S3 (b[0]>0, b[1]<0): t*_left=1, t*_right=0.

### phi_X (negative log-density)
```
phi_X(x) = 0.5*(log x - mu)^T B (log x - mu) + sum_j log(x_j)
           + (n/2)*log(2*pi) + 0.5*log det(A)
```
Uses **B** (not A) for the quadratic form (Mahalanobis distance with B=A^{-1}).

### Quantile Hessian formula (eq 3.27)
```
d^2q/dt^2 = -T(y)
T(y) = (1/y) * (coeff * Var_{mu_y}[gamma] - E_{mu_y}[a*(gamma-E[gamma])^2])
coeff = -beta*log(y) + sum_j(B @ mu)_j + 1
gamma = x1 - x2  (v^T x, v=(1,-1))
```
The f_Y term cancels (d_y[f_Y*Var] = f_Y*T(y) → d^2q/dt^2 = -T(y)).

### E_a_centered_gamma2
Stored in `quadrature_moments_2d` dict as `'E_a_centered_gamma2'`:
```
E[a*(gamma - E_gamma)^2] = E[a*gamma^2] - 2*E_gamma*E[a*gamma] + E_gamma^2*E[a]
```
This is distinct from Cov(a,(gamma-Egamma)^2) = E_a_centered_gamma2 - E_a*Var_gamma.

## Running smoke tests
```bash
cd /workspace/project
python -c "from src.distributions import get_scenarios; print(get_scenarios())"
python -m data.generate_data --n_samples 10000 --output_dir data/
```

## Imports
All src modules use absolute imports from the `src` package (e.g., `from src.distributions import ...`).
Run scripts from `/workspace/project` (repo root) with `sys.path` including `.`.

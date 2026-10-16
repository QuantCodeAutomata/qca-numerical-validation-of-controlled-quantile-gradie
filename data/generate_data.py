"""Generate and save synthetic lognormal samples for all scenarios.

Each scenario's data is saved as a compressed NumPy archive:
    data/<scenario_name>_X.npz

The archive contains:
    X      : np.ndarray, shape (n_samples, 2)  lognormal samples
    mu     : np.ndarray, shape (2,)
    B      : np.ndarray, shape (2, 2)
    A      : np.ndarray, shape (2, 2)
    b      : np.ndarray, shape (2,)
    beta   : float scalar
    regime : str

Usage
-----
    python -m data.generate_data [--n_samples 500000] [--seed 42] [--output_dir data/]
"""

import os
import argparse
import numpy as np

from src.distributions import get_scenarios, simulate_lognormal


def generate_and_save(
    n_samples: int = 500_000,
    seed: int = 42,
    output_dir: str = 'data',
) -> None:
    """Generate lognormal samples for every scenario and persist to disk.

    Parameters
    ----------
    n_samples  : int, number of Monte Carlo draws per scenario
    seed       : int, base random seed (each scenario uses seed + offset)
    output_dir : str, directory where .npz files are written
    """
    os.makedirs(output_dir, exist_ok=True)
    scenarios = get_scenarios()

    for offset, (name, scenario) in enumerate(scenarios.items()):
        rng_seed = seed + offset
        X = simulate_lognormal(scenario, n_samples=n_samples, seed=rng_seed)

        out_path = os.path.join(output_dir, f'{name}_X.npz')
        np.savez_compressed(
            out_path,
            X=X,
            mu=scenario.mu,
            B=scenario.B,
            A=scenario.A,
            b=scenario.b,
            beta=np.array(scenario.beta),
            regime=np.array(scenario.regime),
        )
        print(f'[{name}] saved {n_samples:,} samples -> {out_path}')


def load_scenario_data(name: str, output_dir: str = 'data') -> dict:
    """Load previously generated data for a scenario.

    Parameters
    ----------
    name       : str, scenario name ('S1', 'S2', 'S3')
    output_dir : str

    Returns
    -------
    dict with keys: X, mu, B, A, b, beta, regime
    """
    path = os.path.join(output_dir, f'{name}_X.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}. Run generate_and_save() first."
        )
    data = np.load(path, allow_pickle=False)
    return {
        'X': data['X'],
        'mu': data['mu'],
        'B': data['B'],
        'A': data['A'],
        'b': data['b'],
        'beta': float(data['beta']),
        'regime': str(data['regime']),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate lognormal simulation data for all scenarios.'
    )
    parser.add_argument(
        '--n_samples', type=int, default=500_000,
        help='Number of Monte Carlo draws per scenario (default: 500 000).'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Base random seed (default: 42).'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data',
        help='Directory to write output files (default: data/).'
    )
    args = parser.parse_args()
    generate_and_save(
        n_samples=args.n_samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()

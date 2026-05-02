#!/usr/bin/env python3
"""
Run logistic regression experiments for N=10, 20, 30 and create comparison plots.
"""
import argparse
import os
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import numpyro
from numpyro.infer import NUTS, MCMC
import numpy as np

from projection_vi import ComponentwiseFlow, MFVIStep
from experiments.targets import BLR
from experiments.compare_rotations.rotations import (
    compute_active_subspace_rotation,
    compute_relative_score_pca_rotation
)

numpyro.set_host_device_count(5)
    
def setup_target(d: int, N: int, prior_scale: float, seed: int, kappa: float = 1.0):
    """Create BLR target with synthetic data."""
    key1, key2, key3 = jax.random.split(jax.random.key(seed), 3)

    diag = jnp.logspace(-kappa, kappa, d)
    U_ = jnp.linalg.qr(jax.random.normal(key1, shape=(d, d)))[0]
    cov_X = U_ @ jnp.diag(diag) @ U_.T
    X = jax.random.multivariate_normal(key2, mean=jnp.zeros(d), cov=cov_X, shape=(N,))
    y = jax.random.bernoulli(key3, shape=(N,))

    target = BLR(X=X, y=y, prior_scale=prior_scale)
    return target

def run_mcmc(target, num_samples, thinning=1):
    """Generate MCMC reference samples."""
    num_warmup = 10*target.d
    num_chains = 5

    nuts_kernel = NUTS(target.numpyro_model)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples*thinning, num_chains=num_chains, thinning=thinning, progress_bar=False)
    mcmc.run(jax.random.key(0))

    mcmc_samples = mcmc.get_samples()

    param_names = target.param_unc_names()
    samples_unc = None
    for key in param_names:
        sample = mcmc_samples[key]
        if sample.ndim == 1:
            sample = sample.reshape(-1, 1)
        if samples_unc is None:
            samples_unc = sample
        else:
            samples_unc = jnp.concatenate([samples_unc, sample], axis=1)

    samples = target.param_constrain(samples_unc)

    samples = jnp.array(samples)
    moments_1 = jnp.mean(samples, axis=0)
    moments_2 = jnp.mean(samples**2, axis=0)
    return samples, {'moments_1': moments_1, 'moments_2': moments_2}

def run_method_on_target(target, method_name: str, key, fit_config, scale=None, shift=None, return_samples=False):
    """Run MFVI method and return training losses and optionally samples."""
    if scale is None:
        scale = jnp.ones(target.d)
    if shift is None:
        shift = jnp.zeros(target.d)

    @jax.jit
    def log_prob_fn(x):
        return target.log_prob(x * scale + shift) + jnp.sum(jnp.log(scale))

    key1, key2, key3 = jax.random.split(key, 3)

    # Compute rotation matrix
    rotation_matrix = None
    if method_name == "active_subspace":
        rotation_matrix = compute_active_subspace_rotation(
            log_prob_fn=log_prob_fn,
            d=target.d,
            num_samples=2000,
            key=key1
        )
    elif method_name == "relative_score_pca":
        rotation_matrix = compute_relative_score_pca_rotation(
            log_prob_fn=log_prob_fn,
            d=target.d,
            num_samples=2000,
            key=key1
        )

    flow = ComponentwiseFlow(
        target.d,
        num_bins=fit_config['num_bins'],
        range_min=fit_config['range_min'],
        range_max=fit_config['range_max']
    )

    if rotation_matrix is not None:
        def rotated_log_prob_fn(x):
            return log_prob_fn(rotation_matrix @ x)
        fit_log_prob = rotated_log_prob_fn
    else:
        fit_log_prob = log_prob_fn

    params, training_losses = MFVIStep(
        fit_log_prob,
        d=target.d,
        flow=flow,
        nsample=fit_config['num_samples_train'],
        key=key2,
        learning_rate=fit_config['learning_rate'],
        max_iter=fit_config['num_iterations']
    )

    if return_samples:
        # Generate samples
        base_samples = jax.random.normal(key3, (fit_config.get('num_samples_eval', 2000), target.d))
        spline_samples, log_det = flow.apply(params, base_samples)

        if rotation_matrix is not None:
            samples = jax.vmap(lambda x: rotation_matrix @ x)(spline_samples)
        else:
            samples = spline_samples

        samples = samples * scale + shift
        return training_losses, samples
    else:
        return training_losses


def run_all_methods(target_config, fit_config, num_repeats=5):
    """Run all methods multiple times and return all losses and samples from first run."""
    target = setup_target(**target_config)

    # Generate MCMC reference samples
    print(f"Running MCMC for N={target_config['N']}...")
    mcmc_samples, _ = run_mcmc(target, num_samples=2000, thinning=5)

    if fit_config['laplace_init']:
        def neg_logp_fn(x):
            return -target.log_prob(x)

        bfgs_res = minimize(neg_logp_fn, jnp.zeros(target.d), method='BFGS', options={'maxiter': 1000})
        shift = bfgs_res.x
        laplace_cov = bfgs_res.hess_inv
        scale = jnp.sqrt(jnp.maximum(jnp.diag(laplace_cov), .01))
    else:
        shift = jnp.zeros(target.d)
        scale = jnp.ones(target.d)

    methods = ['standard', 'active_subspace', 'relative_score_pca']
    all_runs_results = {method: [] for method in methods}
    first_run_samples = {}

    print(f"Running {num_repeats} repetitions for N={target_config['N']}...")
    print("="*80)

    for run_idx in range(num_repeats):
        key = jax.random.key(run_idx)
        print(f"  Run {run_idx + 1}/{num_repeats}")

        for method in methods:
            # Get samples only from first run
            return_samples = (run_idx == 0)
            result = run_method_on_target(
                target, method, key, fit_config, shift=shift, scale=scale,
                return_samples=return_samples
            )

            if return_samples:
                training_losses, samples = result
                all_runs_results[method].append(training_losses)
                first_run_samples[method] = samples
            else:
                all_runs_results[method].append(result)

    return all_runs_results, first_run_samples, mcmc_samples


def plot_comparison(all_results_dict, d, prior_scale, kappa, num_iterations, output_dir):
    """Create comparison plot with 3 subplots for N=10, 20, 30."""
    sns.set_theme(context='paper', style='whitegrid', font_scale=1.3)
    methods = ['standard', 'active_subspace', 'relative_score_pca']
    method_labels = {
        'standard': 'Standard',
        'active_subspace': 'Fisher information',
        'relative_score_pca': 'Relative Score PCA'
    }

    colors = {
        'standard': 'b',
        'active_subspace': 'teal',
        'relative_score_pca': 'crimson'
    }

    fig, axes = plt.subplots(1, 3, figsize=(7, 3))
    N_values = [10, 20, 30]

    for ax_idx, N in enumerate(N_values):
        ax = axes[ax_idx]
        all_runs_results = all_results_dict[N]['losses']

        for method in methods:
            all_losses = np.array(all_runs_results[method])

            # Compute mean and std across runs
            mean_losses = np.mean(all_losses, axis=0)
            std_losses = np.std(all_losses, axis=0)
            iterations = np.arange(len(mean_losses))

            # Plot mean with shaded std
            ax.plot(iterations, mean_losses, label=method_labels[method],
                   color=colors[method], linewidth=1)
            ax.fill_between(iterations,
                           mean_losses - std_losses,
                           mean_losses + std_losses,
                           color=colors[method],
                           alpha=0.2)

        ax.set_xlabel('Iteration')
        if ax_idx == 0:
            ax.set_ylabel('Training Loss')
        ax.set_title(fr'$n$={N}')

        if ax_idx == 2:
            ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plot_filename = os.path.join(
        output_dir,
        f"training_loss_comparison_d{d}_prior{prior_scale}_kappa{kappa}_iter{num_iterations}.pdf"
    )
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()

    print(f"Comparison plot saved to: {plot_filename}")


def plot_scatter_comparison(all_results_dict, d, prior_scale, kappa, num_iterations, output_dir):
    """Create scatter plots for first 2 variables for each N."""
    sns.set_theme(context='paper', style='whitegrid', font_scale=1.5)
    methods = ['standard', 'relative_score_pca']
    method_labels = {
        'standard': 'Standard',
        'relative_score_pca': 'Relative Score PCA'
    }

    colors = {
        'standard': 'blue',
        'relative_score_pca': 'crimson',
        'mcmc': 'deepskyblue'
    }

    N_values = [10, 20, 30]

    for N in N_values:
        
        
        first_run_samples = all_results_dict[N]['samples']
        mcmc_samples = all_results_dict[N]['mcmc_samples']

        # First subfigure: MCMC + Standard
        def make_scatter_plot(ax, samples):
            ax.scatter(mcmc_samples[:, 0], mcmc_samples[:, 2], marker='o', alpha=0.4, label='Target', c='deepskyblue')
            ax.scatter(samples[:, 0], samples[:, 2], marker='v', alpha=0.5, label='Samples', c='crimson')

        fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
        make_scatter_plot(axes[0], first_run_samples['standard'])
        axes[0].set_title('Standard MFVI')
        make_scatter_plot(axes[1], first_run_samples['relative_score_pca'])
        axes[1].set_title('MFVI with PCA rotation')

        plt.tight_layout()

        plot_filename = os.path.join(
            output_dir,
            f"scatter_N{N}_d{d}_prior{prior_scale}_kappa{kappa}_iter{num_iterations}.pdf"
        )
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()

        print(f"Scatter plot for N={N} saved to: {plot_filename}")


def main():
    parser = argparse.ArgumentParser(description='Run MFVI comparison for N=10,20,30')
    parser.add_argument('--d', default=10, type=int)
    parser.add_argument('--prior_scale', default=2.0, type=float)
    parser.add_argument('--kappa', default=1.0, type=float)
    parser.add_argument('--output_dir', default='results', help='Output directory')
    parser.add_argument('--num_iterations', type=int, default=100, help='MFVI iterations')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--num_bins', type=int, default=10, help='Spline bins')
    parser.add_argument('--num_samples_train', type=int, default=2000, help='Training samples')
    parser.add_argument('--nrep', type=int, default=20, help='Number of repeats')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fit_config = {
        'num_iterations': args.num_iterations,
        'learning_rate': args.learning_rate,
        'num_bins': args.num_bins,
        'num_samples_train': args.num_samples_train,
        'num_samples_eval': 2000,
        'range_min': -5.0,
        'range_max': 5.0,
        'laplace_init': True
    }

    N_values = [10, 20, 30]
    all_results_dict = {}

    for N in N_values:
        target_config = {
            'd': args.d,
            'N': N,
            'prior_scale': args.prior_scale,
            'seed': 2025,
            'kappa': args.kappa,
        }

        losses, samples, mcmc_samples = run_all_methods(target_config, fit_config, num_repeats=args.nrep)
        all_results_dict[N] = {'losses': losses, 'samples': samples, 'mcmc_samples': mcmc_samples}

    # Create comparison plot
    plot_comparison(
        all_results_dict,
        args.d,
        args.prior_scale,
        args.kappa,
        args.num_iterations,
        args.output_dir
    )

    # Create scatter plots
    plot_scatter_comparison(
        all_results_dict,
        args.d,
        args.prior_scale,
        args.kappa,
        args.num_iterations,
        args.output_dir
    )

    return 0


if __name__ == "__main__":
    main()

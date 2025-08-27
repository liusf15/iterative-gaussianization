import os
import argparse
import pickle
import time
import pandas as pd
import jax
import jax.numpy as jnp 
from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.special import logsumexp

from experiments.targets import irt_2pl
from projection_vi.flows import NeuralSplineFlow
from projection_vi.iterative_gaussianization import iterative_gaussianization, iterative_forward_map, MFVIStep
from projection_vi.utils import median_heuristic, compute_ksd, compute_mmd, wasserstein_1d

def load_reference_moments():
    with open ("experiments/posteriordb_experiment/references/irt_2pl_mcmc_moments.pkl", "rb") as f:
        reference_moments = pickle.load(f)
    ref_moment_1 = reference_moments['moments_1'].mean(0)
    ref_moment_2 = reference_moments['moments_2'].mean(0)
    return ref_moment_1, ref_moment_2

def load_initialization():
    initialization = pd.read_csv('experiments/posteriordb_experiment/references/irt_2pl_laplace_initialization.csv', index_col=0)
    scale = initialization.loc['scale'].values
    shift = initialization.loc['mean'].values  
    return shift, scale

def load_reference_samples():
    filename = f'experiments/posteriordb_experiment/references/irt_2pl_mcmc_samples_unc.csv'
    samples = pd.read_csv(filename, index_col=0).values
    return samples

def fit_gaussianization(logp_fn, seed=0, nsample=1000, ntrain=1000, gamma=0.9, n_layer=4, beta_0=0.1, learning_rate=0.1, num_bins=10, range_max=7., max_iter=100):
    
    key, subkey = jax.random.split(jax.random.key(seed))
    flow, transforms = iterative_gaussianization(logp_fn, 
                                                d, 
                                                nsample=ntrain, 
                                                key=subkey, 
                                                gamma=gamma, 
                                                niter=n_layer, 
                                                opt_params={'beta_0': beta_0, 'learning_rate': learning_rate, 'max_iter': max_iter},
                                                flow_params={'num_bins': num_bins, 'range_min': -range_max, 'range_max': range_max, 'boundary_slopes': 'unconstrained'})
    
    key, subkey = jax.random.split(key)
    base_samples = jax.random.normal(subkey, (nsample, d))
    transformed_samples, logdet = iterative_forward_map(flow, transforms, base_samples)

    log_density = mvn.logpdf(base_samples, mean=jnp.zeros(d), cov=jnp.eye(d)) - logdet

    return transformed_samples, log_density

def fit_NSF(logp_fn, seed=0, nsample=1000, ntrain=1000, n_layer=4, beta_0=0.1, learning_rate=0.1, num_bins=10, range_max=7., max_iter=100, hidden_dim=1):
    model_nf = NeuralSplineFlow(dim=d, n_layers=n_layer, hidden_dims=[hidden_dim], num_bins=num_bins, range_min=-range_max, range_max=range_max, boundary_slopes='unconstrained')
    
    key, subkey = jax.random.split(jax.random.key(seed))
    params_nf, losses_nf = MFVIStep(logp_fn, 
                                    d, 
                                    model_nf, 
                                    nsample=ntrain, 
                                    key=subkey, 
                                    beta_0=beta_0, 
                                    learning_rate=learning_rate, 
                                    max_iter=max_iter)
    key, subkey = jax.random.split(key)
    base_samples = jax.random.normal(subkey, (nsample, d))
    transformed_samples, logdet = model_nf.apply(params_nf, base_samples, method=model_nf.forward) 
    log_density = mvn.logpdf(base_samples, mean=jnp.zeros(d), cov=jnp.eye(d)) - logdet

    return transformed_samples, log_density

def evaluate(mcmc_samples_unc, ref_moment_1, ref_moment_2, flow_samples, log_q):
    bandwidth = median_heuristic(mcmc_samples_unc)
    metrics = {}
    metrics['ksd_imq'] = compute_ksd(flow_samples, jax.grad(target.log_prob), bandwidth=bandwidth, kernel_type='imq')
    metrics['ksd_rbf'] = compute_ksd(flow_samples, jax.grad(target.log_prob), bandwidth=bandwidth, kernel_type='rbf')
    metrics['mmd_imq'] = compute_mmd(mcmc_samples_unc, flow_samples, bandwidth=bandwidth, kernel_type='imq')
    metrics['mmd_rbf'] = compute_mmd(mcmc_samples_unc, flow_samples, bandwidth=bandwidth, kernel_type='rbf')

    # sliced version
    eigvecs = jnp.linalg.eigh(jnp.cov(mcmc_samples_unc.T))[1]
    eigvecs = eigvecs[:, ::-1]  
    for r in [0, 1, 2, 3, 4, d-5, d-4, d-3, d-2, d-1]:
        bandwidth_r = median_heuristic(mcmc_samples_unc @ eigvecs[:, r:r+1])
        metrics[f'mmd_imq_proj_{r}'] = compute_mmd(mcmc_samples_unc @ eigvecs[:, r:r+1], flow_samples @ eigvecs[:, r:r+1], bandwidth=bandwidth_r, kernel_type='imq')
        metrics[f'mmd_rbf_proj_{r}'] = compute_mmd(mcmc_samples_unc @ eigvecs[:, r:r+1], flow_samples @ eigvecs[:, r:r+1], bandwidth=bandwidth_r, kernel_type='rbf')
        metrics[f'W2_proj_{r}'] = wasserstein_1d(mcmc_samples_unc @ eigvecs[:, r:r+1], flow_samples @ eigvecs[:, r:r+1], p=2)

    log_weights = jax.vmap(target.log_prob)(flow_samples) - log_q
    metrics['ess'] = jnp.exp(2 * logsumexp(log_weights) - logsumexp(2 * log_weights))

    metrics['elbo'] = jnp.nanmean(log_weights)

    samples_constrain = target.param_constrain(flow_samples)
    metrics['mse1'] = jnp.mean((jnp.mean(samples_constrain, 0) - ref_moment_1)**2 / jnp.maximum(ref_moment_1**2, 1))
    metrics['mse2'] = jnp.mean((jnp.mean(samples_constrain**2, 0) - ref_moment_2)**2 / jnp.maximum(ref_moment_2**2, 1))
    return metrics

def main(args):
    ref_moment_1, ref_moment_2 = load_reference_moments()
    shift, scale = load_initialization()
    scale = jnp.maximum(scale, args.min_scale)
    mcmc_samples_unc = load_reference_samples()

    @jax.jit
    def logp_fn_shifted(x):
        return target.log_prob(x * scale + shift) + jnp.sum(jnp.log(scale))

    if args.algo == 'gaussianization':
        start = time.time()
        transformed_samples, logq = fit_gaussianization(logp_fn_shifted, args.seed, args.nsample, args.ntrain, args.gamma, args.n_layer, args.beta_0, args.learning_rate, args.num_bins, args.range_max, args.max_iter)
        time_elapsed = time.time() - start
    elif args.algo == 'nsf':
        start = time.time()
        transformed_samples, logq = fit_NSF(logp_fn_shifted, args.seed, args.nsample, args.ntrain, args.n_layer, args.beta_0, args.learning_rate, args.num_bins, args.range_max, max_iter=args.nf_max_iter, hidden_dim=args.hidden_dim)
        time_elapsed = time.time() - start
    else:
        raise NotImplementedError(f"Algorithm {args.algo} is not implemented.")

    transformed_samples = transformed_samples * scale + shift

    metrics = evaluate(mcmc_samples_unc, ref_moment_1, ref_moment_2, transformed_samples, logq)
    metrics['time'] = time_elapsed

    savepath = args.savepath
    os.makedirs(savepath, exist_ok=True)
    if args.algo == 'gaussianization':
        filename = f"{args.algo}_seed_{args.seed}_nsample_{args.nsample}_ntrain_{args.ntrain}_gamma_{args.gamma}_n_layer_{args.n_layer}_beta_0_{args.beta_0}_learning_rate_{args.learning_rate}_num_bins_{args.num_bins}_range_max_{args.range_max}_max_iter_{args.max_iter}_scale_{args.min_scale}.pkl"
    else:
        filename = f"{args.algo}_seed_{args.seed}_nsample_{args.nsample}_ntrain_{args.ntrain}_hidden_{args.hidden_dim}_n_layer_{args.n_layer}_beta_0_{args.beta_0}_learning_rate_{args.learning_rate}_num_bins_{args.num_bins}_range_max_{args.range_max}_max_iter_{args.nf_max_iter}_scale_{args.min_scale}.pkl"

    with open(os.path.join(savepath, filename), 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Results saved to {os.path.join(savepath, filename)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='gaussianization', choices=['gaussianization', 'nsf'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsample', type=int, default=1000)
    parser.add_argument('--ntrain', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--beta_0', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--num_bins', type=int, default=10)
    parser.add_argument('--range_max', type=float, default=7.0)
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--nf_max_iter', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=1)
    parser.add_argument('--min_scale', type=float, default=0.5)
    parser.add_argument('--savepath', type=str, default='experiments/results')
    args = parser.parse_args()

    target = irt_2pl("stan/irt_2pl.json")
    d = target.d

    main(args)

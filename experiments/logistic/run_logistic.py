import os
import argparse
import pickle
import pandas as pd
import jax
import jax.numpy as jnp 
from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.special import logsumexp
from jax.scipy.optimize import minimize

from experiments.targets import BLR
from projection_vi.iterative_gaussianization import iterative_gaussianization, iterative_forward_map
from projection_vi.utils import median_heuristic, compute_ksd, compute_mmd

def get_reference_samples():
    return pd.read_csv("experiments/logistic/bayesian_logistic_reference_samples_unc.csv", index_col=0).values

def laplace_approximation():
    def neg_logp_fn(x):
        return -target.log_prob(x)

    bfgs_res = minimize(neg_logp_fn, jnp.zeros(d), method='BFGS', options={'maxiter': 1000})
    laplace_mean = bfgs_res.x
    laplace_cov = bfgs_res.hess_inv
    laplace_scale = jnp.sqrt(jnp.maximum(jnp.diag(laplace_cov), .01))
    return laplace_mean, laplace_scale

def get_scaled_target(shift, scale):
    @jax.jit
    def logp_fn_shifted(x):
        return target.log_prob(x * scale + shift) + jnp.sum(jnp.log(scale))
    return logp_fn_shifted

def fit(logp_fn, seed=0, nsample=2000, ntrain=1000, gamma=0.95, random_rotate=False, niter=1, beta_0=1., learning_rate=0.1, num_bins=10, range_max=8., max_iter=100):
    key, subkey = jax.random.split(jax.random.key(seed))
    flow, transforms = iterative_gaussianization(logp_fn, 
                                                d, 
                                                nsample=ntrain, 
                                                key=subkey, 
                                                gamma=gamma, 
                                                random_rotate=random_rotate,
                                                niter=niter, 
                                                opt_params={'beta_0': beta_0, 'learning_rate': learning_rate, 'max_iter': max_iter},
                                                flow_params={'num_bins': num_bins, 'range_min': -range_max, 'range_max': range_max, 'boundary_slopes': 'unconstrained'})
    
    key, subkey = jax.random.split(key)
    base_samples = jax.random.normal(subkey, (nsample, d))
    transformed_samples_list = []
    logdensity_list = []
    for i in range(niter):  
        transformed_samples, logdet = iterative_forward_map(flow, transforms[:i+1], base_samples)
        transformed_samples_list.append(transformed_samples)

        log_density = mvn.logpdf(base_samples, mean=jnp.zeros(d), cov=jnp.eye(d)) - logdet
        logdensity_list.append(log_density)

    return transformed_samples_list, logdensity_list

def evaluate(target, mcmc_samples_unc, flow_samples, log_q):
    bandwidth = median_heuristic(mcmc_samples_unc)
    metrics = {}
    metrics['ksd_imq'] = compute_ksd(flow_samples, jax.grad(target.log_prob), bandwidth=bandwidth, kernel_type='imq')
    metrics['ksd_rbf'] = compute_ksd(flow_samples, jax.grad(target.log_prob), bandwidth=bandwidth, kernel_type='rbf')
    metrics['mmd_imq'] = compute_mmd(mcmc_samples_unc, flow_samples, bandwidth=bandwidth, kernel_type='imq')
    metrics['mmd_rbf'] = compute_mmd(mcmc_samples_unc, flow_samples, bandwidth=bandwidth, kernel_type='rbf')

    log_weights = jax.vmap(target.log_prob)(flow_samples) - log_q
    metrics['ess'] = jnp.exp(2 * logsumexp(log_weights) - logsumexp(2 * log_weights))

    metrics['elbo'] = jnp.nanmean(log_weights)

    samples_constrain = target.param_constrain(flow_samples)
    mcmc_samples_constrain = target.param_constrain(mcmc_samples_unc)
    ref_moment_1 = jnp.mean(mcmc_samples_constrain, 0)
    ref_moment_2 = jnp.mean(mcmc_samples_constrain**2, 0)

    metrics['mse1'] = jnp.mean((jnp.mean(samples_constrain, 0) - ref_moment_1)**2 / jnp.maximum(ref_moment_1**2, 1))
    metrics['mse2'] = jnp.mean((jnp.mean(samples_constrain**2, 0) - ref_moment_2)**2 / jnp.maximum(ref_moment_2**2, 1))
    return metrics

def main(args):
    
    x0 = jnp.zeros(d)
    x0 = x0.at[0].set(-2.)
    shift, scale = laplace_approximation()
    logp_fn_shifted = get_scaled_target(shift, scale)

    samples_mfvi, logq_mfvi = fit(logp_fn_shifted, args.seed, gamma=0.)
    samples_pca, logq_pca = fit(logp_fn_shifted, seed=args.seed, gamma=0.95)
    samples_random, logq_random = fit(logp_fn_shifted, seed=args.seed, random_rotate=True, niter=args.n_layer)

    mcmc_samples_unc = get_reference_samples()
    metrics = {}
    metrics['mfvi'] = evaluate(target, mcmc_samples_unc, samples_mfvi[-1] * scale + shift, logq_mfvi[-1])
    metrics['pca'] = evaluate(target, mcmc_samples_unc, samples_pca[-1] * scale + shift, logq_pca[-1])
    for i in range(len(samples_random)):
        metrics[f'random_{i+1}'] = evaluate(target, mcmc_samples_unc, samples_random[i] * scale + shift, logq_random[i])

    savepath = args.savepath
    os.makedirs(savepath, exist_ok=True)

    filename = f"logistic_iter_{args.n_layer}_seed_{args.seed}.pkl"

    with open(os.path.join(savepath, filename), 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Results saved to {os.path.join(savepath, filename)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_layer', type=int, default=5)
    parser.add_argument('--savepath', type=str, default='experiments/results')
    args = parser.parse_args()

    n = 20
    d = 10

    key = jax.random.key(2025)
    cov_X = jnp.logspace(-1, 1, d)
    key, subkey = jax.random.split(key)
    U_ = jnp.linalg.qr(jax.random.normal(subkey, shape=(d, d)))[0]
    cov_X = U_ @ jnp.diag(cov_X) @ U_.T
    key, subkey = jax.random.split(key)
    X = jax.random.multivariate_normal(subkey, mean=jnp.zeros(d), cov=cov_X, shape=(n,))

    key, subkey = jax.random.split(key)
    y = jax.random.bernoulli(subkey, shape=(n, ))

    prior_scale = 2.

    target = BLR(X, y, prior_scale=prior_scale)

    main(args)

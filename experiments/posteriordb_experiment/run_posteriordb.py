import jax
import jax.numpy as jnp
import os
import argparse
import pickle
import pandas as pd
from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.special import logsumexp

from projection_vi import ComponentwiseFlow
import experiments.targets as Targets
from projection_vi.utils import median_heuristic, compute_ksd, compute_mmd
from projection_vi.iterative_gaussianization import iterative_gaussianization, MFVIStep, iterative_forward_map

def load_reference_moments(posterior_name):
    with open (f"experiments/results/{posterior_name}_mcmc_moments.pkl", "rb") as f:
        reference_moments = pickle.load(f)
    ref_moment_1 = reference_moments['moments_1'].mean(0)
    ref_moment_2 = reference_moments['moments_2'].mean(0)
    return ref_moment_1, ref_moment_2

def load_initialization(posterior_name):
    initialization = pd.read_csv(f'experiments/results/{posterior_name}_laplace_initialization.csv', index_col=0)
    scale = initialization.loc['scale'].values
    shift = initialization.loc['mean'].values  
    return shift, scale

def load_reference_samples(posterior_name):
    filename = f'~/ceph/projection_vi/posteriordb_reference/{posterior_name}_mcmc_samples_unc.csv'
    samples = pd.read_csv(filename, index_col=0).values
    return samples

def fit_gaussianization(logp_fn, seed=0, nsample=1000, ntrain=1000, gamma=0.9, beta_0=0.1, learning_rate=0.1, num_bins=10, range_max=7., max_iter=100):
    key, subkey = jax.random.split(jax.random.key(seed))
    flow, transforms = iterative_gaussianization(logp_fn, 
                                                d, 
                                                nsample=ntrain, 
                                                key=subkey, 
                                                gamma=gamma, 
                                                niter=1, 
                                                opt_params={'beta_0': beta_0, 'learning_rate': learning_rate, 'max_iter': max_iter},
                                                flow_params={'num_bins': num_bins, 'range_min': -range_max, 'range_max': range_max, 'boundary_slopes': 'unconstrained'})
    
    key, subkey = jax.random.split(key)
    base_samples = jax.random.normal(subkey, (nsample, d))
    transformed_samples, logdet = iterative_forward_map(flow, transforms, base_samples)

    log_density = mvn.logpdf(base_samples, mean=jnp.zeros(d), cov=jnp.eye(d)) - logdet

    return transformed_samples, log_density

def fit_MFVI(logp_fn, seed=0, nsample=1000, ntrain=1000, beta_0=0.1, learning_rate=0.1, num_bins=10, range_max=7., max_iter=100):
    flow = ComponentwiseFlow(d, num_bins=num_bins, range_min=-range_max, range_max=range_max, boundary_slopes='unconstrained')
    key, subkey = jax.random.split(jax.random.key(seed))
    params, losses = MFVIStep(logp_fn, d, flow, ntrain, subkey, beta_0=beta_0, learning_rate=learning_rate, max_iter=max_iter)

    key, subkey = jax.random.split(key)
    base_samples = jax.random.normal(subkey, (nsample, d))
    transformed_samples, logdet = flow.apply(params, base_samples, method=flow.forward) 
    log_density = mvn.logpdf(base_samples, mean=jnp.zeros(d), cov=jnp.eye(d)) - logdet
    return transformed_samples, log_density

def evaluate(mcmc_samples_unc, ref_moment_1, ref_moment_2, flow_samples, log_q):
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
    metrics['mse1'] = jnp.mean((jnp.mean(samples_constrain, 0) - ref_moment_1)**2 / jnp.maximum(ref_moment_1**2, 1))
    metrics['mse2'] = jnp.mean((jnp.mean(samples_constrain**2, 0) - ref_moment_2)**2 / jnp.maximum(ref_moment_2**2, 1))
    return metrics

def main(args):
    ref_moment_1, ref_moment_2 = load_reference_moments(args.posterior_name)
    shift, scale = load_initialization(args.posterior_name)

    mcmc_samples_unc = load_reference_samples(args.posterior_name)

    @jax.jit
    def logp_fn_shifted(x):
        return target.log_prob(x * scale + shift) + jnp.sum(jnp.log(scale))

    all_results = {}
    gamma_list = [0., 0.9, 0.95, 0.99, 1.1]
    for gamma in gamma_list:
        print('gamma', gamma)
        if gamma == 0.:
            transformed_samples, logdet = fit_MFVI(logp_fn_shifted, seed=args.seed, nsample=args.nsample, ntrain=args.ntrain, beta_0=args.beta_0, learning_rate=args.learning_rate, num_bins=args.num_bins, range_max=args.range_max, max_iter=args.max_iter)
        else:
            transformed_samples, logdet = fit_gaussianization(logp_fn_shifted, seed=args.seed, nsample=args.nsample, ntrain=args.ntrain, gamma=gamma, beta_0=args.beta_0, learning_rate=args.learning_rate, num_bins=args.num_bins, range_max=args.range_max, max_iter=args.max_iter)
        transformed_samples = transformed_samples * scale + shift
        all_results[gamma] = evaluate(mcmc_samples_unc, ref_moment_1, ref_moment_2, transformed_samples, logdet)
        print(all_results[gamma])
    
    print(all_results)
    savepath = os.path.join(args.savepath, args.date, args.posterior_name)
    os.makedirs(savepath, exist_ok=True)
    filename = os.path.join(savepath, f'{args.posterior_name}_{args.nsample}_{args.ntrain}_lr_{args.learning_rate}_maxiter_{args.max_iter}_bin_{args.num_bins}_range_{args.range_max}_beta_0_{args.beta_0}_{args.seed}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(all_results, f)
    print('Results saved to', filename)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--posterior_name', type=str, default='hmm')
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--max_iter', type=int, default=100)
    argparser.add_argument('--learning_rate', type=float, default=1e-2)
    argparser.add_argument('--nsample', type=int, default=1000)
    argparser.add_argument('--ntrain', type=int, default=1000)
    argparser.add_argument('--num_bins', type=int, default=10)
    argparser.add_argument('--range_max', type=float, default=7.)
    argparser.add_argument('--beta_0', type=float, default=.5)
    argparser.add_argument('--savepath', type=str, default='.')
    argparser.add_argument('--date', type=str, default='')

    args = argparser.parse_args()

    data_file = f"stan/{args.posterior_name}.json"
    target = getattr(Targets, args.posterior_name)(data_file)
    d = target.d
    
    main(args)
    
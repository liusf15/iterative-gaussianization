import jax
import jax.numpy as jnp
import numpy as np
import os
import argparse
import pickle
import pandas as pd
from jax.scipy.optimize import minimize

from projection_vi import ComponentwiseFlow, AffineFlow, RealNVP, ComponentwiseCDF
from projection_vi.train import train, iterative_projection_mfvi, iterative_AS_mfvi
import experiments.targets as Targets
from projection_vi.utils import softplus, inverse_softplus

def Laplace_approximation(logp_fn, d):
    def neg_logp_fn(x):
        return -logp_fn(x)
    bfgs_res = minimize(neg_logp_fn, jnp.zeros(d,), method='BFGS')
    laplace_mean = bfgs_res.x
    laplace_cov = bfgs_res.hess_inv
    laplace_scale = jnp.sqrt(jnp.maximum(jnp.diag(laplace_cov), 1))
    if (not jnp.isnan(laplace_mean).any()) and (not jnp.isinf(laplace_scale).any()):
        return laplace_mean, laplace_scale
    else:
        x, losses = train(neg_logp_fn, jnp.zeros(d,), learning_rate=1e-2, max_iter=2000)
        laplace_mean = x
        laplace_cov = jnp.linalg.inv(jax.hessian(neg_logp_fn)(x))
        laplace_scale = jnp.sqrt(jnp.maximum(jnp.diag(laplace_cov), 1))
        return laplace_mean, laplace_scale

def run_experiment(posterior_name='arK', seed=0, n_train=1000, n_val=1000, niter=5, learning_rate=1e-3, max_iter=1000,n_layers=8, init='MFG', savepath=None):
    # set up target distribution
    data_file = f"stan/{posterior_name}.json"
    target = getattr(Targets, posterior_name)(data_file)
    logp_fn = jax.jit(target.log_prob)
    d = target.d

    # load reference moments
    with open (f"experiments/results/{posterior_name}_mcmc_moments.pkl", "rb") as f:
        reference_moments = pickle.load(f)
    ref_moment_1 = reference_moments['moments_1'].mean(0)
    ref_moment_2 = reference_moments['moments_2'].mean(0)

    # generate base samples
    key, subkey = jax.random.split(jax.random.key(seed))
    base_samples = jax.random.normal(subkey, (n_train + n_val, d))
    val_samples = base_samples[n_train:]
    base_samples = base_samples[:n_train]

    all_results = {}

    print("MFG")
    affine_model = AffineFlow(d=d)
    affine_params = affine_model.init(jax.random.key(0), jnp.zeros((1, d)))
    
    @jax.jit
    def loss_fn(affine_params):
        return affine_model.apply(affine_params, base_samples, logp_fn, method=affine_model.reverse_kl)
    affine_params, losses = train(loss_fn, affine_params, learning_rate=learning_rate, max_iter=max_iter)
    mfg_mean = affine_params['params']['shift']
    mfg_scale = softplus(affine_params['params']['scale_logit'] + inverse_softplus(1.))
    all_results['mfg'] = {'moment_1': mfg_mean, 
                          'moment_2': mfg_scale**2 + mfg_mean**2, 
                          'mse_1': np.sum((mfg_mean - ref_moment_1)**2 / ref_moment_1**2), 
                          'mse_2': np.sum((mfg_scale**2 + mfg_mean**2 - ref_moment_2)**2 / ref_moment_2**2)}

    if init == 'MFG':
        shift, scale = mfg_mean, mfg_scale
    elif init == 'Laplace':
        shift, scale = Laplace_approximation(logp_fn, d)
    else:
        shift = jnp.zeros(d)
        scale = jnp.ones(d)

    @jax.jit
    def logp_fn_shifted(x):
        return logp_fn(x * scale + shift) + jnp.sum(jnp.log(scale))

    def process_raw_samples(raw_samples):
        samples_constrain = target.param_constrain(raw_samples * scale + shift)
        moment_1 = np.mean(samples_constrain, axis=0)
        moment_2 = np.mean(samples_constrain**2, axis=0)
        mse_1 = np.sum((moment_1 - ref_moment_1)**2 / ref_moment_1**2)
        mse_2 = np.sum((moment_2 - ref_moment_2)**2 / ref_moment_2**2)
        return {'moment_1': moment_1, 'moment_2': moment_2, 'mse_1': mse_1, 'mse_2': mse_2}


    model = ComponentwiseFlow(d=d, num_bins=10, range_min=-5, range_max=5, boundary_slopes='unconstrained')

    print("MF")
    key, subkey = jax.random.split(key)
    mfvi_samples_unc, mfvi_val_samples, mfvi_logs = iterative_AS_mfvi(model, logp_fn_shifted, niter=1, key=subkey, base_samples=base_samples, val_samples=val_samples, learning_rate=learning_rate, max_iter=max_iter, rank0=-1, weighted=False, rotate_first_iter=False)
    all_results['mfvi'] = process_raw_samples(mfvi_samples_unc[0])

    print("iterative random projection")
    key, subkey = jax.random.split(key)
    rp_samples_unc, rp_val_samples, rp_logs = iterative_AS_mfvi(model, logp_fn_shifted, niter=niter, key=subkey, base_samples=base_samples, val_samples=val_samples, learning_rate=learning_rate, max_iter=max_iter, rank0=0, rank=0, weighted=False, rotate_first_iter=True)
    for j in range(niter):
        all_results[f'rp_iter{j}'] = process_raw_samples(rp_samples_unc[j])

    print("iterative AS projection")
    if d <= 3:
        rank = 0
    else:
        rank = 1
    as_samples_unc, as_val_samples, as_logs = iterative_AS_mfvi(model, logp_fn_shifted, niter=niter, key=subkey, base_samples=base_samples, val_samples=val_samples, learning_rate=learning_rate, max_iter=max_iter, rank0=d, rank=rank, weighted=False, rotate_first_iter=True)
    for j in range(niter):
        all_results[f'as_iter{j}'] = process_raw_samples(as_samples_unc[j])

    print("Fit RealNVP")
    model_nvp = RealNVP(dim=d, n_layers=n_layers, hidden_dims=[d])
    key, subkey = jax.random.split(key)
    params_nvp = model_nvp.init(subkey, jnp.zeros((1, d)))

    @jax.jit
    def loss_nvp(params_nvp):
        return model_nvp.apply(params_nvp, base_samples, logp_fn_shifted, method=model_nvp.reverse_kl)

    params_nvp, losses_nvp = train(loss_nvp, params_nvp, learning_rate=learning_rate, max_iter=max_iter)
    transformed_samples_nvp, _ = model_nvp.apply(params_nvp, base_samples, method=model_nvp.forward)
    all_results['realnvp'] = process_raw_samples(transformed_samples_nvp)
    print(pd.DataFrame(all_results))

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        filename = os.path.join(savepath, f'{posterior_name}_init_{init}_train_{n_train}_val_{n_val}_iter_{niter}_lr_{learning_rate}_maxiter_{max_iter}_layer_{n_layers}_{seed}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(all_results, f)
        print('Results saved to', filename)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--posterior_name', type=str, default='hmm', choices=['arK', 'gp_regr', 'hmm', 'nes_logit', 'normal_mixture', 'eight_schools', 'garch', 'mesquite'])
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--max_iter', type=int, default=500)
    argparser.add_argument('--lr', type=float, default=1e-2)
    argparser.add_argument('--n_train', type=int, default=3000)
    argparser.add_argument('--n_val', type=int, default=1000)
    argparser.add_argument('--niter', type=int, default=3)
    argparser.add_argument('--n_layers', type=int, default=8)
    argparser.add_argument('--init', type=str, default='MFG', choices=['MFG', 'Laplace'])
    argparser.add_argument('--savepath', type=str, default='/mnt/home/sliu1/ceph/projection_vi')
    argparser.add_argument('--date', type=str, default='20250415')

    args = argparser.parse_args()
    savepath = os.path.join(args.savepath, args.date)
    
    run_experiment(args.posterior_name, 
                   args.seed,
                   n_train=args.n_train, 
                   n_val=args.n_val,
                   niter=args.niter,
                   learning_rate=args.lr,
                   max_iter=args.max_iter, 
                   n_layers=args.n_layers,
                   init=args.init,
                   savepath=savepath)
    
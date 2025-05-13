import jax
import jax.numpy as jnp
import numpy as np
import os
import argparse
import pickle
import pandas as pd

from projection_vi import ComponentwiseFlow, RealNVP
from projection_vi.train import train, iterative_AS_mfvi
import experiments.targets as Targets


def run_experiment(posterior_name='arK', 
                   seed=0, 
                   n_train=2000, 
                   n_val=1000, niter=3, 
                   learning_rate=1e-2, 
                   max_iter=500, 
                   n_layers=8, 
                   savepath=None, 
                   boundary_slopes='unconstrained', 
                   num_bins=10, 
                   range_max=5.0,
                   IS_score=False,
                   init="laplace"):
    
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

    # load initialization
    initialization = pd.read_csv(f'experiments/results/{posterior_name}_{init}_initialization.csv', index_col=0)
    scale = initialization.loc['scale'].values
    shift = initialization.loc['mean'].values  

    # generate base samples
    key, subkey = jax.random.split(jax.random.key(seed))
    base_samples = jax.random.normal(subkey, (n_train + n_val, d))
    val_samples = base_samples[n_train:]
    base_samples = base_samples[:n_train]

    all_results = {}

    @jax.jit
    def logp_fn_shifted(x):
        return logp_fn(x * scale + shift) + jnp.sum(jnp.log(scale))

    def process_raw_samples(raw_samples, scale=1., shift=0.):
        samples_constrain = target.param_constrain(raw_samples * scale + shift)
        moment_1 = np.mean(samples_constrain, axis=0)
        moment_2 = np.mean(samples_constrain**2, axis=0)
        mse_1 = np.mean((moment_1 - ref_moment_1)**2 / jnp.clip(ref_moment_1**2, min=0.1))
        mse_2 = np.mean((moment_2 - ref_moment_2)**2 / jnp.clip(ref_moment_2**2, min=0.1))
        return {'moment_1': moment_1, 'moment_2': moment_2, 'mse_1': mse_1, 'mse_2': mse_2}

    model = ComponentwiseFlow(d=d, 
                              num_bins=num_bins, 
                              range_min=-range_max, 
                              range_max=range_max, 
                              boundary_slopes=boundary_slopes)

    print("MF")
    key, subkey = jax.random.split(key)
    mfvi_samples_unc, mfvi_logs = iterative_AS_mfvi(model, logp_fn_shifted, niter=1, key=subkey, base_samples=base_samples, val_samples=val_samples, learning_rate=learning_rate, max_iter=max_iter, rank0=-1, weighted=False)
    all_results['mfvi'] = process_raw_samples(mfvi_samples_unc[0], scale, shift)

    print("iterative random projection")
    key, subkey = jax.random.split(key)
    rp_samples_unc, rp_logs = iterative_AS_mfvi(model, logp_fn_shifted, niter=niter, key=subkey, base_samples=base_samples, val_samples=val_samples, learning_rate=learning_rate, max_iter=max_iter, rank0=0, rank=0, weighted=False)
    for j in range(niter):
        all_results[f'rp_iter{j}'] = process_raw_samples(rp_samples_unc[j], scale, shift)

    print("iterative AS projection")
    if d <= 2:
        rank = 0
    else:
        rank = d // 2
    as_samples_unc, as_logs = iterative_AS_mfvi(model, logp_fn_shifted, niter=niter, key=subkey, base_samples=base_samples, val_samples=val_samples, learning_rate=learning_rate, max_iter=max_iter, rank0=d, rank=rank, weighted=IS_score)
    for j in range(niter):
        all_results[f'as_iter{j}'] = process_raw_samples(as_samples_unc[j], scale, shift)

    print("Fit RealNVP")
    model_nvp = RealNVP(dim=d, n_layers=n_layers, hidden_dims=[d])
    key, subkey = jax.random.split(key)
    params_nvp = model_nvp.init(subkey, jnp.zeros((1, d)))

    @jax.jit
    def loss_nvp(params_nvp):
        return model_nvp.apply(params_nvp, base_samples, logp_fn_shifted, method=model_nvp.reverse_kl)

    params_nvp, losses_nvp = train(loss_nvp, params_nvp, learning_rate=learning_rate, max_iter=max_iter)
    if jnp.isnan(jnp.array(losses_nvp)).any():
        print('NaN loss, reducing learning rate to', learning_rate/10)
        key, subkey = jax.random.split(key)
        params_nvp = model_nvp.init(subkey, jnp.zeros((1, d)))
        params_nvp, losses_nvp = train(loss_nvp, params_nvp, learning_rate=learning_rate/10, max_iter=max_iter)

    transformed_samples_nvp, _ = model_nvp.apply(params_nvp, base_samples, method=model_nvp.forward)
    all_results['realnvp'] = process_raw_samples(transformed_samples_nvp, scale, shift)
    print(pd.DataFrame(all_results).loc['mse_1'])
    print(pd.DataFrame(all_results).loc['mse_2'])

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        if not IS_score:
            filename = os.path.join(savepath, f'{posterior_name}_init_{init}_train_{n_train}_val_{n_val}_iter_{niter}_lr_{learning_rate}_maxiter_{max_iter}_boundary_{boundary_slopes}_bin_{num_bins}_range_{range_max}_layer_{n_layers}_{seed}.pkl')
        else:
            filename = os.path.join(savepath, f'{posterior_name}_IS_init_{init}_train_{n_train}_val_{n_val}_iter_{niter}_lr_{learning_rate}_maxiter_{max_iter}_boundary_{boundary_slopes}_bin_{num_bins}_range_{range_max}_layer_{n_layers}_{seed}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(all_results, f)
        print('Results saved to', filename)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--posterior_name', type=str, default='hmm')
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--max_iter', type=int, default=500)
    argparser.add_argument('--lr', type=float, default=1e-2)
    argparser.add_argument('--n_train', type=int, default=2000)
    argparser.add_argument('--n_val', type=int, default=1000)
    argparser.add_argument('--niter', type=int, default=3)
    argparser.add_argument('--boundary_slopes', type=str, default='unconstrained')
    argparser.add_argument('--num_bins', type=int, default=10)
    argparser.add_argument('--range_max', type=float, default=5)
    argparser.add_argument('--n_layers', type=int, default=8)
    argparser.add_argument('--IS_score', action='store_true', default=False)
    argparser.add_argument('--init', type=str, default='laplace', choices=['laplace', 'mfg'])
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
                   savepath=savepath,
                   boundary_slopes=args.boundary_slopes,
                   num_bins=args.num_bins,
                   range_max=args.range_max,
                   IS_score=args.IS_score,
                   init=args.init)
    
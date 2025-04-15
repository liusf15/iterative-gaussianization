import jax
import jax.numpy as jnp
import numpy as np
import os
import argparse
import pickle

from projection_vi import ComponentwiseFlow, AffineFlow
from projection_vi.train import train, iterative_projection_mfvi
import experiments.targets as Targets


def run_experiment(posterior_name='arK', seed=0, n_train=1000, niter=5, learning_rate=1e-3, max_iter=1000, savepath=None):
    # set up target distribution
    data_file = f"stan/{posterior_name}.json"
    target = getattr(Targets, posterior_name)(data_file)
    logp_fn = jax.jit(target.log_prob)
    d = target.d

    print("Diagonal Gaussian VI")
    affine_model = AffineFlow(d=d)
    affine_params = affine_model.init(jax.random.key(0), jnp.zeros((1, d)))
    key, subkey = jax.random.split(jax.random.key(seed))
    base_samples = jax.random.normal(subkey, (n_train, d))

    @jax.jit
    def loss_fn(affine_params):
        return affine_model.apply(affine_params, base_samples, logp_fn, method=affine_model.reverse_kl)
    affine_params, losses = train(loss_fn, affine_params, learning_rate=learning_rate, max_iter=max_iter)
    shift = affine_params['params']['shift']
    scale = jax.nn.softplus(affine_params['params']['scale_logit']) * 1.2

    @jax.jit
    def logp_fn_shifted(x):
        return logp_fn(x * scale + shift) + jnp.sum(jnp.log(scale))

    model = ComponentwiseFlow(d=d, num_bins=10)
    key, subkey = jax.random.split(key)
    base_samples = jax.random.normal(subkey, (n_train, d))
    key, subkey = jax.random.split(key)
    log_weights_hist, samples_hist, loss_hist = iterative_projection_mfvi(model, logp_fn_shifted, niter=niter, key=subkey, base_samples=base_samples, learning_rate=learning_rate, max_iter=max_iter)

    moments_1 = [shift]
    moments_2 = [jax.nn.softplus(affine_params['params']['scale_logit'])**2 + shift**2]
    for k in range(niter):
        transformed_samples = samples_hist[k] * scale + shift
        transformed_samples = target.param_constrain(transformed_samples)
        moments_1.append(jnp.mean(transformed_samples, 0)) 
        moments_2.append(jnp.mean(transformed_samples**2, 0))

    with open (f"experiments/results/{posterior_name}_mcmc_moments.pkl", "rb") as f:
        reference_moments = pickle.load(f)
    ref_moment_1 = reference_moments['moments_1'].mean(0)
    ref_moment_2 = reference_moments['moments_2'].mean(0)

    mse_1 = np.sum((np.array(moments_1) - ref_moment_1)**2, 1)
    mse_2 = np.sum((np.array(moments_2) - ref_moment_2)**2, 1)
    print(mse_1, mse_2)
    if savepath is not None:
        results = {'moment1': moments_1,
                   'moment2': moments_2,
                   'mse1': mse_1,
                   'mse2': mse_2}
        os.makedirs(savepath, exist_ok=True)
        filename = os.path.join(savepath, f'{posterior_name}_train_{n_train}_iter_{niter}_lr_{learning_rate}_maxiter_{max_iter}_{seed}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print('Results saved to', filename)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--posterior_name', type=str, default='hmm', choices=['arK', 'gp_regr', 'hmm', 'nes_logit', 'normal_mixture'])
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--max_iter', type=int, default=500)
    argparser.add_argument('--lr', type=float, default=1e-2)
    argparser.add_argument('--n_train', type=int, default=1000)
    argparser.add_argument('--savepath', type=str, default='/mnt/home/sliu1/ceph/projection_vi')
    argparser.add_argument('--date', type=str, default='20250415')

    args = argparser.parse_args()
    savepath = os.path.join(args.savepath, args.date)
    
    run_experiment(args.posterior_name, 
                   args.seed,
                   n_train=args.n_train, 
                   learning_rate=args.lr,
                   max_iter=args.max_iter, 
                   savepath=savepath)
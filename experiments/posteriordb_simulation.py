import jax
import jax.numpy as jnp
import numpy as np
import os
import argparse
import pickle

from projection_vi import ComponentwiseFlow, AffineFlow, RealNVP, ComponentwiseCDF
from projection_vi.train import train, iterative_projection_mfvi, iterative_AS_mfvi
import experiments.targets as Targets

inverse_softplus = lambda x: jnp.log(jnp.exp(x) - 1.)


def run_experiment(posterior_name='arK', seed=0, n_train=1000, n_val=1000, niter=5, learning_rate=1e-3, max_iter=1000, AS=False, n_layers=8, savepath=None):
    # set up target distribution
    data_file = f"stan/{posterior_name}.json"
    target = getattr(Targets, posterior_name)(data_file)
    logp_fn = jax.jit(target.log_prob)
    d = target.d

    print("Diagonal Gaussian VI")
    affine_model = AffineFlow(d=d)
    affine_params = affine_model.init(jax.random.key(0), jnp.zeros((1, d)))
    key, subkey = jax.random.split(jax.random.key(seed))
    base_samples = jax.random.normal(subkey, (n_train + n_val, d))
    val_samples = base_samples[n_train:]
    base_samples = base_samples[:n_train]

    @jax.jit
    def loss_fn(affine_params):
        return affine_model.apply(affine_params, base_samples, logp_fn, method=affine_model.reverse_kl)
    affine_params, losses = train(loss_fn, affine_params, learning_rate=learning_rate, max_iter=max_iter)
    shift = affine_params['params']['shift']
    scale = jax.nn.softplus(affine_params['params']['scale_logit'] + inverse_softplus(1.))
    transformed_samples = target.param_constrain(base_samples * scale + shift)
    moments_1_mf = jnp.mean(transformed_samples, 0)
    moments_2_mf = jnp.mean(transformed_samples**2, 0)

    scale = scale * 1.2

    @jax.jit
    def logp_fn_shifted(x):
        return logp_fn(x * scale + shift) + jnp.sum(jnp.log(scale))

    model = ComponentwiseFlow(d=d, num_bins=10, range_min=-10, range_max=10)
    key, subkey = jax.random.split(key)
    
    if AS:
        rank = d // 2
    else:
        rank = 0
    train_samples_hist, val_samples_hist, validation_results = iterative_AS_mfvi(model, logp_fn_shifted, niter=niter, key=subkey, base_samples=base_samples, val_samples=val_samples, learning_rate=learning_rate, max_iter=max_iter, rank=rank, weighted=False)
    print('Validation KL:', validation_results['KL'])
    print('Validation ESS:', validation_results['ESS'])
    moments_1_train = []
    moments_2_train = []
    moments_1_val = []
    moments_2_val = []
    for k in range(niter):
        transformed_samples = target.param_constrain(train_samples_hist[k] * scale + shift)
        moments_1_train.append(jnp.mean(transformed_samples, 0)) 
        moments_2_train.append(jnp.mean(transformed_samples**2, 0))

        transformed_samples = target.param_constrain(val_samples_hist[k] * scale + shift)
        moments_1_val.append(jnp.mean(transformed_samples, 0)) 
        moments_2_val.append(jnp.mean(transformed_samples**2, 0))

    print("Fit RealNVP")
    model_nvp = RealNVP(dim=d, n_layers=n_layers, hidden_dims=[d])
    key, subkey = jax.random.split(key)
    params_nvp = model_nvp.init(subkey, jnp.zeros((1, d)))

    @jax.jit
    def loss_nvp(params_nvp):
        return model_nvp.apply(params_nvp, base_samples, logp_fn, method=model_nvp.reverse_kl)

    params_nvp, losses_nvp = train(loss_nvp, params_nvp, learning_rate=learning_rate, max_iter=max_iter)
    transformed_samples_nvp, _ = model_nvp.apply(params_nvp, base_samples, method=model_nvp.forward)
    transformed_samples_nvp = target.param_constrain(transformed_samples_nvp)
    nvp_moment_1 = jnp.mean(transformed_samples_nvp, 0)
    nvp_moment_2 = jnp.mean(transformed_samples_nvp**2, 0)

    with open (f"experiments/results/{posterior_name}_mcmc_moments.pkl", "rb") as f:
        reference_moments = pickle.load(f)
    ref_moment_1 = reference_moments['moments_1'].mean(0)
    ref_moment_2 = reference_moments['moments_2'].mean(0)

    mse_1_train = np.sum((np.array(moments_1_train) - ref_moment_1)**2 / ref_moment_1**2, 1)
    mse_2_train = np.sum((np.array(moments_2_train) - ref_moment_2)**2 / ref_moment_2**2, 1)
    mse_1_val = np.sum((np.array(moments_1_val) - ref_moment_1)**2 / ref_moment_1**2, 1)
    mse_2_val = np.sum((np.array(moments_2_val) - ref_moment_2)**2 / ref_moment_2**2, 1)
    if savepath is not None:
        results = {'mse1_train': mse_1_train,
                   'mse2_train': mse_2_train,
                   'mse1_val': mse_1_val,
                   'mse2_val': mse_2_val,
                   'mfvi_mse1': np.sum((moments_1_mf - ref_moment_1)**2 / ref_moment_1**2),
                    'mfvi_mse2': np.sum((moments_2_mf - ref_moment_2)**2 / ref_moment_2**2),
                   'nvp_mse1': np.sum((nvp_moment_1 - ref_moment_1)**2 / ref_moment_1**2),
                   'nvp_mse2': np.sum((nvp_moment_2 - ref_moment_2)**2 / ref_moment_2**2),
                   'validation_metrics': validation_results}
        print(results)
        os.makedirs(savepath, exist_ok=True)
        if AS:
            filename = os.path.join(savepath, f'AS_{posterior_name}_train_{n_train}_val_{n_val}_iter_{niter}_lr_{learning_rate}_maxiter_{max_iter}_layer_{n_layers}_{seed}.pkl')
        else:
            filename = os.path.join(savepath, f'random_{posterior_name}_train_{n_train}_val_{n_val}_iter_{niter}_lr_{learning_rate}_maxiter_{max_iter}_layer_{n_layers}_{seed}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print('Results saved to', filename)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--posterior_name', type=str, default='hmm', choices=['arK', 'gp_regr', 'hmm', 'nes_logit', 'normal_mixture', 'eight_schools', 'garch'])
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--max_iter', type=int, default=500)
    argparser.add_argument('--lr', type=float, default=1e-2)
    argparser.add_argument('--n_train', type=int, default=3000)
    argparser.add_argument('--n_val', type=int, default=1000)
    argparser.add_argument('--niter', type=int, default=3)
    argparser.add_argument('--n_layers', type=int, default=8)
    argparser.add_argument('--AS', default=False, action='store_true')
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
                   AS=args.AS,
                   n_layers=args.n_layers,
                   savepath=savepath)
    
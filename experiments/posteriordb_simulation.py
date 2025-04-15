import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn
from tqdm import trange
import numpy as np
import os
import argparse
import pickle

from projection_vi import ComponentwiseFlow, AffineFlow
from projection_vi.train import train
import experiments.targets as Targets

def run_experiment(posterior_name='arK', nsample=64, num_composition=1, max_deg=3, optimizer='lbfgs', max_iter=50, lr=1., savepath=None):
    # set up target distribution
    data_file = f"stan/{posterior_name}.json"
    target = getattr(Targets, posterior_name)(data_file)
    d = target.d

    print("training normalizing flow ......")
    model = TransportQMC(d, target, num_composition=num_composition, max_deg=max_deg)
    params = model.init_params()
    best_params = params
    max_ess = 0
    get_kl = jax.jit(model.reverse_kl)
    get_ess = jax.jit(model.ess)
    
    for seed in range(10):
        print("Seed:", seed)
        rng = np.random.default_rng(seed)

        U = jnp.array(sample_uniform(nsample, d, rng, 'rqmc'))
        loss_fn = lambda params: get_kl(params, U)

        U_val = jnp.array(sample_uniform(nsample, d, rng, 'rqmc'))
        val_fn = lambda params: get_ess(params, U_val)
        
        if optimizer == 'lbfgs':
            final_state, logs_ess = lbfgs(loss_fn, params, val_fn, max_iter=max_iter, max_lr=lr)
        else:
            final_state, logs_ess = sgd(loss_fn, params, val_fn, max_iter=max_iter, lr=lr)
        if logs_ess[-1] > max_ess:
            best_params = final_state[0]
            max_ess = logs_ess[-1]
    print("Effective sample size:", max_ess)

    print("estimating posterior moments ......")
    @jax.jit
    def get_samples(U):
        X, log_det = jax.vmap(model.forward, in_axes=(None, 0))(best_params, U)
        log_p = jax.vmap(target.log_prob)(X)
        log_weights = log_p + log_det
        log_weights = jnp.nan_to_num(log_weights, nan=-jnp.inf)
        log_weights -= jnp.max(log_weights)
        return X, log_weights

    def get_moments(X, log_weights):
        if getattr(target, 'param_constrain', None):
            X = target.param_constrain(np.array(X, float))
        
        weights = jnp.exp(log_weights - jnp.max(log_weights))
        moments_1 = jnp.sum(weights[:, None] * X, axis=0) / jnp.sum(weights)
        moments_2 = jnp.sum(weights[:, None] * X**2, axis=0) / jnp.sum(weights)
        return moments_1, moments_2
    
    m_list = np.arange(3, 12, 2)
    moments_1 = {}
    moments_2 = {}
    nrep = 50
    rng = np.random.default_rng(2024)
    for i in trange(nrep):
        for m in m_list:
            for sampler in ['mc', 'rqmc']:
                U = sample_uniform(2**m, d, rng, sampler)
                X, log_weights = get_samples(U)

                # compute first and second moments
                moments_1[(sampler, m, i)], moments_2[(sampler, m, i)] = get_moments(X, log_weights)
                
    if savepath is not None:
        results = {'flow_parameters': best_params, 
                   'moment1': moments_1,
                   'moment2': moments_2}
        os.makedirs(savepath, exist_ok=True)
        with open(os.path.join(savepath, f'{posterior_name}_comp_{num_composition}.pkl'), 'wb') as f:
            pickle.dump(results, f)
        print('Results saved to', os.path.join(savepath, f'{posterior_name}.pkl'))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--posterior_name', type=str, default='hmm', choices=['arK', 'gp_regr', 'hmm', 'nes_logit', 'normal_mixture'])
    argparser.add_argument('--m', type=int, default=8)
    argparser.add_argument('--num_composition', type=int, default=3)
    argparser.add_argument('--max_deg', type=int, default=7)
    argparser.add_argument('--optimizer', type=str, default='lbfgs', choices=['lbfgs', 'sgd'])
    argparser.add_argument('--max_iter', type=int, default=400)
    argparser.add_argument('--lr', type=float, default=1.)
    argparser.add_argument('--savepath', type=str, default=None)

    args = argparser.parse_args()
    nsample = 2**args.m
    
    run_experiment(args.posterior_name, 
                   nsample=nsample, 
                   num_composition=args.num_composition, 
                   max_deg=args.max_deg, 
                   optimizer=args.optimizer, 
                   max_iter=args.max_iter, 
                   lr=args.lr, 
                   savepath=args.savepath)
import jax
import jax.numpy as jnp
import argparse
import os
import pickle
import pandas as pd

from projection_vi.utils import softplus, inverse_softplus
from projection_vi.iterative_gaussianization import *
from projection_vi.flows import BlockAffineFlow, AffineFlow
from experiments.targets import GLMM

def load_reference_moments():
    with open (f"experiments/posteriordb_experiment/references/glmm_n_{n}_mcmc_moments.pkl", "rb") as f:
        reference_moments = pickle.load(f)
    ref_moment_1 = reference_moments['moments_1'].mean(0)
    ref_moment_2 = reference_moments['moments_2'].mean(0)
    return ref_moment_1, jnp.sqrt(ref_moment_2 - ref_moment_1**2)

def load_reference_samples():
    filename = f'experiments/posteriordb_experiment/references/GLMM_n_{n}_mcmc_samples_unc.csv'
    samples = pd.read_csv(filename, index_col=0).values
    samples = target.param_constrain(samples)
    return samples

def metrics(samples):
    metrics = {}
    transformed_samples = target.param_constrain(jax.vmap(target.reparametrize_inv)(samples))
    metrics['mean'] = jnp.mean(transformed_samples, 0)
    metrics['std'] = jnp.std(transformed_samples, 0)
    return transformed_samples, metrics

def main(args):
    key1, key2, key3 = jax.random.split(jax.random.key(args.seed), 3)
    ntrain = args.ntrain
    nsample = args.nsample
    beta_0 = args.beta_0
    lr = args.lr
    max_iter = args.max_iter

    # Initialization by mean-field Gaussian VI
    mfg_flow = AffineFlow(d)
    params_mfg, _ = MFVIStep(target.log_prob_reparam, 
                             d, 
                             mfg_flow, 
                             ntrain, 
                             key1, 
                             beta_0=1., 
                             learning_rate=0.1, 
                             max_iter=200)

    shift = params_mfg['params']['shift']
    scale = softplus(params_mfg['params']['scale_logit'] + inverse_softplus(1.))

    @jax.jit
    def logp_fn_shifted(x):
        return target.log_prob_reparam(x * scale + shift) + jnp.sum(jnp.log(scale))

    # Block Gaussian VI
    gvi_flow = BlockAffineFlow(d, 3)
    params_gvi, _ = MFVIStep(logp_fn_shifted, 
                             d, 
                             gvi_flow, 
                             ntrain, 
                             key1, 
                             beta_0=beta_0, 
                             learning_rate=lr, 
                             max_iter=max_iter)

    # MFVI+PCA
    V_r = ScorePCA(logp_fn_shifted, d, ntrain, key2, args.gamma)
    W = get_householder_matrix(V_r)
    logp_rotated = RotateTarget(logp_fn_shifted, W)
    params_mfg_pca, _ = MFVIStep(logp_rotated, 
                                 d, 
                                 mfg_flow, 
                                 ntrain, 
                                 key1, 
                                 beta_0=beta_0, 
                                 learning_rate=lr, 
                                 max_iter=max_iter)

    base_samples = jax.random.normal(key3, (nsample, d))

    gvi_samples, _ = gvi_flow.apply(params_gvi, base_samples, method=gvi_flow.forward)
    gvi_samples = gvi_samples * scale + shift
    
    mfg_pca_samples, _ = mfg_flow.apply(params_mfg_pca, base_samples, method=mfg_flow.forward)
    mfg_pca_samples = jax.vmap(apply_householder, in_axes=(None, 0))(W, mfg_pca_samples)
    mfg_pca_samples = mfg_pca_samples * scale + shift

    all_samples = {}
    all_samples['gvi'] = metrics(gvi_samples)
    all_samples['mfg_pca'] = metrics(mfg_pca_samples)

    savepath = args.savepath
    os.makedirs(savepath, exist_ok=True)
    filename = os.path.join(savepath, f'GLMM_n_{args.n}_{args.seed}.pkl')
    with open (filename, "wb") as f:
        pickle.dump(all_samples, f)
    print(f'Saved to {filename}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ntrain', type=int, default=5000)
    parser.add_argument('--nsample', type=int, default=5000)
    parser.add_argument('--beta_0', type=float, default=.5)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--savepath', type=str, default='experiments/results/glmm')

    args = parser.parse_args()

    n = args.n
    target = GLMM(f"stan/GLMM_n_{n}.json", family='poisson')
    d = target.d

    main(args)

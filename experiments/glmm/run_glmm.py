import jax
import jax.numpy as jnp
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jax.scipy.special import digamma, logsumexp

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

def reparametrize(x):
    sigmasq_inv = jnp.exp(x[0])
    beta0 = x[1]
    beta1 = x[2]
    b = x[3:]
    cond_var = 1 / (sigmasq_inv + jnp.sum(hess_at_mle, 1))
    cond_mean = -cond_var * (jnp.sum(hess_at_mle * (beta0 + beta1 * target.x - eta_mle), 1))
    b_new = (b - cond_mean) / jnp.sqrt(cond_var)
    x_new = jnp.concatenate([x[:3], b_new])
    return x_new

def reparametrize_inv(x):
    sigmasq_inv = jnp.exp(x[0])
    beta0 = x[1]
    beta1 = x[2]
    b = x[3:]
    cond_var = 1 / (sigmasq_inv + jnp.sum(hess_at_mle, 1))
    cond_mean = -cond_var * (jnp.sum(hess_at_mle * (beta0 + beta1 * target.x - eta_mle), 1))
    b_orig = b * jnp.sqrt(cond_var) + cond_mean
    x_orig = jnp.concatenate([x[:3], b_orig])
    return x_orig

@jax.jit
def log_prob_reparam(x):
    x_orig = reparametrize_inv(x)
    logp = target.log_prob(x_orig)
    sigmasq_inv = jnp.exp(x[0])
    cond_var = 1 / (sigmasq_inv + jnp.sum(hess_at_mle, 1))
    log_det = 0.5 * jnp.sum(jnp.log(cond_var))
    return logp + log_det

def metrics(samples, log_weights):
    metrics = {}
    log_weights = log_weights - jnp.max(log_weights)
    log_weights = log_weights - jnp.log(jnp.sum(jnp.exp(log_weights)))
    metrics['ess'] = jnp.exp(2 * logsumexp(log_weights) - logsumexp(2 * log_weights))
    metrics['elbo'] = jnp.nanmean(log_weights)

    samples_constrain = jax.vmap(reparametrize_inv)(samples)
    samples_constrain = target.param_constrain(samples_constrain)
    metrics['mean'] = jnp.mean(samples_constrain, 0)
    metrics['std'] = jnp.std(samples_constrain, 0)
    return samples_constrain, metrics

def main(args):
    key1, key2, key3 = jax.random.split(jax.random.key(args.seed), 3)
    ntrain = args.ntrain
    nsample = args.nsample
    beta_0 = args.beta_0
    lr = args.lr

    # Block Gaussian VI
    gvi_flow = BlockAffineFlow(d, 3)
    params_gvi, losses = MFVIStep(log_prob_reparam, d, gvi_flow, ntrain, key1, beta_0=beta_0, learning_rate=lr, max_iter=300)

    # Mean-field Gaussian VI
    mfg_flow = AffineFlow(d)
    params_mfg, losses = MFVIStep(log_prob_reparam, d, mfg_flow, ntrain, key1, learning_rate=lr, max_iter=200)

    # MFV + score PCA
    shift = params_mfg['params']['shift']
    scale = softplus(params_mfg['params']['scale_logit'] + inverse_softplus(1.))

    @jax.jit
    def logp_fn_shifted(x):
        return log_prob_reparam(x * scale + shift) + jnp.sum(jnp.log(scale))

    V_r = ScorePCA(logp_fn_shifted, d, ntrain, key2, args.gamma)
    r = V_r.shape[1]
    W = get_householder_matrix(V_r)
    logp_rotated = RotateTarget(logp_fn_shifted, W)
    params_mfg_pca, losses = MFVIStep(logp_rotated, d, mfg_flow, ntrain, key1, beta_0=beta_0, learning_rate=lr, max_iter=100)

    base_samples = jax.random.normal(key3, (nsample, d))
    logq = -jnp.sum(base_samples**2, 1) / 2 - d / 2 * jnp.log(2 * jnp.pi)
    mfg_samples, logdet_mfg = mfg_flow.apply(params_mfg, base_samples, method=mfg_flow.forward)
    mfg_log_weight = jax.vmap(log_prob_reparam)(mfg_samples) - logq + logdet_mfg

    gvi_samples, logdet_gvi = gvi_flow.apply(params_gvi, base_samples, method=gvi_flow.forward)
    gvi_log_weight = jax.vmap(log_prob_reparam)(gvi_samples) - logq + logdet_gvi

    mfg_pca_samples, logdet_mfg_pca = mfg_flow.apply(params_mfg_pca, base_samples, method=mfg_flow.forward)
    mfg_pca_samples = jax.vmap(apply_householder, in_axes=(None, 0))(W, mfg_pca_samples)
    mfg_pca_samples = mfg_pca_samples * scale + shift
    mfg_pca_log_weight = jax.vmap(log_prob_reparam)(mfg_pca_samples) - logq + logdet_mfg_pca + jnp.sum(jnp.log(scale))

    all_samples = {}
    all_samples['gvi'] = metrics(gvi_samples, gvi_log_weight)
    all_samples['mfg'] = metrics(mfg_samples, mfg_log_weight)
    all_samples['mfg_pca'] = metrics(mfg_pca_samples, mfg_pca_log_weight)

    savepath = args.savepath
    os.makedirs(savepath, exist_ok=True)
    filename = os.path.join(savepath, f'GLMM_n_{args.n}_lr_{args.lr}_gamma_{args.gamma}_beta0_{args.beta_0}_{args.seed}.pkl')
    with open (filename, "wb") as f:
        pickle.dump(all_samples, f)

    ref_samples = load_reference_samples()
    fig, ax = plt.subplots(1, 3, figsize=(6, 2))
    for j in range(3):
        sns.kdeplot(ref_samples[:, j], ax=ax[j], label='MCMC')
        for method in ['gvi', 'mfg', 'mfg_pca']:
            sns.kdeplot(all_samples[method][0][:, j], ax=ax[j], label=method)
    plt.legend()
    plt.savefig(f'{savepath}/GLMM_n_{args.n}_lr_{args.lr}_gamma_{args.gamma}_beta0_{args.beta_0}_{args.seed}.png', bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ntrain', type=int, default=5000)
    parser.add_argument('--nsample', type=int, default=5000)
    parser.add_argument('--beta_0', type=float, default=.5)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--max_iter', type=int, default=300)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--savepath', type=str, default='experiments/results/glmm')

    args = parser.parse_args()

    n = args.n
    target = GLMM(f"stan/GLMM_n_{n}.json", family='poisson')
    d = target.d
    data = target.data
    eta_mle = digamma(target.y + 0.5)
    hess_at_mle = jnp.exp(eta_mle)

    main(args)
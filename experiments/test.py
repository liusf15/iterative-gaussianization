import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn
from src.projection_vi.flows import ComponentwiseFlow
from src.projection_vi.train import train

d = 2
mean = jnp.array([1., -1.])
cov = jnp.array([[1., 0.5], [0.5, 1.]])


def logp_fn(x):
    return mvn.logpdf(x, mean=mean, cov=cov)


model = ComponentwiseFlow(d=d, num_bins=10)

params = model.init(jax.random.key(0), jnp.zeros((1, d)))

base_samples = jax.random.normal(jax.random.key(3), (100, d))

@jax.jit
def loss_fn(params):
    return model.apply(params, base_samples, logp_fn, method=model.reverse_kl)

params, losses = train(loss_fn, params, learning_rate=1e-3, max_iter=500)



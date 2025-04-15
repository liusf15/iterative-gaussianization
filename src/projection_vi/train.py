import jax
import jax.numpy as jnp
import optax
import jax_tqdm
from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.special import logsumexp
from projection_vi.utils import sample_ortho

def train(loss_fn, params, learning_rate=0.01, max_iter=500):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    @jax_tqdm.scan_tqdm(max_iter)
    def train_step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    init_carry = (params, opt_state)
    carry, losses = jax.lax.scan(train_step, init_carry, jnp.arange(max_iter))
    params, opt_state = carry
    losses = list(losses)
    return params, losses


def train_minibatch(loss_fn, params, key, learning_rate=0.01, max_iter=500):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    keys = jax.random.split(key, max_iter)

    @jax_tqdm.scan_tqdm(max_iter)
    def train_step(carry, t):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, keys[t])
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    init_carry = (params, opt_state)
    carry, losses = jax.lax.scan(train_step, init_carry, jnp.arange(max_iter))
    params, opt_state = carry
    losses = list(losses)
    return params, losses

def iterative_projection_mfvi(model, logp_fn, niter, key, base_samples, learning_rate=1e-3, max_iter=1000):
    d = model.d

    logq = mvn.logpdf(base_samples, mean=jnp.zeros(d), cov=jnp.eye(d))

    @jax.jit
    def loss_fn(params, base_samples, rot):
        return model.apply(params, base_samples, logp_fn, rot=rot, method=model.reverse_kl)

    def _step(rot, base_samples):
        params = model.init(jax.random.key(0), jnp.zeros((1, d)))
        loss = lambda params: loss_fn(params, base_samples, rot)
        params, losses = train(loss, params, learning_rate=learning_rate, max_iter=max_iter)
        transformed_samples, logdet = model.apply(params, base_samples, rot=rot, method=model.forward)
        return transformed_samples, logdet, losses

    log_weights_hist = []
    loss_hist = []
    samples_hist = []
    for k in range(niter):
        key, subkey = jax.random.split(key)
        rot = sample_ortho(d, subkey)

        base_samples, ld, losses = _step(rot, base_samples)
        logq = logq - ld
        log_weights_hist.append(jax.vmap(logp_fn)(base_samples) - logq)
        samples_hist.append(base_samples)
        loss_hist.append(losses)
        ess = jnp.exp(2 * logsumexp(log_weights_hist[-1]) - logsumexp(2 * log_weights_hist[-1]))
        print("Iteration:", k, 'KL:', -log_weights_hist[-1].mean(), 'ESS:', ess)
    return jnp.stack(log_weights_hist), jnp.stack(samples_hist), jnp.array(loss_hist)

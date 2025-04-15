import jax
import jax.numpy as jnp

def sample_ortho(d, key):
    A = jax.random.normal(key=key, shape=(d, d))
    Q = jnp.linalg.qr(A)[0]
    return Q
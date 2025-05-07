import jax
import jax.numpy as jnp
from jax.nn import softplus
inverse_softplus = lambda x: jnp.log(jnp.exp(x) - 1.)

def sample_ortho(d, key):
    A = jax.random.normal(key=key, shape=(d, d))
    Q = jnp.linalg.qr(A)[0]
    return Q

def complete_orthonormal_basis(U_r, key):
    d, r = U_r.shape

    random_matrix = jax.random.normal(key, shape=(d, d - r))
    orthogonal_component = random_matrix - U_r @ (U_r.T @ random_matrix)
    Q, _ = jnp.linalg.qr(orthogonal_component)

    return jnp.hstack([U_r, Q])

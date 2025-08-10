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

def median_bandwidth(X: jnp.ndarray) -> float:
    """
    Compute the RBF bandwidth sigma using the median heuristic on samples X.
    
    Args:
        X: (n, d) array of samples.
    
    Returns:
        sigma: median of pairwise distances.
    """
    # Pairwise squared distances
    sq_norms = jnp.sum(X**2, axis=1, keepdims=True)  # (n, 1)
    D2 = sq_norms + sq_norms.T - 2 * X @ X.T         # (n, n)
    
    # Extract upper triangle indices i < j
    n = X.shape[0]
    i, j = jnp.triu_indices(n, k=1)
    
    # Compute distances and take median
    dists = jnp.sqrt(jnp.maximum(D2[i, j], 0.0))
    return jnp.median(dists)

def rbf_kernel(X: jnp.ndarray, Y: jnp.ndarray = None, sigma: float = 1.0) -> jnp.ndarray:
    """
    RBF kernel matrix via JAX.
    """
    if Y is None:
        Y = X
    # pairwise squared distances
    sq_dists = jnp.sum((X[:, None, :] - Y[None, :, :])**2, axis=-1)
    return jnp.exp(-sq_dists / (2 * sigma**2))

def compute_mmd(X: jnp.ndarray, Y: jnp.ndarray, sigma: float = 1.0, biased: bool = False) -> jnp.ndarray:
    """
    Squared MMD between X and Y using RBF kernel.
    """
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)

    if biased:
        return jnp.mean(Kxx) + jnp.mean(Kyy) - 2 * jnp.mean(Kxy)
    else:
        n = X.shape[0]
        m = Y.shape[0]
        sum_xx = (jnp.sum(Kxx) - jnp.trace(Kxx)) / (n * (n - 1))
        sum_yy = (jnp.sum(Kyy) - jnp.trace(Kyy)) / (m * (m - 1))
        sum_xy = jnp.mean(Kxy)
        return sum_xx + sum_yy - 2 * sum_xy

def compute_ksd(X: jnp.ndarray, score_fn: callable, sigma: float = 1.0) -> jnp.ndarray:
    """
    Squared Kernel Stein Discrepancy using the Langevin Stein kernel.
    score_fn: a function X -> ∇ log p(X)
    """
    n, d = X.shape
    score = score_fn(X)        # (n, d)
    K = rbf_kernel(X, sigma=sigma)  # (n, n)

    # pairwise differences
    X_diff = X[:, None, :] - X[None, :, :]  # (n, n, d)
    # gradient of kernel
    grad_K = - (X_diff / (sigma**2)) * K[..., None]  # (n, n, d)
    # laplacian of kernel
    sq_dist = jnp.sum(X_diff**2, axis=-1)             # (n, n)
    laplacian_K = ((sq_dist - sigma**2) / (sigma**4)) * K  # (n, n)

    term1 = jnp.einsum('id,jd,ij->ij', score, score, K)
    term2 = jnp.einsum('id,ijd->ij', score, grad_K)
    term3 = jnp.einsum('jd,ijd->ij', score, grad_K)
    H = term1 + term2 + term3 + laplacian_K

    return (jnp.sum(H) - jnp.trace(H)) / (n * (n - 1))

# Example usage (with jit-compiled functions):
# X = jax.random.normal(jax.random.PRNGKey(0), (128, 2))
# Y = jax.random.normal(jax.random.PRNGKey(1), (128, 2))
# sigma = median_bandwidth(np.vstack([np.array(X), np.array(Y)]))  # from earlier
# mmd2 = compute_mmd(X, Y, sigma, biased=False)
# def score_fn(x): return -x  # for standard normal
# ksd2 = compute_ksd(X, score_fn, sigma)

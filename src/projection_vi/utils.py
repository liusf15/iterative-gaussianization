import jax
import jax.numpy as jnp
from typing import Callable, Optional

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

def median_heuristic(x_samples: jnp.ndarray, y_samples: jnp.ndarray = None) -> float:
    """
    Compute median heuristic for bandwidth selection.
    If y_samples is None, uses only x_samples for pairwise distances.
    """
    if y_samples is None:
        samples = x_samples
        n = samples.shape[0]

        diffs = samples[:, None, :] - samples[None, :, :]  
        distances = jnp.sqrt(jnp.sum(diffs**2, axis=-1))  

        triu_indices = jnp.triu_indices(n, k=1)
        pairwise_distances = distances[triu_indices]
    else:
        nx, ny = x_samples.shape[0], y_samples.shape[0]
        diffs = x_samples[:, None, :] - y_samples[None, :, :]  
        distances = jnp.sqrt(jnp.sum(diffs**2, axis=-1))  
        pairwise_distances = distances.flatten()
    
    return jnp.median(pairwise_distances)

def compute_ksd(samples: jnp.ndarray,
                score_fn: Callable[[jnp.ndarray], jnp.ndarray],
                kernel_type: str = 'rbf',
                bandwidth: Optional[float] = None,
                beta: float = -0.5) -> float:
    n_samples, dim = samples.shape
    
    scores = jax.vmap(score_fn)(samples)
    
    if bandwidth is None:
        bandwidth = median_heuristic(samples)
        if bandwidth == 0:
            bandwidth = 1.0
    
    x_diff = samples[:, None, :] - samples[None, :, :]  
    distances_sq = jnp.sum(x_diff**2, axis=-1)  
    
    if kernel_type == 'rbf':
        # k(x,y) = exp(-||x-y||^2 / (2*h^2))
        k_xy = jnp.exp(-distances_sq / (2 * bandwidth**2))  
        
        # ∇_x k(x,y) = k(x,y) * (y-x) / h^2
        grad_k = k_xy[:, :, None] * (-x_diff) / (bandwidth**2)  
        
        # ∇_y k(x,y) = k(x,y) * (x-y) / h^2  
        grad_k_y = k_xy[:, :, None] * x_diff / (bandwidth**2) 
        
        # trace(∇_x ∇_y k(x,y)) = k(x,y) * (d/h^2 - ||x-y||^2/h^4)
        trace_hess = k_xy * (dim / (bandwidth**2) - distances_sq / (bandwidth**4))
        
    elif kernel_type == 'imq':
        # k(x,y) = (h^2 + ||x-y||^2)^β
        k_base = bandwidth**2 + distances_sq  # (n, n)
        k_xy = jnp.power(k_base, beta)  # (n, n)
        
        # ∇_x k(x,y) = 2β * k(x,y) * (x-y) / (h^2 + ||x-y||^2)
        grad_k = 2 * beta * k_xy[:, :, None] * x_diff / k_base[:, :, None] 
        grad_k_y = -grad_k  # ∇_y k(x,y) = -∇_x k(x,y)
        
        # trace(∇_x ∇_y k(x,y)) for IMQ
        trace_hess = 2 * beta * k_xy * (dim * (beta - 1) / k_base + 2 * beta * distances_sq / (k_base**2))
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Term 1: score_x^T score_y k(x,y)
    term1 = jnp.sum(scores[:, None, :] * scores[None, :, :], axis=-1) * k_xy  
    
    # Term 2: score_x^T ∇_y k(x,y)
    term2 = jnp.sum(scores[:, None, :] * grad_k_y, axis=-1) 
    
    # Term 3: score_y^T ∇_x k(x,y)
    term3 = jnp.sum(scores[None, :, :] * grad_k, axis=-1) 
    
    # Term 4: trace(∇_x ∇_y k(x,y))
    term4 = trace_hess 
    
    stein_kernel_matrix = term1 + term2 + term3 + term4
    
    ksd_squared = jnp.mean(stein_kernel_matrix)
    
    return jnp.sqrt(jnp.maximum(ksd_squared, 0.0))

def mmd_squared(x_samples: jnp.ndarray, 
                y_samples: jnp.ndarray,
                kernel_type: str = 'rbf',
                bandwidth: Optional[float] = None,
                **kernel_kwargs) -> float:
    nx, ny = x_samples.shape[0], y_samples.shape[0]
    
    if bandwidth is None and kernel_type in ['rbf', 'imq', 'laplace']:
        bandwidth = median_heuristic(x_samples, y_samples)
        if bandwidth == 0:
            bandwidth = 1.0
    
    if kernel_type == 'rbf':
        def compute_rbf_terms():
            # ||x_i - x_j||² for all i,j
            x_sq = jnp.sum(x_samples**2, axis=1, keepdims=True) 
            x_dists_sq = x_sq + x_sq.T - 2 * jnp.dot(x_samples, x_samples.T)  
            k_xx = jnp.exp(-x_dists_sq / (2 * bandwidth**2))
            
            # ||y_i - y_j||² for all i,j  
            y_sq = jnp.sum(y_samples**2, axis=1, keepdims=True)  
            y_dists_sq = y_sq + y_sq.T - 2 * jnp.dot(y_samples, y_samples.T)  
            k_yy = jnp.exp(-y_dists_sq / (2 * bandwidth**2))
            
            # ||x_i - y_j||² for all i,j
            xy_dists_sq = x_sq + y_sq.T - 2 * jnp.dot(x_samples, y_samples.T)  
            k_xy = jnp.exp(-xy_dists_sq / (2 * bandwidth**2))
            
            return k_xx, k_yy, k_xy
        
        k_xx, k_yy, k_xy = compute_rbf_terms()
        
    elif kernel_type == 'imq':
        beta = kernel_kwargs.get('beta', -0.5)
        
        def compute_imq_terms():
            x_sq = jnp.sum(x_samples**2, axis=1, keepdims=True)
            x_dists_sq = x_sq + x_sq.T - 2 * jnp.dot(x_samples, x_samples.T)
            k_xx = jnp.power(bandwidth**2 + x_dists_sq, beta)
            
            y_sq = jnp.sum(y_samples**2, axis=1, keepdims=True)
            y_dists_sq = y_sq + y_sq.T - 2 * jnp.dot(y_samples, y_samples.T)
            k_yy = jnp.power(bandwidth**2 + y_dists_sq, beta)
            
            xy_dists_sq = x_sq + y_sq.T - 2 * jnp.dot(x_samples, y_samples.T)
            k_xy = jnp.power(bandwidth**2 + xy_dists_sq, beta)
            
            return k_xx, k_yy, k_xy
        
        k_xx, k_yy, k_xy = compute_imq_terms()
        
    else:
        raise NotImplementedError(f"Kernel type '{kernel_type}' not implemented.")
    
    k_xx_off_diag = k_xx - jnp.diag(jnp.diag(k_xx))
    k_yy_off_diag = k_yy - jnp.diag(jnp.diag(k_yy))
    
    term1 = jnp.sum(k_xx_off_diag) / (nx * (nx - 1))
    term2 = jnp.sum(k_yy_off_diag) / (ny * (ny - 1))
    term3 = jnp.sum(k_xy) / (nx * ny)
    
    return term1 + term2 - 2 * term3

def compute_mmd(x_samples: jnp.ndarray, 
        y_samples: jnp.ndarray,
        kernel_type: str = 'rbf',
        bandwidth: Optional[float] = None,
        **kernel_kwargs) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between two sets of samples.
    
    Args:
        x_samples: Samples from first distribution, shape (n_x, dim)
        y_samples: Samples from second distribution, shape (n_y, dim)
        kernel_type: Type of kernel ('rbf', 'imq', 'polynomial', 'laplace')
        bandwidth: Kernel bandwidth (auto-selected if None)
        **kernel_kwargs: Additional kernel parameters
        
    Returns:
        MMD value (always non-negative)
    """
    mmd_sq = mmd_squared(x_samples, y_samples, kernel_type, bandwidth, **kernel_kwargs)
    return jnp.sqrt(jnp.maximum(mmd_sq, 0.0))  

def wasserstein_1d(x, y, p=2):
    x = jnp.sort(x)
    y = jnp.sort(y)
    n, m = x.size, y.size

    k = max(n, m)
    qs = (jnp.arange(k) + 0.5) / k

    xq = jnp.quantile(x, qs, method="linear")
    yq = jnp.quantile(y, qs, method="linear")
    return (jnp.mean(jnp.abs(xq - yq) ** p)) ** (1.0 / p)


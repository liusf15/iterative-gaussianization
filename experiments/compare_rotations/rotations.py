import jax
import jax.numpy as jnp
from typing import Callable, Optional

def compute_active_subspace_rotation(log_prob_fn: Callable[[jnp.ndarray], float], 
                                    d: int,
                                    num_samples: int = 2000,
                                    key: Optional[jax.Array] = None) -> jnp.ndarray:
    """
    Compute active subspace rotation matrix based on covariance of (grad_log_p + x).
    This captures the difference between target score and Gaussian score.
    
    Args:
        log_prob_fn: Target log probability function
        d: Dimension of the target distribution
        num_samples: Number of samples to estimate score covariance
        key: JAX random key
        
    Returns:
        Orthogonal rotation matrix (d x d)
    """
    if key is None:
        key = jax.random.key(123)
    
    # Sample from standard Gaussian
    gaussian_samples = jax.random.normal(key, (num_samples, d))
    
    # Compute score function (gradient of log probability)
    def score_fn(x):
        return jax.grad(log_prob_fn)(x)
    
    # Compute score differences: grad_log_p(x) + x
    # This is the difference between target score and Gaussian score (-x)
    score_differences = jax.vmap(lambda x: score_fn(x) + x)(gaussian_samples)
    
    score_cov = jnp.cov(score_differences.T)
    
    # Eigendecomposition for active subspace
    eigenvalues, eigenvectors = jnp.linalg.eigh(score_cov)
    
    # Sort eigenvalues/eigenvectors in descending order
    idx = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Return eigenvectors as rotation matrix (columns are active subspace directions)
    rotation_matrix = eigenvectors
    
    return rotation_matrix

def compute_relative_score_pca_rotation(log_prob_fn: Callable[[jnp.ndarray], float], 
                                       d: int,
                                       num_samples: int = 2000,
                                       key: Optional[jax.Array] = None) -> jnp.ndarray:
    """
    Compute rotation matrix using relative score PCA based on cross-covariance 
    between x (standard normal) and relative score (grad_log_p + x).
    
    Args:
        log_prob_fn: Target log probability function
        d: Dimension of the target distribution
        num_samples: Number of samples to estimate cross-covariance
        key: JAX random key
        
    Returns:
        Orthogonal rotation matrix (d x d)
    """
    if key is None:
        key = jax.random.key(123)
    
    # Sample from standard Gaussian
    gaussian_samples = jax.random.normal(key, (num_samples, d))
    
    # Compute score function (gradient of log probability)
    def score_fn(x):
        return jax.grad(log_prob_fn)(x)
    
    # Compute relative scores: grad_log_p(x) + x
    # This is the difference between target score and Gaussian score (-x)
    relative_scores = jax.vmap(lambda x: score_fn(x) + x)(gaussian_samples)
    
    # Compute empirical cross-covariance matrix between x and relative score
    # Cross-cov[i,j] = E[x_i * relative_score_j] (without centering as requested)
    cross_cov = jnp.dot(gaussian_samples.T, relative_scores) / num_samples
    
    # Symmetrize the cross-covariance matrix
    cross_cov_sym = 0.5 * (cross_cov + cross_cov.T)
    
    # Eigendecomposition for relative score PCA
    eigenvalues, eigenvectors = jnp.linalg.eigh(cross_cov_sym)
    
    # Sort eigenvalues/eigenvectors in descending order by absolute value
    idx = jnp.argsort(jnp.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Return eigenvectors as rotation matrix (columns are relative score PCA directions)
    rotation_matrix = eigenvectors
    
    return rotation_matrix


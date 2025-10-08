import jax
import jax.numpy as jnp
import optax
import jax_tqdm
from jax.scipy.stats import multivariate_normal as mvn
from projection_vi.flows import ComponentwiseFlow
from projection_vi.utils import sample_ortho

def MFVIStep(logp_fn, d, flow, nsample, key, beta_0=1., learning_rate=1e-3, max_iter=1000, T_anneal=None):
    key, subkey = jax.random.split(key)
    params = flow.init(subkey, jnp.zeros((1, d)))

    key, subkey = jax.random.split(key)
    base_samples = jax.random.normal(subkey, shape=(nsample, d))

    if T_anneal is None:
        T_anneal = int(0.8 * max_iter)

    @jax.jit
    def reverse_kl(params, t):
        X, log_det = flow.apply(params, base_samples)
        t_ = jnp.clip(t, 0, T_anneal)
        beta_t = 1 - .5 * (1 + jnp.cos(jnp.pi * t_ / T_anneal)) * (1 - beta_0)
        logp = jax.vmap(logp_fn)(X) * beta_t
        logp = jnp.where(jnp.abs(logp) < 1e10, logp, jnp.nan)
        logq = mvn.logpdf(base_samples, mean=jnp.zeros(d), cov=jnp.eye(d))
        return jnp.nanmean(logq - log_det - logp)
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    @jax_tqdm.scan_tqdm(max_iter)
    def train_step(carry, t):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(reverse_kl)(params, t)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    init_carry = (params, opt_state)
    carry, losses = jax.lax.scan(train_step, init_carry, jnp.arange(max_iter))
    params, opt_state = carry
    losses = list(losses)
    return params, losses

def RotateTarget(logp_fn, W):
    def logp_rotated(x):
        x_rotated = apply_householder(W, x)
        logp = logp_fn(x_rotated)
        return logp
    return logp_rotated

def PullbackTarget(logp_fn, flow, params):
    def logp_pullback(x):
        x, logdet = flow.apply(params, x)
        logp = logp_fn(x)
        return logp + logdet
    return logp_pullback

def ScorePCA(logp_fn, d, nsample, key, gamma=0.9):
    """
    Computes principal components of H = Cov(x, \nabla \log p(x) + x)
    Args:
        logp_fn (Callable): log density function of the target distribution.
        d (int): Dimensionality of the input space.
        nsample (int): Number of samples to use for estimating the score.
        key (jax.random.PRNGKey): JAX random key
        gamma (float, optional): Fraction of total variance to retain in PCA. If 0, returns identity matrix. 
            If < 1, selects enough components to explain gamma fraction of variance. Defaults to 0.9.
    Returns:
        jnp.ndarray: Matrix of principal component vectors (shape: [d, rank]), where 'rank' is the number of components retained.
    """
    if gamma == 0:
        return jnp.eye(d)
    base_samples = jax.random.normal(key, shape=(nsample, d))
    scores = jax.vmap(jax.grad(logp_fn))(base_samples) + base_samples
    H = scores.T @ base_samples / nsample
    H_2 = H @ H.T
    eigvals, eigvecs, = jnp.linalg.eigh(H_2)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    if gamma < 1:
        rank = jnp.where(jnp.cumsum(eigvals) >= jnp.sum(eigvals) * gamma)[0][0] + 1
    else:
        rank = d
    print("rank", rank)
    return eigvecs[:, :rank]

@jax.jit
def get_householder_matrix(U):
    U = jnp.array(U)
    d, r = U.shape
    W = jnp.zeros_like(U)

    def body(i, carry):
        U, W = carry
        e_i = jax.nn.one_hot(i, d, dtype=U.dtype) 
        w = U[:, i] - e_i
        w = w / (jnp.linalg.norm(w) + 1e-12)
        proj_full = w @ U                    
        mask = (jnp.arange(r) >= i).astype(U.dtype)  
        update = -2.0 * jnp.outer(w, proj_full * mask) 
        U = U + update

        W = W.at[:, i].set(w)
        return U, W

    _, W = jax.lax.fori_loop(0, r, body, (U, W))
    return W

@jax.jit
def apply_householder(W, x):
    W = jnp.array(W)
    x = jnp.array(x)
    d, r = W.shape

    def body(i, x):
        w = W[:, i]
        return x - 2.0 * (w @ x) * w

    x = jax.lax.fori_loop(0, r, lambda k, x: body(r - 1 - k, x), x)
    return x

@jax.jit
def apply_householder_transpose(W, x):
    W = jnp.array(W)
    x = jnp.array(x)
    d, r = W.shape

    def body(i, x):
        w = W[:, i]
        return x - 2.0 * (w @ x) * w

    x = jax.lax.fori_loop(0, r, lambda k, x: body(k, x), x)
    return x

def iterative_gaussianization(logp_fn, 
                              d, 
                              nsample, 
                              key, 
                              gamma, 
                              npca=None, 
                              random_rotate=False, 
                              niter=5, 
                              opt_params={'beta_0': 1., 'learning_rate': 1e-3, 'max_iter': 1000}, 
                              flow_params={'num_bins': 10, 'range_min': -5., 'range_max': 5., 'boundary_slopes': 'unconstrained'}):
    """
    Constructs a sequence of rotations and coordinatewise maps to Gaussianize a target distribution.
    Args:
        logp_fn (callable): Log-probability function of the target distribution.
        d (int): Dimensionality of the data.
        nsample (int): Number of samples to use in each iteration.
        key (jax.random.PRNGKey): JAX random key
        gamma (float): the top eigenvectors that explain at least `gamma` fraction of the total variance are used for the rotation
        npca (int, optional): Number of samples to use for PCA. Defaults to `nsample` if None.
        random_rotate (bool, optional): If True, uses random orthogonal rotations; otherwise uses ScorePCA. Defaults to False.
        niter (int, optional): Number of Gaussianization iterations. Defaults to 5.
        opt_params (dict, optional): Optimization parameters for MFVIStep. Defaults to {'beta_0': 1., 'learning_rate': 1e-3, 'max_iter': 1000}.
        flow_params (dict, optional): Parameters for the ComponentwiseFlow. Defaults to {'num_bins': 10, 'range_min': -5., 'range_max': 5., 'boundary_slopes': 'unconstrained'}.
    Returns:
        flow (ComponentwiseFlow): The trained componentwise flow model.
        transforms (list): List of tuples containing rotation matrices and flow parameters for each iteration.
    """
    flow = ComponentwiseFlow(d, **flow_params)
    logp_k = logp_fn
    if npca is None:
        npca = nsample
    transforms = []
    for i in range(niter):
        print(f"Iteration {i+1}/{niter}")
        key, subkey = jax.random.split(key)
        if random_rotate:
            V_r = sample_ortho(d, subkey)
        else:
            V_r = ScorePCA(logp_k, d, npca, subkey, gamma)
        
        W = get_householder_matrix(V_r)
        logp_k = RotateTarget(logp_k, W)

        key, subkey = jax.random.split(key)
        params, losses = MFVIStep(logp_k, d, flow, nsample, subkey, **opt_params)

        logp_k = PullbackTarget(logp_k, flow, params)

        transforms.append((W, params))
        print(f"Loss", losses[-1])
    return flow, transforms

def iterative_forward_map(flow, transforms, samples):
    """
    Applies a sequence of transformations to the input samples using the provided flow and Householder matrices.
    Args:
        flow (ComponentwiseFlow): ComponentwiseFlow object
        transforms (list): A list of tuples, where each tuple contains a Householder matrix `W_k` and parameters `param_k` for the flow. The transformations are applied in reverse order.
        samples (jax.numpy.ndarray): The input samples to be transformed.
    Returns:
        tuple:
            - samples (jax.numpy.ndarray): The transformed samples after applying all flows and Householder transformations.
            - logdet (float): The log-determinant of the Jacobian from all flow transformations.
    """

    logdet = 0.
    for W_k, param_k in transforms[::-1]:
        samples, _logdet = flow.apply(param_k, samples)
        logdet += _logdet
        samples = jax.vmap(apply_householder, in_axes=(None, 0))(W_k, samples)
    return samples, logdet

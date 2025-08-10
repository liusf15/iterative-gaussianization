import jax
import jax.numpy as jnp
import optax
import jax_tqdm
from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.special import logsumexp
from projection_vi.utils import sample_ortho, complete_orthonormal_basis

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

def iterative_AS_mfvi(model, logp_fn, niter, key, base_samples, val_samples, learning_rate=1e-3, max_iter=1000, rank0=0, rank=0, weighted=False):
    d = model.d
    
    @jax.jit
    def loss_fn(params, base_samples, rot):
        return model.apply(params, base_samples, logp_fn, rot=rot, method=model.reverse_kl)

    def _step(rot, base_samples):
        params = model.init(jax.random.key(0), jnp.zeros((1, d)))
        loss = lambda params: loss_fn(params, base_samples, rot)
        params, losses = train(loss, params, learning_rate=learning_rate, max_iter=max_iter)
        transformed_samples, logdet = model.apply(params, base_samples, rot=rot, method=model.forward)
        return transformed_samples, logdet, params, losses
    
    def logjac(x, params):
        return model.apply(params, x, inverse=False, return_jac=False)[1]

    def grad_logjac(x, params):
        return jax.grad(logjac)(x, params)

    def update_score_q(base_samples, scores_q, params, rot):
        def fn(base_sample, score_q):
            _logjac = model.apply(params, rot @ base_sample, inverse=False, return_jac=True)[1]
            Jac = rot.T @ jnp.diag(jnp.exp(-_logjac)) @ rot
            return Jac @ (score_q - rot.T @ grad_logjac(rot @ base_sample, params))
        return jax.vmap(fn, in_axes=(0, 0))(base_samples, scores_q)

    def score_active_subspace(base_samples, transformed_samples, scores_q, params, rot, log_weights=None):
        scores_p_ = jax.vmap(jax.grad(logp_fn))(transformed_samples)
        scores_q_ = update_score_q(base_samples, scores_q, params, rot)
        relative_scores = scores_p_ - scores_q_
        if log_weights is None:
            # expectation is taken over q
            H = relative_scores.T @ relative_scores / base_samples.shape[0]
        else:
            # expectation is taken over p, using importance weighting
            log_weights = log_weights - logsumexp(log_weights)
            weights = jnp.exp(log_weights)
            H = (weights[:, None] * relative_scores).T @ relative_scores
        eigvals, eigvecs = jnp.linalg.eigh(H)
        return eigvecs[:, ::-1], eigvals[::-1], scores_q_

    logq = mvn.logpdf(base_samples, mean=jnp.zeros(d), cov=jnp.eye(d))
    val_logq = mvn.logpdf(val_samples, mean=jnp.zeros(d), cov=jnp.eye(d))
    
    if rank0 < 0:
        rot = jnp.eye(d)
    elif rank0 == 0:
        key, subkey = jax.random.split(key)
        rot = sample_ortho(d, subkey)
    else:
        scores_p = jax.vmap(jax.grad(logp_fn))(val_samples)
        scores_q = -val_samples
        relative_scores = scores_p - scores_q
        if not weighted:
            # expectation is taken over q
            H = relative_scores.T @ relative_scores / val_samples.shape[0]
        else:
            # expectation is taken over p, using importance weighting
            log_weights = jax.vmap(logp_fn)(val_samples) - val_logq
            log_weights = log_weights - logsumexp(log_weights)
            weights = jnp.exp(log_weights)
            H = (weights[:, None] * relative_scores).T @ relative_scores
        eigvals, eigvecs = jnp.linalg.eigh(H)
        eigvecs = eigvecs[:, ::-1]
        rot = eigvecs.T 

    val_KL_hist = []
    val_ess_hist = []
    samples_hist = []
    val_samples_hist = []
    for k in range(niter):
        transformed_samples, ld, optim_params, losses = _step(rot, base_samples)
        logq = logq - ld
        log_weights = jax.vmap(logp_fn)(transformed_samples) - logq
        samples_hist.append(transformed_samples)
        ess = jnp.exp(2 * logsumexp(log_weights) - logsumexp(2 * log_weights))
        
        # update val samples and metrics
        transformed_val_samples, val_ld = model.apply(optim_params, val_samples, rot=rot, method=model.forward)
        val_logq = val_logq - val_ld
        val_log_weights = jax.vmap(logp_fn)(transformed_val_samples) - val_logq
        val_KL_hist.append(-jnp.mean(val_log_weights))
        val_ess_hist.append(jnp.exp(2 * logsumexp(val_log_weights) - logsumexp(2 * val_log_weights)))

        # find AS and update rotation matrix
        if rank == 0:
            key, subkey = jax.random.split(key)
            rot = sample_ortho(d, subkey)
        else:
            if weighted:
                eigvecs, eigvals, scores_q = score_active_subspace(val_samples, transformed_val_samples, scores_q, optim_params, rot, log_weights=val_log_weights)
            else:
                eigvecs, eigvals, scores_q = score_active_subspace(val_samples, transformed_val_samples, scores_q, optim_params, rot, log_weights=None)
            eigvals = jnp.clip(eigvals, a_min=0)
            var_explained = jnp.cumsum(eigvals) / jnp.sum(eigvals)
            idx = jnp.where(var_explained > 0.95)[0][0]
            print("First", idx+1, "components explain", round(var_explained[idx], 3), "of variance")
            U_r = eigvecs[:, :min(idx + 1, rank)]
            key, subkey = jax.random.split(key)
            U = complete_orthonormal_basis(U_r, subkey)
            rot = U.T

        # update base samples
        base_samples = transformed_samples
        val_samples = transformed_val_samples

        print("Iteration:", k, 'KL:', -log_weights.mean(), 'ESS:', ess)
    validation_metrics = {'KL': val_KL_hist, 'ESS': val_ess_hist}
    return jnp.stack(samples_hist), validation_metrics


def iterative_AS_mfvi_with_annealing(model, logp_fn, temperatures, key, base_samples, val_samples, learning_rate=1e-3, max_iter=1000, rank0=0, rank=0, weighted=False):
    d = model.d
    niter = len(temperatures)
    
    @jax.jit
    def loss_fn(params, base_samples, rot, temperature):
        return model.apply(params, base_samples, logp_fn, rot=rot, temperature=temperature, method=model.reverse_kl)

    def _step(rot, base_samples, temperature):
        params = model.init(jax.random.key(0), jnp.zeros((1, d)))
        loss = lambda params: loss_fn(params, base_samples, rot, temperature)
        params, losses = train(loss, params, learning_rate=learning_rate, max_iter=max_iter)
        transformed_samples, logdet = model.apply(params, base_samples, rot=rot, method=model.forward)
        return transformed_samples, logdet, params, losses
    
    def logjac(x, params):
        return model.apply(params, x, inverse=False, return_jac=False)[1]

    def grad_logjac(x, params):
        return jax.grad(logjac)(x, params)

    def update_score_q(base_samples, scores_q, params, rot):
        def fn(base_sample, score_q):
            _logjac = model.apply(params, rot @ base_sample, inverse=False, return_jac=True)[1]
            Jac = rot.T @ jnp.diag(jnp.exp(-_logjac)) @ rot
            return Jac @ (score_q - rot.T @ grad_logjac(rot @ base_sample, params))
        return jax.vmap(fn, in_axes=(0, 0))(base_samples, scores_q)

    def score_active_subspace(base_samples, transformed_samples, scores_q, params, rot, log_weights=None):
        scores_p_ = jax.vmap(jax.grad(logp_fn))(transformed_samples)
        scores_q_ = update_score_q(base_samples, scores_q, params, rot)
        relative_scores = scores_p_ - scores_q_
        if log_weights is None:
            # expectation is taken over q
            H = relative_scores.T @ relative_scores / base_samples.shape[0]
        else:
            # expectation is taken over p, using importance weighting
            log_weights = log_weights - logsumexp(log_weights)
            weights = jnp.exp(log_weights)
            H = (weights[:, None] * relative_scores).T @ relative_scores
        eigvals, eigvecs = jnp.linalg.eigh(H)
        return eigvecs[:, ::-1], eigvals[::-1], scores_q_

    logq = mvn.logpdf(base_samples, mean=jnp.zeros(d), cov=jnp.eye(d))
    val_logq = mvn.logpdf(val_samples, mean=jnp.zeros(d), cov=jnp.eye(d))
    
    if rank0 < 0:
        rot = jnp.eye(d)
    elif rank0 == 0:
        key, subkey = jax.random.split(key)
        rot = sample_ortho(d, subkey)
    else:
        scores_p = jax.vmap(jax.grad(logp_fn))(val_samples)
        scores_q = -val_samples
        relative_scores = scores_p - scores_q
        if not weighted:
            # expectation is taken over q
            H = relative_scores.T @ relative_scores / val_samples.shape[0]
        else:
            # expectation is taken over p, using importance weighting
            log_weights = jax.vmap(logp_fn)(val_samples) - val_logq
            log_weights = log_weights - logsumexp(log_weights)
            weights = jnp.exp(log_weights)
            H = (weights[:, None] * relative_scores).T @ relative_scores
        eigvals, eigvecs = jnp.linalg.eigh(H)
        eigvecs = eigvecs[:, ::-1]
        rot = eigvecs.T 

    val_KL_hist = []
    val_ess_hist = []
    samples_hist = []
    val_samples_hist = []
    for k, temperature in enumerate(temperatures):
        transformed_samples, ld, optim_params, losses = _step(rot, base_samples, temperature)
        logq = logq - ld
        log_weights = jax.vmap(logp_fn)(transformed_samples) - logq
        samples_hist.append(transformed_samples)
        ess = jnp.exp(2 * logsumexp(log_weights) - logsumexp(2 * log_weights))
        
        # update val samples and metrics
        transformed_val_samples, val_ld = model.apply(optim_params, val_samples, rot=rot, method=model.forward)
        val_logq = val_logq - val_ld
        val_log_weights = jax.vmap(logp_fn)(transformed_val_samples) - val_logq
        val_KL_hist.append(-jnp.mean(val_log_weights))
        val_ess_hist.append(jnp.exp(2 * logsumexp(val_log_weights) - logsumexp(2 * val_log_weights)))

        # find AS and update rotation matrix
        if rank == 0:
            key, subkey = jax.random.split(key)
            rot = sample_ortho(d, subkey)
        else:
            if weighted:
                eigvecs, eigvals, scores_q = score_active_subspace(val_samples, transformed_val_samples, scores_q, optim_params, rot, log_weights=val_log_weights)
            else:
                eigvecs, eigvals, scores_q = score_active_subspace(val_samples, transformed_val_samples, scores_q, optim_params, rot, log_weights=None)
            eigvals = jnp.clip(eigvals, a_min=0)
            var_explained = jnp.cumsum(eigvals) / jnp.sum(eigvals)
            idx = jnp.where(var_explained > 0.95)[0][0]
            print("First", idx+1, "components explain", round(var_explained[idx], 3), "of variance")
            U_r = eigvecs[:, :min(idx + 1, rank)]
            key, subkey = jax.random.split(key)
            U = complete_orthonormal_basis(U_r, subkey)
            rot = U.T

        # update base samples
        base_samples = transformed_samples
        val_samples = transformed_val_samples

        print("Iteration:", k, 'KL:', -log_weights.mean(), 'ESS:', ess)
    validation_metrics = {'KL': val_KL_hist, 'ESS': val_ess_hist}
    return jnp.stack(samples_hist), validation_metrics
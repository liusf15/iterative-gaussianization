import jax
import jax.numpy as jnp
import optax

def train(loss_fn, params, learning_rate=0.01, max_iter=500):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    init_carry = (params, opt_state)
    carry, losses = jax.lax.scan(train_step, init_carry, None, length=max_iter)
    params, opt_state = carry
    losses = list(losses)
    return params, losses


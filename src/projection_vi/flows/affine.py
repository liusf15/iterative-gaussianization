import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn import softplus
import distrax

inverse_softplus = lambda x: jnp.log(jnp.exp(x) - 1.)

class AffineFlow(nn.Module):
    d: int  

    def setup(self):
        self.shift = self.param(
            "shift",
            nn.initializers.zeros_init(),
            (self.d,)
        )
        
        self.scale_logit = self.param(
            "scale_logit",
            nn.initializers.zeros_init(),
            (self.d,)
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray, inverse: bool = False):

        scale = softplus(self.scale_logit + inverse_softplus(1.))  

        affine_bij = distrax.ScalarAffine(
            shift=self.shift,  # shape (d,)
            scale=scale       # shape (d,)
        )

        if not inverse:
            y, logdet = affine_bij.forward_and_log_det(x)
        else:
            y, logdet = affine_bij.inverse_and_log_det(x)
        logdet = jnp.sum(logdet, axis=1)
        return y, logdet

    def forward(self, x: jnp.ndarray):
        return self(x, inverse=False)

    def inverse(self, y: jnp.ndarray):
        return self(y, inverse=True)

    def reverse_kl(self, base_samples, logp_fn):
        X, log_det = self.forward(base_samples)
        logp = jax.vmap(logp_fn)(X)
        return -jnp.nanmean(log_det + logp)
    
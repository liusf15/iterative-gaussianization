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
    

class ComponentwiseFlow(nn.Module):
    d: int
    num_bins: int = 10
    range_min: float = -5.0
    range_max: float = 5.0

    def setup(self):
        param_shape = (self.d, 3 * self.num_bins + 1)
        self.spline_params = self.param(
            'spline',
            nn.initializers.zeros_init(),
            param_shape
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray, inverse: bool = False):
        
        def spline_1d(params_i, x_i):
            spline = distrax.RationalQuadraticSpline(
                params=params_i,
                range_min=self.range_min,
                range_max=self.range_max,
                boundary_slopes='unconstrained'
            )

            if not inverse:
                y_i, logdet_i = spline.forward_and_log_det(x_i)
            else:
                y_i, logdet_i = spline.inverse_and_log_det(x_i)
            return y_i, logdet_i

        y_t, logdet_t = jax.vmap(spline_1d, in_axes=(0, 1))(self.spline_params, x)
        logdet = jnp.sum(logdet_t, axis=0)  
        y = y_t.T
        return y, logdet

    def forward(self, x, rot=None):
        if rot is not None:
            x = x @ rot.T
        x, logdet = self(x, inverse=False)
        if rot is not None:
            x = x @ rot
        return x, logdet

    def inverse(self, z):
        return self(z, inverse=True)

    def reverse_kl(self, base_samples, logp_fn, rot=None):
        X, log_det = self.forward(base_samples, rot=rot)
        logp = jax.vmap(logp_fn)(X)
        return -jnp.nanmean(log_det + logp)
        
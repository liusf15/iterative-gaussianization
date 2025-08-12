import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn import softplus
from jax.scipy.stats import norm
import distrax
from typing import Sequence, Callable

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
        logp = jnp.where(jnp.abs(logp) < 1e10, logp, jnp.nan)
        return -jnp.nanmean(log_det + logp)
    

class ComponentwiseFlow(nn.Module):
    d: int
    num_bins: int = 10
    range_min: float = -5.0
    range_max: float = 5.0
    boundary_slopes: str = 'identity'

    def setup(self):
        param_shape = (self.d, 3 * self.num_bins + 1)
        self.spline_params = self.param(
            'spline',
            nn.initializers.zeros_init(),
            param_shape
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray, inverse: bool = False, return_jac=False):
        
        if x.ndim == 1:
            x = x[None, :]  # Add batch dimension
            single_input = True
        else:
            single_input = False
            
        def spline_1d(params_i, x_i):
            spline = distrax.RationalQuadraticSpline(
                params=params_i,
                range_min=self.range_min,
                range_max=self.range_max,
                boundary_slopes=self.boundary_slopes
            )

            if not inverse:
                y_i, logdet_i = spline.forward_and_log_det(x_i)
            else:
                y_i, logdet_i = spline.inverse_and_log_det(x_i)
            return y_i, logdet_i

        y_t, logdet_t = jax.vmap(spline_1d, in_axes=(0, 1))(self.spline_params, x)
        y = y_t.T
        logdet = logdet_t.T

        if single_input:
            y = y[0]
            logdet = logdet[0]

            if not return_jac:
                logdet = jnp.sum(logdet)
        else:
            if not return_jac:
                logdet = jnp.sum(logdet, axis=1)  

        return y, logdet

    def forward(self, x, rot=None):
        if rot is not None:
            x = x @ rot.T
        x, logdet = self(x, inverse=False)
        if rot is not None:
            x = x @ rot
        return x, logdet

    def inverse(self, z, rot=None):
        if rot is not None:
            z = z @ rot.T
        z, logdet = self(z, inverse=True)
        if rot is not None:
            z = z @ rot
        return z, logdet

    def reverse_kl(self, base_samples, logp_fn, rot=None, temperature=1.):
        X, log_det = self.forward(base_samples, rot=rot)
        logp = jax.vmap(logp_fn)(X) * temperature + (1 - temperature) * (-.5 * jnp.sum(X**2, axis=-1))
        logp = jnp.where(jnp.abs(logp) < 1e10, logp, jnp.nan)
        return -jnp.nanmean(log_det + logp)
    
class ComponentwiseCDF(nn.Module):
    d: int
    num_bins: int = 10

    def setup(self):
        param_shape = (self.d, 2 * self.num_bins)
        self.params = self.param(
            'shift_scale',
            nn.initializers.zeros_init(),
            param_shape
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray, inverse: bool = False):
        
        def bij(params_i, x_i):
            shift, scale_logit = jnp.split(params_i, 2, axis=-1)
            scale = jax.nn.softplus(scale_logit + inverse_softplus(1.))
            if not inverse:
                u_i = norm.cdf(x_i, loc=shift[:, None], scale=scale[:, None]).mean(0)
                u_i = jnp.clip(u_i, -1e10, 1e10)
                y_i = norm.ppf(u_i)
                logdet_i = norm.logpdf(x_i, loc=shift[:, None], scale=scale[:, None]).mean(0) - norm.logpdf(y_i)
            else:
                raise NotImplementedError("Inverse is not implemented.")    
            return y_i, logdet_i

        y_t, logdet_t = jax.vmap(bij, in_axes=(0, 1))(self.params, x)
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
        logp = jnp.where(jnp.abs(logp) < 1e10, logp, jnp.nan)
        return -jnp.nanmean(log_det + logp)


class ConditionerMLP(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int
    activation: Callable = nn.relu
    zero_init: bool = False  # Option to use zero initialization

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_dims:
            x = self.activation(
                nn.Dense(
                    h,
                    kernel_init=nn.initializers.zeros_init() if self.zero_init else nn.initializers.variance_scaling(scale=0.1, mode="fan_in", distribution="normal")
                )(x)
            )
        x = nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.zeros_init() if self.zero_init else nn.initializers.variance_scaling(scale=0.01, mode="fan_in", distribution="truncated_normal"),
            bias_init=nn.initializers.zeros_init()
        )(x)
        return x

class RealNVP(nn.Module):
    dim: int
    n_layers: int
    hidden_dims: Sequence[int]

    def setup(self):
        self.masks = [
            jnp.array([((i + j) % 2) == 0 for j in range(self.dim)], dtype=bool)
            for i in range(self.n_layers)
        ]

        self.conditioners = [
            ConditionerMLP(
                hidden_dims=self.hidden_dims, 
                output_dim=self.dim * 2,
                name=f"conditioner_mlp_{i}"
            )
            for i in range(self.n_layers)
        ]

    @nn.compact
    def __call__(self, x, inverse=False):
        logdet = 0.
        for i in range(self.n_layers):
            mask = self.masks[i]
            conditioner_mlp = self.conditioners[i]
            
            def bijector_fn(params) -> distrax.Bijector:
                scale_logit, shift = jnp.split(params, 2, axis=-1)
                scale = jax.nn.softplus(scale_logit + inverse_softplus(1.))
                return distrax.ScalarAffine(shift=shift, scale=scale)
            
            def conditioner_fn(x_masked):
                return conditioner_mlp(x_masked)  
            
            bij = distrax.MaskedCoupling(
                mask=mask,
                conditioner=conditioner_fn,
                bijector=bijector_fn,
            )
            if not inverse:
                x, ld = bij.forward_and_log_det(x)
            else:
                x, ld = bij.inverse_and_log_det(x)

            logdet += ld

        return x, logdet

    def forward(self, x):
        return self(x, inverse=False)
    
    def inverse(self, y):
        return self(y, inverse=True)

    def reverse_kl(self, base_samples, logp_fn):
        X, log_det = self.forward(base_samples)
        logp = jax.vmap(logp_fn)(X)
        logp = jnp.where(jnp.abs(logp) < 1e10, logp, jnp.nan)
        return -jnp.nanmean(log_det + logp)

class NeuralSplineFlow(nn.Module):
    dim: int
    n_layers: int
    hidden_dims: Sequence[int]
    num_bins: int = 10
    range_min: float = -5.0
    range_max: float = 5.0
    boundary_slopes: str = 'identity'

    def setup(self):
        self.masks = [
            jnp.array([((i + j) % 2) == 0 for j in range(self.dim)], dtype=bool)
            for i in range(self.n_layers)
        ]

        self.conditioners = [
            ConditionerMLP(
                hidden_dims=self.hidden_dims, 
                output_dim=self.dim * (3 * self.num_bins + 1),
                name=f"conditioner_mlp_{i}"
            )
            for i in range(self.n_layers)
        ]

    @nn.compact
    def __call__(self, x, inverse=False):
        logdet = 0.
        for i in range(self.n_layers):
            mask = self.masks[i]
            conditioner_mlp = self.conditioners[i]
            
            def bijector_fn(raw_params):
                params = raw_params.reshape(raw_params.shape[:-1] + (self.dim, 3 * self.num_bins + 1))
                spline = distrax.RationalQuadraticSpline(
                    params=params,
                    range_min=self.range_min,
                    range_max=self.range_max,
                    boundary_slopes=self.boundary_slopes
                )
                return spline

            def conditioner_fn(x_masked):
                return conditioner_mlp(x_masked)  
            
            bij = distrax.MaskedCoupling(
                mask=mask,
                conditioner=conditioner_fn,
                bijector=bijector_fn,
            )
            if not inverse:
                x, ld = bij.forward_and_log_det(x)
            else:
                x, ld = bij.inverse_and_log_det(x)

            logdet += ld

        return x, logdet

    def forward(self, x):
        return self(x, inverse=False)
    
    def inverse(self, y):
        return self(y, inverse=True)

    def reverse_kl(self, base_samples, logp_fn):
        X, log_det = self.forward(base_samples)
        logp = jax.vmap(logp_fn)(X)
        logp = jnp.where(jnp.abs(logp) < 1e10, logp, jnp.nan)
        return -jnp.nanmean(log_det + logp)
    
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn import softplus
from jax.scipy.stats import norm
import jax.scipy as jsp
import distrax
from typing import Sequence, Callable, Tuple

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


class BlockAffineFlow(nn.Module):
    """
    Block-Gaussian VI flow:
      - Head (first r dims): full-covariance Gaussian via Cholesky L (lower-triangular)
      - Tail (last d-r dims): product-Gaussian (diagonal scales)
    Acts as an affine bijection per block.
    """
    d: int
    r: int
    min_scale: float = 1e-5  # positivity margin for Cholesky diag and tail scales

    def setup(self):
        assert 0 <= self.r <= self.d, "r must be in [0, d]"

        # --- Head (full-covariance over first r dims) ---
        if self.r > 0:
            self.shift_head = self.param(
                "shift_head",
                nn.initializers.zeros_init(),
                (self.r,),
            )
            # Raw lower-triangular params: we'll softplus the diagonal for positivity.
            self.tril_raw = self.param(
                "tril_raw",
                nn.initializers.zeros_init(),
                (self.r, self.r),
            )

        # --- Tail (diagonal/product over last d-r dims) ---
        tail = self.d - self.r
        if tail > 0:
            self.shift_tail = self.param(
                "shift_tail",
                nn.initializers.zeros_init(),
                (tail,),
            )
            self.scale_logit_tail = self.param(
                "scale_logit_tail",
                nn.initializers.zeros_init(),
                (tail,),
            )
            # Initialize scale to 1.0 like in your AffineFlow
            self._inv_sp_1 = inverse_softplus(1.0)

    # --- helpers to build the affine pieces ---
    def _build_L(self) -> jnp.ndarray:
        """
        Build a valid Cholesky factor L (lower triangular with positive diagonal)
        from unconstrained 'tril_raw'.
        """
        # strictly lower part (free)
        L_lower = jnp.tril(self.tril_raw, k=-1)
        # diagonal via softplus for positivity + margin
        diag = softplus(jnp.diag(self.tril_raw) + self._inv_sp_1) + self.min_scale
        L = L_lower + jnp.diag(diag)
        return L  # shape (r, r)

    def _tail_affine(self) -> distrax.ScalarAffine:
        scale_tail = softplus(self.scale_logit_tail + self._inv_sp_1) + self.min_scale
        return distrax.ScalarAffine(shift=self.shift_tail, scale=scale_tail)

    # --- core call ---
    @nn.compact
    def __call__(self, x: jnp.ndarray, inverse: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        x: (batch, d)
        Returns (y, logdet_per_sample), where logdet is a scalar per sample.
        """
        # Split head/tail
        x_head = x[:, :self.r] if self.r > 0 else None
        x_tail = x[:, self.r:] if (self.d - self.r) > 0 else None

        logdet_head = 0.0
        ys = []

        # --- Head block (full-covariance) ---
        if self.r > 0:
            L = self._build_L()  # (r, r), lower-triangular
            shift_h = self.shift_head  # (r,)
            # log|det L| (same for all samples)
            logabsdetL = jnp.sum(jnp.log(jnp.diag(L)))  # scalar

            if not inverse:
                # y_h = shift + x_h @ L^T
                y_head = x_head @ L.T + shift_h  # (batch, r)
                logdet_head = logabsdetL
            else:
                # x_h = (y_h - shift) @ L^{-T}
                b = x_head - shift_h  # we interpret input as 'y' in inverse branch
                # solve (L^T) x^T = b^T  => x = b @ L^{-T}
                x_head_inv = jsp.linalg.solve_triangular(L.T, b.T, lower=False).T
                y_head = x_head_inv
                logdet_head = -logabsdetL

            ys.append(y_head)

        # --- Tail block (diagonal/product) ---
        logdet_tail = 0.0
        if (self.d - self.r) > 0:
            aff_tail = self._tail_affine()
            if not inverse:
                y_tail, ld_tail = aff_tail.forward_and_log_det(x_tail)       # (batch, tail), (batch, tail)
            else:
                y_tail, ld_tail = aff_tail.inverse_and_log_det(x_tail)       # (batch, tail), (batch, tail)
            ys.append(y_tail)
            # sum per-sample contributions across tail dims
            logdet_tail = jnp.sum(ld_tail, axis=1)  # (batch,)

        # --- Concatenate outputs ---
        if len(ys) == 2:
            y = jnp.concatenate(ys, axis=1)
        elif len(ys) == 1:
            y = ys[0]
        else:
            # d == 0 edge case (not typical)
            y = x

        # Combine head scalar logdet (same for all samples) with tail per-sample logdet
        if isinstance(logdet_head, float) or jnp.ndim(logdet_head) == 0:
            batch = x.shape[0]
            logdet_head_vec = jnp.full((batch,), logdet_head, dtype=x.dtype)
        else:
            logdet_head_vec = logdet_head  # already per-sample

        logdet_total = logdet_head_vec + (logdet_tail if jnp.ndim(logdet_tail) > 0 else 0.0)  # (batch,)

        return y, logdet_total

    # --- convenience wrappers ---
    def forward(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self(x, inverse=False)

    def inverse(self, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self(y, inverse=True)

    # --- reverse KL (same style as yours) ---
    def reverse_kl(self, base_samples: jnp.ndarray, logp_fn) -> jnp.ndarray:
        """
        Reverse KL objective:  E_q[ -log p(f(x)) - log|det J_f(x)| ],
        where x ~ base (e.g., N(0,I)), y = f(x).
        """
        X, log_det = self.forward(base_samples)          # log_det: (batch,)
        logp = jax.vmap(logp_fn)(X)                      # (batch,)
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

class HybridFlow(nn.Module):
    d: int
    r: int
    # pass-through params for ComponentwiseFlow (RQS head)
    num_bins: int = 10
    range_min: float = -5.0
    range_max: float = 5.0
    boundary_slopes: str = 'identity'

    def setup(self):
        assert 0 <= self.r <= self.d, "`r` must be in [0, d]."
        head_dim = self.r
        tail_dim = self.d - self.r

        if head_dim > 0:
            self.head = ComponentwiseFlow(
                d=head_dim,
                num_bins=self.num_bins,
                range_min=self.range_min,
                range_max=self.range_max,
                boundary_slopes=self.boundary_slopes,
            )
        else:
            self.head = None

        if tail_dim > 0:
            self.tail = AffineFlow(d=tail_dim)
        else:
            self.tail = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, inverse: bool = False):
        # normalize to (batch, d) because AffineFlow expects batching
        if x.ndim == 1:
            x = x[None, :]
            single = True
        else:
            single = False

        head_dim = self.r
        tail_dim = self.d - self.r

        x_head = x[:, :head_dim] if head_dim > 0 else None
        x_tail = x[:, head_dim:] if tail_dim > 0 else None

        # --- apply RQS to head ---
        if self.head is not None:
            # ComponentwiseFlow supports inverse and returns (y, summed_logdet)
            y_head, ld_head = self.head(x_head, inverse=inverse)
        else:
            y_head = jnp.zeros((x.shape[0], 0), dtype=x.dtype)
            ld_head = jnp.zeros((x.shape[0],), dtype=x.dtype)

        # --- apply affine to tail ---
        if self.tail is not None:
            y_tail, ld_tail = self.tail(x_tail, inverse=inverse)
        else:
            y_tail = jnp.zeros((x.shape[0], 0), dtype=x.dtype)
            ld_tail = jnp.zeros((x.shape[0],), dtype=x.dtype)

        y = jnp.concatenate([y_head, y_tail], axis=1)
        logdet = ld_head + ld_tail  # both already summed over dims

        if single:
            y = y[0]
            logdet = logdet[0]

        return y, logdet

    def forward(self, x, rot=None):
        if rot is not None:
            x = x @ rot.T
        y, logdet = self(x, inverse=False)
        if rot is not None:
            y = y @ rot
        return y, logdet

    def inverse(self, z, rot=None):
        if rot is not None:
            z = z @ rot.T
        x, logdet = self(z, inverse=True)
        if rot is not None:
            x = x @ rot
        return x, logdet

    def reverse_kl(self, base_samples, logp_fn, rot=None, temperature: float = 1.0):
        X, log_det = self.forward(base_samples, rot=rot)
        logp = jax.vmap(logp_fn)(X) * temperature + (1 - temperature) * (-0.5 * jnp.sum(X**2, axis=-1))
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
    
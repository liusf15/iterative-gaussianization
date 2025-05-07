import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import BernoulliLogits
from numpyro.infer.util import log_density
import json

class ImproperUniform(dist.Distribution):
    """
    A dummy 'improper uniform' on the real line.
    It always returns 0 for the log probability.
    """
    support = numpyro.distributions.constraints.real
    has_rsample = True

    def __init__(self):
        super().__init__(batch_shape=(), event_shape=())

    def sample(self, key, sample_shape=()):
        return dist.Normal(0, 1).sample(key, sample_shape)

    def log_prob(self, value):
        return jnp.zeros_like(value)
    
class arK:
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        self.K = self.data['K']
        self.T = self.data['T']
        self.y = jnp.array(self.data['y'])
        self.d = 1 + self.K + 1

        def _numpyro_model():
            alpha = numpyro.sample("alpha", dist.Normal(0, 1))
            beta = numpyro.sample("beta", dist.Normal(0, 1).expand([self.K]))
            sigma_unc = numpyro.sample("sigma_unc", ImproperUniform())
            sigma = numpyro.deterministic("sigma", jnp.exp(sigma_unc))
            numpyro.factor("sigma_prior", dist.HalfNormal(1).log_prob(sigma) + sigma_unc)
            
            for t in range(self.K, self.T):
                mean_t = alpha + jnp.dot(beta, self.y[t-self.K:t])
                numpyro.sample(f"y_{t}", dist.Normal(mean_t, sigma), obs=self.y[t])

        self.numpyro_model = _numpyro_model
        self._seeded_model = numpyro.handlers.seed(_numpyro_model, jax.random.key(0))

    def _log_prob(self, x):
        params = {
            "alpha": x[0],
            "beta": x[1:self.K+1],
            "sigma_unc": x[self.K+1]
        }
        logp = log_density(self._seeded_model, (), {}, params)[0]
        return logp

    log_prob = jax.jit(_log_prob, static_argnums=(0,))

    def param_constrain(self, X):
        if X.ndim == 1:
            X = X.at[-1].set(jnp.exp(X[-1]))
            return X
        X = X.at[:, -1].set(jnp.exp(X[:, -1]))
        return X
    
    def param_unc_names(self):
        return ['alpha', 'beta', 'sigma_unc']
 
def gp_exp_quad_cov(x, alpha, rho, sigma):
    N = x.shape[0]
    x = x[:, None]  
    dists = jnp.square(x - x.T)  
    cov = alpha**2 * jnp.exp(-0.5 * dists / rho**2)
    cov += jnp.eye(N) * sigma
    return cov

class gp_regr:
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        self.N = self.data['N']
        self.x = jnp.array(self.data['x'])
        self.y = jnp.array(self.data['y'])
        self.d = 3

        def _numpyro_model():
            rho_unc = numpyro.sample("rho_unc", ImproperUniform())
            rho = numpyro.deterministic("rho", jnp.exp(rho_unc))
            numpyro.factor("rho_prior", dist.Gamma(25, 4).log_prob(rho) + rho_unc)

            alpha_unc = numpyro.sample("alpha_unc", ImproperUniform())
            alpha = numpyro.deterministic("alpha", jnp.exp(alpha_unc))
            numpyro.factor("alpha_prior", dist.HalfNormal(2).log_prob(alpha) + alpha_unc)

            sigma_unc = numpyro.sample("sigma_unc", ImproperUniform()) 
            sigma = numpyro.deterministic("sigma", jnp.exp(sigma_unc))
            numpyro.factor("sigma_prior", dist.HalfNormal(1).log_prob(sigma) + sigma_unc)

            cov = gp_exp_quad_cov(self.x, alpha, rho, sigma)
            L_cov = jax.scipy.linalg.cholesky(cov, lower=True)
            numpyro.sample("y", dist.MultivariateNormal(jnp.zeros(self.N), scale_tril=L_cov), obs=self.y)

        self.numpyro_model = _numpyro_model
        self._seeded_model = numpyro.handlers.seed(_numpyro_model, jax.random.key(0))

    def _log_prob(self, x):
        params = {
            "rho_unc": x[0],
            "alpha_unc": x[1],
            "sigma_unc": x[2]
        }
        logp = log_density(self._seeded_model, (), {}, params)[0]
        return logp

    log_prob = jax.jit(_log_prob, static_argnums=(0,))

    def param_constrain(self, X):
        return jnp.exp(X)
    
    def param_unc_names(self):
        return ['rho_unc', 'alpha_unc', 'sigma_unc']

class garch:
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            data = json.load(f)
        self.T = data["T"]
        self.y = jnp.array(data["y"])
        self.sigma1 = data["sigma1"]
        self.d = 4

        def _numpyro_model():
            mu = numpyro.sample("mu", ImproperUniform())

            alpha0_unc = numpyro.sample("alpha0_unc", ImproperUniform())
            alpha0 = jnp.exp(alpha0_unc)                               # >0
            numpyro.factor("alpha0_jac", alpha0_unc)  

            alpha1_unc = numpyro.sample("alpha1_unc", ImproperUniform())
            alpha1 = 1 / (1 + jnp.exp(-alpha1_unc))  # (0,1)
            numpyro.factor("alpha1_jac", jnp.log(alpha1) + jnp.log(1 - alpha1))  # Jacobian −log(scale)

            beta1_unc = numpyro.sample("beta1_unc", ImproperUniform())    # (0,1)
            beta1  = (1. - alpha1) / (1 + jnp.exp(-beta1_unc))                         # (0,1−α1)
            numpyro.factor("beta1_jac", jnp.log(beta1) + jnp.log(1 - beta1 / (1 - alpha1)))         # Jacobian −log(scale)

            # ---- GARCH recursion --------------------------------
            sigma = jnp.zeros(self.T)
            sigma = sigma.at[0].set(self.sigma1)
            for t in range(1, self.T):
                sigma_t = jnp.sqrt(alpha0
                                   + alpha1 * (self.y[t-1] - mu)**2
                                   + beta1  * sigma[t-1]**2)
                sigma   = sigma.at[t].set(sigma_t)

            # ---- likelihood -------------------------------------
            ll = dist.Normal(mu, sigma).log_prob(self.y).sum()
            numpyro.factor("obs_ll", ll)

        self.numpyro_model = _numpyro_model
        # Seed the model for log_prob calculations
        self._seeded_model = numpyro.handlers.seed(_numpyro_model, jax.random.PRNGKey(0))

    def _log_prob(self, x):
        params = {
            "mu": x[0],
            "alpha0_unc": x[1],
            "alpha1_unc": x[2],
            "beta1_unc": x[3]
        }
        logp = log_density(self._seeded_model, (), {}, params)[0]
        return logp
    
    log_prob = jax.jit(_log_prob, static_argnums=(0,))

    def param_constrain(self, X):
        if X.ndim == 1:
            return jnp.array([
                X[0],
                jnp.exp(X[1]),
                1 / (1 + jnp.exp(-X[2])),
                (1. - 1 / (1 + jnp.exp(-X[2]))) / (1 + jnp.exp(-X[3]))
            ])

        return jnp.hstack([X[:, 0:1], 
                           jnp.exp(X[:, 1:2]), 
                           1 / (1 + jnp.exp(-X[:, 2:3])), 
                           (1 - 1 / (1 + jnp.exp(-X[:, 2:3]))) * (1 + jnp.exp(-X[:, 3:4]))])
    
    def param_unc_names(self):
        return ['mu', 'alpha0_unc', 'alpha1_unc', 'beta1_unc']

class hmm:
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        self.N = self.data['N']
        self.y = jnp.array(self.data['y'])
        self.d = 4

        def _numpyro_model():
            theta_unc = numpyro.sample("theta_unc", ImproperUniform().expand([2]))
            theta = numpyro.deterministic("theta", jax.nn.sigmoid(theta_unc))  # Ensures 0 < theta < 1
            numpyro.factor("theta_prior", jnp.sum(jnp.log(theta) + jnp.log(1 - theta)))

            mu_unc = numpyro.sample("mu_unc", ImproperUniform().expand([2]))  # Unconstrained
            mu = jnp.array([jnp.exp(mu_unc[0]), jnp.exp(mu_unc[0]) + jnp.exp(mu_unc[1])]) 

            numpyro.factor("mu_prior", dist.Normal(jnp.array([3., 10.0]), 1.0).log_prob(mu).sum() + mu_unc.sum())

            log_theta = jnp.log(jnp.array([
                [theta[0], theta[1]],
                [1 - theta[0], 1 - theta[1]]
            ]))
            
            gamma = jnp.zeros((self.N, 2))
            
            for k in range(2):
                gamma = gamma.at[0, k].set(dist.Normal(mu[k], 1.0).log_prob(self.y[0]))
            
            for t in range(1, self.N):
                for k in range(2):
                    norm_lpdf = dist.Normal(mu[k], 1.0).log_prob(self.y[t])
                    acc = jnp.array([gamma[t - 1, j] + log_theta[j, k] + norm_lpdf for j in range(2)])
                    gamma = gamma.at[t, k].set(logsumexp(acc))
            
            numpyro.factor("log_prob", logsumexp(gamma[self.N - 1]))

        self.numpyro_model = _numpyro_model
        self._seeded_model = numpyro.handlers.seed(_numpyro_model, jax.random.key(0))

    def _log_prob(self, x):
        params = {
            "theta_unc": x[:2],
            "mu_unc": x[2:]
        }
        logp = log_density(self._seeded_model, (), {}, params)[0]
        return logp

    log_prob = jax.jit(_log_prob, static_argnums=(0,))

    def param_constrain(self, X):
        if X.ndim == 1:
            return jnp.array([jax.nn.sigmoid(X[0]), jax.nn.sigmoid(X[1]), jnp.exp(X[2]), jnp.exp(X[2]) + jnp.exp(X[3])])
        if X.ndim == 2:
            return jnp.hstack([jax.nn.sigmoid(X[:, 0:1]), jax.nn.sigmoid(X[:, 1:2]), jnp.exp(X[:, 2:3]), jnp.exp(X[:, 2:3]) + jnp.exp(X[:, 3:4])])
        
    def param_unc_names(self):
        return ['theta_unc', 'mu_unc']
    
class nes_logit:
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        self.N = self.data['N']
        self.income = jnp.array(self.data['income'])
        self.vote = jnp.array(self.data['vote'])
        self.d = 2

        def _numpyro_model():
            alpha = numpyro.sample("alpha", dist.Normal(0, 1))
            beta = numpyro.sample("beta", dist.Normal(jnp.zeros(1), jnp.ones(1)))
            x = self.income[:, None]  
            numpyro.sample("vote", BernoulliLogits(x @ beta + alpha), obs=self.vote)

        self.numpyro_model = _numpyro_model
        self._seeded_model = numpyro.handlers.seed(_numpyro_model, jax.random.key(0))
    
    def _log_prob(self, x):
        params = {
            "alpha": x[0],
            "beta": x[1:]
        }
        logp = log_density(self._seeded_model, (), {}, params)[0]
        return logp
    
    log_prob = jax.jit(_log_prob, static_argnums=(0,))

    def param_constrain(self, X):
        return X
    
    def param_unc_names(self):
        return ['alpha', 'beta']
    
class normal_mixture:
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        self.N = self.data['N']
        self.y = jnp.array(self.data['y'])
        self.d = 5

        def _numpyro_model():
            mu_unc = numpyro.sample("mu_unc", ImproperUniform().expand([2]))  
            mu = jnp.array([mu_unc[0], mu_unc[0] + jnp.exp(mu_unc[1])]) 
            numpyro.factor("mu_prior", dist.Normal(0., 2.0).log_prob(mu).sum() + mu_unc[1])

            sigma_unc = numpyro.sample("sigma_unc", ImproperUniform().expand([2]))
            sigma = numpyro.deterministic("sigma", jnp.exp(sigma_unc))
            numpyro.factor("sigma_prior", dist.HalfNormal(2).log_prob(sigma).sum() + sigma_unc.sum())

            theta_unc = numpyro.sample("theta_unc", ImproperUniform())
            theta = numpyro.deterministic("theta", jax.nn.sigmoid(theta_unc))  
            numpyro.factor("theta_prior", dist.Beta(5, 5).log_prob(theta) + jnp.sum(jnp.log(theta) + jnp.log(1 - theta)))

            log_prob_1 = dist.Normal(mu[0], sigma[0]).log_prob(self.y)  
            log_prob_2 = dist.Normal(mu[1], sigma[1]).log_prob(self.y) 
            
            log_mix = logsumexp(
                jnp.stack([
                    jnp.log(theta) + log_prob_1, 
                    jnp.log(1 - theta) + log_prob_2  
                ]),
                axis=0
            )  
        
            numpyro.factor("obs", log_mix.sum())

        self.numpyro_model = _numpyro_model
        self._seeded_model = numpyro.handlers.seed(_numpyro_model, jax.random.key(0))
    
    def _log_prob(self, x):
        params = {
            "mu_unc": x[:2],
            "sigma_unc": x[2:4],
            "theta_unc": x[4]
        }
        logp = log_density(self._seeded_model, (), {}, params)[0]
        return logp
    
    log_prob = jax.jit(_log_prob, static_argnums=(0,))

    def param_constrain(self, X):
        if X.ndim == 1:
            return jnp.array([X[0], X[0] + jnp.exp(X[1]), jnp.exp(X[2]), jnp.exp(X[3]), jax.nn.sigmoid(X[4])])
        if X.ndim == 2:
            return jnp.hstack([X[:, 0:1], X[:, 0:1] + jnp.exp(X[:, 1:2]), jnp.exp(X[:, 2:3]), jnp.exp(X[:, 3:4]), jax.nn.sigmoid(X[:, 4:5])])
    
    def param_unc_names(self):
        return ['mu_unc', 'sigma_unc', 'theta_unc']

class eight_schools:
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        # Stan data: J schools, y, sigma
        self.J = self.data['J']                         # number of schools
        self.y = jnp.array(self.data['y'])              # treatment effects
        self.sigma = jnp.array(self.data['sigma'])      # std errors
        self.d = self.J + 2  # dimension of unconstrained parameters = (mu_unc, tau_unc, theta_unc of length J)

        def _numpyro_model():
            
            mu = numpyro.sample("mu", dist.Normal(0, 5))

            tau_unc = numpyro.sample("tau_unc", ImproperUniform())
            tau = numpyro.deterministic("tau", jnp.exp(tau_unc))
            numpyro.factor("tau_prior", dist.Normal(0, 10).log_prob(tau) + tau_unc)

            # theta_unc = numpyro.sample("theta_unc", ImproperUniform().expand([self.J]))
            # theta = numpyro.deterministic("theta", mu + tau * theta_unc)
            # numpyro.factor("theta_prior", dist.Normal(mu, tau).log_prob(theta).sum() + tau_unc * self.J)
            theta_unc = numpyro.sample("theta_unc", dist.Normal(0, 1).expand([self.J]))
            theta = mu + tau * theta_unc
            
            ll = dist.Normal(theta, self.sigma).log_prob(self.y).sum()
            numpyro.factor("obs_ll", ll)

        self.numpyro_model = _numpyro_model
        # Seed the model for log_prob calculations
        self._seeded_model = numpyro.handlers.seed(_numpyro_model, jax.random.PRNGKey(0))

    def _log_prob(self, x):
        J = self.J
        params = {
            "tau_unc": x[0],
            "mu": x[1],
            "theta_unc": x[2 : 2 + J]
        }
        logp = log_density(self._seeded_model, (), {}, params)[0]
        return logp
    
    # JIT-compile for speed
    log_prob = jax.jit(_log_prob, static_argnums=(0,))

    def param_constrain(self, X):
        if X.ndim == 1:
            tau = jnp.exp(X[0:1])
            mu = X[1:2]
            theta_unc = X[2:]
            theta = mu + tau * theta_unc
            return jnp.concatenate([mu, tau, theta])

        # If X.ndim == 2, we handle multiple samples
        tau = jnp.exp(X[:, 0:1])
        mu = X[:, 1:2]
        theta_unc = X[:, 2:]
        # shape (N, J)
        theta = mu + tau * theta_unc
        # shape (N, J+2)
        return jnp.hstack([mu, tau, theta])

    def param_unc_names(self):
        return ['tau_unc', 'mu', 'theta_unc']


class mesquite:
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        self.N = self.data['N']
        self.weight = jnp.array(self.data['weight'])
        self.diam1 = jnp.array(self.data['diam1'])
        self.diam2 = jnp.array(self.data['diam2'])
        self.canopy_height = jnp.array(self.data['canopy_height'])
        self.total_height = jnp.array(self.data['total_height'])
        self.density = jnp.array(self.data['density'])
        self.group = jnp.array(self.data['group'])
        self.d = 8

        def _numpyro_model():
            beta = numpyro.sample("beta", ImproperUniform().expand([7]))
            sigma_unc = numpyro.sample("sigma_unc", ImproperUniform())
            sigma = jnp.exp(sigma_unc)
            numpyro.factor("sigma_jac", sigma_unc)

            mean = beta[0] + beta[1] * self.diam1 + beta[2] * self.diam2 + beta[3] * self.canopy_height + beta[4] * self.total_height + beta[5] * self.density + beta[6] * self.group

            numpyro.sample("weight", dist.Normal(mean, sigma), obs=self.weight)

        self.numpyro_model = _numpyro_model
        self._seeded_model = numpyro.handlers.seed(_numpyro_model, jax.random.key(0))
    
    def _log_prob(self, x):
        params = {
            "beta": x[:-1],
            "sigma_unc": x[-1]
        }
        logp = log_density(self._seeded_model, (), {}, params)[0]
        return logp
    
    log_prob = jax.jit(_log_prob, static_argnums=(0,))

    def param_constrain(self, X):
        if X.ndim == 1:
            return jnp.array([
                X[:-1],
                jnp.exp(X[-1:]),
            ])

        return jnp.hstack([X[:, :-1], jnp.exp(X[:, -1:])])
    
    def param_unc_names(self):
        return ['beta', 'sigma_unc']

class radon:
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.N = self.data['N']
        self.floor_measure = jnp.array(self.data['floor_measure'])
        self.log_radon = jnp.array(self.data['log_radon'])
        self.d = 3

        def _numpyro_model():
            sigma_y_unc = numpyro.sample("sigma_y_unc", ImproperUniform())
            sigma_y = jnp.exp(sigma_y_unc)
            numpyro.factor("sigma_y_jac", dist.Normal().log_prob(sigma_y) + sigma_y_unc)
            alpha = numpyro.sample("alpha", dist.Normal(0, 10))
            beta = numpyro.sample("beta", dist.Normal(0, 10))

            mu = alpha + beta * self.floor_measure
            numpyro.sample("log_radon", dist.Normal(mu, sigma_y), obs=self.log_radon)
        
        self.numpyro_model = _numpyro_model
        self._seeded_model = numpyro.handlers.seed(_numpyro_model, jax.random.PRNGKey(0))

    def _log_prob(self, x):
        params = {
            "alpha": x[0],
            "beta": x[1],
            "sigma_y_unc": x[2]
        }
        logp = log_density(self._seeded_model, (), {}, params)[0]
        return logp
    log_prob = jax.jit(_log_prob, static_argnums=(0,))

    def param_constrain(self, X):
        if X.ndim == 1:
            return jnp.array([
                X[0],
                X[1],
                jnp.exp(X[2]),
            ])
        return jnp.hstack([X[:, 0:1], X[:, 1:2], jnp.exp(X[:, 2:3])])
    
    def param_unc_names(self):
        return ['alpha', 'beta', 'sigma_y_unc']

class irt_2pl:
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.I = self.data['I']
        self.J = self.data['J']
        self.y = jnp.array(self.data['y']) # shape (I, J)
        self.d = self.I * 2 + self.J + 3

        def _numpyro_model():
            theta = numpyro.sample("theta", dist.Normal(0, 1).expand([self.J]))

            sigma_log_a_unc = numpyro.sample("sigma_log_a_unc", dist.Normal(0, 2))
            sigma_log_a = jnp.exp(sigma_log_a_unc)

            log_a_std = numpyro.sample("log_a_std", dist.Normal(0, 1).expand([self.I]))
            log_a = log_a_std * sigma_log_a

            a = jnp.exp(log_a) # (I, )
            
            mu_b = numpyro.sample("mu_b", dist.Normal(0, 5))

            sigma_b_unc = numpyro.sample("sigma_b_unc", dist.Normal(0, 2))
            sigma_b = jnp.exp(sigma_b_unc)

            b_std = numpyro.sample("b_std", dist.Normal(0, 1).expand([self.I]))
            b = sigma_b * b_std + mu_b

            logits = a[:, None] * (theta - b[:, None])

            numpyro.sample("y", dist.Bernoulli(logits=logits), obs=self.y)
        self.numpyro_model = _numpyro_model
        self._seeded_model = numpyro.handlers.seed(_numpyro_model, jax.random.PRNGKey(0))

    def _log_prob(self, x):
        params = {
            "theta": x[:self.J],
            "sigma_log_a_unc": x[self.J],
            "log_a_std": x[(self.J+1) : (self.J+1+self.I)],
            "mu_b": x[(self.J+1+self.I)],
            "sigma_b_unc": x[(self.J+1+self.I+1)],
            "b_std": x[(self.J+1+self.I+2):],
        }
        logp = log_density(self._seeded_model, (), {}, params)[0]
        return logp
    
    log_prob = jax.jit(_log_prob, static_argnums=(0,))

    def param_constrain(self, x):
        X = jnp.atleast_2d(x)
        theta = X[:, :self.J]
        sigma_log_a = jnp.exp(X[:, self.J:self.J + 1])
        log_a = X[:, self.J + 1: self.J + 1 + self.I] * sigma_log_a
        mu_b = X[:, self.J + 1 + self.I: self.J + 1 + self.I + 1]
        sigma_b = jnp.exp(X[:, self.J + 1 + self.I + 1: self.J + 1 + self.I + 2])
        b = sigma_b * X[:, self.J + 1 + self.I + 2:] + mu_b

        X = jnp.hstack([theta, sigma_log_a, log_a, mu_b, sigma_b, b])
        X = X.reshape(x.shape)
        return X
    
    def param_unc_names(self):
        return ['theta', 'sigma_log_a_unc', 'log_a_std', 'mu_b', 'sigma_b_unc', 'b_std']

class glmm_poisson:
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.n = self.data['n']
        self.C = jnp.array(self.data['C'])
        self.year = jnp.array(self.data['year'])
        self.year_squared = self.year**2
        self.year_cubed = self.year**3
        self.x = jnp.hstack([self.year[:, None], self.year_squared[:, None], self.year_cubed[:, None]]) # (n, 3)
        self.d = 5 + self.n

        def _numpyro_model():
            alpha = numpyro.sample("alpha", dist.Normal(0, 5))
            beta = numpyro.sample("beta", dist.Normal(0, 5).expand([3]))
            sigma_unc = numpyro.sample("sigma_unc", dist.Normal(0, 1))
            sigma = jnp.exp(sigma_unc)
            eps_std = numpyro.sample("eps_std", dist.Normal(0, 1).expand([self.n]))

            rate = jnp.exp(alpha + self.x @ beta + eps_std * sigma)
            numpyro.sample("C", dist.Poisson(rate), obs=self.C)
        self.numpyro_model = _numpyro_model
        self._seeded_model = numpyro.handlers.seed(_numpyro_model, jax.random.PRNGKey(0))

    def _log_prob(self, x):
        params = {
            "alpha": x[0],
            "beta": x[1:4],
            "sigma_unc": x[4],
            "eps_std": x[5:],
        }
        logp = log_density(self._seeded_model, (), {}, params)[0]
        return logp
    log_prob = jax.jit(_log_prob, static_argnums=(0,))

    def param_constrain(self, x):
        if x.ndim == 1:
            return jnp.array([
                x[:4],
                jnp.exp(x[4]),
                x[5:] * jnp.exp(x[4]),
            ])
        return jnp.hstack([x[:, :4], jnp.exp(x[:, 4:5]), x[:, 5:] * jnp.exp(x[:, 4:5])])
    
    def param_unc_names(self):
        return ['alpha', 'beta', 'sigma_unc', 'eps_std']

class kidscore_interaction:
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.N = self.data['N']
        self.kid_score = jnp.array(self.data['kid_score'])
        self.mom_iq = jnp.array(self.data['mom_iq'])
        self.mom_hs = jnp.array(self.data['mom_hs'])
        self.inter= self.mom_iq * self.mom_hs
        self.d = 5

        def _numpyro_model():
            sigma_unc = numpyro.sample("sigma_unc", ImproperUniform())
            sigma = jnp.exp(sigma_unc)
            numpyro.factor("sigma_jac", dist.Cauchy(0, 2.5).log_prob(sigma) + sigma_unc)
            beta = numpyro.sample("beta", ImproperUniform().expand([4]))
            mu = beta[0] + beta[1] * self.mom_hs + beta[2] * self.mom_iq + beta[3] * self.inter
            numpyro.sample("kid_score", dist.Normal(mu, sigma), obs=self.kid_score)
        self.numpyro_model = _numpyro_model
        self._seeded_model = numpyro.handlers.seed(_numpyro_model, jax.random.PRNGKey(0))
    
    def _log_prob(self, x):
        params = {
            "beta": x[:4],
            "sigma_unc": x[4]
        }
        logp = log_density(self._seeded_model, (), {}, params)[0]
        return logp
    
    log_prob = jax.jit(_log_prob, static_argnums=(0,))

    def param_constrain(self, x):
        if x.ndim == 1:
            return jnp.array([
                x[:4],
                jnp.exp(x[4]),
            ])
        return jnp.hstack([x[:, :4], jnp.exp(x[:, 4:5])])

    def param_unc_names(self):
        return ['beta', 'sigma_unc']

class rosenbrock:
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.D = self.data['D']
        self.d = 2*self.D

        def _numpyro_model():
            v = numpyro.sample("v", dist.Normal(1, 1).expand([self.D]))
            theta_sample = numpyro.sample("theta", dist.Normal(v**2, .1).expand([self.D]))
            theta = numpyro.deterministic("theta_deterministic", theta_sample)  # Include theta in the model's output

        self.numpyro_model = _numpyro_model
        # Seed the model for log_prob calculations
        self._seeded_model = numpyro.handlers.seed(_numpyro_model, jax.random.PRNGKey(0))
    
    def _log_prob(self, x):
        params = {
            "v": x[:self.D],
            "theta": x[self.D:],
        }
        logp = log_density(self._seeded_model, (), {}, params)[0]
        return logp
    
    # JIT-compile for speed
    log_prob = jax.jit(_log_prob, static_argnums=(0,))

    def param_constrain(self, X):
        return X

    def param_unc_names(self):
        return ['v', 'theta']

class funnel:
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.D = self.data['D']
        self.d = self.D + 1

        def _numpyro_model():
            double_log_sigma = numpyro.sample("double_log_sigma", dist.Normal(0, 3))
            alpha_sample = numpyro.sample("alpha", dist.Normal(0, jnp.exp(0.5 * double_log_sigma)).expand([self.D]))
            alpha = numpyro.deterministic("alpha_deterministic", alpha_sample)  # Include theta in the model's output

        self.numpyro_model = _numpyro_model
        self._seeded_model = numpyro.handlers.seed(_numpyro_model, jax.random.PRNGKey(0))
    
    def _log_prob(self, x):
        params = {
            "double_log_sigma": x[0],
            "alpha": x[1:],
        }
        logp = log_density(self._seeded_model, (), {}, params)[0]
        return logp
    
    log_prob = jax.jit(_log_prob, static_argnums=(0,))

    def param_constrain(self, X):
        return X

    def param_unc_names(self):
        return ['double_log_sigma', 'alpha']

class arma:
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            data = json.load(f)
        self.T = data["T"]
        self.y = jnp.array(data["y"])
        self.d = 4

        def _numpyro_model():

            mu     = numpyro.sample("mu",     dist.Normal(0, 10))
            phi    = numpyro.sample("phi",    dist.Normal(0, 2))
            theta  = numpyro.sample("theta",  dist.Normal(0, 2))
            sigma_unc = numpyro.sample("sigma_unc", ImproperUniform())
            sigma = numpyro.deterministic("sigma", jnp.exp(sigma_unc))
            numpyro.factor("sigma_prior", dist.Normal(0, 2.5).log_prob(sigma) + sigma_unc)

            nu  = []
            err = []

            nu_1  = mu + phi*mu
            err_1 = self.y[0] - nu_1  
            nu.append(nu_1)
            err.append(err_1)

            for t in range(1, self.T):
                nu_t  = mu + phi * self.y[t-1] + theta * err[t-1]
                err_t = self.y[t] - nu_t
                nu.append(nu_t)
                err.append(err_t)

            err = jnp.stack(err)  

            numpyro.sample("err", dist.Normal(0, sigma), obs=err)

        self.numpyro_model = _numpyro_model
        self._seeded_model = numpyro.handlers.seed(_numpyro_model, jax.random.PRNGKey(0))

    def _log_prob(self, x):
        params = {
            "mu": x[0],
            "phi": x[1],
            "theta": x[2],
            "sigma_unc": x[3]
        }
        logp = log_density(self._seeded_model, (), {}, params)[0]
        return logp
    
    log_prob = jax.jit(_log_prob, static_argnums=(0,))

    def param_constrain(self, X):
        if X.ndim == 1:
            X = X.at[-1].set(jnp.exp(X[-1]))
            return X
        X = X.at[:, -1].set(jnp.exp(X[:, -1]))
        return X

    def param_unc_names(self):
        return ['mu', 'phi', 'theta', 'sigma_unc']

class BLR:
    def __init__(self, X, y, prior_scale=1.):
        self.X = X
        self.y = y
        self.prior_scale = prior_scale
        self.n, self.d = X.shape

        def _numpyro_model():
            beta = numpyro.sample("beta", dist.Normal(0, prior_scale).expand([self.d]))
            logits = jnp.dot(self.X, beta)
            numpyro.sample("y", dist.Bernoulli(logits=logits), obs=self.y)

        self.numpyro_model = _numpyro_model
        self._seeded_model = numpyro.handlers.seed(_numpyro_model, jax.random.key(0))
    
    def _log_prob(self, x):
        params = {
            "beta": x
        }
        logp = log_density(self._seeded_model, (), {}, params)[0]
        return logp
    
    log_prob = jax.jit(_log_prob, static_argnums=(0,))

    def param_constrain(self, X):
        return X
    
    def param_unc_names(self):
        return ['beta']

class BananaNormal:
    def __init__(self, d):
        self.d = d

    def log_prob(self, x):
        x1, x2 = x[0], x[1]
        term1 = -0.5 * x1**2 - 0.5 * jnp.log(2 * jnp.pi)
        term2 = -0.5 * 2 * (x2 - (x1**2 - 0.5 - 0.5))**2 - 0.5 * jnp.log(2 * jnp.pi * .5)
        term3 = -0.5 * jnp.sum(x[2:]**2) - 0.5 * (self.d - 2) * jnp.log(2 * jnp.pi)
        return term1 + term2 + term3
    
    def density(self, x):
        return jnp.exp(self.log_prob(x))

class BayesianLogisticRegression:
    def __init__(self, X, y, prior_scale=1.0):
        self.d = X.shape[1]
        self.X = X
        self.y = y # 1 or 0
        self.prior_scale = prior_scale

    def log_prob(self, w):
        logits = jnp.dot(self.X, w)
        log_prior = -0.5 * jnp.sum((w / self.prior_scale)**2) - 0.5 * self.d * jnp.log(2 * jnp.pi)
        log_lik = jnp.sum(logits * self.y - jnp.logaddexp(0, logits))
        return log_prior + log_lik

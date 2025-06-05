import numpy as np
from scipy.stats import (
    norm, truncnorm, lognorm, beta, truncexpon, loguniform
)

# Uniform prior
def transform_uniform(x, a, b):
    return a + (b - a) * x

# Log-uniform prior
def transform_loguniform(x, a, b):
    return np.exp(np.log(a) + x * (np.log(b) - np.log(a)))

# Gaussian prior
def transform_normal(x, mu, sigma):
    return norm.ppf(x, loc=mu, scale=sigma)

# Truncated normal prior
def transform_truncated_normal(x, mu, sigma, a, b):
    ar, br = (a - mu) / sigma, (b - mu) / sigma
    return truncnorm.ppf(x, ar, br, loc=mu, scale=sigma)

# Log-normal prior
def transform_lognormal(x, scale, s):
    return lognorm.ppf(x, scale=scale, s=s)

# Beta prior (bounded between 0 and 1)
def transform_beta(x, a, b):
    return beta.ppf(x, a, b)

# Truncated exponential prior
def transform_truncexpon(x, b, loc, scale):
    return truncexpon.ppf(x, b / scale, loc=loc, scale=scale)

# Empirical (histogram-based) prior
def transform_histogram(x, hist_cum, hist_bins):
    return np.interp(x, hist_cum, hist_bins)

# Compose a generic prior transform function
def make_prior_transform(transforms, hyperparams):
    def prior_transform(utheta):
        transformed = np.copy(utheta)
        for i, (transform, params) in enumerate(zip(transforms, hyperparams)):
            transformed[:, i] = transform(utheta[:, i], *params)
        return transformed
    return prior_transform


# Default configuration
DEFAULT_PRIORS = {
    "K": dict(transform="uniform", bounds=[0., 100.]),              # semi-amplitude [km/s]
    "P": dict(transform="loguniform", bounds=[0.1, 1000.]),         # period [days]
    "tau": dict(transform="uniform", bounds=[0., 1.]),              # phase
    "e": dict(transform="uniform", bounds=[0., 0.99]),              # eccentricity
    "omega": dict(transform="uniform", bounds=[0., 2*np.pi]),       # argument of periastron
    "offset": dict(transform="uniform", bounds=[-100., 100.])       # RV offset
}

TRANSFORM_FUNCS = {
    "uniform": transform_uniform,
    "loguniform": transform_loguniform,
    "normal": transform_normal,
    "truncated_normal": transform_truncated_normal,
    "beta": transform_beta,
}


class PriorConfig:
    def __init__(self, user_priors=None):
        self.priors = {**DEFAULT_PRIORS}
        if user_priors:
            for key, val in user_priors.items():
                self.priors[key] = val

        self.param_names = list(self.priors.keys())
        self.transforms = []
        self.hyperparams = []

        for key in self.param_names:
            prior = self.priors[key]
            try:
                func = TRANSFORM_FUNCS[prior['transform']]
            except KeyError:
                raise ValueError(f"Transform '{prior['transform']}' not implemented for prior '{key}'")

            # Extract parameters for transform functions, usually 'bounds' or 'params'
            if 'bounds' in prior:
                params = prior['bounds']
            elif 'params' in prior:
                params = prior['params']
            else:
                raise ValueError(f"Prior '{key}' missing 'bounds' or 'params'")

            self.transforms.append(func)
            self.hyperparams.append(params)

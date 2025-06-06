# sampler.py

import os
import numpy as np
import ultranest
import ultranest.stepsampler
from .model import rv_model

class OrbitalSampler:
    def __init__(self, times, rvs, rvs_err,
                 prior_transform=None,
                 param_labels=None, periodic=None,
                 results_dir='./orbits/', star_id='star',
                 nlive=1000):
        self.times = times
        self.rvs = rvs
        self.rvs_err = rvs_err
        self.prior_transform = prior_transform
        self.star_id = star_id
        self.results_dir = results_dir
        self.nlive = nlive

        # Default parameter names and periodic flags if not provided
        self.param_labels = param_labels or ['K [km/s]', 'P [d]', 'tau', 'e', 'omega', 'offset']
        self.periodic = periodic or [False, False, True, False, True, False]

        # Initialize sampler object but do not run yet
        self._sampler = ultranest.ReactiveNestedSampler(
            self.param_labels,
            self.log_likelihood,
            self.prior_transform,
            wrapped_params=self.periodic,
            vectorized=True,
            log_dir=os.path.join(self.results_dir, f'orbit_{self.star_id}'),
            resume='resume'
        )
        self._sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=20,generate_direction=ultranest.stepsampler.generate_mixture_random_direction)

        self.result = None

    def log_likelihood(self, theta):
        theta = np.atleast_2d(theta)
        model = rv_model(theta, self.times)
        inv_sigma2 = 1.0 / (self.rvs_err ** 2)
        # Negative chi^2 / 2 (normal log likelihood without constant terms)
        loglike = -0.5 * np.sum(((self.rvs - model) ** 2) * inv_sigma2, axis=1)
        return loglike

    def run(self, dlogz=0.01):
        self.result = self._sampler.run(min_num_live_points=self.nlive, frac_remain=dlogz)
        return self.result

    def summary(self):
        if self.result is None:
            print("No results to summarize; please run sampler first.")
            return
        self._sampler.print_results()

    def plot(self):
        if self.result is None:
            print("No results to plot; please run sampler first.")
            return
        self._sampler.plot_run()
        self._sampler.plot_trace()
        self._sampler.plot_corner()

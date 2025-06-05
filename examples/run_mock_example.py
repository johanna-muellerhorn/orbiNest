#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os

from orbiNest.model import rv_model
from orbiNest.data import RVDataLoader
from orbiNest.priors import PriorConfig, make_prior_transform
from orbiNest.sampler import OrbitalSampler

# ------------------------------------------------------------------------------
# Generate mock data
rng = np.random.default_rng(seed=26201)

true_params = [25., 40., 0.3, 0.15, np.pi/4, -10.]  # [K, P, tau, e, omega, offset]
N_obs = 10
mjds = np.sort(rng.uniform(0, 100, N_obs))
rvs_true = rv_model(np.array([true_params]), mjds).flatten()
rv_err = 2.0  # km/s
rvs_obs = rvs_true + rng.normal(0, rv_err, N_obs)

# Save to CSV
mock_df = pd.DataFrame({
    'Star_Id': ['test_star'] * N_obs,
    'MJD': mjds,
    'STAR V': rvs_obs,
    'STAR V err': [rv_err] * N_obs
})
mock_path = './mock_data.csv'
mock_df.to_csv(mock_path, index=False)

# ------------------------------------------------------------------------------
# Load the mock data
loader = RVDataLoader(mock_path, rv_col='STAR V', rv_err_col='STAR V err', time_col='MJD')
data = loader.get_star_data('test_star')

# ------------------------------------------------------------------------------
# Custom priors: widen the prior on K
user_priors = {
    'K': {"transform": "uniform", "bounds": [0., 200.]},  # Adjusted K range
    # Others remain at default
}
prior_config = PriorConfig()

prior_transform = make_prior_transform(
    prior_config.transforms,
    prior_config.hyperparams
)


# ------------------------------------------------------------------------------
# Fit
sampler = OrbitalSampler(
    star_id='test_star',
    times=data.times,
    rvs=data.rvs,
    rvs_err=data.rvs_err,
    prior_transform=prior_transform,
    param_labels=prior_config.param_names,
    periodic=[False, False, True, False, True, False],  # example
    results_dir='./orbit_results'
)

result = sampler.run()
sampler.summary()
sampler.plot()

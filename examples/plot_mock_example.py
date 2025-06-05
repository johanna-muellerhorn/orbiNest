#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os


from orbiNest.model import rv_model
from orbiNest.data import RVDataLoader, OrbitFitLoader
from orbiNest.plot import *

rng = np.random.default_rng(seed=26201)

# ------------------------------------------------------------------------------
# Load the mock data
star_id = 'test_star'
mock_path = './mock_data.csv'

loader = RVDataLoader(mock_path, rv_col='STAR V', rv_err_col='STAR V err', time_col='MJD')
data = loader.get_star_data(star_id)

# ------------------------------------------------------------------------------
# Custom priors: widen the prior on K
results_dir = f'./orbit_results/orbit_{star_id}/'
plot_dir = results_dir+'plots/'
orbitfit = OrbitFitLoader(results_dir=results_dir)

print(f"Nested Sampling Orbit Inference Results for Star {star_id}")
for elem in orbitfit.results:
    print(elem, orbitfit.results[elem])

print("Plot orbit...")
plot_sample(data.star_id, data.rvs, data.rvs_err, data.times, orbitfit,plot_path=plot_dir)

print("Plot phase-folded orbit...")
plot_sample(data.star_id, data.rvs, data.rvs_err, data.times, orbitfit, phase=True,plot_path=plot_dir)

print("Plot prediction band...")
plot_prediction_band(data.star_id, data.rvs, data.rvs_err, data.times, orbitfit, plot_path=plot_dir)

print("Plot summary...")
plot_summary(data.star_id, data.rvs, data.rvs_err, data.times, orbitfit, plot_path=plot_dir)

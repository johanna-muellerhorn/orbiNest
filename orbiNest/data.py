import pandas as pd
import numpy as np
import json
from collections import namedtuple

StarData = namedtuple('StarData', ['star_id','times', 'rvs', 'rvs_err'])

class RVDataLoader:
    def __init__(self, file, id_col='Star_Id', time_col='MJD',
                 rv_col='RV', rv_err_col='RV_err'):
        self.df = pd.read_csv(file)
        self.id_col = id_col
        self.time_col = time_col
        self.rv_col = rv_col
        self.rv_err_col = rv_err_col

    def get_star_ids(self):
        return sorted(self.df[self.id_col].unique())

    def get_star(self, star_id):
        star_df = self.df[self.df[self.id_col] == star_id]
        return star_df.sort_values(by=self.time_col)

    def get_star_data(self, star_id):
        star_df = self.get_star(star_id)
        return StarData(
            star_id=star_id,
            times=star_df[self.time_col].values,
            rvs=star_df[self.rv_col].values,
            rvs_err=star_df[self.rv_err_col].values
        )

class OrbitFitLoader:
    def __init__(self, results_dir=None):
        self.results_dir = results_dir
        self.samples = self._get_orbit_samples()
        self.results = self._get_orbit_results()

    def _get_orbit_samples(self):
        orbit_samples = np.genfromtxt(self.results_dir+'chains/equal_weighted_post.txt', skip_header=1)
        return orbit_samples

    def _get_orbit_results(self):
        with open(self.results_dir+'info/results.json') as d:
            orbit_results = json.load(d)
        return orbit_results

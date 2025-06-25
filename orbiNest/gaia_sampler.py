# sampler.py

import os
import numpy as np
import ultranest
import ultranest.stepsampler
from .model import rv_model

class OrbitalSamplerGaia:
    def __init__(self, target, times, rvs, rvs_err,
                 prior_transform=None,
                 param_labels=None, periodic=None,
                 results_dir='./orbits/', star_id='star',
                 nlive=1000, fit_type='rvs'):
        self.times = times
        self.rvs = rvs
        self.rvs_err = rvs_err
        self.prior_transform = prior_transform
        self.star_id = star_id
        self.results_dir = results_dir
        self.nlive = nlive
        self.fit_type = fit_type

        if (self.fit_type == 'rvs'):
            self.log_likelihood = self.log_likelihood_rvs
        if (self.fit_type == 'gaia_mean_and_amp'):
            self.log_likelihood = self.log_likelihood_mean_and_amp
        if (self.fit_type == 'gaia_ruwe'):
            self.log_likelihood = self.log_likelihood_ruwe

        #if (likelihood_type == 'gaia_mean_and_amp') or (likelihood_type == 'gaia_ruwe'):
        #    rv_err = np.sqrt((np.std(radial_velocities) / np.sqrt(len(radial_velocities)) * np.sqrt(np.pi / 2))**2 + 0.113**2)
        # Default parameter names and periodic flags if not provided
        self.param_labels = param_labels or ['K [km/s]', 'P [d]', 'tau', 'e', 'omega', 'offset', 'cosi']
        self.periodic = periodic or [False, False, True, False, True, False, False]

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

    def log_likelihood_rvs(self, theta):
        theta = np.atleast_2d(theta)
        model = rv_model(theta, self.times)
        inv_sigma2 = 1.0 / (self.rvs_err ** 2)
        # Negative chi^2 / 2 (normal log likelihood without constant terms)
        loglike = -0.5 * np.sum(((self.rvs - model) ** 2) * inv_sigma2, axis=1)
        return loglike

    def log_likelihood_mean_and_amp(self, theta):
        # log-likelihood, for parameter vector theta=(logK, logf, mean anomaly, e, omega, jitter, offset)
        theta = np.atleast_2d(theta)
        model = rv_model(theta, self.times)
        inv_sigma2 = 1.0 / (self.rvs_err ** 2)
        loglike = -0.5 * np.sum(((self.rvs - model) ** 2) * inv_sigma2, axis=1)

        gaia_rv_pred = rv_model(theta, self.star.times_gaia)
        gaia_rv_median_pred = np.median(gaia_rv_pred)
        gaia_rv_amp_pred = np.max(gaia_rv_pred) - np.min(gaia_rv_pred)
        gaia_rv_err_pred = np.median(gaia_rv_pred)

        gaia_rv_amp_pred = np.max(gaia_rv_pred) - np.min(gaia_rv_pred)
        loglike_gaia_mean = - 0.5 * np.log(self.star.rv_error_gaia**2) - 0.5 * (gaia_rv_mean_pred-self.star.rv_gaia)**2 /(self.star.rv_error_gaia**2)
        loglike_gaia_amp  = - 0.5 * np.log(self.star.rv_error_gaia**2) - 0.5 * (gaia_rv_amp_pred-self.star.rv_amplitude_gaia)**2 /(self.star.rv_error_gaia**2)

        #print('RV mean and amplitude:', self.star.rv_gaia, self.star.rv_amplitude_gaia, gaia_rv_mean_pred, gaia_rv_amp_pred, loglike, loglike_gaia_mean, loglike_gaia_amp)
        return (loglike + loglike_gaia_mean + loglike_gaia_amp).sum(axis=1)

    def log_likelihood_ruwe(self,theta):
        theta = np.atleast_2d(theta)
        model = rv_model(theta, self.times)
        inv_sigma2 = 1.0 / (self.rvs_err ** 2)
        # Negative chi^2 / 2 (normal log likelihood without constant terms)
        loglike = -0.5 * np.sum(((self.rvs - model) ** 2) * inv_sigma2, axis=1)

        gaia_rv_pred = RVmodel(theta, self.star.times_gaia)
        gaia_rv_mean_pred = np.mean(gaia_rv_pred)
        gaia_rv_amp_pred = np.max(gaia_rv_pred) - np.min(gaia_rv_pred)

        loglike_gaia_mean = - 0.5 * np.log(self.star.rv_error_gaia**2) - 0.5 * (gaia_rv_mean_pred-self.star.rv_gaia)**2 /(self.star.rv_error_gaia**2)
        loglike_gaia_amp  = - 0.5 * np.log(self.star.rv_error_gaia**2) - 0.5 * (gaia_rv_amp_pred-self.star.rv_amplitude_gaia)**2 /(self.star.rv_error_gaia**2)

        #tau = ((Tp+gaia_ref-T_ref)/P)%1
        Tp = tau*P+T_ref-gaia_ref

        fbin = binary_mass_function(K=K,period=P,e=e)
        mcomp = np.maximum(min_companion_mass(fbin/np.sin(i)**3, self.star.mass), 0.01)
        #print(fbin, mcomp)

        astrometry = calculate_astrometry(self.star.ra_deg,self.star.dec_deg,self.star.parallax,self.star.GaiaG,self.star.pmra,self.star.pmdec,self.star.mass,
                                           P, mcomp, e,i, Tp=Tp, omega=np.pi/2, w=w,f=f)

        gaia_ruwe_pred = gaiamock.check_ruwe(*astrometry,binned=True)[0]
        #loglike_gaia_ruwe  = - 1000. * int((ruwe_pred-self.star.ruwe)>self.star.ruwe_error)
        loglike_gaia_ruwe  = - 0.5 * np.log(self.star.ruwe_error**2) - 0.5 * (gaia_ruwe_pred-self.star.ruwe)**2 /(self.star.ruwe_error**2)

        #print(theta.shape, loglike.shape, loglike_gaia_mean.shape, loglike_gaia_amp.shape, loglike_gaia_ruwe.shape)
        #print(theta, loglike, loglike_gaia_mean, loglike_gaia_amp, loglike_gaia_ruwe)
        #ypred_photo = estimate_ellipsoidal_amplitude(self.star.mass,mcomp,self.star.radius,P,i)

        #print('RV mean and amplitude:', self.star.rv_gaia, self.star.rv_amplitude_gaia, gaia_rv_mean_pred, gaia_rv_amp_pred, loglike, loglike_gaia_mean, loglike_gaia_amp)
        #print(theta, fbin, mcomp, gaia_rv_mean_pred, gaia_rv_amp_pred, gaia_ruwe_pred, loglike,loglike_gaia_mean,loglike_gaia_amp,loglike_gaia_ruwe)
        return (loglike + loglike_gaia_mean + loglike_gaia_amp+loglike_gaia_ruwe).sum()

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

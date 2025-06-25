# sampler.py

import os
import numpy as np
import ultranest
import ultranest.stepsampler
from .model import rv_model
import gaiamock.gaiamock as gaiamock
_C_FUNCS = gaiamock.read_in_C_functions()

class OrbitalSamplerGaia:
    def __init__(self, target, hp, times, rvs, rvs_err,
                 prior_transform=None,
                 param_labels=None, periodic=None,
                 results_dir='./orbits/', #star_id='star',
                 nlive=1000, fit_type='rvs'):
        self.target = target
        self.hp = hp
        self.times = times
        self.rvs = rvs
        self.rvs_err = rvs_err
        self.prior_transform = prior_transform
        self.star_id = self.target.source_id
        self.results_dir = results_dir
        self.nlive = nlive
        self.fit_type = fit_type


        if (self.fit_type == 'rvs'):
            self.log_likelihood = self.log_likelihood_rvs
        if (self.fit_type == 'gaia_mean_and_amp'):
            self.log_likelihood = self.log_likelihood_mean_and_amp
        if (self.fit_type == 'gaia_ruwe'):
            self.log_likelihood = self.log_likelihood_ruwe

        #    rv_err = np.sqrt((np.std(radial_velocities) / np.sqrt(len(radial_velocities)) * np.sqrt(np.pi / 2))**2 + 0.113**2)
        # Default parameter names and periodic flags if not provided
        self.param_labels = param_labels or ['K [km/s]', 'P [d]', 'tau', 'e', 'omega', 'offset', 'cosi']
        self.periodic = periodic or [False, False, True, False, True, False, False]

        print('INITIALIZING STAR AND SAMPLER:')
        print(self.star_id, self.results_dir)
        print('Fit type:', self.fit_type)
        print('Fit parameters:', self.param_labels)

        print('----------------- STAR INFO ------------------')
        print('Gaia RV:', self.target.radial_velocity)
        print('Gaia RV err:', self.target.radial_velocity_error)
        print('Gaia RV Amplitude:', self.target.rv_amplitude_robust)
        print('Gaia N epochs / RVs:', len(self.target.t_ast_yr), self.target.rv_nb_transits)
        print(self.hp.eps_astro, self.hp.eps_spectro)
        print('----------------------------------------------')

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

    def log_likelihood_mean_and_amp(self, theta, T_ref=51544., gaia_ref=57388.5):
        # log-likelihood, for parameter vector theta=(logK, logf, mean anomaly, e, omega, jitter, offset)
        theta = np.atleast_2d(theta)
        model = rv_model(theta, self.times)
        inv_sigma2 = 1.0 / (self.rvs_err ** 2)
        loglike = -0.5 * np.sum(((self.rvs - model) ** 2) * inv_sigma2, axis=1)

        K = theta[:, [0]]
        P = theta[:, [1]]
        tau = theta[:, [2]]
        e = np.abs(theta[:, [3]])
        w = theta[:, [4]]
        off = theta[:, [5]]
        cosi = theta[:, [6]]

        t_rvs_day = self.target.t_ast_yr * 365.25 + gaia_ref # converting to modified julian dates

        if self.target.rv_nb_transits is not None and self.target.rv_nb_transits < len(t_rvs_day):
            t_rvs_day = np.random.choice(t_rvs_day, size=int(self.target.rv_nb_transits), replace=False)

        gaia_rvs = rv_model(theta, t_rvs_day)

        Tp = tau*P+T_ref-gaia_ref
        pred_rv_amplitude_robust = (np.max(gaia_rvs,axis=1) - np.min(gaia_rvs,axis=1))*self.hp.bias_spectro
        pred_radial_velocity = (np.median(gaia_rvs,axis=1))

        loglike_gaia_mean = - 0.5 * (pred_radial_velocity-self.target.radial_velocity)**2 /(self.target.radial_velocity_error**2)
        loglike_gaia_amp  = - 0.5 * ((np.log(self.target.rv_amplitude_robust / pred_rv_amplitude_robust) / self.hp.eps_spectro))**2
        #print('RV mean and amplitude:', self.target.rv_gaia, self.target.rv_amplitude_gaia, gaia_rv_mean_pred, gaia_rv_amp_pred, loglike, loglike_gaia_mean, loglike_gaia_amp)
        return (loglike + loglike_gaia_mean + loglike_gaia_amp)

    def log_likelihood_ruwe(self,theta):
        # log-likelihood, for parameter vector theta=(logK, logf, mean anomaly, e, omega, jitter, offset)
        theta = np.atleast_2d(theta)
        model = rv_model(theta, self.times)
        inv_sigma2 = 1.0 / (self.rvs_err ** 2)
        loglike = -0.5 * np.sum(((self.rvs - model) ** 2) * inv_sigma2, axis=1)

        K = theta[:, [0]]
        P = theta[:, [1]]
        tau = theta[:, [2]]
        e = np.abs(theta[:, [3]])
        w = theta[:, [4]]
        off = theta[:, [5]]
        cosi = theta[:, [6]]

        t_rvs_day = self.target.t_ast_yr * 365.25 + gaia_ref # converting to modified julian dates

        if self.target.rv_nb_transits is not None and self.target.rv_nb_transits < len(t_rvs_day):
            t_rvs_day = np.random.choice(t_rvs_day, size=int(self.target.rv_nb_transits), replace=False)

        gaia_rvs = rv_model(theta, t_rvs_day)

        Tp = tau*P+T_ref-gaia_ref
        pred_rv_amplitude_robust = (np.max(gaia_rvs,axis=1) - np.min(gaia_rvs,axis=1))*self.hp.bias_spectro
        pred_radial_velocity = (np.median(gaia_rvs,axis=1))

        loglike_gaia_mean = - 0.5 * (pred_radial_velocity-self.target.radial_velocity)**2 /(self.target.radial_velocity_error**2)
        loglike_gaia_amp  = - 0.5 * ((np.log(self.target.rv_amplitude_robust / pred_rv_amplitude_robust) / self.hp.eps_spectro))**2
        #print('RV mean and amplitude:', self.target.rv_gaia, self.target.rv_amplitude_gaia, gaia_rv_mean_pred, gaia_rv_amp_pred, loglike, loglike_gaia_mean, loglike_gaia_amp)

        #tau = ((Tp+gaia_ref-T_ref)/P)%1
        Tp = tau*P+T_ref-gaia_ref

        fbin = binary_mass_function(K=K,period=P,e=e)
        mcomp = np.maximum(min_companion_mass(fbin/np.sin(i)**3, self.target.mass), 0.01)
        #print(fbin, mcomp)

        astrometry = calculate_astrometry(self.target.ra_deg,self.target.dec_deg,self.target.parallax,self.target.GaiaG,self.target.pmra,self.target.pmdec,self.target.mass,
                                           P, mcomp, e,i, Tp=Tp, omega=np.pi/2, w=w,f=f)

        # Predict astrometric observable (RUWE)
        gaia_ruwe_pred = astrometric.predict_ruwe(ra=self.target.ra,
                                    dec=self.target.dec,
                                    parallax=self.target.parallax,
                                    pmra=self.target.pmra,
                                    pmdec=self.target.pmdec,
                                    m1=self.target.mass,
                                    m2=mcompanion,
                                    period=period,
                                    Tp=Tp,
                                    ecc=ecc,
                                    omega=omega,
                                    inc=inc,
                                    w=w,
                                    phot_g_mean_mag=self.target.phot_g_mean_mag,
                                    f=self.hp.f,
                                    t_ast_yr=self.target.t_ast_yr,
                                    psi=self.target.psi,
                                    plx_factor=self.target.plx_factor,
                                    epoch_err_per_transit=self.target.epoch_err_per_transit,
                                    data_release=self.hp.data_release,
                                    bias_factor=self.hp.bias_astro)

        #loglike_gaia_ruwe  = - 1000. * int((ruwe_pred-self.target.ruwe)>self.target.ruwe_error)
        loglike_gaia_ruwe  = - 0.5 * ((np.log(self.target.ruwe / gaia_ruwe_pred) / self.hp.eps_astro))**2
        return (loglike + loglike_gaia_mean + loglike_gaia_amp+loglike_gaia_ruwe)

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

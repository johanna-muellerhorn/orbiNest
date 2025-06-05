#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#@author: j. mueller-horn


import ultranest
import ultranest.stepsampler
from ultranest.plot import cornerplot, PredictionBand
from scipy.stats import uniform,gamma,norm,lognorm,beta,truncnorm,truncexpon,loguniform
from scipy.optimize import newton
import json
import pandas as pd

# system functions
import time, sys, os

# basic numeric setup
import numpy as np
from numpy.random import default_rng

# plotting
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

# seed the random number generator
rng = default_rng(seed=26201)
plt.rcParams.update({'font.size': 12})

# astro constants
G    = 6.67408e-11 # [m^3 kg^-1 s^-2]
Msun = 1.98e30     # [kg]
s2yr = 60.**2.*24.*365.25
d2s  = 24*60.**2
AU   = 1.495978707e11 # [m]
pi05 = np.pi/2

################################################################################

def get_anomaly_MA(P,e,Ma):
    '''
    - solve Kepler's equation using the Newton Raphson method
    - determine the eccentric anomaly and the true anomaly
    '''
    Ea   = Ma
    diff = 1.
    count = 0
    while (diff > 1.e-6) and (count < 100):
        Enew = Ea - (Ea - np.abs(e)*np.sin(Ea) -Ma)/(1. - np.abs(e)*np.cos(Ea))
        diff   = np.max(np.abs(Enew-Ea)/np.pi)
        Ea     = Enew*1.
        count += 1
    Ta  = 2.*np.arctan(np.sqrt((1.+np.abs(e))/(1.-np.abs(e)))*np.tan(0.5*Ea))
    return Ta

def orbital_period(m0,m1,a):
    'Kepler\'s third law'
    period = 2.*np.pi * (a*AU)**(3./2.) * (G*(m0+m1)*Msun)**(-1./2.)
    return period/d2s
print('earth period = ', orbital_period(1.,3.003e-6,1.))

def RV_amplitude(m0,m1,T,e,i=np.pi/2.):
    'computes RV semi-amplitude in km/s for two stars of massses m0 and m1'
    amplitude = m1/(m0+m1) * (2*np.pi/(T*d2s)*G*(m0+m1)*Msun)**(1./3.) * np.sin(i)/(1-e**2)**0.5
    return amplitude/1e3

def RVmodel(theta, t, T_ref=51544.):
    K=theta[:,[0]]; P=theta[:,[1]]; tau=theta[:,[2]]; e=theta[:,[3]]
    w=theta[:,[4]]; off=theta[:,[5]]
    e = np.abs(e)
    frac_date = ((t - T_ref) / P)%1
    Ma = ((frac_date - tau) * 2 * np.pi)%(2*np.pi)
    Ta = get_anomaly_MA(P,e,Ma)
    model = K * (np.cos(w+Ta)+e*np.cos(w)) + off
    return model

################################################################################
# Prior transforms for nested samplers:
def prior_uniform(x, hyperparameters=[]):
    a, b = hyperparameters
    return uniform.pdf(x,loc=a,scale=np.abs(a)+np.abs(b))

def prior_loguniform(x, hyperparameters=[]):
    a, b = hyperparameters
    return loguniform.pdf(x,a,b)

def prior_beta(x, hyperparameters=[]):
    a, b = hyperparameters
    return beta.pdf(x,a,b,loc=-1e-10)

def prior_lognormal(x, hyperparameters=[]):
    a,b = hyperparameters
    return lognorm.pdf(x,scale=a, s=b)

def prior_normal(x, hyperparameters=[]):
    mu, sigma = hyperparameters
    return norm.pdf(x, loc=mu, scale=sigma)

def prior_truncated_normal(x, hyperparameters=[]):
    mu, sigma, a, b = hyperparameters
    ar, br = (a - mu) / sigma, (b - mu) / sigma
    return truncnorm.pdf(x, ar, br, loc=mu, scale=sigma)

def freq_dep_amp_sig(f, e,f0=0.1,sigK0=30):
    # for creating a RV-amplitude prior dependent on frequency and eccentricity
    sigK = (sigK0**2*(f0/f)**(-2./3.)/(1-e**2))**(1./2.)
    return sigK

################################################################################

class spectra_sample:
    # constructor function
    def __init__(self, file=''):
        self.df = pd.read_csv(file)
    def single_star(self,star_id):
        return self.df.loc[lambda df: df['Star_Id']==star_id]

class star:
    # constructor function
    def __init__(self,star_id):
        self.star_id = star_id

    def get_data(self,spectra):
        #self.mean_params = params.single_star(star_id=self.star_id)
        self.spectra = spectra.single_star(star_id=self.star_id)
        self.rvs     = self.spectra['STAR V'].values
        self.rvs_err = self.spectra['STAR V err'].values
        self.times   = self.spectra['MJD-OBS'].values
        #self.pointing_ids = self.spectra['Pointing_Id'].values
        #self.npointings = len(np.unique(self.pointing_ids))

################################################################################

def plot_sample(plot_id,phase_plot=False):
    textargs = dict(horizontalalignment='center',verticalalignment='bottom')
    plt.rcParams.update({'font.size': 12})

    # true values
    truths = median_orbit_params
    y      = spectra_sample.df.loc[lambda df: df['Star_Id'] == plot_id,:]['STAR V'].values #data[k] #+18.8
    yerr   = spectra_sample.df.loc[lambda df: df['Star_Id'] == plot_id,:]['STAR V err'].values#errs[k]
    time   = spectra_sample.df.loc[lambda df: df['Star_Id'] == plot_id,:]['MJD-OBS'].values#times[k]
    truelog = loglike(truths,time,y,yerr)


    phase_array = np.linspace(0.,2*np.pi,1000)
    y_true    = RVmodel(truths, phase_array/(2*np.pi*truths[1]), np.min(phase_array/(2*np.pi*truths[1])))
    errs_true = y-RVmodel(truths,(2*np.pi*truths[1]*(time))%(2*np.pi), np.min(phase_array/(2*np.pi*truths[1])))

    time_array = np.linspace(np.min(time),np.max(time),10000)#np.max(time),1000)

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8, 5), gridspec_kw={'height_ratios': [3,1]},sharex=True)

    fig.suptitle(r'$P=${:.1f} d, $e=${:.2f}'.format(1./truths[1], truths[3]),
                 fontsize=14, x=0.57,y=0.93)
    if phase_plot:
        ax1.errorbar(((2*np.pi*median_orbit_params[1]*(time-np.min(time)))%(2*np.pi))/(2*np.pi)/median_orbit_params[1], y, yerr=yerr, fmt='ko', ecolor='C3',lw=2, zorder=3,markersize=4, label='data, phasefolded')
        im = ax1.scatter(((2*np.pi*median_orbit_params[1]*(time-np.min(time)))%(2*np.pi))/(2*np.pi)/median_orbit_params[1], y, s=30, color='k')
        ax1.plot(phase_array/(2*np.pi)/median_orbit_params[1], y_true, lw=2., label='Keplerian orbit')

        ax2.errorbar(((2*np.pi*median_orbit_params[1]*(time-np.min(time)))%(2*np.pi))/(2*np.pi)/median_orbit_params[1], y-RVmodel(truths,time,np.min(time)), yerr=yerr, fmt='ko', ecolor='C3',lw=2, zorder=3,markersize=4, label='data, phasefolded')
        im = ax2.scatter(((2*np.pi*truths[1]*(time-np.min(time)))%(2*np.pi))/(2*np.pi)/truths[1], y-RVmodel(truths,time,np.min(time)), s=30, color='k')#c=bin_pointings[k],
        ax2.plot([0,1/truths[1]],[0,0],linestyle='dashed',color='lightgrey')
        ax2.set_xlabel('time [d] (phase-folded)')
    else:
        ax1.errorbar(time, y, yerr=yerr,zorder=2,lw=0,elinewidth=1,ecolor='black')
        ax1.scatter(time, y, zorder=3, s=35, label='data',color='black',edgecolor='k',lw=0.3)
        ax1.plot(time_array,RVmodel(median_orbit_params, time_array, np.min(time_array)),label='median',color='crimson')
        #ax1.plot(time_array,RVmodel(mean_orbit_params, time_array),label='mean',lw=0.8)
        #ax1.plot(time_array,RVmodel(maxlogl_orbit_params, time_array),label='max logl',lw=0.8)
        #ax1.plot(time_array,RVmodel(maxpost_orbit_params, time_array),label='MAP',color='dodgerblue')
        #ax1.plot(time_array,RVmodel(guess_orbit_params, time_array),label='initial guess',lw=0.8)

        ax2.errorbar(time, y-RVmodel(truths,time,np.min(time)),yerr=yerr,lw=0,elinewidth=1,ecolor='black')
        ax2.scatter(time, y-RVmodel(truths,time,np.min(time)),zorder=3, s=35, label='data',color='black',edgecolor='k',lw=0.3)
        ax2.plot(np.linspace(np.min(time)-10,np.max(time)+10,200), np.zeros(200),linestyle='dashed',color='lightgrey')


        ax1.set_ylabel(r'$v_{\mathrm{rad}}$ [km/s]')
        ax2.set_ylabel(r'$v_{\mathrm{rad}} -$ model [km/s]')
        ax2.set_xlabel(r'MJD [d]')

        ax1.legend(loc='upper left')
        #ax2.legend()

        plt.tight_layout()
        plt.savefig(path_export+f'plot_orbit_{plot_id}.pdf')
        #plt.show()

        plt.close()

def plot_band(plot_id):
    textargs = dict(horizontalalignment='center',verticalalignment='bottom')
    plt.rcParams.update({'font.size': 12})

    y      = spectra_sample.df.loc[lambda df: df['Star_Id'] == plot_id,:]['STAR V'].values #data[k] #+18.8
    yerr   = spectra_sample.df.loc[lambda df: df['Star_Id'] == plot_id,:]['STAR V err'].values#errs[k]
    time   = spectra_sample.df.loc[lambda df: df['Star_Id'] == plot_id,:]['MJD-OBS'].values#times[k]

    time_array = np.linspace(np.min(time),np.max(time),2000)#np.max(time),1000)

    fig, ax1 = plt.subplots(1,1,figsize=(8, 4))
    band =PredictionBand(time_array)
    for sample in samples:
        band.add(RVmodel(sample,time_array,np.min(time_array)))
    band.line(color='midnightblue', label='UltraNest median ')
    band.shade(color='midnightblue', alpha=0.3,label=r'$1-\sigma$')
    # add wider quantile (0.01 .. 0.99)
    band.shade(q=0.455, color='cornflowerblue', alpha=0.3,label=r'$2-\sigma$')
    ax1.errorbar(time, y, yerr=yerr,zorder=2,lw=0,elinewidth=1,ecolor='black')
    ax1.scatter(time, y, zorder=3, s=35, label='data',color='black',edgecolor='k',lw=0.3)
    #ax1.set_ylim(-110,70)
    ax1.set_ylabel(r'$v_{\mathrm{rad}}$ [km/s]')
    ax1.set_xlabel(r'MJD [d]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(path_export+f'plot_prediction_band_{plot_id}.pdf')
    #plt.show()
    plt.close()

def plot_summary(plot_id, mean_logP=None,sigma_logP=None,n_modes=None):

    textargs = dict(horizontalalignment='center',verticalalignment='bottom')

    #plt.close()
    # true values
    y      = spectra_sample.df.loc[lambda df: df['Star_Id'] == plot_id,:]['STAR V'].values #data[k] #+18.8
    yerr   = spectra_sample.df.loc[lambda df: df['Star_Id'] == plot_id,:]['STAR V err'].values#errs[k]
    time   = spectra_sample.df.loc[lambda df: df['Star_Id'] == plot_id,:]['MJD-OBS'].values#times[k]
    print(np.min(np.sort(time)[1:]-np.sort(time)[:-1]))

    fit_vals = maxpost_orbit_params # median_orbit_params
    fig = plt.figure(constrained_layout=True, figsize=[12,6])
    #fig.suptitle(r'$P=${:.1f} d, $e=${:.2f}'.format(1./median_orbit_params[1], median_orbit_params[3]),
    #             fontsize=14)#, x=0.57,y=0.93)
    gs = GridSpec(2, 6, figure=fig)
    ax1 = fig.add_subplot(gs[0, :4])
    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
    ax2 = fig.add_subplot(gs[0, 4:])
    ax3 = fig.add_subplot(gs[1, 4:])
    ax4 = fig.add_subplot(gs[1, :2])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[1, 3])

    ax1.errorbar(time, y, yerr=yerr,zorder=2, linestyle='None',elinewidth=0.8,ecolor='black')
    ax1.scatter(time, y, zorder=3, s=30, label='data',color='black')

    phase_array = np.linspace(0.,2*np.pi,1000)
    y_phase    = RVmodel(fit_vals, phase_array/(2*np.pi*fit_vals[1]),np.min(phase_array/(2*np.pi*fit_vals[1])))
    ax2.errorbar(((2*np.pi*fit_vals[1]*(time-np.min(time)))%(2*np.pi))/fit_vals[1]/(2*np.pi), y, yerr=yerr, fmt='k', ecolor='black',elinewidth=1, linestyle='None',markersize=4, label='data, phase-folded')
    ax2.scatter(((2*np.pi*fit_vals[1]*(time-np.min(time)))%(2*np.pi))/fit_vals[1]/(2*np.pi), y, zorder=3, s=30,color='k')
    ax2.plot(phase_array/fit_vals[1]/(2*np.pi), y_phase, lw=2., label='median',color='crimson')

    ax3.errorbar(((2*np.pi*fit_vals[1]*(time-np.min(time)))%(2*np.pi))/fit_vals[1]/(2*np.pi), y - RVmodel(fit_vals,time,np.min(time)), yerr=yerr,zorder=2,lw=0,elinewidth=1,ecolor='black')
    ax3.scatter(((2*np.pi*fit_vals[1]*(time-np.min(time)))%(2*np.pi))/fit_vals[1]/(2*np.pi), y - RVmodel(fit_vals,time,np.min(time)), zorder=3, s=30, label='data, phase-folded',color='black')
    ax3.plot(np.linspace(0,1/fit_vals[1],100), np.zeros(100),linestyle='dashed',color='lightgrey')


    tt = np.linspace(np.min(time)-50,np.max(time)+50,2500)
    # go through the solutions
    #with open(path_results+'{}/info/results.json'.format(bin_IDs[k])) as d:
    #    results = json.load(d)
    print('Loglike: ', results['maximum_likelihood']['logl'],loglike(results['posterior']['median'],time,y,yerr),loglike(fit_vals,time,y,yerr))
    #post_samples[:,[0,1]] = 10**post_samples[:,[0,1]]

    weights = np.ones(len(samples[:,1]))/len(samples[:,1])
    for params in samples[::10]:
        ax1.plot(tt,RVmodel(params,tt, np.min(time)),alpha=0.8,color='cornflowerblue',zorder=1,lw=0.3)
    #ax1.plot(tt,RVmodel(gaia_orbit_params, tt, np.min(time)),label='median',color='dimgrey',alpha=0.7, zorder=2)
    ax1.plot(tt,RVmodel(fit_vals, tt, np.min(time)),label='median',color='crimson',zorder=2)

    ax4.scatter(1./samples[:,1],samples[:,3],alpha=0.8,s=15,facecolor='cornflowerblue',
                 edgecolor='k')
    ax5.hist(samples[:,3],weights=weights,bins=7,color='cornflowerblue',edgecolor='k')
    ax6.hist(1./samples[:,1],weights=weights,bins=7,color='cornflowerblue',edgecolor='k')
    #ax6.set_xscale('log')
    ax4.set_ylabel(r'$e$')
    ax4.set_xlabel(r'$P$ [d]')
    #ax4.set_xscale('log')
    ax1.set_xlabel(r'$t - t_{\mathrm{initial}}$ [d]')
    ax2.set_xlabel(r'time [d]')
    ax3.set_xlabel(r'time [d]')
    ax5.set_xlabel(r'$e$')
    ax6.set_xlabel(r'$P$ [d]')
    ax1.set_ylabel(r'$v_{\mathrm{rad}}$ [km/s]')
    ax2.set_ylabel(r'$v_{\mathrm{rad}}$ [km/s]')
    ax3.set_ylabel(r'$v_{\mathrm{rad}}-$ model [km/s]')
    ax5.set_ylabel(r'count')
    ax6.set_ylabel(r'count')
    #ax5.set_xlim(0.32,0.68)
    #ax6.set_xlim(210,230)
    #ax4.set_xlim(210,230)
    #plt.tight_layout()#pad=0.4, w_pad=0.5, h_pad=0.5)
    plt.savefig(path_export+f'plot_summary_{plot_id}.pdf')
    plt.show()
################################################################################
# load data: y,yerr,t (and orbital parameters)

# path names
path_data    = './'    #location of input data files
path_results = f'{plot_id}/'
path_export = f'{plot_id}/plots/'

spectra_sample = spectra_sample(data_path+'spectra_sample.csv')
all_star_ids = sorted(set(spectra_sample.df['Star_Id']))
print(spectra_sample.df)
print(f'fitting {len(all_star_ids)} binaries with {len(spectra_sample.df)} spectra...')

################################################################################


for star_id in all_star_ids:

    with open(path_results+'info/results.json') as d:
        results = json.load(d)
    samples = np.genfromtxt(path_results+'chains/equal_weighted_post.txt', skip_header=1)
    samples = samples[::50]
    median_orbit_params  = results['posterior']['median']
    mean_orbit_params    = results['posterior']['mean']
    maxlogl_orbit_params = results['maximum_likelihood']['point']
    maxpost_orbit_params = MAP(plot_id)

    plot_sample(plot_id,phase_plot=True)
    plot_band(plot_id)
    plot_summary(plot_id)

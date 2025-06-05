import numpy as np
import pandas as pd
import os

from orbiNest.model import *


from ultranest.plot import PredictionBand

import matplotlib.pyplot as plt
import numpy as np

def plot_sample(star_id, rvs, rvs_err, times, orbitfit, phase=False, plot_path='./',T_ref=51544.):
    truths = orbitfit.results['maximum_likelihood']['point']
    time_array = np.linspace(np.min(times), np.max(times), 10000)
    phase_array = np.linspace(0., 2 * np.pi, 1000)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.suptitle(r'$P=${:.1f} d, $e=${:.2f}'.format(truths[1], truths[3]), fontsize=14, x=0.57, y=0.93)

    if phase:
        phase_array = np.linspace(0.,1,1000)
        # Compute fitted time of periastron passage
        Tp = T_ref + truths[2]*truths[1]

        # Compute model at phase values
        time_model = phase_array * truths[1] + Tp
        y_phase = rv_model(truths, time_model).reshape(-1)

        # Compute data phases
        phases = ((times - Tp) / truths[1]) % 1  # Ensure phase is between 0 and 1
        # Plot data points (error bars and scatter)
        ax1.errorbar(phases*truths[1], rvs, yerr=rvs_err, fmt='ko', ecolor='C3', lw=2, zorder=3, markersize=4)
        ax1.plot(phase_array*truths[1], y_phase, lw=2., label='Keplerian orbit')

        ax2.errorbar(phases*truths[1], rvs - rv_model(truths, times, T_ref=51544.).reshape(-1),
                     yerr=rvs_err, fmt='ko', ecolor='C3', lw=2, zorder=3, markersize=4)
        ax2.axhline(0, linestyle='dashed', color='lightgrey')
        ax2.set_xlabel('time [d] (phase-folded)')
    else:
        ax1.errorbar(times, rvs, yerr=rvs_err, fmt='none', ecolor='black', lw=1)
        ax1.scatter(times, rvs, zorder=3, s=35, label='data', color='black', edgecolor='k', lw=0.3)
        ax1.plot(time_array, rv_model(truths, time_array, T_ref=51544.).reshape(-1), label='median', color='crimson')

        ax2.errorbar(times, rvs - rv_model(truths, times, T_ref=51544.).reshape(-1), yerr=rvs_err, fmt='none', ecolor='black', lw=1)
        ax2.scatter(times, rvs - rv_model(truths, times, T_ref=51544.).reshape(-1), s=35, color='black', edgecolor='k', lw=0.3)
        ax2.axhline(0, linestyle='dashed', color='lightgrey',zorder=0)

        ax2.set_xlabel(r'MJD [d]')

    ax1.set_ylabel(r'$v_{\mathrm{rad}}$ [km/s]')
    ax2.set_ylabel(r'$v_{\mathrm{rad}} -$ model [km/s]')
    ax1.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(plot_path+f'plot_orbit_{"phase" if phase else "time"}.pdf')
    plt.close()


def plot_prediction_band(star_id, rvs, rvs_err, times, orbitfit, plot_path='./'):
    textargs = dict(horizontalalignment='center',verticalalignment='bottom')
    plt.rcParams.update({'font.size': 12})
    time_array = np.linspace(np.min(times),np.max(times),2000)#np.max(time),1000)

    fig, ax1 = plt.subplots(1,1,figsize=(8, 4))
    band =PredictionBand(time_array)
    for sample in orbitfit.samples[::10]:
        band.add(rv_model(sample,time_array,T_ref=51544).reshape(-1))
    band.line(color='midnightblue', label='UltraNest median ')
    band.shade(color='midnightblue', alpha=0.3,label=r'$1-\sigma$')
    # add wider quantile (0.01 .. 0.99)
    band.shade(q=0.455, color='cornflowerblue', alpha=0.3,label=r'$2-\sigma$')
    ax1.errorbar(times, rvs, yerr=rvs_err,zorder=2,lw=0,elinewidth=1,ecolor='black')
    ax1.scatter(times, rvs, zorder=3, s=35, label='data',color='black',edgecolor='k',lw=0.3)
    #ax1.set_ylim(-110,70)
    ax1.set_ylabel(r'$v_{\mathrm{rad}}$ [km/s]')
    ax1.set_xlabel(r'MJD [d]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(plot_path+f'plot_band_{star_id}.pdf')
    #plt.show()
    plt.close()

from matplotlib.gridspec import GridSpec

def plot_summary(star_id, rvs, rvs_err, times, orbitfit, plot_path='./',T_ref=51544.):
    samples = orbitfit.samples
    fit_vals = orbitfit.results['maximum_likelihood']['point']

    fig = plt.figure(constrained_layout=True, figsize=[12, 6])
    gs = GridSpec(2, 6, figure=fig)

    ax1 = fig.add_subplot(gs[0, :4])
    ax2 = fig.add_subplot(gs[0, 4:])
    ax3 = fig.add_subplot(gs[1, 4:])
    ax4 = fig.add_subplot(gs[1, :2])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[1, 3])

    tt = np.linspace(np.min(times) - 50, np.max(times) + 50, 2500)

    ax1.errorbar(times, rvs, yerr=rvs_err, fmt='none', elinewidth=0.8, ecolor='black')
    ax1.scatter(times, rvs, s=30, color='black', label='data')
    for params in samples[::50]:
        ax1.plot(tt, rv_model(params, tt, T_ref=51544.).reshape(-1), alpha=0.3, color='cornflowerblue', lw=0.5)
    ax1.plot(tt, rv_model(fit_vals, tt, T_ref=51544.).reshape(-1), label='median', color='crimson')

    # Phase-folded orbit and residuals
    phase_array = np.linspace(0.,1,1000)
    # Compute fitted time of periastron passage
    Tp = T_ref + fit_vals[2]*fit_vals[1]

    # Compute model at phase values
    time_model = phase_array * fit_vals[1] + Tp
    y_phase = rv_model(fit_vals, time_model).reshape(-1)

    # Compute data phases
    phases = ((times - Tp) / fit_vals[1]) % 1  # Ensure phase is between 0 and 1
    # Plot data points (error bars and scatter)
    ax2.errorbar(phases*fit_vals[1], rvs, yerr=rvs_err, fmt='ko', ecolor='C3', lw=2, zorder=3, markersize=4)
    ax2.plot(phase_array*fit_vals[1], y_phase, lw=2., label='Keplerian orbit')

    ax3.errorbar(phases*fit_vals[1], rvs - rv_model(fit_vals, times, T_ref=51544.).reshape(-1),
                 yerr=rvs_err, fmt='ko', ecolor='C3', lw=2, zorder=3, markersize=4)

    ax3.axhline(0, linestyle='dashed', color='lightgrey',zorder=0)

    # Histograms & scatter
    weights = np.ones(len(samples)) / len(samples)
    ax4.scatter(samples[:, 1], samples[:, 3], alpha=0.8, s=15, facecolor='cornflowerblue', edgecolor='k')
    ax5.hist(samples[:, 3], weights=weights, bins=7, color='cornflowerblue', edgecolor='k')
    ax6.hist(samples[:, 1], weights=weights, bins=7, color='cornflowerblue', edgecolor='k')

    # Labels
    ax1.set_xlabel(r'MJD [d]')
    ax1.set_ylabel(r'$v_{\mathrm{rad}}$ [km/s]')
    ax2.set_xlabel('Phase')
    ax2.set_ylabel(r'$v_{\mathrm{rad}}$ [km/s]')
    ax3.set_xlabel('Phase')
    ax3.set_ylabel(r'$v_{\mathrm{rad}} -$ model [km/s]')
    ax4.set_xlabel(r'$P$ [d]')
    ax4.set_ylabel(r'$e$')
    ax5.set_xlabel(r'$e$')
    ax6.set_xlabel(r'$P$ [d]')
    ax5.set_ylabel('Count')
    ax6.set_ylabel('Count')

    plt.savefig(plot_path+f'plot_summary_{star_id}.pdf')
    plt.close()

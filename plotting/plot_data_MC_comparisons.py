#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import dask
from dask import delayed, multiprocessing, compute, bag
from dask.diagnostics import ProgressBar

from icecube.weighting.weighting import from_simprod
from icecube import dataclasses

import comptools as comp
import comptools.analysis.plotting as plotting


def plot_rate(array, weights, bins, xlabel=None, color='C0',
              label=None, legend=True, alpha=0.8, ax=None):

    if ax is None:
        ax = plt.gca()

    # Remove inf and nan values
    array = check_array_finite(array)
    weights = check_array_finite(weights, check_array=array)

    rate = np.histogram(array, bins=bins, weights=weights)[0]
    rate_err = np.sqrt(np.histogram(array, bins=bins, weights=weights**2)[0])
    plotting.plot_steps(bins, rate, yerr=rate_err, color=color,
                        label=label, alpha=alpha, ax=ax)
    ax.set_yscale('log', nonposy='clip')
    ax.set_ylabel('Rate [Hz]')
    if xlabel:
        ax.set_xlabel(xlabel)
    if legend:
        ax.legend()
    ax.grid(True)

    return ax


def check_array_finite(array, check_array=None):
    if check_array is None:
        isfinite_mask = np.isfinite(array)
    else:
        isfinite_mask = np.isfinite(check_array)

    return array[isfinite_mask]


def plot_data_MC_ratio(sim_array, sim_weights, data_array, data_weights, bins,
                       xlabel=None, color='C0', alpha=0.8, label=None,
                       legend=False, ylim=None, ax=None):

    if ax is None:
        ax = plt.gca()

    # Remove inf and nan values
    sim_array = check_array_finite(sim_array)
    sim_weights = check_array_finite(sim_weights, check_array=sim_array)
    data_array = check_array_finite(data_array)
    data_weights = check_array_finite(data_weights, check_array=data_array)

    sim_rate = np.histogram(sim_array, bins=bins, weights=sim_weights)[0]
    sim_rate_err = np.sqrt(np.histogram(sim_array, bins=bins, weights=sim_weights**2)[0])

    data_rate = np.histogram(data_array, bins=bins, weights=data_weights)[0]
    data_rate_err = np.sqrt(np.histogram(data_array, bins=bins, weights=data_weights**2)[0])

    ratio, ratio_err = comp.analysis.ratio_error(data_rate, data_rate_err, sim_rate, sim_rate_err)

    plotting.plot_steps(bins, ratio, yerr=ratio_err,
                        color=color, label=label, alpha=alpha, ax=ax)
    ax.grid(True)
    ax.set_ylabel('Data/MC')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylim:
        ax.set_ylim(ylim)
    if legend:
        ax.legend()

    ax.axhline(1, marker='None', ls='-.', color='k')

    return ax


def plot_data_MC_comparison(df_sim, df_data, var_key, var_bins, var_label,
                            livetime, comp_list=None, ylim_ratio=None):

    if comp_list is None:
        comp_list = ['PPlus', 'Fe56Nucleus']

    color_dict = comp.analysis.get_color_dict()

    gs = gridspec.GridSpec(2, 1, height_ratios=[2,1], hspace=0.0)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    for composition in comp_list:
        comp_mask = df_sim['MC_comp'] == composition
        sim_variable = df_sim[comp_mask][var_key]
        sim_weights = df_sim[comp_mask]['weights']
        # Add simulation rate distribution to ax1
        plot_rate(sim_variable, sim_weights, bins=var_bins,
                  color=color_dict[composition], label=composition, ax=ax1)

        # Add data/MC rate ratio distribution to ax2
        data_weights = np.array([1/livetime]*len(df_data[var_key]))
        ax2 = plot_data_MC_ratio(sim_variable, sim_weights,
                           df_data[var_key], data_weights, var_bins,
                           xlabel=var_label, color=color_dict[composition],
                           label=composition, ax=ax2)
        if ylim_ratio:
            ax2.set_ylim(ylim_ratio)

    # Add data rate distribution to ax1
    plot_rate(df_data[var_key], data_weights, bins=var_bins,
              color=color_dict['data'], label='Data', ax=ax1)

    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode='expand', borderaxespad=0., frameon=False)

    return gs


def flux(E):
    ''' Broken power law flux

    This is a "realistic" flux (simple broken power law with a knee @ 3PeV)
    to weight simulation to. More information can be found on the
    IT73-IC79 data-MC comparison wiki page
    https://wiki.icecube.wisc.edu/index.php/IT73-IC79_Data-MC_Comparison
    '''
    phi_0 = 3.6e-6
    # phi_0 = 3.1e-6
    # phi_0 = 2.95e-6
    gamma_1 = -2.7
    # gamma_2 = -3.0
    gamma_2 = -3.1
    eps = 100
    E = np.array(E) * 1e-6

    return (1e-6) * phi_0 * E**gamma_1 *(1+(E/3.)**eps)**((gamma_2-gamma_1)/eps)


def get_sim_weights(df_sim):

    simlist = np.unique(df_sim['sim'])
    print('simlist = {}'.format(simlist))
    for i, sim in enumerate(simlist):
        gcd_file, sim_files = comp.simfunctions.get_level3_sim_files(sim)
        num_files = len(sim_files)
        print('Simulation set {}: {} files'.format(sim, num_files))
        if i == 0:
            generator = num_files*from_simprod(int(sim))
        else:
            generator += num_files*from_simprod(int(sim))
    energy = df_sim['MC_energy'].values
    ptype = df_sim['MC_type'].values
    num_ptypes = np.unique(ptype).size
    cos_theta = np.cos(df_sim['MC_zenith']).values
    weights = 1.0/generator(energy, ptype, cos_theta)

    return weights


def save_data_MC_plots(config, june_july_only):

    df_sim = comp.load_sim(config='IC86.2012', test_size=0, verbose=False)
    # energy_mask_sim = (df_sim['lap_log_energy'] > 6.0)
    # energy_mask_sim = (df_sim['lap_log_energy'] > 6.4) & (df_sim['lap_log_energy'] < 8.0)
    # df_sim = df_sim[energy_mask_sim]

    df_data = comp.load_data(config=config, verbose=False)
    df_data = df_data[np.isfinite(df_data['log_dEdX'])]
    # energy_mask_data = (df_data['lap_log_energy'] > 6.4) & (df_data['lap_log_energy'] < 8.0)
    # df_data = df_data[energy_mask_data]

    if june_july_only:
        print('Masking out all data events not in June or July')
        def is_june_july(time):
            i3_time = dataclasses.I3Time(time)
            return i3_time.date_time.month in [6, 7]
        june_july_mask = df_data.end_time_mjd.apply(is_june_july)
        df_data = df_data[june_july_mask].reset_index(drop=True)

    months = (6, 7) if june_july_only else None
    livetime, livetime_err = comp.get_detector_livetime(config, months=months)

    weights = get_sim_weights(df_sim)
    df_sim['weights'] = flux(df_sim['MC_energy'])*weights

    MC_comp_mask = {}
    comp_list = ['PPlus', 'Fe56Nucleus']
    for composition in comp_list:
        MC_comp_mask[composition] = df_sim['MC_comp'] == composition
    #     MC_comp_mask[composition] = df_sim['MC_comp_class'] == composition

    # S125 data-MC plot
    log_s125_bins = np.linspace(-0.5, 3.5, 50)
    gs_s125 = plot_data_MC_comparison(df_sim, df_data, 'log_s125',
                log_s125_bins, '$\mathrm{\log_{10}(S_{125})}$',
                livetime, ylim_ratio=(0, 2))
    s125_outfile = os.path.join(comp.paths.figures_dir, 'data-MC-comparison',
                                's125_{}.png'.format(config))
    plt.savefig(s125_outfile)

    # dE/dX data-MC plot
    log_dEdX_bins = np.linspace(-2, 4, 50)
    gs_dEdX = plot_data_MC_comparison(df_sim, df_data, 'log_dEdX',
                log_dEdX_bins, '$\mathrm{\log_{10}(dE/dX)}$',
                livetime, ylim_ratio=(0, 5.5))
    dEdX_outfile = os.path.join(comp.paths.figures_dir, 'data-MC-comparison',
                                'dEdX_{}.png'.format(config))
    plt.savefig(dEdX_outfile)

    # cos(zenith) data-MC plot
    cos_zenith_bins = np.linspace(0.8, 1.0, 50)
    gs_zenith = plot_data_MC_comparison(df_sim, df_data, 'lap_cos_zenith',
                cos_zenith_bins, '$\mathrm{\cos(\\theta_{reco})}$',
                livetime, ylim_ratio=(0, 3))
    zenith_outfile = os.path.join(comp.paths.figures_dir, 'data-MC-comparison',
                                'zenith_{}.png'.format(config))
    plt.savefig(zenith_outfile)

    # InIce median radius data-MC plot
    inice_radius_bins = np.linspace(0, 200, 50)
    gs_inice_radius = plot_data_MC_comparison(df_sim, df_data, 'median_inice_radius',
                inice_radius_bins, '$\mathrm{\cos(\\theta_{reco})}$',
                livetime, ylim_ratio=(0, 3))
    inice_radius_outfile = os.path.join(comp.paths.figures_dir, 'data-MC-comparison',
                                'median_inice_radius_{}.png'.format(config))
    plt.savefig(inice_radius_outfile)

    # log_d4r_peak_energy data-MC plot
    log_d4r_peak_energy_bins = np.linspace(-0.5, 3.5, 50)
    gs_d4R_peak_energy = plot_data_MC_comparison(df_sim, df_data, 'log_d4r_peak_energy',
                log_d4r_peak_energy_bins, '$\mathrm{\log_{10}(E_{D4R}/GeV)}$',
                livetime, ylim_ratio=(0, 5.5))
    d4R_peak_energy_outfile = os.path.join(comp.paths.figures_dir, 'data-MC-comparison',
                                'd4R_peak_energy_{}.png'.format(config))
    plt.savefig(d4R_peak_energy_outfile)

    # log_d4r_peak_sigma data-MC plot
    log_d4r_peak_sigma_bins = np.linspace(-1, 3, 50)
    gs_d4R_peak_sigma = plot_data_MC_comparison(df_sim, df_data, 'log_d4r_peak_sigma',
                log_d4r_peak_sigma_bins, '$\mathrm{\log_{10}(E_{D4R}/GeV)}$',
                livetime, ylim_ratio=(0, 5.5))
    d4R_peak_sigma_outfile = os.path.join(comp.paths.figures_dir, 'data-MC-comparison',
                                'd4R_peak_sigma_{}.png'.format(config))
    plt.savefig(d4R_peak_sigma_outfile)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Extracts and saves desired information from simulation/data .i3 files')
    parser.add_argument('-c', '--config', dest='config', nargs='*',
                   choices=comp.datafunctions.get_data_configs(),
                   help='Detector configuration')
    parser.add_argument('--june_july_only', dest='june_july_only',
                   default=False, action='store_true',
                   help='Option to only use June and July data')
    args = parser.parse_args()

    config_bag = bag.from_sequence(args.config)
    save_plots = config_bag.map(save_data_MC_plots, args.june_july_only)
    with ProgressBar() as bar:
        save_plots.compute(num_workers=len(args.config))

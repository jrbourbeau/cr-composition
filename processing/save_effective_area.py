#!/usr/bin/env python

from __future__ import division
from collections import defaultdict
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import comptools as comp
from comptools.analysis import plotting
color_dict = comp.analysis.get_color_dict()


def thrown_showers_per_ebin(sim_list, log_energy_bins=None):
    e_bins = []
    for f in comp.simfunctions.get_level3_sim_files_iterator(sim_list):
        start_idx = f.find('Run')
        run = int(f[start_idx+3: start_idx+9])
        e_bin = comp.simfunctions.run_to_energy_bin(run)
        e_bins.append(e_bin)

    if log_energy_bins is None:
        log_energy_bins = np.arange(5, 8.1, 0.1)
    log_energy_midpoints = (log_energy_bins[1:] + log_energy_bins[:-1]) / 2
    vals = np.histogram(e_bins, bins=log_energy_bins)[0]

    n_resamples = 100
    n_showers_per_file = n_resamples
    thrown_showers = vals * n_showers_per_file

    return thrown_showers


def sigmoid_flat(log_energy, p0, p1, p2):
    return p0 / (1 + np.exp(-p1*log_energy + p2))


def sigmoid_slant(log_energy, p0, p1, p2, p3):
    '''Fit function for effective area vs. energy

    Parameters
    ----------
    log_energy : numpy.ndarray
        Log energy values

    '''
    return (p0 + p3*log_energy) / (1 + np.exp(-p1*log_energy + p2))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Calculates, plots, and saves detector effective area')
    parser.add_argument('-s', '--sim', dest='sim', nargs='*', type=int,
                        help='Simulation to run over')
    parser.add_argument('-c', '--config', dest='config',
                        default='IC86.2012',
                        choices=comp.simfunctions.get_sim_configs(),
                        help='Detector configuration')
    parser.add_argument('-n', '--n_samples', dest='n_samples', type=int,
                        default=1000,
                        help='Number of random samples to use in calcuating the fit error')
    parser.add_argument('--n_groups', dest='n_groups', type=int,
                        default=2,
                        help='Number of composition groups to use')
    parser.add_argument('--sigmoid', dest='sigmoid', default='slant',
                        choices=['flat', 'slant'],
                        help='Sigmoid function to fit to effective area')

    args = parser.parse_args()

    if not args.sim:
        args.sim = comp.simfunctions.config_to_sim(args.config)

    num_groups = args.n_groups
    comp_list = comp.get_comp_list(num_groups=num_groups)

    energybins = comp.analysis.get_energybins()
    bins = np.arange(5, 8.1, 0.1)
    bin_midpoints = (bins[1:] + bins[:-1]) / 2
    bin_midpoints_mask = np.logical_and(
                    bin_midpoints >= energybins.log_energy_min,
                    bin_midpoints <= energybins.log_energy_max)

    df_sim = comp.load_sim(config=args.config, test_size=0,
                           log_energy_min=None, log_energy_max=None)

    geom_factor = (df_sim.lap_cos_zenith.max() + df_sim.lap_cos_zenith.min()) / 2

    # Get simulation thrown areas for each energy bin
    thrown_radii = comp.simfunctions.get_sim_thrown_radius(bin_midpoints)
    thrown_areas = np.pi * thrown_radii**2

    # Calculate efficiencies and effective areas for each composition group
    efficiencies, efficiencies_err = {}, {}
    effective_area, effective_area_err = {}, {}
    for composition in comp_list:
        # Get list of simulation sets for composition
        comp_mask = df_sim['comp_group_{}'.format(num_groups)] == composition
        sim_list = df_sim.loc[comp_mask, 'sim'].unique()
        # Get number of thrown showers and number of showers that pass cuts
        # for energy bins for sim_list
        thrown_showers = thrown_showers_per_ebin(sim_list, log_energy_bins=bins)

        passed_showers = np.histogram(df_sim.loc[comp_mask, 'MC_log_energy'],
                                      bins=bins)[0]

        efficiency, efficiency_err = comp.ratio_error(
                                    passed_showers, np.sqrt(passed_showers),
                                    thrown_showers, np.sqrt(thrown_showers),
                                    nan_to_num=False)
        efficiencies[composition] = efficiency
        efficiencies_err[composition] = efficiency_err

        # Calculate effective area from efficiencies and thrown areas
        effective_area[composition] = efficiency * thrown_areas
        effective_area_err[composition] = efficiency_err * thrown_areas

    # Fit effective area
    fit_func = sigmoid_flat if args.sigmoid == 'flat' else sigmoid_slant
    p0 = [7e4, 8.0, 50.0] if args.sigmoid == 'flat' else [7e4, 8.5, 50.0, 800]
    effective_area_fit = {}
    energy_min_fit, energy_max_fit = 5.8, 7.9
    midpoints_fitmask = np.logical_and(bin_midpoints > energy_min_fit,
                                       bin_midpoints < energy_max_fit)
    for composition in comp_list:
        popt, pcov = curve_fit(
            fit_func, bin_midpoints[midpoints_fitmask],
            effective_area[composition][midpoints_fitmask],
            sigma=effective_area_err[composition][midpoints_fitmask], p0=p0)
        eff_area_fit = fit_func(bin_midpoints, *popt)
        effective_area_fit[composition] = eff_area_fit

        chi2 = np.sum((effective_area[composition][midpoints_fitmask] - eff_area_fit[midpoints_fitmask])**2 / (effective_area_err[composition][midpoints_fitmask]) ** 2)
        ndof = len(eff_area_fit[midpoints_fitmask]) - len(p0)
        print('({}) chi2 / ndof = {} / {} = {}'.format(composition, chi2,
                                                       ndof, chi2/ndof))

    # Perform several fits to random fluxuations of the effective area
    effective_area_sample_fits = defaultdict(list)
    for _ in range(args.n_samples):
        for composition in comp_list:
            # Get new random sample to fit
            eff_area_sample = np.random.normal(effective_area_fit[composition][midpoints_fitmask],
                                               effective_area_err[composition][midpoints_fitmask])
            # Fit with error bars
            popt, pcov = curve_fit(fit_func, bin_midpoints[midpoints_fitmask],
               eff_area_sample, p0=p0,
               sigma=effective_area_err[composition][midpoints_fitmask])

            eff_area_fit_sample = fit_func(bin_midpoints, *popt)
            effective_area_sample_fits[composition].append(eff_area_fit_sample)

    # Calculate median and error of effective area fit
    eff_area_fit = pd.DataFrame()
    eff_fit = pd.DataFrame()
    for composition in comp_list:
        fit_median, fit_err_low, fit_err_high = np.percentile(
            effective_area_sample_fits[composition], (50, 16, 84), axis=0)
        fit_err_low = np.abs(fit_err_low - fit_median)
        fit_err_high = np.abs(fit_err_high - fit_median)

        eff_area_fit['eff_area_median_{}'.format(composition)] = fit_median
        eff_area_fit['eff_area_err_low_{}'.format(composition)] = fit_err_low
        eff_area_fit['eff_area_err_high_{}'.format(composition)] = fit_err_high

        # Calculate fit efficiencies from the fit effective areas
        eff_fit['eff_median_{}'.format(composition)] = fit_median / thrown_areas
        eff_fit['eff_err_low_{}'.format(composition)] = fit_err_low / thrown_areas
        eff_fit['eff_err_high_{}'.format(composition)] = fit_err_high / thrown_areas

    # Save fit effective areas and efficiencies to disk
    eff_area_outfile = os.path.join(comp.paths.comp_data_dir,
                                    args.config + '_sim',
                                    'effective_area_fit.hdf')
    eff_area_fit[bin_midpoints_mask].reset_index(drop=True).to_hdf(eff_area_outfile, 'dataframe')

    eff_outfile = os.path.join(comp.paths.comp_data_dir,
                               args.config + '_sim',
                               'efficiency_fit.hdf')
    eff_fit[bin_midpoints_mask].reset_index(drop=True).to_hdf(eff_outfile, 'dataframe')

    # Plot effective area
    fig, ax = plt.subplots()
    for composition in comp_list:
        # Plot raw binned effective area
        plotting.plot_steps(bins, effective_area[composition],
                            yerr=effective_area_err[composition],
                            color=color_dict[composition], label=composition,
                            ax=ax)
        # Plot fit effective area
        ax.errorbar(bin_midpoints, eff_area_fit['eff_area_median_{}'.format(composition)],
                    yerr=[eff_area_fit['eff_area_err_low_{}'.format(composition)],
                          eff_area_fit['eff_area_err_high_{}'.format(composition)]],
                    marker='.', ls=':', color=color_dict[composition],
                    alpha=0.9, label=composition + ' (fit)')

    ax.axvline(6.4, marker='None', ls='-.', color='k')
    ax.set_xlabel('$\mathrm{\log_{10}(E_{true}/GeV)}$')
    ax.set_ylabel('Effective area [$\mathrm{m^2}$]')
    ax.set_xlim(bins.min(), 8.0)
    ax.set_ylim(0)
    # ax.set_yscale("log", nonposy='clip')
    ax.ticklabel_format(style='sci',axis='y')
    ax.yaxis.major.formatter.set_powerlimits((0,0))
    ax.grid()
    ax.legend(title='True composition')
    outfile = os.path.join(comp.paths.figures_dir,
                           'effarea_{}_{}-sigmoid.png'.format(args.config,
                                                              args.sigmoid))
    plt.savefig(outfile)

    # Plot efficiencies
    fig, ax = plt.subplots()
    for composition in comp_list:
        # Plot raw binned effective area
        plotting.plot_steps(bins, efficiencies[composition],
                            yerr=efficiencies_err[composition],
                            color=color_dict[composition], label=composition,
                            ax=ax)
        # Plot fit effective area
        ax.errorbar(bin_midpoints, eff_fit['eff_median_{}'.format(composition)],
                    yerr=[eff_fit['eff_err_low_{}'.format(composition)],
                          eff_fit['eff_err_high_{}'.format(composition)]],
                    marker='.', ls=':', color=color_dict[composition],
                    alpha=0.9, label=composition + ' (fit)')

    ax.axvline(6.4, marker='None', ls='-.', color='k')
    ax.set_xlabel('$\mathrm{\log_{10}(E_{true}/GeV)}$')
    ax.set_ylabel('Detection efficiency \n($\mathrm{N_{passed}/N_{thrown}}$)')
    ax.set_xlim(bins.min(), 8.0)
    ax.set_ylim(0)
    # ax.set_yscale("log", nonposy='clip')
    ax.ticklabel_format(style='sci',axis='y')
    ax.yaxis.major.formatter.set_powerlimits((0,0))
    ax.grid()
    ax.legend(title='True composition')
    outfile = os.path.join(comp.paths.figures_dir,
                           'efficiencies_{}_{}-sigmoid.png'.format(args.config,
                                                              args.sigmoid))
    plt.savefig(outfile)

    # # Plot correspondingn detector efficiency
    # def eff_area_to_efficiency(eff_area, eff_area_err_low, eff_area_err_upper):
    #     eff = eff_area / thrown_areas
    #     eff_err_low = eff_area_err_low / thrown_areas
    #     eff_err_low = eff_area_err_upper / thrown_areas
    #     # eff = eff_area / (thrown_areas * geom_factor)
    #     # eff_err_low = eff_area_err_low / (thrown_areas * geom_factor)
    #     # eff_err_low = eff_area_err_upper / (thrown_areas * geom_factor)
    #     return eff, eff_err_low, eff_err_low
    #
    # fig, ax = plt.subplots()
    # efficiencies_pyunfold = np.empty(len(energybins.log_energy_midpoints)*2)
    # efficiencies_err_pyunfold = np.empty_like(efficiencies_pyunfold)
    # for composition in ['light', 'heavy']:
    #
    #     plotting.plot_steps(bins, efficiencies[composition],
    #                         yerr=efficiencies_err[composition],
    #                         color=color_dict[composition], label=composition,
    #                         ax=ax)
    #
    #     eff, eff_err_low, eff_err_high = eff_area_to_efficiency(
    #                         eff_area_fit_params[composition]['median'],
    #                         eff_area_fit_params[composition]['err_low'],
    #                         eff_area_fit_params[composition]['err_high'])
    #     ax.errorbar(bin_midpoints, eff, yerr=[eff_err_low, eff_err_high],
    #                 color=color_dict[composition], label=composition + '(fit)',
    #                 marker='.', ls=':')
    #
    #     # Save efficiency array for unfolding
    #     start_idx = 0 if composition == 'light' else 1
    #     efficiencies_pyunfold[start_idx::2] = eff[bin_midpoints_mask]
    #     efficiencies_err_pyunfold[start_idx::2] = eff_err_low[bin_midpoints_mask]
    #
    # eff_outfile = os.path.join(comp.paths.comp_data_dir, 'unfolding',
    #                            'efficiency.npy')
    # np.save(eff_outfile, efficiencies_pyunfold)
    #
    # eff_err_outfile = os.path.join(comp.paths.comp_data_dir, 'unfolding',
    #                                'efficiency-err.npy')
    # np.save(eff_err_outfile, efficiencies_err_pyunfold)
    #
    # ax.axvline(6.4, marker='None', ls='-.', color='k')
    # ax.set_xlabel('$\mathrm{\log_{10}(E_{true}/GeV)}$')
    # ax.set_ylabel('Detection efficiency \n($\mathrm{N_{passed}/N_{thrown}}$)')
    # ax.set_xlim(bins.min(), 8.0)
    # ax.set_ylim(0)
    # # ax.set_yscale("log", nonposy='clip')
    # ax.ticklabel_format(style='sci',axis='y')
    # ax.yaxis.major.formatter.set_powerlimits((0,0))
    # ax.grid()
    # ax.legend(title='True composition', fontsize=14)
    # outfile = os.path.join(comp.paths.figures_dir,
    #                        'efficiency-fit.png')
    # plt.savefig(outfile)

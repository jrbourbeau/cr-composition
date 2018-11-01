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

color_dict = comp.get_color_dict()


def sim_file_to_run(file):
    """Extracts run number from a simulation file path

    Parameters
    ----------
    file : str
        Simulation file path.

    Returns
    -------
    run : int
        Run number for simulation file

    Examples
    --------
    >>> file = '/data/ana/CosmicRay/IceTop_level3/sim/IC79/7241/Level3_IC79_7241_Run005347.i3.gz'
    >>> sim_file_to_run(file)
    5347
    """
    start_idx = file.find('Run')
    run = int(file[start_idx+3: start_idx+9])
    return run


def thrown_showers_per_ebin(sim_list, log_energy_bins=None):
    """Calculate the number of thrown showers in each energy bin

    Parameters
    ----------
    sim_list : array_like
        Sequence of simulation dataset numbers.
    log_energy_bins : array_like or None, optional
        Log energy bins to use (defaults to np.arange(5, 8.1, 0.1)).

    Returns
    -------
    thrown_showers : np.ndarray
        Array containing the number of thrown showers in each energy bin.
    """
    e_bins = []
    for sim in sim_list:
        file_iter = comp.simfunctions.get_level3_sim_files_iterator(sim)
        runs = (sim_file_to_run(f) for f in file_iter)
        for run in runs:
            e_bin = comp.simfunctions.run_to_energy_bin(run, sim)
            e_bins.append(e_bin)

    if log_energy_bins is None:
        log_energy_bins = np.arange(5, 8.1, 0.1)
    vals, _ = np.histogram(e_bins, bins=log_energy_bins)

    n_resamples = 100
    n_showers_per_file = n_resamples
    thrown_showers = vals * n_showers_per_file

    return thrown_showers


def sigmoid_flat(log_energy, p0, p1, p2):
    return p0 / (1 + np.exp(-p1*log_energy + p2))


def sigmoid_slant(log_energy, p0, p1, p2, p3):
    """Fit function for effective area vs. energy

    Parameters
    ----------
    log_energy : numpy.ndarray
        Log energy values
    """
    return (p0 + p3*log_energy) / (1 + np.exp(-p1*log_energy + p2))


def generate_fit_func(degree=0):
    """Returns polynomial sigmoid fit function for efficiency vs. energy

    Parameters
    ----------
    degree : int, optional 
        Degree of polynomial to use (default is 0).
    """

    def fit_func(log_energy, *p): 
        """Fit function for effective area vs. energy

        Parameters
        ----------
        log_energy : numpy.ndarray
            Log energy values
        """
        extra_params = p[2:]
        # extra_term = [param * log_energy**i
        #               for i, param in zip(range(0, len(extra_params)), extra_params)]
        # np.poly1d(extra_params)(log_energy)
        return np.poly1d(p[2:])(log_energy) / (1 + np.exp(-p[0]*log_energy + p[1]))
    
    return fit_func 


def fit_efficiencies(df_file=None, config='IC86.2012', num_groups=2,
                     sigmoid='slant', n_samples=1000): 
    print('Loading df_file: {}'.format(df_file))

    comp_list = comp.get_comp_list(num_groups=num_groups)

    energybins = comp.get_energybins(config=config)
    # Want to include energy bins for energies below the normal analysis energy
    # range so we can get a better estimate of how the detector efficiencies turn on
    low_energy_bins = np.arange(5.0, energybins.log_energy_min, 0.1)
    bins = np.concatenate((low_energy_bins, energybins.log_energy_bins))
    bin_midpoints = (bins[1:] + bins[:-1]) / 2

    df_sim = comp.load_sim(df_file=df_file,
                           config=config,
                           test_size=0,
                           log_energy_min=None,
                           log_energy_max=None)

    # Thrown areas are different for different energy bin
    thrown_radii = comp.simfunctions.get_sim_thrown_radius(bin_midpoints)
    thrown_areas = np.pi * thrown_radii**2
    thrown_areas_max = thrown_areas.max()

    # Calculate efficiencies and effective areas for each composition group
    efficiencies = pd.DataFrame()
    effective_area, effective_area_err = {}, {}
    for composition in comp_list + ['total']:
        compositions = df_sim['comp_group_{}'.format(num_groups)]
        # Need list of simulation sets for composition to get number of thrown showers
        if composition == 'total':
            comp_mask = np.full_like(compositions, True)
        else:
            comp_mask = compositions == composition
        sim_list = df_sim.loc[comp_mask, 'sim'].unique()
        thrown_showers = thrown_showers_per_ebin(sim_list, log_energy_bins=bins)
        print('thrown_showers ({}) = {}'.format(composition, thrown_showers))
        passed_showers = np.histogram(df_sim.loc[comp_mask, 'MC_log_energy'], bins=bins)[0]

        efficiency, efficiency_err = comp.ratio_error(num=passed_showers,
                                                      num_err=np.sqrt(passed_showers),
                                                      den=thrown_showers,
                                                      den_err=np.sqrt(thrown_showers))

        # Calculate effective area from efficiencies and thrown areas
        effective_area[composition] = efficiency * thrown_areas
        effective_area_err[composition] = efficiency_err * thrown_areas

        # Scale efficiencies by geometric factor to take into account
        # different simulated thrown radii
        thrown_radius_factor = thrown_areas / thrown_areas_max
        efficiencies['eff_{}'.format(composition)] = efficiency * thrown_radius_factor
        efficiencies['eff_err_{}'.format(composition)] = efficiency_err * thrown_radius_factor


    # Fit sigmoid function to efficiency vs. energy distribution
    # fit_func = sigmoid_flat if sigmoid == 'flat' else sigmoid_slant
    poly_degree = 1
    num_params = poly_degree + 3
    fit_func = generate_fit_func(degree=poly_degree) 
    # p0 = [7e4, 8.0, 50.0] if sigmoid == 'flat' else [7e4, 8.5, 50.0, 800]
    init_params = [8.5, 50.0, 7e4, 800]
    p0 = np.empty(num_params)
    p0[:min(num_params, len(init_params))] = init_params[:num_params]

    efficiencies_fit = {}
    energy_min_fit, energy_max_fit = 5.8, energybins.log_energy_max
    midpoints_fitmask = np.logical_and(bin_midpoints > energy_min_fit,
                                       bin_midpoints < energy_max_fit)
    # Find best-fit sigmoid function
    for composition in comp_list + ['total']:
        eff = efficiencies.loc[midpoints_fitmask, 'eff_{}'.format(composition)]
        eff_err = efficiencies.loc[midpoints_fitmask, 'eff_err_{}'.format(composition)]
        popt, pcov = curve_fit(fit_func, bin_midpoints[midpoints_fitmask],
                               eff,
                               p0=p0,
                               sigma=eff_err)
        eff_fit = fit_func(bin_midpoints, *popt)
        efficiencies_fit[composition] = eff_fit

        chi2 = np.sum((eff - eff_fit[midpoints_fitmask])**2 / (eff_err) ** 2)
        ndof = len(eff_fit[midpoints_fitmask]) - len(p0)
        print('({}) chi2 / ndof = {} / {} = {}'.format(composition, chi2,
                                                       ndof, chi2/ndof))

    # Perform many fits to random statistical fluxuations of the best fit efficiency
    # This will be used to estimate the uncertainty in the best fit efficiency
    np.random.seed(2)
    efficiencies_fit_samples = defaultdict(list)
    for _ in xrange(n_samples):
        for composition in comp_list+['total']:
            # Get new random sample to fit
            eff_err = efficiencies.loc[midpoints_fitmask, 'eff_err_{}'.format(composition)] 
            eff_sample = np.random.normal(efficiencies_fit[composition][midpoints_fitmask],
                                          eff_err)
            # Fit with error bars
            popt, pcov = curve_fit(fit_func,
                                   bin_midpoints[midpoints_fitmask],
                                   eff_sample,
                                   p0=p0,
                                   sigma=eff_err)

            eff_fit_sample = fit_func(bin_midpoints, *popt)
            efficiencies_fit_samples[composition].append(eff_fit_sample)

    # Calculate median and error of efficiency fits
    eff_fit = pd.DataFrame()
    for composition in comp_list+['total']:
        fit_median, fit_err_low, fit_err_high = np.percentile(efficiencies_fit_samples[composition],
                                                              (50, 16, 84),
                                                              axis=0)
        fit_err_low = np.abs(fit_err_low - fit_median)
        fit_err_high = np.abs(fit_err_high - fit_median)

        eff_fit['eff_median_{}'.format(composition)] = fit_median
        eff_fit['eff_err_low_{}'.format(composition)] = fit_err_low
        eff_fit['eff_err_high_{}'.format(composition)] = fit_err_high

    return efficiencies.loc[midpoints_fitmask, :], eff_fit

    # # Plot effective area
    # fig, ax = plt.subplots()
    # for composition in comp_list:
    #     # Plot raw binned effective area
    #     comp.plot_steps(bins, effective_area[composition],
    #                     yerr=effective_area_err[composition],
    #                     color=color_dict[composition],
    #                     label=composition,
    #                     ax=ax)
    # ax.axvline(6.4, marker='None', ls='-.', color='k')
    # ax.set_xlabel('$\mathrm{\log_{10}(E_{true}/GeV)}$')
    # ax.set_ylabel('Effective area [$\mathrm{m^2}$]')
    # ax.set_xlim(bins.min(), bins.max())
    # ax.set_ylim(0)
    # ax.ticklabel_format(style='sci',axis='y')
    # ax.yaxis.major.formatter.set_powerlimits((0,0))
    # ax.grid()
    # ax.legend(title='True composition')
    # outfile = os.path.join(comp.paths.figures_dir, 'efficiencies',
    #                        'effarea_{}_{}-sigmoid_num_groups_{}.png'.format(
    #                             config, sigmoid, num_groups))
    # comp.check_output_dir(outfile)
    # plt.savefig(outfile)

    # # Plot efficiencies
    # fig, ax = plt.subplots()
    # for composition in comp_list:
    #     # Plot raw binned effective area
    #     comp.plot_steps(bins, efficiencies[composition],
    #                     yerr=efficiencies_err[composition],
    #                     color=color_dict[composition], label=composition,
    #                     ax=ax)
    #     # Plot fit effective area
    #     ax.errorbar(bin_midpoints, eff_fit['eff_median_{}'.format(composition)],
    #                 yerr=[eff_fit['eff_err_low_{}'.format(composition)],
    #                       eff_fit['eff_err_high_{}'.format(composition)]],
    #                 marker='.', ls=':', color=color_dict[composition],
    #                 alpha=0.9, label=composition + ' (fit)')
    # ax.axvline(6.4, marker='None', ls='-.', color='k')
    # ax.set_xlabel('$\mathrm{\log_{10}(E_{true}/GeV)}$')
    # ax.set_ylabel('Detection efficiency')
    # ax.set_xlim(bins.min(), bins.max())
    # ax.set_ylim(0)
    # ax.ticklabel_format(style='sci',axis='y')
    # ax.yaxis.major.formatter.set_powerlimits((0,0))
    # ax.grid()
    # ax.legend(title='True composition')
    # outfile = os.path.join(comp.paths.figures_dir, 'efficiencies',
    #                        'efficiencies_{}_{}-sigmoid_num_groups_{}.png'.format(
    #                             config, sigmoid, num_groups))
    # comp.check_output_dir(outfile)
    # plt.savefig(outfile)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Calculates, plots, and saves detector effective area')
    parser.add_argument('-c',
                        '--config',
                        dest='config',
                        default='IC86.2012',
                        choices=comp.simfunctions.get_sim_configs(),
                        help='Detector configuration')
    parser.add_argument('--df_file',
                        dest='df_file',
                        default=None,
                        help='Simulation DataFrame file(s) to load. Can contain wildcards.')
    parser.add_argument('-n',
                        '--n_samples',
                        dest='n_samples',
                        type=int,
                        default=1000,
                        help='Number of random samples to use in calcuating the fit error')
    parser.add_argument('--num_groups',
                        dest='num_groups',
                        type=int,
                        default=2,
                        help='Number of composition groups to use')
    parser.add_argument('--sigmoid',
                        dest='sigmoid',
                        default='slant',
                        choices=['flat', 'slant'],
                        help='Sigmoid function to fit to effective area')

    args = parser.parse_args()

    config = args.config
    num_groups = args.num_groups
    sigmoid = args.sigmoid
    n_samples = args.n_samples

    eff_fit = fit_efficiencies(df_file=args.df_file,
                               config=config,
                               num_groups=num_groups,
                               sigmoid=sigmoid,
                               n_samples=n_samples)
    print(eff_fit)
    
    eff_outfile = os.path.join(comp.paths.comp_data_dir,
                               config,
                               'efficiencies',
                               'efficiency_fit_num_groups_{}_sigmoid-{}.hdf'.format(num_groups, sigmoid),
                               )
    comp.check_output_dir(eff_outfile)
    # Only want to save fitted efficiencies for energies in analysis range
    bin_midpoints_mask = np.logical_and(bin_midpoints >= energybins.log_energy_min,
                                        bin_midpoints <= energybins.log_energy_max)
    # eff_fit.loc[bin_midpoints_mask, :]
    #        .reset_index(drop=True)
    #        .to_hdf(eff_outfile, 'dataframe')
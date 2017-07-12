#!/usr/bin/env python

from __future__ import division, print_function
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import multiprocessing as mp

import comptools as comp
import comptools.analysis.plotting as plotting

color_dict = comp.analysis.get_color_dict()


def get_binned_energy_counts(config):

    df_data = comp.load_dataframe(datatype='data', config=config)

    energybins = comp.analysis.get_energybins()
    counts, _ = np.histogram(df_data['lap_log_energy'],
                             bins=energybins.log_energy_bins)

    print('{} complete!'.format(config))

    return counts


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Extracts and saves desired information from simulation/data .i3 files')
    parser.add_argument('-c', '--config', dest='config', nargs='*',
                   choices=comp.datafunctions.get_data_configs(),
                   help='Detector configuration')
    args = parser.parse_args()

    # Energy distribution comparison plot
    energy_dist_pool = mp.Pool(processes=len(args.config))
    energy_counts = energy_dist_pool.map(get_binned_energy_counts, args.config)

    config_counts_dict = dict(zip(args.config, energy_counts))

    energybins = comp.analysis.get_energybins()

    gs = gridspec.GridSpec(2, 1, height_ratios=[1,1], hspace=0.1)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)

    for idx, config in enumerate(args.config):
        counts = config_counts_dict[config]
        frequency = counts/np.sum(counts)
        frequency_err = np.sqrt(counts)/np.sum(counts)
        plotting.plot_steps(energybins.log_energy_bins, frequency, yerr=frequency_err,
                            color='C{}'.format(idx), label=config, alpha=0.8,
                            ax=ax1)
    ax1.set_ylabel('Frequency')
    ax1.tick_params(labelbottom='off')
    ax1.grid()
    ax1.legend()

    for idx, config in enumerate(args.config):
        if config == 'IC86.2012': continue
        counts = config_counts_dict[config]
        frequency = counts/np.sum(counts)
        frequency_err = np.sqrt(counts)/np.sum(counts)

        counts_2012 = config_counts_dict['IC86.2012']
        frequency_2012 = counts_2012/np.sum(counts_2012)
        frequency_err_2012 = np.sqrt(counts_2012)/np.sum(counts_2012)

        ratio, ratio_err = comp.analysis.ratio_error(frequency, frequency_err,
                               frequency_2012, frequency_err_2012)

        plotting.plot_steps(energybins.log_energy_bins, ratio, yerr=ratio_err,
                            color='C{}'.format(idx), label=config, alpha=0.8,
                            ax=ax2)
    ax2.axhline(1, marker='None', linestyle='-.', color='k', lw=1.5)
    ax2.set_ylabel('$\mathrm{f/f_{2012}}$')
    # ax2.set_ylabel('Ratio with IC86.2012')
    ax2.set_xlabel('$\mathrm{\log_{10}(E_{reco}/GeV)}$')
    # ax2.set_ylim(0)
    ax2.set_xlim(energybins.log_energy_min, energybins.log_energy_max)
    ax2.grid()

    energy_dist_outfile = os.path.join(comp.paths.figures_dir,
                                'yearly_data_comparisons', 'energy_dist.png')
    comp.check_output_dir(energy_dist_outfile)
    plt.savefig(energy_dist_outfile)

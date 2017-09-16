#!/usr/bin/env python

from __future__ import division
from collections import OrderedDict
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from dask import delayed, multiprocessing
from dask.diagnostics import ProgressBar

import comptools as comp
import comptools.analysis.plotting as plotting


def data_config_to_sim_config(data_config):
    if not data_config in comp.datafunctions.get_data_configs():
        raise ValueError('Invalid data config, {}, entered...'.format(data_config))
    if 'IC86' in data_config:
        sim_config = 'IC86.2012'
    else:
        sim_config = 'IC79.2010'
    return sim_config


def get_config_flux(config):

    sim_config = data_config_to_sim_config(config)

    pipeline_str = 'BDT'
    pipeline = comp.get_pipeline(pipeline_str)
    energybins = comp.analysis.get_energybins()
    # Load simulation and training features
    df_sim_train, df_sim_test = comp.load_sim(config=sim_config, verbose=False)
    feature_list, feature_labels = comp.analysis.get_training_features()
    # Load data
    df_data = comp.load_data(config=config)
    X_data = comp.dataframe_functions.dataframe_to_array(df_data,
                                    feature_list + ['lap_log_energy'])
    log_energy = X_data[:,-1]
    X_data = X_data[:,:-1]

    pipeline.fit(df_sim_train[feature_list], df_sim_train['target'])
    data_predictions = pipeline.predict(X_data)
    # Get composition masks
    data_labels = np.array([comp.dataframe_functions.label_to_comp(pred) for pred in data_predictions])
    data_light_mask = data_labels == 'light'
    data_heavy_mask = data_labels == 'heavy'
    # Get number of identified comp in each energy bin
    df_flux = {}
    comp_list = ['light', 'heavy']
    for composition in comp_list:
        comp_mask = data_labels == composition
        df_flux['counts_' + composition] = np.histogram(log_energy[comp_mask],
                                                bins=energybins.log_energy_bins)[0]
        df_flux['counts_' + composition + '_err'] = np.sqrt(df_flux['counts_' + composition])

    df_flux['counts_total'] = np.histogram(log_energy, bins=energybins.log_energy_bins)[0]
    df_flux['counts_total_err'] = np.sqrt(df_flux['counts_total'])
    # Solid angle
    max_zenith_rad = df_sim_train['lap_zenith'].max()
    solid_angle = 2*np.pi*(1-np.cos(max_zenith_rad))
    df_flux['solid_angle'] = solid_angle
    # Livetime
    livetime, livetime_err = comp.get_detector_livetime(config=config)
    df_flux['livetime'] = livetime
    df_flux['livetime_err'] = livetime_err

    return df_flux


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculates and saves flux plot')
    parser.add_argument('-c', '--config', dest='config', nargs='*',
                   choices=comp.datafunctions.get_data_configs(),
                   help='Detector configuration')
    parser.add_argument('--correct_eff_area', dest='correct_eff_area',
                   default=False, action='store_true',
                   help='Option to normalize effective areas to IC86.2012 simulation')
    args = parser.parse_args()

    results = [delayed(get_config_flux)(config) for config in args.config]
    df_flux = delayed(pd.DataFrame)(results, index=args.config)
    with ProgressBar():
        print('Computing flux for {}'.format(args.config))
        df_flux = df_flux.compute(get=multiprocessing.get,
                                  num_workers=len(results))

    # df_flux.to_hdf('flux_dataframe.hdf', 'dataframe')
    # df_flux = pd.read_hdf('flux_dataframe.hdf', mode='r')

    energybins = comp.analysis.get_energybins()
    # Effective area
    eff_area = comp.get_effective_area_fit(config='IC86.2012',
                            energy_points=energybins.energy_midpoints)

    print(df_flux)

    # Flux vs energy
    color_dict = comp.analysis.get_color_dict()
    comp_list = ['light', 'heavy']

    # Plot flux for each year separately
    for config in args.config:
        fig, ax = plt.subplots()
        df_flux_config = df_flux.loc[config]
        for composition in comp_list + ['total']:
            flux, flux_err = comp.analysis.get_flux(
                                    df_flux_config['counts_' + composition],
                                    energybins=energybins.energy_bins,
                                    eff_area=eff_area,
                                    livetime=df_flux_config['livetime'],
                                    livetime_err=df_flux_config['livetime_err'],
                                    solid_angle=df_flux_config['solid_angle'])
            plotting.plot_steps(energybins.log_energy_bins, flux, yerr=flux_err,
                                ax=ax, color=color_dict[composition], label=composition)
        ax.set_yscale("log", nonposy='clip')
        ax.set_xlabel('$\mathrm{\log_{10}(E_{reco}/GeV)}$')
        ax.set_ylabel('$\mathrm{ E^{2.7} \ J(E) \ [GeV^{1.7} m^{-2} sr^{-1} s^{-1}]}$')
        ax.set_xlim([energybins.log_energy_min, energybins.log_energy_max])
        ax.set_ylim([10**3, 10**5])
        ax.grid(linestyle='dotted', which="both")

        leg = plt.legend(loc='upper center', frameon=False,
                  bbox_to_anchor=(0.5,  # horizontal
                                  1.15),# vertical
                  ncol=len(comp_list)+1, fancybox=False)
        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0)

        outfile = os.path.join(comp.paths.figures_dir, 'flux',
                               'flux-{}.png'.format(config))
        comp.check_output_dir(outfile)
        plt.savefig(outfile)

    # Plot combined flux for all years
    fig, ax = plt.subplots()
    for composition in comp_list + ['total']:
        livetime_err = comp.get_summation_error(df_flux['livetime_err'])
        counts = df_flux['counts_' + composition].sum()
        print('counts = {}'.format(counts))
        counts_err = np.sqrt(np.sum(df_flux['counts_' + composition + '_err']**2, axis=0))
        print('counts_err = {}'.format(counts_err))
        flux, flux_err = comp.analysis.get_flux(
                                counts, counts_err=counts_err,
                                energybins=energybins.energy_bins,
                                eff_area=eff_area,
                                livetime=df_flux['livetime'].sum(),
                                livetime_err=livetime_err,
                                solid_angle=df_flux['solid_angle'].mean())
        plotting.plot_steps(energybins.log_energy_bins, flux, yerr=flux_err,
                            ax=ax, color=color_dict[composition], label=composition)
    ax.set_yscale("log", nonposy='clip')
    ax.set_xlabel('$\mathrm{\log_{10}(E_{reco}/GeV)}$')
    ax.set_ylabel('$\mathrm{ E^{2.7} \ J(E) \ [GeV^{1.7} m^{-2} sr^{-1} s^{-1}]}$')
    ax.set_xlim([energybins.log_energy_min, energybins.log_energy_max])
    ax.set_ylim([10**3, 10**5])
    ax.grid(linestyle='dotted', which="both")

    leg = plt.legend(loc='upper center', frameon=False,
              bbox_to_anchor=(0.5,  # horizontal
                              1.15),# vertical
              ncol=len(comp_list)+1, fancybox=False)
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)

    config_str = '_'.join(args.config)
    outfile = os.path.join(comp.paths.figures_dir, 'flux',
                           'flux-combined-{}.png'.format(config_str))
    comp.check_output_dir(outfile)
    plt.savefig(outfile)

    # Individual years on single plot
    if len(args.config) > 1:
        # Get colors
        df_flux['light_color'] = sns.color_palette('Blues', len(args.config)).as_hex()
        df_flux['heavy_color'] = sns.color_palette('Oranges', len(args.config)).as_hex()
        df_flux['total_color'] = sns.color_palette('Greens', len(args.config)).as_hex()

        if args.correct_eff_area:
            ratio = OrderedDict()
            df_flux_2012 = df_flux.loc['IC86.2012']
            for config in args.config:
                df_flux_config = df_flux.loc[config]
                rate = df_flux_config['counts_total'] / df_flux_config['livetime']
                rate_2012 = df_flux_2012['counts_total'] / df_flux_2012['livetime']
                ratio[config] = rate[6]/rate_2012[6]
        else:
            ratio = {config: 1.0 for config in args.config}

        print(ratio)

        # Plot rate for each year on single plot
        fig, ax = plt.subplots()
        for composition in comp_list + ['total']:
            for config in args.config:
                df_flux_config = df_flux.loc[config]
                rate, rate_err = comp.ratio_error(
                            df_flux_config['counts_' + composition],
                            np.sqrt(df_flux_config['counts_' + composition]),
                            df_flux_config['livetime'],
                            df_flux_config['livetime_err'])
                plotting.plot_steps(energybins.log_energy_bins, rate, yerr=rate_err,
                                    ax=ax, color=df_flux_config[composition + '_color'],
                                    label=config + ' ' + composition)
        ax.set_yscale("log", nonposy='clip')
        ax.set_xlabel('$\mathrm{\log_{10}(E_{reco}/GeV)}$')
        ax.set_ylabel('Rate $\mathrm{[s^{-1}]}$')
        ax.set_xlim([energybins.log_energy_min, energybins.log_energy_max])
        # ax.set_ylim([10**3, 10**5])
        ax.grid(linestyle='dotted', which="both")

        leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                         ncol=1, frameon=False)
                        #  ncol=len(comp_list)+1, frameon=False)

        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0)

        config_str = '_'.join(args.config)
        outfile = os.path.join(comp.paths.figures_dir, 'flux',
                               'rate-{}.png'.format(config_str))
        comp.check_output_dir(outfile)
        plt.savefig(outfile)


        # Plot flux for each year on single plot
        gamma_corrections_file = os.path.join(comp.paths.comp_data_dir,
                                    'gamma_eff_area_corrections.hdf')
        df_gamma_corrections = pd.read_hdf(gamma_corrections_file)
        print(df_gamma_corrections)
        print(df_gamma_corrections.index)
        print('IC86.2011 corrections = {}'.format(df_gamma_corrections.loc['IC86.2011']))

        fig, ax = plt.subplots()
        for composition in comp_list + ['total']:
            for config in args.config:
                df_flux_config = df_flux.loc[config]
                flux, flux_err = comp.analysis.get_flux(
                                        df_flux_config['counts_' + composition],
                                        energybins=energybins.energy_bins,
                                        # eff_area=eff_area*df_gamma_corrections.loc[config],
                                        eff_area=eff_area*ratio[config],
                                        livetime=df_flux_config['livetime'],
                                        livetime_err=df_flux_config['livetime_err'],
                                        solid_angle=df_flux_config['solid_angle'])

                plotting.plot_steps(energybins.log_energy_bins, flux, yerr=flux_err,
                                    ax=ax, color=df_flux_config[composition + '_color'],
                                    label=config + ' ' + composition)

                # ax.plot(energybins.log_energy_midpoints,
                #     energybins.energy_midpoints**2.7*flux_func(energybins.energy_midpoints),
                #     marker='None', ls='-')

        ax.set_yscale("log", nonposy='clip')
        ax.set_xlabel('$\mathrm{\log_{10}(E_{reco}/GeV)}$')
        ax.set_ylabel('$\mathrm{ E^{2.7} \ J(E) \ [GeV^{1.7} m^{-2} sr^{-1} s^{-1}]}$')
        ax.set_xlim([energybins.log_energy_min, energybins.log_energy_max])
        ax.set_ylim([10**3, 10**5])
        ax.grid(linestyle='dotted', which="both")

        leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                         ncol=1, frameon=False)
                        #  ncol=len(comp_list)+1, frameon=False)

        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0)

        config_str = '_'.join(args.config)
        outfile = os.path.join(comp.paths.figures_dir, 'flux',
                               'flux-{}.png'.format(config_str))
        comp.check_output_dir(outfile)
        plt.savefig(outfile)

        # Plot correction ratio vs. year
        fig, ax = plt.subplots()
        x = range(len(ratio.keys()))
        ax.plot(x, ratio.values(), color='C0', ls=':', markersize=10)
        plt.xticks(x, ratio.keys())
        ax.set_xlabel('Year')
        ax.set_ylabel('Rate / IC86.2012 Rate')
        ax.grid()
        config_str = '_'.join(args.config)
        outfile = os.path.join(comp.paths.figures_dir, 'flux',
                               'correction_ratio-{}.png'.format(config_str))
        comp.check_output_dir(outfile)
        plt.savefig(outfile)

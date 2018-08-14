#!/usr/bin/env python

from __future__ import division, print_function
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import comptools as comp
import comptools.plotting as plotting

color_dict = comp.color_dict
sns.set_context(context='talk', font_scale=1.5)


if __name__ == '__main__':

    description = 'Makes performance plots for IceTop Laputop reconstruction'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config', dest='config', nargs='*',
                   choices=comp.simfunctions.get_sim_configs(),
                   help='Detector configuration')
    args = parser.parse_args()

    for config in args.config:

        df_sim = comp.load_sim(config=config, test_size=0, verbose=True)

        comp_list = ['light', 'heavy']
        MC_comp_mask = {}
        for composition in comp_list:
            MC_comp_mask[composition] = df_sim['MC_comp_class'] == composition
        light_mask = df_sim['MC_comp_class'] == 'light'
        heavy_mask = df_sim['MC_comp_class'] == 'heavy'

        energybins = comp.get_energybins()

        # Energy resolution
        energy_res = np.log10(df_sim['lap_energy'] / df_sim['MC_energy'])

        medians_light, stds_light, _ = comp.data_functions.get_median_std(
                df_sim['MC_log_energy'][light_mask], energy_res[light_mask],
                energybins.log_energy_bins)
        medians_heavy, stds_heavy, _ = comp.data_functions.get_median_std(
                df_sim['MC_log_energy'][heavy_mask], energy_res[heavy_mask],
                energybins.log_energy_bins)

        gs = gridspec.GridSpec(2, 1, height_ratios=[1,1], hspace=0.1)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)

        plotting.plot_steps(energybins.log_energy_bins, medians_light, lw=1.5,
                            color=color_dict['light'], label='light', ax=ax1)
        plotting.plot_steps(energybins.log_energy_bins, medians_heavy, lw=1.5,
                            color=color_dict['heavy'], label='heavy', ax=ax1)
        ax1.axhline(0, marker='None', linestyle='-.', color='k', lw=1.5)
        ax1.set_ylabel('Median of\n$\mathrm{\log_{10}(E_{reco}/E_{true})}$')
        ax1.set_ylim(-0.1, 0.1)
        ax1.tick_params(labelbottom='off')
        ax1.grid()
        ax1.legend()

        plotting.plot_steps(energybins.log_energy_bins, stds_light, lw=1.5,
                            color=color_dict['light'], label='light', ax=ax2)
        plotting.plot_steps(energybins.log_energy_bins, stds_heavy, lw=1.5,
                            color=color_dict['heavy'], label='heavy', ax=ax2)

        ax2.set_ylabel('1$\mathrm{\sigma}$ of $\mathrm{\log_{10}(E_{reco}/E_{true})}$')
        ax2.set_xlabel('$\mathrm{\log_{10}(E_{true}/GeV)}$')
        ax2.set_ylim(0)
        ax2.set_xlim(energybins.log_energy_min, energybins.log_energy_max)
        ax2.grid()

        energy_res_outfile = os.path.join(comp.paths.figures_dir,
                                        'laputop_performance',
                                        'energy_res_{}.png'.format(config))
        comp.check_output_dir(energy_res_outfile)
        plt.savefig(energy_res_outfile)


        # Core resolution
        fig, ax = plt.subplots()
        for composition in comp_list:
            core_diff = np.sqrt((df_sim[MC_comp_mask[composition]]['lap_x'] - df_sim[MC_comp_mask[composition]]['MC_x'])**2 \
                                +(df_sim[MC_comp_mask[composition]]['lap_y'] - df_sim[MC_comp_mask[composition]]['MC_y'])**2)
            energy = df_sim[MC_comp_mask[composition]]['MC_energy']
            core_res = comp.data_functions.get_resolution(energy, core_diff, energybins.energy_bins)
            plotting.plot_steps(energybins.log_energy_bins, core_res, lw=1.5,
                                color=color_dict[composition], label=composition, ax=ax)
        ax.set_xlim(energybins.log_energy_min, energybins.log_energy_max)
        ax.set_xlabel('$\mathrm{\log_{10}(E_{true}/GeV)}$')
        ax.set_ylabel('Core resolution [m]')
        ax.grid()
        ax.legend()
        core_res_outfile = os.path.join(comp.paths.figures_dir,
                                        'laputop_performance',
                                        'core_res_CR_group_{}.png'.format(config))
        comp.check_output_dir(core_res_outfile)
        plt.savefig(core_res_outfile)

        # Angular resolution
        fig, ax = plt.subplots()
        for composition in comp_list:
            null_mask = df_sim[MC_comp_mask[composition]]['angle_MCPrimary_Laputop'].isnull()
            ang_diff_deg = df_sim[MC_comp_mask[composition]]['angle_MCPrimary_Laputop'].dropna()*180/np.pi
            energy = df_sim[MC_comp_mask[composition]]['lap_energy'].dropna()
            angular_res = comp.data_functions.get_resolution(energy, ang_diff_deg, energybins.energy_bins)
            plotting.plot_steps(energybins.log_energy_bins, angular_res, lw=1.5,
                                color=color_dict[composition], label=composition, ax=ax)
        ax.set_ylim([0.0, 0.4])
        ax.set_xlim(energybins.log_energy_min, energybins.log_energy_max)
        ax.set_xlabel('$\mathrm{\log_{10}(E_{reco}/GeV)}$')
        ax.set_ylabel('Angular resolution [$^{\circ}$]')
        ax.grid()
        ax.legend()
        angular_res_outfile = os.path.join(comp.paths.figures_dir,
                                        'laputop_performance',
                                        'angular_res_CR_group_{}.png'.format(config))
        comp.check_output_dir(angular_res_outfile)
        plt.savefig(angular_res_outfile)

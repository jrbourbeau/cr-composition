#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import comptools as comp
import comptools.analysis.plotting as plotting

color_dict = comp.analysis.get_color_dict()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Makes and saves classification accuracy vs. energy plot')
    parser.add_argument('-c', '--config', dest='config',
                   choices=comp.simfunctions.get_sim_configs(),
                   help='Detector configuration')
    parser.add_argument('--num_groups', dest='num_groups', type=int,
                   default=2,
                   help='Number of composition groups to use.')
    parser.add_argument('-n', '--n_splits', dest='n_splits', type=int,
                   default=10,
                   help='Detector configuration')
    parser.add_argument('--n_jobs', dest='n_jobs', type=int,
                   default=1,
                   help='Detector configuration')
    args = parser.parse_args()

    energybins = comp.analysis.get_energybins(args.config)
    comp_list = comp.get_comp_list(num_groups=args.num_groups)
    feature_list, feature_labels = comp.get_training_features()
    pipeline_str = 'BDT_comp_{}'.format(args.config)

    df_sim_train, df_sim_test = comp.load_sim(
                                config=args.config,
                                log_energy_min=energybins.log_energy_min,
                                log_energy_max=energybins.log_energy_max)

    df_correct_folds = comp.analysis.get_CV_frac_correct(
        df_sim_train, feature_list, pipeline_str, args.num_groups,
        energybins.log_energy_bins, n_splits=args.n_splits, n_jobs=args.n_jobs)

    fig, ax = plt.subplots()
    for composition in comp_list:
        key = 'frac_correct_{}'.format(composition)
        performance_mean = np.mean(df_correct_folds[key])
        performance_std = np.std(df_correct_folds[key].values)
        plotting.plot_steps(energybins.log_energy_bins, performance_mean, yerr=performance_std,
                            ax=ax, color=color_dict[composition], label=composition)
    ax.set_xlabel('$\mathrm{\log_{10}(E_{MC}/GeV)}$')
    ax.set_ylabel('Classification accuracy [{:d}-fold CV]'.format(args.n_splits))
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim(6.4, energybins.log_energy_max)
    # ax.set_xlim(energybins.log_energy_min, energybins.log_energy_max)
    ax.grid()
    leg = plt.legend(loc='upper center', frameon=False,
                     bbox_to_anchor=(0.5, 1.1), ncol=len(comp_list)+1,
                     fancybox=False)
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)

    # acc = np.nanmean(frac_correct_folds['total'])*100
    # acc_err = np.nanstd(frac_correct_folds['total'])*100
    # cv_str = 'Total accuracy:\n{}\% (+/- {}\%)'.format(int(acc)+1,
    #                                                    int(acc_err)+1)
    # ax.text(7.4, 0.2, cv_str,
    #         ha="center", va="center", size=14,
    #         bbox=dict(boxstyle='round', fc="white", ec="gray", lw=0.8))
    outfile = os.path.join(comp.paths.figures_dir, 'model_evaluation',
                           'frac-correct-{}.png'.format(args.config))
    comp.check_output_dir(outfile)
    plt.savefig(outfile)

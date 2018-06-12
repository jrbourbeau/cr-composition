#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from dask import delayed, multiprocessing
from dask.diagnostics import ProgressBar

import comptools as comp

color_dict = comp.get_color_dict()


@delayed
def get_frac_correct(df_train, df_test, pipeline_str=None, num_groups=4,
                     energy_key='MC_log_energy'):
    '''Calculates the fraction of correctly identified samples in each energy bin
    for each composition in comp_list. In addition, the statisitcal error for the
    fraction correctly identified is calculated.'''

    # Input validation
    if energy_key not in ['MC_log_energy', 'reco_log_energy']:
        raise ValueError("Invalid energy_key ({}) entered. Must be either "
                         "'MC_log_energy' or 'reco_log_energy'.".format(energy_key))

    if pipeline_str is None:
        pipeline_str = 'BDT_comp_IC86.2012_{}-groups'.format(num_groups)

    # Fit pipeline and get mask for correctly identified events
    feature_list, feature_labels = comp.get_training_features()
    pipeline = comp.get_pipeline(pipeline_str)
    comp_target_str = 'comp_target_{}'.format(num_groups)
    pipeline.fit(df_train[feature_list],
                 df_train[comp_target_str])

    test_predictions = pipeline.predict(df_test[feature_list])
    correctly_identified_mask = (test_predictions == df_test[comp_target_str])

    data = {}
    for composition in comp_list + ['total']:
        comp_mask = df_test['comp_group_{}'.format(num_groups)] == composition
        # Get number of MC comp in each energy bin
        num_MC_energy, _ = np.histogram(df_test.loc[comp_mask, energy_key],
                                        bins=energybins.log_energy_bins)
        num_MC_energy_err = np.sqrt(num_MC_energy)

        # Get number of correctly identified comp in each energy bin
        combined_mask = comp_mask & correctly_identified_mask
        num_reco_energy, _ = np.histogram(df_test.loc[combined_mask, energy_key],
                                          bins=energybins.log_energy_bins)
        num_reco_energy_err = np.sqrt(num_reco_energy)

        # Calculate correctly identified fractions as a function of energy
        frac_correct, frac_correct_err = comp.ratio_error(
            num_reco_energy, num_reco_energy_err,
            num_MC_energy, num_MC_energy_err)
        data['frac_correct_{}'.format(composition)] = frac_correct
        data['frac_correct_err_{}'.format(composition)] = frac_correct_err

    return data


if __name__ == '__main__':

    description='Makes and saves classification accuracy vs. energy plot'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config', dest='config',
                        choices=comp.simfunctions.get_sim_configs(),
                        help='Detector configuration')
    parser.add_argument('--num_groups', dest='num_groups', type=int,
                        default=4,
                        help='Number of composition groups to use.')
    parser.add_argument('--n_splits', dest='n_splits', type=int,
                        default=10,
                        help='Detector configuration')
    parser.add_argument('--n_jobs', dest='n_jobs', type=int,
                        default=1,
                        help='Detector configuration')
    parser.add_argument('--energy', dest='energy',
                        default='MC',
                        choices=['MC', 'reco'],
                        help='Energy that should be used.')
    args = parser.parse_args()

    config = args.config
    num_groups = args.num_groups
    n_splits = args.n_splits
    n_jobs = args.n_jobs
    energy_key = 'MC_log_energy' if args.energy == 'MC' else 'reco_log_energy'

    energybins = comp.get_energybins(config)
    comp_list = comp.get_comp_list(num_groups=num_groups)
    feature_list, feature_labels = comp.get_training_features()
    pipeline_str = 'xgboost_comp_{}_{}-groups'.format(config, num_groups)
    # pipeline_str = 'BDT_comp_{}_{}-groups'.format(config, num_groups)

    df_train, df_test = comp.load_sim(config=config,
                                      log_energy_min=energybins.log_energy_min,
                                      log_energy_max=energybins.log_energy_max,
                                      test_size=0.5)


    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)
    folds = []
    for train_index, test_index in skf.split(df_train, df_train['comp_target_{}'.format(num_groups)]):
        df_train_fold = df_train.iloc[train_index]
        df_test_fold = df_train.iloc[test_index]
        frac_correct = get_frac_correct(df_train_fold, df_test_fold,
                                        pipeline_str=pipeline_str,
                                        num_groups=num_groups,
                                        energy_key=energy_key)
        folds.append(frac_correct)

    df_cv = delayed(pd.DataFrame.from_records)(folds)

    # Run get_frac_correct on each fold in parallel
    print('Running {}-fold CV model evaluation...'.format(n_splits))
    with ProgressBar():
        get = multiprocessing.get if n_jobs > 1 else dask.get
        df_cv = df_cv.compute(get=get, num_works=n_jobs)

    # Plot correctly identified vs. energy for each composition
    fig, ax = plt.subplots()
    for composition in comp_list:
        key = 'frac_correct_{}'.format(composition)
        performance_mean = np.mean(df_cv[key].values)
        performance_std = np.std(df_cv[key].values)
        comp.plot_steps(energybins.log_energy_bins, performance_mean, yerr=performance_std,
                        ax=ax, color=color_dict[composition], label=composition)
    if energy_key == 'MC_log_energy':
        xlabel = '$\mathrm{\log_{10}(E_{MC}/GeV)}$'
    else:
        xlabel = '$\mathrm{\log_{10}(E_{reco}/GeV)}$'
    fontsize = 18
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel('Classification accuracy [{:d}-fold CV]'.format(n_splits),
                  fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim(6.4, energybins.log_energy_max)
    # ax.set_xlim(energybins.log_energy_min, energybins.log_energy_max)
    ax.grid()
    leg = plt.legend(loc='upper center', frameon=False,
                     bbox_to_anchor=(0.5,
                                     1.15),
                     ncol=len(comp_list)+1,
                     fancybox=False, fontsize=fontsize)
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
                           'frac-correct_{}_{}_{}-groups.png'.format(
                                energy_key.replace('_', '-'), config, num_groups))
    comp.check_output_dir(outfile)
    plt.savefig(outfile)

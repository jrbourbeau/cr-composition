#!/usr/bin/env python

from __future__ import division, print_function
import os
from itertools import product
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
def get_classified_fractions(df_train, df_test, pipeline_str=None, num_groups=4,
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
    if 'CustomClassifier' in pipeline_str:
        pipeline = comp.get_pipeline(pipeline_str)
    else:
        pipeline = comp.load_trained_model(pipeline_str)
    comp_target_str = 'comp_target_{}'.format(num_groups)

    if 'CustomClassifier' in pipeline_str:
        test_predictions = pipeline.predict(df_test['comp_target_{}'.format(num_groups)])
    else:
        test_predictions = pipeline.predict(df_test[feature_list])
    pred_comp = np.array(comp.decode_composition_groups(test_predictions,
                                                        num_groups=num_groups))

    data = {}
    for true_composition, identified_composition in product(comp_list, comp_list):
        true_comp_mask = df_test['comp_group_{}'.format(num_groups)] == true_composition
        ident_comp_mask = pred_comp == identified_composition

        # Get number of MC comp in each energy bin
        num_true_comp, _ = np.histogram(df_test.loc[true_comp_mask, energy_key],
                                        bins=energybins.log_energy_bins)
        num_true_comp_err = np.sqrt(num_true_comp)

        # Get number of correctly identified comp in each energy bin
        combined_mask = true_comp_mask & ident_comp_mask
        num_identified_comp, _ = np.histogram(df_test.loc[combined_mask, energy_key],
                                              bins=energybins.log_energy_bins)
        num_identified_comp_err = np.sqrt(num_identified_comp)

        # Calculate correctly identified fractions as a function of energy
        frac_identified, frac_identified_err = comp.ratio_error(
            num_identified_comp, num_identified_comp_err,
            num_true_comp, num_true_comp_err)
        data['true_{}_identified_{}'.format(true_composition, identified_composition)] = frac_identified
        data['true_{}_identified_{}_err'.format(true_composition, identified_composition)] = frac_identified_err

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
    parser.add_argument('--prob_correct', dest='prob_correct',
                        type=float,
                        help=('Probability event is correctly classified for '
                              'custom composition classification'))
    args = parser.parse_args()

    config = args.config
    num_groups = args.num_groups
    n_splits = args.n_splits
    n_jobs = args.n_jobs
    energy_key = 'MC_log_energy' if args.energy == 'MC' else 'reco_log_energy'

    p = args.prob_correct

    energybins = comp.get_energybins(config)
    comp_list = comp.get_comp_list(num_groups=num_groups)
    feature_list, feature_labels = comp.get_training_features()

    if p is not None:
        pipeline_str = 'CustomClassifier_{}_2_{}'.format(p, num_groups)
    else:
        # pipeline_str = 'RF_comp_{}_{}-groups'.format(config, num_groups)
        # pipeline_str = 'SVC_comp_{}_{}-groups'.format(config, num_groups)
        # pipeline_str = 'LinearSVC_comp_{}_{}-groups'.format(config, num_groups)
        pipeline_str = 'BDT_comp_{}_{}-groups'.format(config, num_groups)
        # pipeline_str = 'LogisticRegression_comp_{}_{}-groups'.format(config, num_groups)
        # pipeline_str = 'voting_comp_{}_{}-groups'.format(config, num_groups)

    df_train, df_test = comp.load_sim(config=config,
                                      log_energy_min=energybins.log_energy_min,
                                      log_energy_max=energybins.log_energy_max,
                                      test_size=0.5)


    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)
    folds = []
    splitter = skf.split(df_train, df_train['comp_target_{}'.format(num_groups)])
    for fold_idx, (train_index, test_index) in enumerate(splitter):
        df_train_fold = df_train.iloc[train_index]
        df_test_fold = df_train.iloc[test_index]
        frac_correct = get_classified_fractions(df_train_fold, df_test_fold,
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
    fig, axarr = plt.subplots(1, len(comp_list), figsize=(12, 5), sharex=True, sharey=True)
    for idx, true_composition in enumerate(comp_list):
        ax = axarr[idx]
        for identified_composition in comp_list:
            key = 'true_{}_identified_{}'.format(true_composition, identified_composition)
            performance_mean = np.mean(df_cv[key].values)
            performance_std = np.std(df_cv[key].values)
            comp.plot_steps(energybins.log_energy_bins, performance_mean, yerr=performance_std,
                            ax=ax, color=color_dict[identified_composition], label=identified_composition)
        if energy_key == 'MC_log_energy':
            xlabel = '$\mathrm{\log_{10}(E_{MC}/GeV)}$'
        else:
            xlabel = '$\mathrm{\log_{10}(E_{reco}/GeV)}$'
        ax.set_xlabel(xlabel)
        if idx == 0:
            ax.set_ylabel('Classification composition fractions')
        ax.set_title('MC {}'.format(true_composition))
        ax.set_ylim([0.0, 1.0])
        ax.set_xlim(6.4, energybins.log_energy_max)
        ax.grid()
        leg = ax.legend(title='Classified composition', fontsize=8)
        plt.setp(leg.get_title(),fontsize=10)

    outfile = os.path.join(comp.paths.figures_dir, 'model_evaluation',
                           '{}_identification-frac_{}_{}_{}-groups.png'.format(
                                pipeline_str,
                                energy_key.replace('_', '-'),
                                config,
                                num_groups))
    comp.check_output_dir(outfile)
    plt.savefig(outfile)

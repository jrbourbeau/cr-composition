#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import zero_one_loss

import comptools as comp


if __name__ == "__main__":

    description = ('Calculates and plots composition classification '
                   'validation curves')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', dest='config', default='IC86.2012',
                        choices=comp.simfunctions.get_sim_configs(),
                        help='Detector configuration')
    parser.add_argument('--num_groups', dest='num_groups', type=int,
                        default=2,
                        help='Number of composition groups to use.')
    parser.add_argument('--param_name', dest='param_name',
                        default='max_depth',
                        help='Name of hyperparameter to tune.')
    parser.add_argument('--param_label', dest='param_label',
                        default='Maximum depth',
                        help='Name of hyperparameter to go on plot')
    parser.add_argument('--param_values', dest='param_values', nargs='*',
                        help='Values of hyperparameter.')
    parser.add_argument('--param_type', dest='param_type',
                        default='int',
                        choices=['int', 'float', 'string'],
                        help='Type of hyperparameter.')
    parser.add_argument('--cv', dest='cv', type=int,
                        default=10,
                        help='Number of cross-validation folds.')

    args = parser.parse_args()

    color_dict = comp.analysis.get_color_dict()

    energybins = comp.analysis.get_energybins(args.config)
    comp_list = comp.get_comp_list(num_groups=args.num_groups)
    feature_list, feature_labels = comp.get_training_features()
    pipeline_str = 'BDT_comp_{}'.format(args.config)

    df_sim_train, df_sim_test = comp.load_sim(
                config=args.config, log_energy_min=energybins.log_energy_min,
                log_energy_max=energybins.log_energy_max)

    # Calculate CV scores for each composition
    params = np.asarray(args.param_values).astype(args.param_type)
    df_cv = comp.cross_validate_comp(
                            df_sim_train, df_sim_test, pipeline_str,
                            param_name=args.param_name, param_values=params,
                            feature_list=feature_list,
                            target='comp_target_{}'.format(args.num_groups),
                            scoring=zero_one_loss, num_groups=args.num_groups,
                            n_splits=args.cv, verbose=True,
                            n_jobs=min(len(params), 15))

    # Plot validation curve for hyperparameter
    fig, ax = plt.subplots()
    for composition in comp_list:
        # Plot testing curve
        ax.plot(df_cv.index, df_cv['test_mean_{}'.format(composition)],
                marker='^', ls=':', color=color_dict[composition],
                label=composition+' testing set')
        test_mean = df_cv['test_mean_{}'.format(composition)]
        test_std = df_cv['test_std_{}'.format(composition)]
        test_err_high = test_mean + test_std
        test_err_low = test_mean - test_std
        ax.fill_between(df_cv.index, test_err_high, test_err_low,
                        color=color_dict[composition], alpha=0.3)
        # Plot training curve
        ax.plot(df_cv.index, df_cv['train_mean_{}'.format(composition)],
                marker='.', ls='-', color=color_dict[composition],
                label=composition+' training set')
        train_mean = df_cv['train_mean_{}'.format(composition)]
        train_std = df_cv['train_std_{}'.format(composition)]
        train_err_high = train_mean + train_std
        train_err_low = train_mean - train_std
        ax.fill_between(df_cv.index, train_err_high, train_err_low,
                        color=color_dict[composition], alpha=0.3)

    ax.set_xlabel(args.param_label)
    ax.set_ylabel('Classification error')
    ax.grid()
    ax.legend(title='True compositions')
    outfile = os.path.join(comp.paths.figures_dir, 'model_evaluation',
                           'validation-curves',
                           '{}_{}_num_groups-{}_zoomed.png'.format(
                           pipeline_str, args.param_name, args.num_groups))
    comp.check_output_dir(outfile)
    plt.savefig(outfile)

    ax.set_ylim(-0.05, 1.05)
    outfile = os.path.join(comp.paths.figures_dir, 'model_evaluation',
                           'validation-curves',
                           '{}_{}_num_groups-{}.png'.format(
                            pipeline_str, args.param_name, args.num_groups))
    comp.check_output_dir(outfile)
    plt.savefig(outfile)

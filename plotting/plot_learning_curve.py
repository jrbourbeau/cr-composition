#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

import comptools as comp

color_dict = comp.get_color_dict()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculates and saves learning curve plot')
    parser.add_argument('-c', '--config', dest='config',
                        default='IC86.2012',
                        choices=comp.simfunctions.get_sim_configs(),
                        help='Detector configuration')
    parser.add_argument('--num_groups', dest='num_groups', type=int,
                        default=4, choices=[2, 3, 4],
                        help='Number of composition groups')
    parser.add_argument('--cv', dest='cv',
                        default=10, type=int,
                        help='Number CV folds to run')
    parser.add_argument('--n_jobs', dest='n_jobs',
                        default=20, type=int,
                        help='Number of jobs to run in parallel')
    args = parser.parse_args()

    comp_list = comp.get_comp_list(num_groups=args.num_groups)
    energybins = comp.get_energybins(args.config)

    # Load simulation data and pipeline
    df_sim_train, df_sim_test = comp.load_sim(
                                    config=args.config,
                                    log_energy_min=energybins.log_energy_min,
                                    log_energy_max=energybins.log_energy_max)
    feature_list, feature_labels = comp.get_training_features()

    pipeline_str = 'LinearSVC_comp_{}_{}-groups'.format(args.config, args.num_groups)
    # pipeline_str = 'BDT_comp_{}_{}-groups'.format(args.config, args.num_groups)
    pipeline = comp.get_pipeline(pipeline_str)

    # Get learning curve scores
    X = df_sim_train[feature_list]
    y = df_sim_train['comp_target_{}'.format(args.num_groups)]
    train_sizes=np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, test_scores = learning_curve(
                                                estimator=pipeline,
                                                X=X,
                                                y=y,
                                                train_sizes=train_sizes,
                                                cv=args.cv,
                                                n_jobs=args.n_jobs,
                                                verbose=1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot learning curve
    fig, ax = plt.subplots()
    # Training curve
    ax.plot(train_sizes, train_mean,
             color='C0', linestyle='-',
             marker='o', markersize=5,
             label='Training set')
    ax.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.25, color='C0')
    # Validation curve
    ax.plot(train_sizes, test_mean,
             color='C1', linestyle='--',
             marker='s', markersize=5,
             label='Validation set')
    ax.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.25, color='C1')

    ax.grid()
    ax.set_xlabel('Number of training samples')
    ax.set_ylabel('Accuracy')
    ax.legend()
    plt.tight_layout()
    outfile = os.path.join(comp.paths.figures_dir, 'model_evaluation',
                           'learning_curve_{}.png'.format(pipeline_str))
    # outfile = os.path.join(comp.paths.figures_dir, 'model_evaluation',
    #                        'learning_curve_{}_{}-groups.png'.format(args.config,
    #                                                                 args.num_groups))
    comp.check_output_dir(outfile)
    plt.savefig(outfile)

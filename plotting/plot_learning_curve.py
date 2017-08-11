#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

import comptools as comp
import comptools.analysis.plotting as plotting

color_dict = comp.analysis.get_color_dict()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculates and saves learning curve plot')
    parser.add_argument('-c', '--config', dest='config',
                   default='IC86.2012',
                   choices=comp.simfunctions.get_sim_configs(),
                   help='Detector configuration')
    parser.add_argument('--pipeline', dest='pipeline',
                   default='BDT', choices=['BDT', 'GBDT'],
                   help='Detector configuration')
    parser.add_argument('--cv', dest='cv',
                   default=10, type=int,
                   help='Number CV folds to run')
    parser.add_argument('--n_jobs', dest='n_jobs',
                   default=20, type=int,
                   help='Number of jobs to run in parallel')
    args = parser.parse_args()

    # Load simulation data and pipeline
    df_sim_train, df_sim_test = comp.load_sim(config=args.config)
    feature_list, feature_labels = comp.analysis.get_training_features()
    pipeline = comp.get_pipeline(args.pipeline)

    # Get learning curve scores
    train_sizes, train_scores, test_scores = learning_curve(
                                        estimator=pipeline,
                                        X=df_sim_train[feature_list],
                                        y=df_sim_train['target'],
                                        train_sizes=np.linspace(0.1, 1.0, 10),
                                        cv=args.cv,
                                        n_jobs=args.n_jobs,
                                        verbose=3)

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
                     alpha=0.15, color='C0')
    # Validation curve
    ax.plot(train_sizes, test_mean,
             color='C1', linestyle='--',
             marker='s', markersize=5,
             label='Validation set')
    ax.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='C1')

    ax.grid()
    ax.set_xlabel('Number of training samples')
    ax.set_ylabel('Classification accuracy [10-fold CV]')
    ax.legend()
    plt.tight_layout()
    outfile = os.path.join(comp.paths.figures_dir, 'model_evaluation',
                           'learning_curve_{}_{}.png'.format(args.pipeline, args.config))
    comp.check_output_dir(outfile)
    plt.savefig(outfile)

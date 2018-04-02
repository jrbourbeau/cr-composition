#!/usr/bin/env python

from __future__ import division, print_function
import os
from collections import defaultdict
import argparse
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.base import clone
import warnings

import comptools as comp

warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")


if __name__ == '__main__':

    description = 'Saves trained composition classification model for later use'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config', dest='config',
                        choices=comp.simfunctions.get_sim_configs(),
                        help='Detector configuration')
    parser.add_argument('--num_groups', dest='num_groups', type=int,
                        default=4, choices=[2, 3, 4],
                        help='Number of composition groups')
    parser.add_argument('--pipeline', dest='pipeline',
                        default='BDT',
                        help='Composition classification pipeline to use')
    parser.add_argument('--features', dest='features', nargs='*',
                        default=None,
                        help='Training features to use')
    parser.add_argument('--random_feature',
                        dest='random_feature',
                        action='store_true',
                        default=False,
                        help='Add a random training feature column. Useful '
                             'motivator for feature selection.')
    parser.add_argument('--n_jobs', dest='n_jobs', type=int,
                        default=1, choices=list(range(1, 21)),
                        help='Number of jobs to run in parallel for the '
                             'gridsearch. Ignored if gridsearch=False.')
    parser.add_argument('--outfile', dest='outfile',
                        default=None,
                        help='Output file path')
    args = parser.parse_args()

    config = args.config
    num_groups = args.num_groups

    comp_list = comp.get_comp_list(num_groups=num_groups)
    energybins = comp.get_energybins(config=config)
    log_energy_min = energybins.log_energy_min
    log_energy_max = energybins.log_energy_max

    # Load training data and fit model
    df_sim_train, df_sim_test = comp.load_sim(config=config,
                                              energy_reco=True,
                                              log_energy_min=None,
                                              log_energy_max=None,
                                              # log_energy_min=log_energy_min,
                                              # log_energy_max=log_energy_max,
                                              test_size=0.5)

    features, feature_labels = comp.get_training_features(args.features)
    # Add random training feature if specified
    if args.random_feature:
        np.random.seed(2)
        df_sim_train['random'] = np.random.random(size=len(df_sim_train))
        features.append('random')
        feature_labels.append('random')

    X_train = df_sim_train[features].values
    y_train = df_sim_train['comp_target_{}'.format(num_groups)].values

    # Will need energy for each event to make classification performance vs. energy plot
    log_energy_train = df_sim_train['reco_log_energy'].values

    pipeline_str = '{}_comp_{}_{}-groups'.format(args.pipeline, config, num_groups)
    pipeline = comp.get_pipeline(pipeline_str)
    param_grid = comp.get_param_grid(pipeline_name=pipeline_str)
    gridsearch = comp.gridsearch_optimize(pipeline=pipeline,
                                          param_grid=param_grid,
                                          X_train=X_train,
                                          y_train=y_train,
                                          n_jobs=args.n_jobs,
                                          return_gridsearch=True)

    skf = StratifiedKFold(n_splits=10, random_state=2)
    acc = defaultdict(list)
    for train_index, test_index in skf.split(X_train, y_train):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        log_energy_train_fold, log_energy_test_fold = log_energy_train[train_index], log_energy_train[test_index]

        clf = clone(gridsearch.best_estimator_)
        clf.fit(X_train_fold, y_train_fold)

        y_test_pred = clf.predict(X_test_fold)
        # y_train_pred = clf.predict(X_train_fold)

        # # Testing / training accuracy
        # acc_test.append(accuracy_score(y_test_fold, y_test_pred))
        # acc_train.append(accuracy_score(y_train_fold, y_train_pred))

        for idx, composition in enumerate(comp_list):
            comp_mask = y_test_fold == idx
            correctly_classified = y_test_pred == y_test_fold
            correct_counts, _  = np.histogram(log_energy_test_fold[comp_mask & correctly_classified],
                                              bins=energybins.log_energy_bins)
            total_counts, _  = np.histogram(log_energy_test_fold[comp_mask],
                                            bins=energybins.log_energy_bins)

            frac_correct = correct_counts / total_counts

            acc[composition].append(frac_correct)

    # Construct dictionary containing fitted pipeline along with metadata
    # For information on why this metadata is needed see:
    # http://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations
    here = os.path.dirname(os.path.realpath(__file__))
    feature_str = '-'.join(features)
    if args.outfile is None:
        outfile = os.path.join(here,
                               'feature_scan_results',
                               '{}-{}.pkl'.format(pipeline_str, feature_str))
    else:
        outfile = args.outfile
    comp.check_output_dir(outfile)
    gridsearch_results = {'pipeline_name': pipeline_str,
                          'pipeline': gridsearch.best_estimator_,
                          'best_params': gridsearch.best_params_,
                          # Using tuple because items must be pickle-able
                          'features': tuple(features),
                          'feature_labels': tuple(feature_labels),
                          'sklearn_version': sklearn.__version__,
                          'source_code': os.path.realpath(__file__),
                          'config': config,
                          'num_groups': num_groups,
                          'log_energy_bins' : energybins.log_energy_bins,
                          # 'acc_mean': acc_mean,
                          # 'acc_std': acc_std,
                          }
    for idx, composition in enumerate(comp_list):
        gridsearch_results['acc_mean_{}'.format(composition)] = np.mean(acc[composition], axis=0)
        gridsearch_results['acc_std_{}'.format(composition)] = np.std(acc[composition], axis=0)


    joblib.dump(gridsearch_results, outfile)

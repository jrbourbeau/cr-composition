#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
import sklearn
from sklearn.externals import joblib
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
    parser.add_argument('--gridsearch', dest='gridsearch',
                        action='store_true',
                        default=False,
                        help=('Perform a grid search to find optimal '
                              'hyperparameter values.'))
    parser.add_argument('--n_jobs', dest='n_jobs', type=int,
                        default=1, choices=list(range(1, 21)),
                        help='Number of jobs to run in parallel for the '
                             'gridsearch. Ignored if gridsearch=False.')
    args = parser.parse_args()

    config = args.config
    num_groups = args.num_groups
    comp_list = comp.get_comp_list(num_groups=num_groups)

    # Load training data and fit model
    df_sim_train, df_sim_test = comp.load_sim(config=config,
                                              energy_reco=False,
                                              log_energy_min=None,
                                              log_energy_max=None,
                                              test_size=0.5)
    feature_list, feature_labels = comp.get_training_features()
    X_train = df_sim_train[feature_list].values
    y_train = df_sim_train['comp_target_{}'.format(num_groups)].values

    # Load untrained model
    pipeline_str = '{}_comp_{}_{}-groups'.format(args.pipeline, config, num_groups)
    pipeline = comp.get_pipeline(pipeline_str)

    if args.gridsearch:
        param_grid = comp.get_param_grid(pipeline_name=pipeline_str)
        pipeline = comp.gridsearch_optimize(pipeline=pipeline,
                                            param_grid=param_grid,
                                            X_train=X_train,
                                            y_train=y_train,
                                            scoring='accuracy')
    else:
        pipeline.fit(X_train, y_train)

    # Construct dictionary containing fitted pipeline along with metadata
    # For information on why this metadata is needed see:
    # http://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations
    model_dict = {'pipeline': pipeline,
                  # Using tuple because items must be pickle-able
                  'training_features': tuple(feature_list),
                  'sklearn_version': sklearn.__version__,
                  'save_pipeline_code': os.path.realpath(__file__)}
    outfile = os.path.join(comp.paths.comp_data_dir,
                           config,
                           'models',
                           '{}.pkl'.format(pipeline_str))
    comp.check_output_dir(outfile)
    joblib.dump(model_dict, outfile)

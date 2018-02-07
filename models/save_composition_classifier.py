#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
import sklearn
from sklearn.externals import joblib

import comptools as comp


if __name__ == '__main__':

    description='Saves trained composition classification model for later use'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config', dest='config',
                        choices=comp.simfunctions.get_sim_configs(),
                        help='Detector configuration')
    parser.add_argument('--num_groups', dest='num_groups', type=int,
                        default=4, choices=[2, 3, 4],
                        help='Number of composition groups')
    parser.add_argument('--pipeline', dest='pipeline',
                        help=('Option to explicity specify the pipeline to be '
                              'used. If not given, the default BDT model'
                              ' will be used.'))
    args = parser.parse_args()

    config = args.config
    num_groups = args.num_groups

    comp_list = comp.get_comp_list(num_groups=num_groups)
    energybins = comp.get_energybins(config=config)
    log_energy_min = energybins.log_energy_min
    log_energy_max = energybins.log_energy_max

    if args.pipeline:
        pipeline_str = args.pipeline
    else:
        pipeline_str = 'BDT_comp_{}_{}-groups'.format(config, num_groups)
    # Load untrained model
    pipeline = comp.get_pipeline(pipeline_str)
    # Load training data and fit model
    df_sim_train, df_sim_test = comp.load_sim(config=config,
                                              energy_reco=False,
                                              log_energy_min=None,
                                              log_energy_max=None,
                                              # log_energy_min=log_energy_min,
                                              # log_energy_max=log_energy_max,
                                              test_size=0.5)
    feature_list, feature_labels = comp.get_training_features()

    # if 'classifier__n_jobs' in pipeline.get_params():
    #     pipeline.named_steps['classifier'].set_params(n_jobs=1)

    X_train = df_sim_train[feature_list].values
    y_train = df_sim_train['comp_target_{}'.format(num_groups)].values
    pipeline.fit(X_train, y_train)

    # Construct dictionary containing fitted pipeline along with metadata
    # For information on why this metadata is needed see:
    # http://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations
    model_dict = {'pipeline': pipeline,
                  'training_features': tuple(feature_list), # Needs to be pickle-able
                  'sklearn_version': sklearn.__version__,
                  'save_pipeline_code': os.path.realpath(__file__)}
    # Save trained model w/metadata to disk
    outfile_dir = os.path.join(comp.paths.project_root, 'models')
    outfile_basename = '{}.pkl'.format(pipeline_str)
    joblib.dump(model_dict, os.path.join(outfile_dir, outfile_basename))

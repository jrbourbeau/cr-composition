#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
import sklearn
from sklearn.externals import joblib

import comptools as comp


if __name__ == '__main__':

    description='Saves trained energy reconstruction model for later use'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config', dest='config',
                   choices=comp.simfunctions.get_sim_configs(),
                   default='IC86.2012',
                   help='Detector configuration')
    args = parser.parse_args()

    # Load untrained model
    pipeline_str = 'RF_energy_{}'.format(args.config)
    pipeline = comp.get_pipeline(pipeline_str)
    # Load training data and fit model
    feature_list, feature_labels = comp.get_training_features()
    columns = feature_list + ['MC_log_energy']

    energybins = comp.get_energybins(config=args.config)
    # log_energy_min = 5.0
    # log_energy_max = None

    df_sim_train, df_sim_test = comp.load_sim(config=args.config,
                                              energy_reco=False,
                                              # energy_cut_key='MC_log_energy',
                                              log_energy_min=None,
                                              log_energy_max=None,
                                              test_size=0.5)
    X_train = df_sim_train[feature_list].values
    y_train = df_sim_train['MC_log_energy'].values
    pipeline.fit(X_train, y_train)

    # Construct dictionary containing fitted pipeline along with metadata
    # For information on why this metadata is needed see:
    # http://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations
    model_dict = {'pipeline': pipeline,
                  'training_features': tuple(feature_list), # Needs to be pickle-able
                  'sklearn_version': sklearn.__version__,
                  'save_pipeline_code': os.path.realpath(__file__)}
    # Save trained model w/metadata to disk
    outfile = os.path.join(comp.paths.project_root,
                           'models',
                           '{}.pkl'.format(pipeline_str))
    joblib.dump(model_dict, outfile)

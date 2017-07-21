#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
import sklearn
from sklearn.externals import joblib

import comptools as comp


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Extracts and saves desired information from simulation/data .i3 files')
    parser.add_argument('-c', '--config', dest='config',
                   choices=comp.simfunctions.get_sim_configs(),
                   help='Detector configuration')
    args = parser.parse_args()

    comp_class = True
    comp_key = 'MC_comp_class' if comp_class else 'MC_comp'
    comp_list = ['light', 'heavy'] if comp_class else ['P', 'He', 'O', 'Fe']

    pipeline_str = 'BDT'
    pipeline = comp.get_pipeline(pipeline_str)

    df_sim_train, df_sim_test = comp.load_dataframe(datatype='sim',
                                    config=args.config, comp_key=comp_key)

    feature_list, feature_labels = comp.analysis.get_training_features()
    pipeline.fit(df_sim_train[feature_list], df_sim_train.target)

    # Construct dictionary containing fitted pipeline along with metadata
    # For information on why this metadata is needed see:
    # http://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations
    model_dict = {'pipeline': pipeline,
                  'training_features': tuple(feature_list), # Needs to be pickle-able
                  'sklearn_version': sklearn.__version__,
                  'save_pipeline_code': os.path.realpath(__file__)}
    # Save model (w/metadata) to disk
    outfile_dir = os.path.join(comp.paths.project_home, 'models')
    outfile_basename = '{}_{}.pkl'.format(pipeline_str, args.config)
    joblib.dump(model_dict, os.path.join(outfile_dir, outfile_basename))

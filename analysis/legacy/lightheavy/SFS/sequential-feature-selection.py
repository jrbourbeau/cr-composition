#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import composition as comp

if __name__ == "__main__":

    p = argparse.ArgumentParser(description='Runs sequential feature selection on the cluster')
    p.add_argument('--config', dest='config', default='IC79',
                   help='Detector configuration')
    p.add_argument('--pipeline', dest='pipeline',
                   default='xgboost',
                   help='Pipeline to use for classification')
    p.add_argument('--method', dest='method',
                   default='forward', choices=['forward', 'backward'],
                   help='Whether to use forward or backward sequential selection')
    p.add_argument('--floating', dest='floating',
                   default=False, action='store_true',
                   help='Whether to use floating variant')
    p.add_argument('--scoring', dest='scoring', default='accuracy',
                   help='Scoring metric to use in cross-validation')
    p.add_argument('--cv', dest='cv', type=int, default=3,
                   help='Number of folds in cross-validation')
    p.add_argument('--n_jobs', dest='n_jobs', type=int, default=1,
                   help='Number cores to run in parallel')
    p.add_argument('--overwrite', dest='overwrite',
                   default=False, action='store_true',
                   help='Overwrite existing merged files')
    args = p.parse_args()

    # Load simulation DataFrame
    df_sim = comp.load_dataframe(datatype='sim', config=args.config)

    # Extract training features and targets from simulation DataFrame
    feature_list, feature_labels = comp.get_training_features()
    X_train_sim, X_test_sim, y_train_sim, y_test_sim, le = comp.get_train_test_sets(
        df_sim, feature_list, comp_class=True)

    # Load pipeline to use
    pipeline = comp.get_pipeline(args.pipeline)

    k_features = X_train_sim.shape[1] if args.method == 'forward' else 1

    # Set up sequential feature selection algorithm
    sfs = SFS(pipeline,
              k_features=k_features,
              forward=True if args.method == 'forward' else False,
              floating=args.floating,
              scoring=args.scoring,
              print_progress=True,
              cv=args.cv,
              n_jobs=args.n_jobs)
    # Run algorithm
    sfs = sfs.fit(X_train_sim, y_train_sim)

    # Get DataFrame of sfs results
    results_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
    # Save DataFrame to csv file
    output_file = 'SFS-results/{}_{}_{}_{}_cv{}.csv'.format(args.pipeline, args.method,
        'floating' if args.floating else 'nofloat', args.scoring, args.cv)
    results_df.to_csv(output_file)

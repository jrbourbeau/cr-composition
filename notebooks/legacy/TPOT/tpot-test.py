#!/usr/bin/env python

from __future__ import division
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns

from tpot import TPOTClassifier

import composition as comp


if __name__ == '__main__':

    sns.set_palette('muted')
    sns.set_color_codes()

    p = argparse.ArgumentParser(
        description='Runs extra modules over a given fileList')
    p.add_argument('-g', dest='generations', type=int,
                   default=2,
                   help='TPOT population size')
    p.add_argument('--popsize', dest='popsize', type=int,
                   default=20,
                   help='TPOT population size')
    p.add_argument('--cvfolds', dest='cvfolds', type=int,
                   default=2,
                   help='Output directory')
    args = p.parse_args()

    '''Throughout this code, X will represent features,
       while y will represent class labels'''

    # df = load_sim()
    df, cut_dict = comp.load_sim(return_cut_dict=True)
    selection_mask = np.array([True] * len(df))
    standard_cut_keys = ['lap_reco_success', 'lap_zenith', 'num_hits_1_30', 'IT_signal',
                         'max_qfrac_1_30', 'lap_containment',
                         'energy_range_lap']
    for key in standard_cut_keys:
        selection_mask *= cut_dict[key]

    df = df[selection_mask]

    feature_list, feature_labels = comp.get_training_features()
    X_train, X_test, y_train, y_test, le = comp.get_train_test_sets(
        df, feature_list)

    print('number training events = ' + str(y_train.shape[0]))
    tpot = TPOTClassifier(generations=args.generations,
                    population_size=args.popsize, num_cv_folds=args.cvfolds,
                    verbosity=3, scoring='accuracy')
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    outfile = 'tpot_pipeline_generations-{}_popsize-{}_cvfolds-{}.py'.format(args.generations, args.popsize, args.cvfolds)
    tpot.export(outfile)

#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.model_selection import validation_curve, StratifiedKFold
from sklearn.metrics import get_scorer, accuracy_score

import comptools as comp

if __name__ == "__main__":

    p = argparse.ArgumentParser(description='Runs findblobs.py on cluster en masse')
    p.add_argument('--pipeline', dest='pipeline',
                   default='xgboost',
                   help='Pipeline to use for classification.')
    p.add_argument('--param_name', dest='param_name',
                   default='max_depth',
                   help='Name of hyperparameter to tune.')
    p.add_argument('--param_value', dest='param_value',
                   help='Value of hyperparameter.')
    p.add_argument('--param_type', dest='param_type',
                   default='float',
                   choices=['int', 'float', 'string'],
                   help='Type of hyperparameter.')
    p.add_argument('--scoring', dest='scoring',
                   default='accuracy',
                   help='Name of scoring metric to use in validation curve.')
    p.add_argument('--cv', dest='cv', type=int,
                   default=10,
                   help='Number of cross-validation folds.')
    p.add_argument('--n_jobs', dest='n_jobs', type=int,
                   default=1,
                   help='Number of core to use validation_curve on.')
    p.add_argument('--outfile', dest='outfile',
                   default='output.csv',
                   help='Output filename for validation data.')

    args = p.parse_args()

    color_dict = comp.analysis.get_color_dict()

    comp_class = True
    target = 'MC_comp_class' if comp_class else 'MC_comp'
    comp_list = ['light', 'heavy'] if comp_class else ['P', 'He', 'O', 'Fe']
    feature_list, feature_labels = comp.get_training_features()
    sim_train_df, sim_test_df = comp.load_dataframe(datatype='sim', target=target)

    pipeline_str = args.pipeline
    pipeline = comp.get_pipeline(pipeline_str)
    if pipeline_str == 'xgboost':
        pipeline.named_steps['classifier'].set_params(nthread=1)
    else:
        try:
            pipeline.named_steps['classifier'].set_params(n_jobs=1)
        except:
            pass

    type_decoder = {'int': int, 'float': float, 'string': str}
    dtype = type_decoder[args.param_type]
    pipeline.named_steps['classifier'].set_params(**{args.param_name: dtype(args.param_value)})

    data_dict = {'classifier': args.pipeline,
                'param_name': args.param_name, 'param_value': dtype(args.param_value),
                'cv-folds': args.cv}

    train_scores = defaultdict(list)
    test_scores = defaultdict(list)
    ks_pval = defaultdict(list)
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=2)
    for train_index, test_index in skf.split(sim_train_df, sim_train_df['target']):

        sim_train_fold, sim_test_fold = sim_train_df.iloc[train_index], sim_train_df.iloc[test_index]

        X_train, y_train = comp.dataframe_functions.dataframe_to_X_y(sim_train_fold, feature_list)
        X_test, y_test = comp.dataframe_functions.dataframe_to_X_y(sim_test_fold, feature_list)

        pipeline = pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        train_score = accuracy_score(y_train, train_pred)
        train_scores['total'].append(train_score)
        train_probs = pipeline.predict_proba(X_train)

        test_pred = pipeline.predict(X_test)
        test_score = accuracy_score(y_test, test_pred)
        test_scores['total'].append(test_score)
        test_probs = pipeline.predict_proba(X_test)

        for class_ in pipeline.classes_:
            print(class_)
            # composition = sim_train.le.inverse_transform([class_])[0]
            #
            # comp_mask_train = sim_train.le.inverse_transform(train_targets) == composition
            # comp_score_train = accuracy_score(train_targets[comp_mask_train], train_pred[comp_mask_train])
            # train_scores[composition].append(comp_score_train)
            #
            # comp_mask_test = sim_train.le.inverse_transform(test_targets) == composition
            # comp_score_test = accuracy_score(test_targets[comp_mask_test], test_pred[comp_mask_test])
            # test_scores[composition].append(comp_score_test)
            #
            # pval = stats.ks_2samp(train_probs[comp_mask_train, class_], test_probs[comp_mask_test, class_])[1]
            # print(class_, pval)
            # ks_pval[composition].append(pval)

    for label in comp_list + ['total']:
        data_dict['train_mean_{}'.format(label)] = [np.mean(train_scores[label])]
        data_dict['train_std_{}'.format(label)] = [np.std(train_scores[label])]
        data_dict['validation_mean_{}'.format(label)] = [np.mean(test_scores[label])]
        data_dict['validation_std_{}'.format(label)] = [np.std(test_scores[label])]
        # data_dict['ks_mean_{}'.format(label)] = [np.mean(ks_pval[label])]
        # data_dict['ks_std_{}'.format(label)] = [np.std(ks_pval[label])]

    print('data_dict = {}'.format(data_dict))

    dataframe = pd.DataFrame.from_dict(data_dict)
    dataframe.to_csv(args.outfile)

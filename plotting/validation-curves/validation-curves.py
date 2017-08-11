#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from dask import delayed, multiprocessing, compute
from dask.diagnostics import ProgressBar

import comptools as comp


@delayed
def get_param_CV_dict(config, pipeline_str, param_name, param_value, n_splits=10):
    '''Calculates stratified k-fold CV scores for a given hyperparameter value

    Parameters
    ----------
    config : str
        Detector configuration.
    pipeline_str : str
        Name of pipeline to use (e.g. 'BDT').
    param_name : str
        Name of hyperparameter (e.g. 'max_depth', 'learning_rate', etc.).
    param_value : int, float, str
        Value to set hyperparameter to.
    n_splits : int, optional
        Number of (stratified) fold to use in cross validation
        (default is 10).

    Returns
    -------
        data_dict : dict
            Return a dictionary with average scores (accuracy and ks p-value)
            as well as CV errors on those scores.

    '''

    comp_list = ['light', 'heavy']
    feature_list, feature_labels = comp.get_training_features()
    df_sim_train, df_sim_test = comp.load_sim(config=config, verbose=False)

    pipeline = comp.get_pipeline(pipeline_str)
    pipeline.named_steps['classifier'].set_params(**{param_name: param_value})

    data_dict = {'classifier': pipeline_str, 'param_name': param_name,
                 'param_value': param_value, 'n_splits': n_splits}

    train_scores = defaultdict(list)
    test_scores = defaultdict(list)
    ks_pval = defaultdict(list)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)
    for train_index, test_index in skf.split(df_sim_train, df_sim_train['target']):

        df_train_fold, df_test_fold = df_sim_train.iloc[train_index], df_sim_train.iloc[test_index]

        X_train, y_train = comp.dataframe_functions.dataframe_to_X_y(df_train_fold, feature_list)
        X_test, y_test = comp.dataframe_functions.dataframe_to_X_y(df_test_fold, feature_list)

        pipeline = pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        train_score = accuracy_score(y_train, train_pred)
        train_scores['total'].append(train_score)
        train_probs = pipeline.predict_proba(X_train)

        test_pred = pipeline.predict(X_test)
        test_score = accuracy_score(y_test, test_pred)
        test_scores['total'].append(test_score)
        test_probs = pipeline.predict_proba(X_test)

        # Get composition scores
        for class_ in pipeline.classes_:
            composition = comp.dataframe_functions.label_to_comp(class_)

            comp_mask_train = y_train == class_
            comp_score_train = accuracy_score(y_train[comp_mask_train],
                                              train_pred[comp_mask_train])
            train_scores[composition].append(comp_score_train)

            comp_mask_test = y_test == class_
            comp_score_test = accuracy_score(y_test[comp_mask_test],
                                             test_pred[comp_mask_test])
            test_scores[composition].append(comp_score_test)

            pval = stats.ks_2samp(train_probs[comp_mask_train, class_],
                                  test_probs[comp_mask_test, class_])[1]
            ks_pval[composition].append(pval)

    for label in comp_list + ['total']:
        data_dict['train_mean_{}'.format(label)] = np.mean(train_scores[label])
        data_dict['train_std_{}'.format(label)] = np.std(train_scores[label])
        data_dict['validation_mean_{}'.format(label)] = np.mean(test_scores[label])
        data_dict['validation_std_{}'.format(label)] = np.std(test_scores[label])
        if label != 'total':
            data_dict['ks_mean_{}'.format(label)] = np.mean(ks_pval[label])
            data_dict['ks_std_{}'.format(label)] = np.std(ks_pval[label])

    return data_dict


@delayed
def plot_validation_curve_comp(df, outfile, xlabel, ylabel='Classification accuracy',
                               ylim=None):
    '''Makes and saves validation curve plot

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing CV scores (and errors) for each hyperparameter
        value that was scanned over. Note: the index of df are the
        hyperparameter values.
    outfile : str, path
        Output image file to save plot.
    xlabel : str
        Label for hyperparameter (e.g. 'Maximum depth', 'Learning rate',
        etc.). Will be used as the x-axis label.
    ylabel : str, optional
        Y-axis label (default is 'Classification accuracy').
    ylim : int, 2-tuple, None, optional
        Option to set y-axis limits (default is None).

    Returns
    -------
        None

    '''

    for key in ['light', 'heavy']:
        # Plot training curve
        score_mean_train = df['train_mean_{}'.format(key)]
        score_std_train = df['train_std_{}'.format(key)]
        plt.plot(df.index, score_mean_train, color=color_dict[key],
                 linestyle='-', marker='o', markersize=5,
                 label='{} training set'.format(key))
        plt.fill_between(df.index, score_mean_train + score_std_train,
                         score_mean_train - score_std_train, alpha=0.15,
                         color=color_dict[key])

        # Plot validataion curve
        score_mean_valid = df['validation_mean_{}'.format(key)]
        score_std_valid = df['validation_std_{}'.format(key)]
        plt.plot(df.index, score_mean_valid, color=color_dict[key],
                 linestyle=':', marker='^', markersize=5,
                 label='{} validation set'.format(key))
        plt.fill_between(df.index, score_mean_valid + score_std_valid,
                         score_mean_valid - score_std_valid, alpha=0.15,
                         color=color_dict[key])

    plt.grid()
    leg = plt.legend(loc='upper center', frameon=False,
              bbox_to_anchor=(0.5,  # horizontal
                              1.15),# vertical
              ncol=2, fancybox=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim: plt.ylim(ylim)
    plt.savefig(outfile)


if __name__ == "__main__":

    p = argparse.ArgumentParser(description='Validation curve plot maker')
    p.add_argument('--config', dest='config', default='IC86.2012',
                   choices=comp.simfunctions.get_sim_configs(),
                   help='Detector configuration')
    p.add_argument('--pipeline', dest='pipeline',
                   default='BDT',
                   help='Pipeline to use for classification.')
    p.add_argument('--param_name', dest='param_name',
                   default='max_depth',
                   help='Name of hyperparameter to tune.')
    p.add_argument('--param_label', dest='param_label',
                   default='Maximum depth',
                   help='Name of hyperparameter to go on plot')
    p.add_argument('--param_values', dest='param_values', nargs='*',
                   help='Values of hyperparameter.')
    p.add_argument('--param_type', dest='param_type',
                   default='int',
                   choices=['int', 'float', 'string'],
                   help='Type of hyperparameter.')
    p.add_argument('--scoring', dest='scoring',
                   default='accuracy',
                   help='Name of scoring metric to use in validation curve.')
    p.add_argument('--cv', dest='cv', type=int,
                   default=10,
                   help='Number of cross-validation folds.')

    args = p.parse_args()

    color_dict = comp.analysis.get_color_dict()

    # Params need to be converted to the appropreiate dtype
    params = np.asarray(args.param_values).astype(args.param_type)
    data_dicts = []
    for param_value in params:
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=2)
        data_dict = get_param_CV_dict(args.config, args.pipeline,
                        args.param_name, param_value, n_splits=args.cv)
        data_dicts.append(data_dict)

    df = delayed(pd.DataFrame.from_records)(data_dicts, index='param_value')
    outfile = os.path.join(comp.paths.figures_dir, 'model_evaluation',
                'validation-curves', '{}_{}.png'.format(args.config, args.param_name))
    comp.check_output_dir(outfile)
    plot = plot_validation_curve_comp(df, outfile, args.param_label,
                                      ylabel='Classification accuracy')
    print('Making validation curve for {}'.format(args.param_name))
    with ProgressBar():
        compute(plot, get=multiprocessing.get, num_works=min(len(params), 20))

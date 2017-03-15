#!/usr/bin/env python

from __future__ import division
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns

from sklearn.model_selection import learning_curve

from composition.analysis.load_sim import load_sim
from composition.analysis.preprocessing import get_train_test_sets
from composition.analysis.features import get_training_features
from composition.analysis.pipelines import get_pipeline


if __name__ == '__main__':

    sns.set_palette('muted')
    sns.set_color_codes()

    p = argparse.ArgumentParser(
        description='Runs extra modules over a given fileList')
    p.add_argument('-clf', '--classifier', dest='classifier',
                   default='RF',
                   choices=['RF', 'KN'],
                   help='Option to specify classifier used')
    p.add_argument('--outdir', dest='outdir',
                   default='/home/jbourbeau/public_html/figures/composition',
                   help='Output directory')
    args = p.parse_args()

    # Throughout this code, X will represent features while y will represent
    # class labels

    # Load and preprocess training data
    df = load_sim()
    feature_list = get_training_features()
    num_features = len(feature_list)
    X_train_std, X_test_std, y_train, y_test, le = get_train_test_sets(
        df, feature_list)

    pipeline = get_pipeline(args.classifier)

    train_sizes, train_scores, test_scores =\
        learning_curve(estimator=pipeline,
                       X=X_train_std,
                       y=y_train,
                       train_sizes=np.linspace(0.1, 1.0, 10),
                       cv=10,
                       n_jobs=4,
                       verbose=3)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean,
             color='b', linestyle='-',
             marker='o', markersize=5,
             label='training accuracy')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='g', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.title(args.classifier + ' Classifier')
    plt.legend()
    # plt.ylim([0.8, 1.0])
    plt.tight_layout()
    outfile = args.outdir + '/learning-curve_{}.png'.format(args.classifier)
    plt.savefig(outfile)

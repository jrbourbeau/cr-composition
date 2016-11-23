#!/usr/bin/env python

from collections import Counter
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import scipy.stats as stats

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier

from composition.analysis.load_sim import load_sim


if __name__ == '__main__':

    sns.set_palette('muted')

    p = argparse.ArgumentParser(
        description='Runs extra modules over a given fileList')
    p.add_argument('--outdir', dest='outdir',
                   default='/home/jbourbeau/public_html/figures/composition',
                   help='Output directory')
    args = p.parse_args()

    '''Throughout this code, X will represent features,
       while y will represent class labels'''

    df = load_sim()
    # Preprocess training data
    feature_list = np.array(['reco_cos_zenith', 'InIce_log_charge',
                             'NChannels', 'NStations', 'reco_radius', 'reco_InIce_containment'])
    # feature_list = np.array(['reco_log_energy', 'reco_cos_zenith', 'InIce_log_charge',
    #                          'NChannels', 'NStations', 'reco_radius', 'reco_InIce_containment'])
    X, y = df[feature_list].values, df.MC_comp.values
    # Convert comp string labels to numerical labels
    y = LabelEncoder().fit_transform(y)

    # Split data into training and test samples
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=2)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # Scale features and labels
    # NOTE: the scaler is fit only to the training features
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    print('events = ' + str(y_train.shape[0]))

    for depth in range(1, 10):
        forest = RandomForestClassifier(n_estimators=100, max_depth=depth, n_jobs=-1)
        # Train forest on training data
        forest.fit(X_train_std, y_train)
        train_proba = forest.predict_proba(X_train_std)
        test_proba = forest.predict_proba(X_test_std)
        # print(train_proba)
        # print(train_proba[:,0])
        # print(train_proba[:,1])
        print(stats.ks_2samp(train_proba[:,0], test_proba[:,0]))
    # name = forest.__class__.__name__
    # importances = forest.feature_importances_
    # indices = np.argsort(importances)[::-1]
    #
    # fig, ax = plt.subplots()
    # feature_labels = np.array(['Energy', 'MC Zenith', 'InIce charge', 'NChannels'])
    # for f in range(X_train_std.shape[1]):
    #     print('{}) {}'.format(f + 1, importances[indices[f]]))
    #
    # plt.ylabel('Feature Importances')
    # plt.bar(range(X_train_std.shape[1]),
    #         importances[indices],
    #         # color='lightblue',
    #         align='center')
    #
    # plt.xticks(range(X_train_std.shape[1]),
    #            feature_labels[indices], rotation=90)
    # plt.xlim([-1, X_train_std.shape[1]])
    # outfile = args.outdir + '/random_forest_feature_importance.png'
    # plt.savefig(outfile)
    #
    #
    # d = pd.DataFrame(df, columns=feature_list)
    # # Compute the correlation matrix
    # corr = d.corr()
    # # Generate a mask for the upper triangle
    # mask = np.zeros_like(corr, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True
    #
    # fig, ax = plt.subplots()
    # sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
    #             square=True, xticklabels=feature_labels, yticklabels=feature_labels,
    #             linewidths=.5, cbar_kws={'label':'Covariance'}, ax=ax)
    # outfile = args.outdir + '/feature_covariance.png'
    # plt.savefig(outfile)
    #
    # print('='*30)
    # print(name)
    # test_predictions = forest.predict(X_test_std)
    # test_acc = accuracy_score(y_test, test_predictions)
    # print('Test accuracy: {:.4%}'.format(test_acc))
    # train_predictions = forest.predict(X_train_std)
    # train_acc = accuracy_score(y_train, train_predictions)
    # print('Train accuracy: {:.4%}'.format(train_acc))
    # print('='*30)

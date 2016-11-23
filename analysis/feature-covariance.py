#!/usr/bin/env python

from __future__ import division
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

from composition.analysis.load_sim import load_sim
from composition.analysis.preprocessing import get_train_test_sets
from composition.analysis.pipelines import get_pipeline


if __name__ == '__main__':

    sns.set_palette('muted')
    sns.set_color_codes()

    p = argparse.ArgumentParser(
        description='Runs extra modules over a given fileList')
    p.add_argument('-c', '--classifier', dest='classifier',
                   default='RF',
                   choices=['RF', 'KN'],
                   help='Option to specify classifier used')
    p.add_argument('--outdir', dest='outdir',
                   default='/home/jbourbeau/public_html/figures/composition',
                   help='Output directory')
    args = p.parse_args()

    # Load and preprocess training data
    df = load_sim()
    # df, cut_dict = load_sim(return_cut_dict=True)
    # selection_mask = np.array([True] * len(df))
    # standard_cut_keys = ['IT_containment', 'IceTopMaxSignalInEdge',
    #                      'IceTopMaxSignal', 'NChannels', 'InIce_containment']
    # for key in standard_cut_keys:
    #     selection_mask *= cut_dict[key]
    # # Add additional energy cut (so IceTop maximum effective area has been
    # # reached)
    # selection_mask *= (df.MC_log_energy >= 6.2)
    #
    # df = df[selection_mask]
    feature_list = np.array(['reco_log_energy', 'reco_cos_zenith', 'InIce_log_charge',
                             'NChannels', 'NStations', 'reco_radius', 'LLHlap_InIce_containment',
                             'log_s125', 'lap_chi2'])
    num_features = len(feature_list)

    feature_labels = np.array(['$\\log_{10}({\mathrm{E/GeV})$', '$\cos(\\theta)$', 'InIce charge',
                               'NChannels', 'NStations', 'IT reco radius', 'InIce containment', 'log(s125)', 'chi2'])
    d = pd.DataFrame(df, columns=feature_list)
    # Compute the correlation matrix
    corr = d.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots()
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
                square=True, xticklabels=feature_labels, yticklabels=feature_labels,
                linewidths=.5, cbar_kws={'label': 'Covariance'}, annot=True, ax=ax)
    outfile = args.outdir + '/feature_covariance.png'
    plt.savefig(outfile)

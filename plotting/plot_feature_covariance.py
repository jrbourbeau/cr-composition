#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn.apionly as sns

import comptools as comp
import comptools.analysis.plotting as plotting

color_dict = comp.analysis.get_color_dict()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Makes and saves feature covariance plot')
    parser.add_argument('-c', '--config', dest='config',
                   choices=comp.simfunctions.get_sim_configs(),
                   help='Detector configuration')
    args = parser.parse_args()

    df_sim_train, df_sim_test = comp.load_sim(config=args.config)
    feature_list, feature_labels = comp.analysis.get_training_features()

    corr = df_sim_train[feature_list].corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    mask[np.diag_indices_from(mask)] = False

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, square=True,
                xticklabels=feature_labels, yticklabels=feature_labels,
                cbar_kws={'label': 'Covariance'}, annot=True, ax=ax)

    outfile = os.path.join(comp.paths.figures_dir, 'model_evaluation',
                           'feature-covariance-{}.png'.format(args.config))
    comp.check_output_dir(outfile)
    plt.savefig(outfile)

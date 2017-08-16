#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import comptools as comp
import comptools.analysis.plotting as plotting

color_dict = comp.analysis.get_color_dict()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Makes and saves feature importance plot')
    parser.add_argument('-c', '--config', dest='config',
                   choices=comp.simfunctions.get_sim_configs(),
                   help='Detector configuration')
    args = parser.parse_args()

    df_sim_train, df_sim_test = comp.load_sim(config=args.config)
    pipeline_str = 'BDT'
    pipeline = comp.get_pipeline(pipeline_str)
    feature_list, feature_labels = comp.analysis.get_training_features()

    pipeline.fit(df_sim_train[feature_list], df_sim_train['target'])

    num_features = len(feature_list)
    importances = pipeline.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[::-1]

    for f in range(num_features):
        print('{}) {}'.format(f + 1, importances[indices[f]]))

    # Make feature importance plot
    fig, ax = plt.subplots()
    ax.set_ylabel('Feature Importance')
    ax.bar(range(num_features),
            importances[indices],
            align='center')

    plt.xticks(range(num_features), feature_labels[indices], rotation=90)
    ax.set_xlim([-0.5, len(feature_list)-0.5])
    ax.grid(axis='y')

    outfile = os.path.join(comp.paths.figures_dir, 'model_evaluation',
                           'feature-importance-{}.png'.format(args.config))
    comp.check_output_dir(outfile)
    plt.savefig(outfile)

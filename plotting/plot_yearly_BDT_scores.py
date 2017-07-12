#!/usr/bin/env python

from __future__ import division, print_function
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import multiprocessing as mp
from sklearn.externals import joblib

import comptools as comp
import comptools.analysis.plotting as plotting

color_dict = comp.analysis.get_color_dict()


def get_BDT_scores(config):

    df_data = comp.load_dataframe(datatype='data', config=config)
    feature_list, feature_labels = comp.get_training_features()
    df_data.loc[:, feature_list].dropna(axis=0, how='any', inplace=True)

    # Load saved pipeline
    model_file = os.path.join(comp.paths.project_home, 'models/GBDT_IC86.2012.pkl')
    pipeline = joblib.load(model_file)['pipeline']
    # Get BDT scores for each data event
    X_data = comp.dataframe_functions.dataframe_to_array(df_data, feature_list)
    classifier_scores = pipeline.decision_function(X_data)

    print('{} complete!'.format(config))

    return classifier_scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Extracts and saves desired information from simulation/data .i3 files')
    parser.add_argument('-c', '--config', dest='config', nargs='*',
                   choices=comp.datafunctions.get_data_configs(),
                   help='Detector configuration')
    args = parser.parse_args()

    # Energy distribution comparison plot
    score_pool = mp.Pool(processes=len(args.config))
    scores = score_pool.map(get_BDT_scores, args.config)

    config_scores_dict = dict(zip(args.config, scores))

    # fig, ax = plt.subplots()
    min_score, max_score = -3, 3
    score_bins = np.linspace(min_score, max_score, 25)
    # for idx, config in enumerate(args.config):
    #     score_counts = np.histogram(config_scores_dict[config], bins=score_bins)[0]
    #     score_freq = score_counts/np.sum(score_counts)
    #     score_freq_err = np.sqrt(score_counts)/np.sum(score_counts)
    #     plotting.plot_steps(score_bins, score_freq, yerr=score_freq_err,
    #                         color='C{}'.format(idx), label=config, alpha=0.8,
    #                         ax=ax)
    #
    # ax.set_ylabel('Frequency')
    # ax.set_xlabel('BDT score')
    # ax.set_ylim(0)
    # ax.set_xlim(min_score, max_score)
    # ax.grid()
    # ax.legend()

    gs = gridspec.GridSpec(2, 1, height_ratios=[1,1], hspace=0.1)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)

    for idx, config in enumerate(args.config):
        score_counts = np.histogram(config_scores_dict[config], bins=score_bins)[0]
        score_freq = score_counts/np.sum(score_counts)
        score_freq_err = np.sqrt(score_counts)/np.sum(score_counts)
        plotting.plot_steps(score_bins, score_freq, yerr=score_freq_err,
                            color='C{}'.format(idx), label=config, alpha=0.8,
                            ax=ax1)
    ax1.set_ylabel('Frequency')
    ax1.tick_params(labelbottom='off')
    ax1.grid()
    ax1.legend()

    for idx, config in enumerate(args.config):
        if config == 'IC86.2012': continue
        score_counts = np.histogram(config_scores_dict[config], bins=score_bins)[0]
        score_freq = score_counts/np.sum(score_counts)
        score_freq_err = np.sqrt(score_counts)/np.sum(score_counts)

        score_counts_2012 = np.histogram(config_scores_dict['IC86.2012'], bins=score_bins)[0]
        score_freq_2012 = score_counts_2012/np.sum(score_counts_2012)
        score_freq_err_2012 = np.sqrt(score_counts_2012)/np.sum(score_counts_2012)

        ratio, ratio_err = comp.analysis.ratio_error(score_freq, score_freq_err,
                               score_freq_2012, score_freq_err_2012)

        plotting.plot_steps(score_bins, ratio, yerr=ratio_err,
                            color='C{}'.format(idx), label=config, alpha=0.8,
                            ax=ax2)
    ax2.axhline(1, marker='None', linestyle='-.', color='k', lw=1.5)
    ax2.set_ylabel('$\mathrm{f/f_{2012}}$')
    # ax2.set_ylabel('Ratio with IC86.2012')
    ax2.set_xlabel('BDT score')
    # ax2.set_ylim(0)
    ax2.set_xlim(min_score, max_score)
    ax2.grid()

    score_outfile = os.path.join(comp.paths.figures_dir,
                                'yearly_data_comparisons', 'BDT_scores.png')
    comp.check_output_dir(score_outfile)
    plt.savefig(score_outfile)

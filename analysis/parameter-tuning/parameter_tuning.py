#!/usr/bin/env python

from __future__ import division
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn.apionly as sns

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve, GridSearchCV, cross_val_score

from composition.analysis.load_sim import load_sim
from composition.analysis.preprocessing import get_train_test_sets, LabelEncoder
from composition.analysis.pipelines import get_pipeline
from composition.analysis.plotting_functions import plot_decision_regions
import composition.analysis.data_functions as data_functions
from composition.support_functions.checkdir import checkdir


if __name__ == '__main__':

    sns.set_palette('muted')
    sns.set_color_codes()

    p = argparse.ArgumentParser(
        description='Runs extra modules over a given fileList')
    p.add_argument('-e', dest='energy',
                   choices=['MC', 'reco'],
                   default='reco',
                   help='Output directory')
    p.add_argument('-clf', dest='classifier',
                   choices=['RF', 'KN'],
                   default='KN',
                   help='Output directory')
    p.add_argument('--outdir', dest='outdir',
                   default='/home/jbourbeau/public_html/figures/composition/parameter-tuning',
                   help='Output directory')
    args = p.parse_args()
    checkdir(args.outdir + '/')

    '''Throughout this code, X will represent features,
       while y will represent class labels'''

    # df = load_sim()
    df, cut_dict = load_sim(return_cut_dict=True)
    selection_mask = np.array([True] * len(df))
    standard_cut_keys = ['reco_exists', 'reco_zenith', 'num_hits', 'IceTopMaxSignalInEdge', 'IceTopMaxSignal',
                         'IceTopNeighbourMaxSignal', 'StationDensity', 'max_charge_frac', 'reco_containment', 'min_energy']
    for key in standard_cut_keys:
        selection_mask *= cut_dict[key]

    df = df[selection_mask]

    feature_list = np.array(
        ['{}_log_energy'.format(args.energy), 'InIce_log_charge'])
    X_train, X_test, y_train, y_test, le = get_train_test_sets(
        df, feature_list)

    print('events = ' + str(y_train.shape[0]))

    max_depth_list = np.arange(1, 30)
    # pipeline.fit(X_train, y_train)
    # scaler = pipeline.named_steps['scaler']
    # clf = pipeline.named_steps['classifier']
    # clf_name = clf.__class__.__name__

    score_mean_list = []
    score_std_list = []
    for max_depth in max_depth_list:
        print('max_depth = {}'.format(max_depth))
        pipeline = get_pipeline(args.classifier)
        pipeline.named_steps['classifier'].set_params(max_depth=max_depth)
        scores = cross_val_score(pipeline, X_train, y_train,
                                 scoring='accuracy', cv=5, n_jobs=4)
        score_mean_list.append(np.mean(scores))
        score_std_list.append(np.std(scores))

    fig, ax = plt.subplots()
    score_mean_list = np.array(score_mean_list)
    score_std_list = np.array(score_std_list)
    ax.plot(max_depth_list, score_mean_list, marker='.', linestyle='-.',
            color='b')
    ax.fill_between(max_depth_list, score_mean_list + score_std_list,
        score_mean_list - score_std_list, color='b', alpha=0.15)
    ax.set_xlabel('Max depth')
    ax.set_ylabel('CV Accuracy')
    outfile = args.outdir + '/{}-parameter-tuning.png'.format(args.classifier)
    plt.savefig(outfile)
    #
    # if len(feature_list) == 2:
    #     fig, ax = plt.subplots()
    #     X_test_std = scaler.transform(X_test)
    #     plot_decision_regions(X_test_std, y_test, clf, scatter_fraction=None)
    #     # Adding axes annotations
    #     plt.xlabel('Scaled Energy')
    #     plt.ylabel('Scaled Charge')
    #     plt.title(clf_name)
    #     plt.legend()
    #     outfile = args.outdir + \
    #         '/{}-decision-regions_{}-energy.png'.format(
    #             args.classifier, args.energy)
    #     plt.savefig(outfile)
    #
    #     fig, ax = plt.subplots()
    #     plot_decision_regions(X_test_std, y_test, clf)
    #     # Adding axes annotations
    #     plt.xlabel('Scaled Energy')
    #     plt.ylabel('Scaled Charge')
    #     plt.title(clf_name)
    #     plt.legend()
    #     outfile = args.outdir + \
    #         '/{}-decision-regions-scatter_{}-energy.png'.format(
    #             args.classifier, args.energy)
    #     plt.savefig(outfile)
    #
    # print('=' * 30)
    # print(clf_name)
    # test_predictions = pipeline.predict(X_test)
    # print('test_predictions = {}'.format(test_predictions))
    # test_acc = accuracy_score(y_test, test_predictions)
    # print('Test accuracy: {:.4%}'.format(test_acc))
    # train_predictions = pipeline.predict(X_train)
    # train_acc = accuracy_score(y_train, train_predictions)
    # print('Train accuracy: {:.4%}'.format(train_acc))
    # print('=' * 30)
    #
    # correctly_identified_mask = (test_predictions == y_test)
    #
    # energy_bin_width = 0.2
    # energy_bins = np.arange(6.2, 9.51, energy_bin_width)
    # energy_midpoints = (energy_bins[1:] + energy_bins[:-1]) / 2
    # log_energy = X_test[:, 0]
    #
    # MC_proton_mask = (le.inverse_transform(y_test) == 'P')
    # MC_iron_mask = (le.inverse_transform(y_test) == 'Fe')
    # # Get number of MC proton and iron as a function of MC energy
    # num_MC_protons_energy = np.histogram(log_energy[MC_proton_mask],
    #                                      bins=energy_bins)[0]
    # num_MC_protons_energy_err = np.sqrt(num_MC_protons_energy)
    # num_MC_irons_energy = np.histogram(log_energy[MC_iron_mask],
    #                                    bins=energy_bins)[0]
    # num_MC_irons_energy_err = np.sqrt(num_MC_irons_energy)
    # num_MC_total_energy = np.histogram(log_energy, bins=energy_bins)[0]
    # num_MC_total_energy_err = np.sqrt(num_MC_total_energy)
    #
    # # Get number of reco proton and iron as a function of MC energy
    # num_reco_proton_energy = np.histogram(
    #     log_energy[MC_proton_mask & correctly_identified_mask],
    #     bins=energy_bins)[0]
    # num_reco_proton_energy_err = np.sqrt(num_reco_proton_energy)
    # num_reco_iron_energy = np.histogram(
    #     log_energy[MC_iron_mask & correctly_identified_mask],
    #     bins=energy_bins)[0]
    # num_reco_iron_energy_err = np.sqrt(num_reco_iron_energy)
    # num_reco_total_energy = np.histogram(
    #     log_energy[correctly_identified_mask],
    #     bins=energy_bins)[0]
    # num_reco_total_energy_err = np.sqrt(num_reco_total_energy)
    #
    # # Calculate reco proton and iron fractions as a function of MC energy
    # reco_proton_frac, reco_proton_frac_err = data_functions.ratio_error(
    #     num_reco_proton_energy, num_reco_proton_energy_err,
    #     num_MC_protons_energy, num_MC_protons_energy_err)
    #
    # reco_iron_frac, reco_iron_frac_err = data_functions.ratio_error(
    #     num_reco_iron_energy, num_reco_iron_energy_err,
    #     num_MC_irons_energy, num_MC_irons_energy_err)
    #
    # reco_total_frac, reco_total_frac_err = data_functions.ratio_error(
    #     num_reco_total_energy, num_reco_total_energy_err,
    #     num_MC_total_energy, num_MC_total_energy_err)
    #
    # # Plot fraction of events vs energy
    # fig, ax = plt.subplots()
    # ax.errorbar(energy_midpoints, reco_proton_frac,
    #             yerr=reco_proton_frac_err,
    #             # xerr=energy_bin_width / 2,
    #             marker='.', markersize=10,
    #             label='Proton')
    # ax.errorbar(energy_midpoints, reco_iron_frac,
    #             yerr=reco_iron_frac_err,
    #             # xerr=energy_bin_width / 2,
    #             marker='.', markersize=10,
    #             label='Iron')
    # ax.errorbar(energy_midpoints, reco_total_frac,
    #             yerr=reco_total_frac_err,
    #             # xerr=energy_bin_width / 2,
    #             marker='.', markersize=10,
    #             label='Total')
    # if args.energy == 'MC':
    #     plt.xlabel('$\log_{10}(E_{\mathrm{MC}}/\mathrm{GeV})$')
    # if args.energy == 'reco':
    #     plt.xlabel('$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$')
    # ax.set_ylabel('Fraction correctly identified')
    # ax.set_ylim([0.0, 1.0])
    # ax.set_xlim([6.2, 9.5])
    # # ax.set_xscale('log', nonposx='clip')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
    #            borderaxespad=0.)
    # if args.energy == 'MC':
    #     outfile = args.outdir + \
    #         '/fraction-reco-correct_vs_MC-energy_{}.png'.format(
    #             args.classifier)
    # if args.energy == 'reco':
    #     outfile = args.outdir + \
    #         '/fraction-reco-correct_vs_reco-energy_{}.png'.format(
    #             args.classifier)
    # plt.savefig(outfile)

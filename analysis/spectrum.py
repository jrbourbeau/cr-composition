#!/usr/bin/env python

from __future__ import division
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from icecube.weighting.weighting import from_simprod
from icecube.weighting.fluxes import GaisserH3a, GaisserH4a

from composition.analysis.load_sim import load_sim
from composition.analysis.preprocessing import get_train_test_sets
from composition.analysis.features import get_training_features
from composition.analysis.pipelines import get_pipeline
import composition.analysis.plotting_functions as plotting
import composition.analysis.data_functions as data_functions


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
                   default='RF',
                   help='Output directory')
    p.add_argument('--outdir', dest='outdir',
                   default='/home/jbourbeau/public_html/figures/composition',
                   help='Output directory')
    args = p.parse_args()

    '''Throughout this code, X will represent features,
       while y will represent class labels'''

    # df = load_sim()
    df, cut_dict = load_sim(return_cut_dict=True)
    selection_mask = np.array([True] * len(df))
    standard_cut_keys = ['reco_exists', 'reco_zenith', 'num_hits', 'IT_signal',
                         'StationDensity', 'max_charge_frac', 'reco_containment',
                         'energy_range']
    for key in standard_cut_keys:
        selection_mask *= cut_dict[key]

    df = df[selection_mask]

    # Calculate simulation event weights
    print('\nCalculating simulation event weights...\n')
    simlist = np.unique(df['sim'])
    num_files_dict = {'7006': 30000, '7007': 30000,
                      '7579': 12000, '7784': 12000}
    for i, sim in enumerate(simlist):
        if i == 0:
            generator = num_files_dict[sim] * from_simprod(int(sim))
        else:
            generator += num_files_dict[sim] * from_simprod(int(sim))
    MC_energy = df['MC_energy'].values
    MC_type = df['MC_type'].values
    flux = GaisserH3a()
    df['weights_H3a'] = flux(MC_energy, MC_type) / \
        generator(MC_energy, MC_type)
    flux = GaisserH4a()
    df['weights_H4a'] = flux(MC_energy, MC_type) / \
        generator(MC_energy, MC_type)

    # Train classifier
    feature_list = ['reco_log_energy', 'InIce_charge', 'reco_cos_zenith', 'lap_chi2',
                    'MC_log_energy', 'weights_H3a']
    X_train, X_test, y_train, y_test, le = get_train_test_sets(
        df, feature_list)
    MC_log_energy_testing = X_test[:, -2]
    reco_log_energy_testing = X_test[:, 0]
    weights_testing = X_test[:, -1]
    MC_proton_mask = (le.inverse_transform(y_test) == 'P')
    MC_iron_mask = (le.inverse_transform(y_test) == 'Fe')
    X_train = X_train[:, :4]
    X_test = X_test[:, :4]

    print('training events = ' + str(y_train.shape[0]))
    print('testing events = ' + str(y_test.shape[0]))

    pipeline = get_pipeline(args.classifier)
    pipeline.fit(X_train, y_train)
    scaler = pipeline.named_steps['scaler']
    clf = pipeline.named_steps['classifier']
    clf_name = clf.__class__.__name__
    print('=' * 30)
    print(clf_name)
    test_predictions = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, test_predictions)
    print('Test accuracy: {:.4%}'.format(test_acc))
    train_predictions = pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, train_predictions)
    print('Train accuracy: {:.4%}'.format(train_acc))
    print('=' * 30)

    energy_bin_width = 0.1
    energy_bins = np.arange(6.2, 8.1, energy_bin_width)
    energy_midpoints = (energy_bins[1:] + energy_bins[:-1]) / 2
    bin_widths = (10**energy_bins[1:] - 10**energy_bins[:-1])
    fig, ax = plt.subplots()
    events = np.histogram(reco_log_energy_testing[MC_proton_mask], bins=energy_bins,
                          weights=weights_testing[MC_proton_mask])[0] / bin_widths
    events_error = np.sqrt(np.histogram(reco_log_energy_testing[MC_proton_mask],
                                        bins=energy_bins, weights=weights_testing[MC_proton_mask]**2)[0]) / bin_widths
    ax.errorbar(energy_midpoints, events,
                yerr=events_error, marker='.', label='Proton', alpha=0.75)
    events = np.histogram(reco_log_energy_testing[MC_iron_mask], bins=energy_bins,
                          weights=weights_testing[MC_iron_mask])[0] / bin_widths
    events_error = np.sqrt(np.histogram(reco_log_energy_testing[MC_iron_mask],
                                        bins=energy_bins, weights=weights_testing[MC_iron_mask]**2)[0]) / bin_widths
    ax.errorbar(energy_midpoints, events,
                yerr=events_error, marker='.', label='Iron', alpha=0.75)
    events = np.histogram(reco_log_energy_testing, bins=energy_bins,
                          weights=weights_testing)[0] / bin_widths
    events_error = np.sqrt(np.histogram(reco_log_energy_testing,
                                        bins=energy_bins, weights=weights_testing**2)[0]) / bin_widths
    ax.errorbar(energy_midpoints, events,
                yerr=events_error, marker='.', label='Total', alpha=0.75)
    ax.set_yscale('log', nonposx='clip')
    ax.set_xlabel('$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$')
    ax.set_ylabel('dN/dE')
    plt.legend(title='MC composition')
    outfile = args.outdir + \
        '/MC-spectrum-{}.png'.format(args.classifier)
    plt.savefig(outfile)


    reco_proton_mask = (le.inverse_transform(test_predictions) == 'P')
    reco_iron_mask = (le.inverse_transform(test_predictions) == 'Fe')
    fig, ax = plt.subplots()
    events = np.histogram(reco_log_energy_testing[reco_proton_mask], bins=energy_bins,
                          weights=weights_testing[reco_proton_mask])[0] / bin_widths
    events_error = np.sqrt(np.histogram(reco_log_energy_testing[reco_proton_mask],
                                        bins=energy_bins, weights=weights_testing[reco_proton_mask]**2)[0]) / bin_widths
    ax.errorbar(energy_midpoints, events,
                yerr=events_error, marker='.', label='Proton', alpha=0.75)
    events = np.histogram(reco_log_energy_testing[reco_iron_mask], bins=energy_bins,
                          weights=weights_testing[reco_iron_mask])[0] / bin_widths
    events_error = np.sqrt(np.histogram(reco_log_energy_testing[reco_iron_mask],
                                        bins=energy_bins, weights=weights_testing[reco_iron_mask]**2)[0]) / bin_widths
    ax.errorbar(energy_midpoints, events,
                yerr=events_error, marker='.', label='Iron', alpha=0.75)
    events = np.histogram(reco_log_energy_testing, bins=energy_bins,
                          weights=weights_testing)[0] / bin_widths
    events_error = np.sqrt(np.histogram(reco_log_energy_testing,
                                        bins=energy_bins, weights=weights_testing**2)[0]) / bin_widths
    ax.errorbar(energy_midpoints, events,
                yerr=events_error, marker='.', label='Total', alpha=0.75)
    ax.set_yscale('log', nonposx='clip')
    ax.set_xlabel('$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$')
    ax.set_ylabel('dN/dE')
    plt.legend(title='Reco composition')
    outfile = args.outdir + \
        '/reco-spectrum-{}.png'.format(args.classifier)
    plt.savefig(outfile)

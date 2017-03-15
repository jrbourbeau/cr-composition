#!/usr/bin/env python

from __future__ import division
from collections import Counter
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from mlxtend.evaluate import plot_decision_regions

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier

from icecube import ShowerLLH

from composition.analysis.load_sim import load_sim


if __name__ == '__main__':

    sns.set_palette('muted')
    sns.set_color_codes()

    p = argparse.ArgumentParser(
        description='Runs extra modules over a given fileList')
    p.add_argument('-e', dest='energy',
                   choices=['MC', 'reco'],
                   default='MC',
                   help='Output directory')
    p.add_argument('--outdir', dest='outdir',
                   default='/home/jbourbeau/public_html/figures/composition/baseline',
                   help='Output directory')
    args = p.parse_args()

    '''Throughout this code, X will represent features,
       while y will represent class labels'''

    df, cut_dict = load_sim(return_cut_dict=True)
    selection_mask = np.array([True] * len(df))
    standard_cut_keys = ['IT_containment', 'IceTopMaxSignalInEdge',
                         'IceTopMaxSignal', 'NChannels', 'InIce_containment']
    for key in standard_cut_keys:
        selection_mask *= cut_dict[key]

    # Add additional energy cut (so IceTop maximum effective area has been
    # reached)
    selection_mask *= (df.MC_log_energy >= 6.2)

    df = df[selection_mask]

    MC_proton_mask = (df.MC_comp == 'P')
    MC_iron_mask = (df.MC_comp == 'Fe')
    # LLH_bins = ShowerLLH.LLHBins(bintype='logdist')
    # energy_bins = LLH_bins.bins['E']
    # energy_bins = energy_bins[energy_bins >= 6.2]
    energy_bin_width = 0.2
    energy_bins = np.arange(6.2, 9.51, energy_bin_width)
    energy_midpoints = (energy_bins[1:] + energy_bins[:-1]) / 2
    if args.energy == 'MC':
        log_energy = df.MC_log_energy
    if args.energy == 'reco':
        log_energy = df.reco_log_energy
    charge = df.InIce_charge

    # Determine reconstructed compositions
    def line_fit(array):
        fit = []
        for x in array:
            if x <= 9.0:
                slope = (5.3-2.55)/(9.5-6.2)
                fit.append(2.55 + slope * (x - 6.2))
            else:
                slope = (5.20-4.9)/(9.5-9.0)
                fit.append(4.9 + slope * (x - 9.0))
        fit = np.array(fit)
        return fit
    charge_cut = line_fit(log_energy)
    reco_comp = []
    for q, cut in zip(np.log10(charge), charge_cut):
        if q <= cut:
            reco_comp.append('P')
        else:
            reco_comp.append('Fe')
    reco_comp = np.array(reco_comp)
    correctly_identified_mask = (reco_comp == df.MC_comp)
    print('fraction correctly identified = {}'.format(sum(correctly_identified_mask)/len(correctly_identified_mask)))

    # Get number of MC proton and iron as a function of MC energy
    num_MC_protons_energy = np.histogram(log_energy[MC_proton_mask],
            bins=energy_bins)[0]
    num_MC_protons_energy_err = np.sqrt(num_MC_protons_energy)
    num_MC_irons_energy = np.histogram(log_energy[MC_iron_mask],
            bins=energy_bins)[0]
    num_MC_irons_energy_err = np.sqrt(num_MC_irons_energy)
    num_MC_total_energy = np.histogram(log_energy, bins=energy_bins)[0]
    num_MC_total_energy_err = np.sqrt(num_MC_total_energy)

    # Get number of reco proton and iron as a function of MC energy
    num_reco_proton_energy = np.histogram(
        log_energy[MC_proton_mask & correctly_identified_mask],
        bins=energy_bins)[0]
    num_reco_proton_energy_err = np.sqrt(num_reco_proton_energy)
    num_reco_iron_energy = np.histogram(
        log_energy[MC_iron_mask & correctly_identified_mask],
        bins=energy_bins)[0]
    num_reco_iron_energy_err = np.sqrt(num_reco_iron_energy)
    num_reco_total_energy = np.histogram(
        log_energy[correctly_identified_mask],
        bins=energy_bins)[0]
    num_reco_total_energy_err = np.sqrt(num_reco_total_energy)

    # Calculate reco proton and iron fractions as a function of MC energy
    reco_proton_frac = num_reco_proton_energy/num_MC_protons_energy
    reco_proton_frac_err = reco_proton_frac*np.sqrt(
            ((num_reco_proton_energy_err)/(num_reco_proton_energy))**2 +
            ((num_MC_protons_energy_err)/(num_MC_protons_energy))**2)

    reco_iron_frac = num_reco_iron_energy/num_MC_irons_energy
    reco_iron_frac_err = reco_iron_frac*np.sqrt(
            ((num_reco_iron_energy_err)/(num_reco_iron_energy))**2 +
            ((num_MC_irons_energy_err/num_MC_irons_energy)**2))

    reco_total_frac = num_reco_total_energy/num_MC_total_energy
    reco_total_frac_err = reco_total_frac*np.sqrt(
            ((num_reco_total_energy_err)/(num_reco_total_energy))**2 +
            ((num_MC_total_energy_err)/(num_MC_total_energy))**2)

    # Plot fraction of events vs energy
    fig, ax = plt.subplots()
    ax.errorbar(energy_midpoints, reco_proton_frac,
            yerr=reco_proton_frac_err,
            marker='.', markersize=10,
            label='Proton')
    ax.errorbar(energy_midpoints, reco_iron_frac,
            yerr=reco_iron_frac_err,
            marker='.', markersize=10,
            label='Iron')
    ax.errorbar(energy_midpoints, reco_total_frac,
            yerr=reco_total_frac_err,
            marker='.', markersize=10,
            label='Total')
    # ax.axhline(0.5, linestyle='-.', marker='None', color='k')
    if args.energy == 'MC':
        plt.xlabel('$\log_{10}(E_{\mathrm{MC}}/\mathrm{GeV})$')
    if args.energy == 'ML':
        plt.xlabel('$\log_{10}(E_{\mathrm{ML}}/\mathrm{GeV})$')
    ax.set_ylabel('Fraction correctly identified')
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([6.2, 9.5])
    # ax.set_xscale('log', nonposx='clip')
    # plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
            borderaxespad=0.)
    if args.energy == 'MC':
        outfile = args.outdir + '/fraction-reco-correct_vs_MC-energy.png'
    if args.energy == 'ML':
        outfile = args.outdir + '/fraction-reco-correct_vs_ML-energy.png'
    plt.savefig(outfile)

#!/usr/bin/env python

from __future__ import division
import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse
import seaborn.apionly as sns

from icecube import ShowerLLH

from composition.analysis.load_sim import load_sim
from composition.support_functions.checkdir import checkdir

if __name__ == "__main__":

    sns.set_palette('muted')
    sns.set_color_codes()

    p = argparse.ArgumentParser(
        description='Creates performance plots for ShowerLLH')
    p.add_argument('-o', '--outdir', dest='outdir',
                   default='/home/jbourbeau/public_html/figures/composition/baseline', help='Output directory')
    p.add_argument('-e', '--energy', dest='energy',
                   default='MC',
                   choices=['MC', 'reco'],
                   help='Option for a variety of preset bin values')
    p.add_argument('--extended', dest='extended',
                   default=False, action='store_true',
                   help='Use extended energy range')
    args = p.parse_args()
    checkdir(args.outdir + '/')

    # Import ShowerLLH sim reconstructions and cuts to be made
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

    nchannels = np.log10(df.NChannels)
    charge = df.InIce_log_charge

    # 2D charge vs nchannels histogram of proton fraction
    charge_bins = np.linspace(0, 7, 100)
    nchannels_bins = np.linspace(0, 4, 100)
    h, xedges, yedges = np.histogram2d(nchannels,
                                                 charge,
                                                 bins=[nchannels_bins,
                                                       charge_bins],
                                                 normed=False)
    h = np.rot90(h)
    h = np.flipud(h)
    h = np.ma.masked_where(h == 0, h)
    h = np.log10(h)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]

    fig, ax = plt.subplots()
    # colormap = 'coolwarm'
    colormap = 'viridis'
    plt.imshow(h, extent=extent, origin='lower',
               interpolation='none', cmap=colormap,
               aspect=0.5)
    # x = np.arange(6.2, 9.51, 0.1)
    # plt.plot([0,10], [0,10], marker='None', linestyle='-.',
    #          color='k')
    if args.energy == 'MC':
        plt.xlabel('$\log_{10}(\mathrm{NChannels})$')
    if args.energy == 'reco':
        plt.xlabel('$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$')
    plt.ylabel('$\log_{10}(Q_{\mathrm{total}})$')
    plt.xlim([0.5, 3.5])
    plt.ylim([0.5, 6])
    cb = plt.colorbar(
        label='$\log_{10}(\mathrm{Counts})$')
    if args.energy == 'MC':
        outfile = args.outdir + '/charge-vs-NChannels.png'
    if args.energy == 'reco':
        outfile = args.outdir + '/charge-vs-reco-energy.png'
    plt.savefig(outfile)
    plt.close()

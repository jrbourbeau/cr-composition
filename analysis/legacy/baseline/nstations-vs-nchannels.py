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

    df = df[selection_mask]

    nchannels = np.log10(df.NChannels)
    nstations = np.log10(df.NStations)

    # 2D charge vs nchannels histogram of proton fraction
    nstations_bins = np.linspace(0, 2, 75)
    nchannels_bins = np.linspace(0, 4, 75)
    h, xedges, yedges = np.histogram2d(nchannels,
                                       nstations,
                                       bins=[nchannels_bins,
                                             nstations_bins],
                                       normed=False)
    h = np.rot90(h)
    h = np.flipud(h)
    h = np.ma.masked_where(h == 0, h)
    h = np.log10(h)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]

    def line_fit(array):
        fit = []
        for x in array:
            if x <= 9.0:
                slope = (5.3 - 2.55) / (9.5 - 6.2)
                fit.append(2.55 + slope * (x - 6.2))
            else:
                slope = (5.20 - 4.9) / (9.5 - 9.0)
                fit.append(4.9 + slope * (x - 9.0))
        fit = np.array(fit)
        return fit

    fig, ax = plt.subplots()
    # colormap = 'coolwarm'
    colormap = 'viridis'
    plt.imshow(h, extent=extent, origin='lower',
               interpolation='none', cmap=colormap,
               aspect=1)
    # x = np.arange(6.2, 9.51, 0.1)
    # plt.plot(x, line_fit(x), marker='None', linestyle='--',
    #          color='k')
    if args.energy == 'MC':
        plt.xlabel('$\log_{10}(\mathrm{NChannels})$')
    if args.energy == 'reco':
        plt.xlabel('$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$')
    plt.ylabel('$\log_{10}(\mathrm{NStations})$')
    # plt.xlim([6.2, 9.5])
    cb = plt.colorbar(
        label='$\log_{10}(\mathrm{Counts})$')
    if args.energy == 'MC':
        outfile = args.outdir + '/NStations-vs-NChannels.png'
    if args.energy == 'reco':
        outfile = args.outdir + '/charge-vs-reco-energy.png'
    plt.savefig(outfile)
    plt.close()

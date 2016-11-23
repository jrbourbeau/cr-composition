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
    args = p.parse_args()
    checkdir(args.outdir + '/')

    # Import ShowerLLH sim reconstructions and cuts to be made
    df = load_sim()

    MC_proton_mask = (df.MC_comp == 'P')
    MC_iron_mask = (df.MC_comp == 'Fe')
    log_s125 = df['log_s125']
    charge = df.InIce_log_charge

    # 2D charge vs nchannels histogram of proton fraction
    charge_bins = np.linspace(0, 7, 100)
    s125_bins = np.linspace(min(log_s125), max(log_s125), 100)
    s125_midpoints = (s125_bins[1:] + s125_bins[:-1]) / 2
    proton_hist, xedges, yedges = np.histogram2d(log_s125[MC_proton_mask],
                                       charge[MC_proton_mask],
                                       bins=[s125_bins, charge_bins],
                                       normed=False)
    proton_hist = np.ma.masked_where(proton_hist == 0, proton_hist)
    iron_hist, xedges, yedges = np.histogram2d(log_s125[MC_iron_mask],
                                       charge[MC_iron_mask],
                                       bins=[s125_bins, charge_bins],
                                       normed=False)

    h = proton_hist/(proton_hist + iron_hist)
    h = np.rot90(h)
    h = np.flipud(h)
    h = np.ma.masked_where(h == 0, h)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    def line_fit(array):
        fit = []
        for x in array:
            slope = (4-3.2)/(2-1)
            fit.append(3.2 + slope * (x - 1))
        fit = np.array(fit)
        return fit

    fig, ax = plt.subplots()
    colormap = 'coolwarm'
    plt.imshow(h, extent=extent, origin='lower',
               interpolation='none', cmap=colormap,
               aspect=0.5, vmin=0, vmax=1)
    x = np.arange(min(log_s125), max(log_s125), 0.1)
    plt.plot(x, line_fit(x), marker='None', linestyle='--',
            color='k')
    plt.xlabel('$\log_{10}(\mathrm{s125})$')
    plt.ylabel('$\log_{10}(Q_{\mathrm{total}})$')
    # plt.xlim([6.2, 9.5])
    cb = plt.colorbar(
        label='P/(P+Fe) Fraction')
    outfile = args.outdir + '/charge-vs-s125.png'
    plt.savefig(outfile)
    plt.close()

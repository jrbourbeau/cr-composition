#!/usr/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
import seaborn.apionly as sns

import composition.support_functions.paths as paths
from composition.support_functions.checkdir import checkdir
from composition.analysis.load_sim import load_sim
# from effective_area import getEff
from ShowerLLH_scripts.analysis.LLH_tools import *
# from LLH_tools import *
# from zfix import zfix

def histogram_2D(x, y, bins, log_counts=False, **opts):
    h, xedges, yedges = np.histogram2d(x, y, bins=bins, normed=False)
    h = np.rot90(h)
    h = np.flipud(h)
    h = np.ma.masked_where(h == 0, h)
    if log_counts:
        h = np.log10(h)
    extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
    colormap = 'viridis'
    plt.imshow(h, extent=extent, origin='lower',
               interpolation='none', cmap=colormap)
    # plt.xlabel('$\log_{10}(E_\mathrm{MC}/\mathrm{GeV})$')
    # plt.ylabel('$\log_{10}(E_{\mathrm{ML}}/\mathrm{GeV})$')
    # plt.title(r'ShowerLLH - IT73 - {} LLH bins'.format(opts['bintype']))
    # plt.xlim([5, 9.5])
    # plt.ylim([5, 9.5])
    # cb = plt.colorbar(
    #     label='$\log_{10}{P(E_{\mathrm{ML}}|E_{\mathrm{MC}})}$')
    # plt.plot([0, 10], [0, 10], linestyle='--', color='k')
    # outfile = opts['outdir'] + '/' + \
    #     'MLenergy_vs_MCenergy_{}.png'.format(opts['bintype'])
    # plt.savefig(outfile)
    # plt.close()



if __name__ == "__main__":
    # Global variables setup for path names
    mypaths = paths.Paths()

    p = argparse.ArgumentParser(
        description='Creates performance plots for ShowerLLH')
    p.add_argument('-c', '--config', dest='config',
                   default='IT73',
                   choices=['IT73', 'IT81'],
                   help='Detector configuration')
    p.add_argument('-o', '--outdir', dest='outdir',
                   default='/home/jbourbeau/public_html/figures/composition/ShowerLLH',
                   help='Output directory')
    p.add_argument('-b', '--bintype', dest='bintype',
                   default='logdist',
                   choices=['standard', 'nozenith', 'logdist'],
                   help='Option for a variety of preset bin values')
    p.add_argument('-n', '--numbins', dest='numbins', type=float,
                   default=30, help='Number of energy bins')
    args = p.parse_args()
    checkdir(args.outdir + '/')
    opts = vars(args).copy()

    # df = load_sim()
    df, cut_dict = load_sim(return_cut_dict=True)
    selection_mask = np.array([True] * len(df))
    standard_cut_keys = ['reco_exists', 'MC_zenith',
                         'IceTopMaxSignalInEdge', 'IceTopMaxSignal']
    for key in standard_cut_keys:
        selection_mask *= cut_dict[key]

    print('n_events before cuts = {}'.format(len(df)))
    df = df[selection_mask]
    print('n_events after cuts = {}'.format(len(df)))

    MC_IT_containment = df.IceTop_FractionContainment
    reco_IT_containment = df.reco_IT_containment

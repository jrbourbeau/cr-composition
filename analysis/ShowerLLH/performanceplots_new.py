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

#==================
# Plotting functions
#==================


def plot_vs_energy(ax, y, sim_reco, weights, cuts, **kwargs):
    energy_bins = get_energy_bins()
    energy_bins = np.linspace(
        energy_bins.min(), energy_bins.max(), kwargs['numbins'] + 1)
    energyMC = sim_reco['MC_energy'][cuts]

    bin_centers, bin_medians, error = get_medians_weighted(
        np.log10(energyMC), y, weights, energy_bins)
    # bin_centers, bin_medians, error = get_medians(
    #     np.log10(energyMC), y, energy_bins)

    try:
        ax.errorbar(bin_centers, bin_medians, yerr=error,
                    marker='.', label=kwargs['label'], color=kwargs['color'])
    except KeyError:
        ax.errorbar(bin_centers, bin_medians, yerr=error,
                    marker='.')
    ax.set_xlabel('$\log_{10}(E_{\mathrm{MC}}/\mathrm{GeV})$')
    ax.set_ylabel(kwargs['ylabel'])
    ax.xaxis.set_ticks([5,6,7,8,9])
    try:
        ax.set_ylim(kwargs['ylim'])
    except:
        return
    # ax.legend(fancybox=True, framealpha=0.5)


def plot_vs_zenith(ax, y, sim_reco, weights, cuts, **kwargs):
    zenith_bins = np.linspace(0.75, 1.0, 46)
    zenithLLH = np.cos(np.pi - sim_reco['zenith'][cuts])

    bin_centers, bin_medians, error = get_medians_weighted(
        zenithLLH, y, weights, zenith_bins)
    # bin_centers, bin_medians, error = get_medians(
    #     zenithLLH, y, zenith_bins)

    try:
        ax.errorbar(bin_centers, bin_medians, yerr=error,
                    marker='.', label=kwargs['label'], color=kwargs['color'])
    except KeyError:
        ax.errorbar(bin_centers, bin_medians, yerr=error,
                    marker='.')
    ax.set_xlabel('$\cos(\\theta_{\mathrm{reco}})$')
    ax.set_ylabel(kwargs['ylabel'])
    try:
        ax.set_ylim(kwargs['ylim'])
    except:
        return
    # ax.legend(fancybox=True, framealpha=0.5)


#========================
# Calculational functions
#========================

def plot_energyres(ax, sim_reco, cuts, **opts):
    energyMC = np.log10(sim_reco['MC_energy'][cuts])
    zenithLLH = np.cos(np.pi - sim_reco['zenith'][cuts])
    energyLLH = np.log10(sim_reco['ML_energy'][cuts])
    # energyLLH = np.log10(sim_reco['ML_energy'][cuts])-sim_reco['fit'](zenithLLH)
    energy_res = energyLLH - energyMC
    # energy_res = np.log10(energyLLH / energyMC)

    weights = sim_reco['weights'][cuts]

    opts['ylabel'] = '$\log_{10}(E_{\mathrm{ML}}/E_{\mathrm{MC}})$'
    if opts['xaxis'] == 'energy':
        plot_vs_energy(ax, energy_res, sim_reco, weights, cuts, **opts)
    if opts['xaxis'] == 'zenith':
        plot_vs_zenith(ax, energy_res, sim_reco, weights, cuts, **opts)


def plot_coreres(ax, sim_reco, cuts, **opts):
    MC_x = sim_reco['MC_x'][cuts]
    MC_y = sim_reco['MC_y'][cuts]
    ML_x = sim_reco['ML_x'][cuts]
    ML_y = sim_reco['ML_y'][cuts]
    core_res = np.sqrt((ML_x - MC_x)**2 + (ML_y - MC_y)**2)

    weights = sim_reco['weights'][cuts]

    opts['ylabel'] = '$\\vec{x}_{ML}-\\vec{x}_{\mathrm{MC}} \ [m]$'
    if opts['xaxis'] == 'energy':
        plot_vs_energy(ax, core_res, sim_reco, weights, cuts, **opts)
    if opts['xaxis'] == 'zenith':
        plot_vs_zenith(ax, core_res, sim_reco, weights, cuts, **opts)


def eres_position(sim_reco, cuts, **opts):
    energyMC = sim_reco['MC_energy'][cuts]
    energyLLH = sim_reco['ML_energy'][cuts]
    energy_res = np.log10(energyLLH / energyMC)
    MC_x = sim_reco['MC_x'][cuts]
    MC_y = sim_reco['MC_y'][cuts]
    core_pos = np.sqrt(MC_x**2 + MC_y**2)

    # Distance bins in [m]
    distance_bins = np.append(np.arange(0, 600, 15), np.arange(600, 1051, 50))

    bin_centers, bin_medians, error = get_medians(
        core_pos, energy_res, distance_bins)

    plt.errorbar(bin_centers, bin_medians, yerr=error, marker='.', )
    plt.xlim([0, 1000])
    plt.xlabel('$\mathrm{Core \ Position} \ [m]$')
    plt.ylabel('$\log_{10}(E_{\mathrm{ML}}/E_{\mathrm{MC}})$')
    plt.title(r'ShowerLLH - IT73 - {} LLH bins'.format(opts['bintype']))
    outfile = opts['outdir'] + '/' + \
        'energy_res_vs_pos_{}.png'.format(opts['bintype'])
    plt.savefig(outfile)
    plt.close()


def LLHenergy_MCenergy(df, cuts, **opts):
    energyMC = df.MC_log_energy[cuts].values
    print('energyMC = {}'.format(energyMC))
    energyLLH = df.reco_log_energy[cuts].values
    print('energyLLH = {}'.format(energyLLH))
    energyLLH = df.reco_log_energy[cuts]
    # Energy bins in Log10(Energy/GeV)
    energy_bins = get_energy_bins()

    h, xedges, yedges = np.histogram2d(energyMC, energyLLH, bins=energy_bins,
                                       normed=False)
    h = np.rot90(h)
    h = np.flipud(h)
    h = np.ma.masked_where(h == 0, h)
    ntot = np.sum(h, axis=0).astype('float')
    ntot[ntot == 0] = 1.
    h /= ntot
    h = np.log10(h)
    # print h.max(), h.min()
    extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]

    colormap = 'viridis'
    plt.imshow(h, extent=extent, origin='lower',
               interpolation='none', cmap=colormap)
    plt.xlabel('$\log_{10}(E_\mathrm{MC}/\mathrm{GeV})$')
    plt.ylabel('$\log_{10}(E_{\mathrm{ML}}/\mathrm{GeV})$')
    plt.title(r'ShowerLLH - IT73 - {} LLH bins'.format(opts['bintype']))
    plt.xlim([5, 9.5])
    plt.ylim([5, 9.5])
    cb = plt.colorbar(
        label='$\log_{10}{P(E_{\mathrm{ML}}|E_{\mathrm{MC}})}$')
    plt.plot([0, 10], [0, 10], linestyle='--', color='k')
    outfile = opts['outdir'] + '/' + \
        'MLenergy_vs_MCenergy_{}.png'.format(opts['bintype'])
    plt.savefig(outfile)
    plt.close()


def effarea(sim_reco, cuts, **opts):
    effarea, sigma, relerr = getEff(sim_reco, cuts)
    energy_bins = get_bin_mids(get_energy_bins(reco=True))
    plt.errorbar(energy_bins, effarea, yerr=sigma, marker='.')
    plt.ylabel('$\mathrm{Effective \ Area} \ [\mathrm{m^2}]$')
    plt.xlabel('$\log_{10}(E_{\mathrm{MC}}/\mathrm{GeV})$')
    plt.title(r'ShowerLLH - IT73 - {} LLH bins'.format(opts['bintype']))
    # plt.xlim([5.5, 9.5])
    outfile = opts['outdir'] + '/' + \
        'effarea_vs_energy_{}.png'.format(opts['bintype'])
    plt.savefig(outfile)
    plt.close()

def make_performance_plots(df, cuts, **kwargs):
    fig, axarr = plt.subplots(2, 2)
    plt.tight_layout()
    # Energy resolution plots
    MC_energy  = df.MC_energy[cuts].values
    # print('MC_energy = {}'.format(MC_energy))
    # ML_energy  = 10**(np.log10(df.reco_energy[cuts])-zfix(np.pi-df.reco_zenith[cuts]))
    ML_energy  = df.reco_energy[cuts].values
    # print('ML_energy = {}'.format(ML_energy))
    # print('ML_energy = {}'.format(ML_energy))
    energy_res = np.log10(ML_energy/MC_energy)
    energy_bins = get_energy_bins()
    energy_bins = np.linspace(energy_bins.min(),
            energy_bins.max(), 31)
    bin_centers, bin_medians, error = get_medians(
        np.log10(MC_energy), energy_res, energy_bins)
    # bin_centers, bin_medians, error = get_medians_weighted(
    #     np.log10(MC_energy), energy_res, weights, energy_bins)
    try:
        axarr[0,0].errorbar(bin_centers, bin_medians, yerr=error,
                    label=kwargs['label'], color=kwargs['color'],
                    marker='.')
    except KeyError:
        axarr[0,0].errorbar(bin_centers, bin_medians, yerr=error,
                    marker='.')
    axarr[0,0].set_xlabel('$\log_{10}(E_{\mathrm{MC}}/\mathrm{GeV})$')
    axarr[0,0].set_ylabel('$\log_{10}(E_{\mathrm{ML}}/E_{\mathrm{MC}})$')
    axarr[0,0].xaxis.set_ticks(np.arange(5.0, 9.6, 0.5))
    axarr[0,0].set_xlim([6.2,9.5])
    # axarr[0,0].set_xlim([5.0,9.5])
    # axarr[0,0].set_ylim([-0.2,0.8])

    MC_zenith = np.cos(np.pi - df.reco_zenith[cuts])
    zenith_bins = np.linspace(0.8, 1.0, 31)
    bin_centers, bin_medians, error = get_medians(
        MC_zenith, energy_res, zenith_bins)
    # bin_centers, bin_medians, error = get_medians_weighted(
    #     MC_zenith, energy_res, weights, zenith_bins)
    try:
        axarr[0,1].errorbar(bin_centers, bin_medians, yerr=error,
                    label=kwargs['label'], color=kwargs['color'],
                    marker='.')
    except KeyError:
        axarr[0,1].errorbar(bin_centers, bin_medians, yerr=error,
                    marker='.')
    axarr[0,1].set_xlabel('$\cos(\\theta_{\mathrm{reco}})$')
    axarr[0,1].set_ylabel('$\log_{10}(E_{\mathrm{ML}}/E_{\mathrm{MC}})$')


    MC_x = df['MC_x'][cuts]
    MC_y = df['MC_y'][cuts]
    ML_x = df['reco_x'][cuts]
    ML_y = df['reco_y'][cuts]
    # core_position = np.sqrt(MC_x**2 + MC_y**2)
    core_position = df.reco_radius[cuts]
    distance_bins = np.linspace(0, 1000, 41)
    bin_centers, bin_medians, error = get_medians(
        core_position, energy_res, distance_bins)
    # bin_centers, bin_medians, error = get_medians_weighted(
    #     core_position, energy_res, weights, distance_bins)
    try:
        axarr[1,0].errorbar(bin_centers, bin_medians, yerr=error,
                    label=kwargs['label'], color=kwargs['color'],
                    marker='.')
    except KeyError:
        axarr[1,0].errorbar(bin_centers, bin_medians, yerr=error,
                    marker='.')
    axarr[1,0].set_xlabel('Core posision [m]')
    axarr[1,0].set_ylabel('$\log_{10}(E_{\mathrm{ML}}/E_{\mathrm{MC}})$')
    # axarr[1,0].xaxis.set_ticks([5,6,7,8,9])

    core_res = np.sqrt((ML_x - MC_x)**2 + (ML_y - MC_y)**2)
    bin_centers, bin_medians, error = get_medians(
        np.log10(MC_energy), core_res, energy_bins)
    # bin_centers, bin_medians, error = get_medians_weighted(
    #     np.log10(MC_energy), core_res, weights, energy_bins)
    try:
        axarr[1,1].errorbar(bin_centers, bin_medians, yerr=error,
                    label=kwargs['label'], color=kwargs['color'],
                    marker='.')
    except KeyError:
        axarr[1,1].errorbar(bin_centers, bin_medians, yerr=error,
                    marker='.')
    axarr[1,1].set_xlabel('$\log_{10}(E_{\mathrm{MC}}/\mathrm{GeV})$')
    axarr[1,1].set_ylabel('$\\vec{x}_{ML}-\\vec{x}_{\mathrm{MC}} \ [m]$')
    axarr[1,1].xaxis.set_ticks(np.arange(5.0, 9.6, 0.5))
    axarr[1,1].set_xlim([5.0,9.5])
    axarr[1,1].set_ylim([0.0,90])
    outfile = kwargs['outdir'] + \
        '/performance-plots_combined.png'.format(opts['bintype'])
    plt.savefig(outfile)
    plt.close()


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

    # sim_reco = load_sim(config=args.config, bintype=args.bintype)
    df = load_sim()
    # df, cut_dict = load_sim(return_cut_dict=True)
    # print(cut_dict.keys())
    # selection_mask = np.array([True] * len(df))
    # standard_cut_keys = ['reco_exists', 'reco_zenith', 'reco_IT_containment',
    #                      'IceTopMaxSignalInEdge', 'IceTopMaxSignal']
    # for key in standard_cut_keys:
    #     selection_mask *= cut_dict[key]
    #
    # print('n_events before cuts = {}'.format(len(df)))
    # df = df[selection_mask]
    # print('n_events after cuts = {}'.format(len(df)))

    # Specify various cut arrays
    MC_proton_mask = (df.MC_comp == 'P')
    MC_iron_mask = (df.MC_comp == 'Fe')

    # Specify dictionary with composition cuts as keys and
    # composition labels as the values
    comp_to_cuts = {}
    comp_to_cuts['Proton'] = MC_proton_mask
    comp_to_cuts['Iron'] = MC_iron_mask
    # Specify color dictionary
    palette = sns.color_palette('muted').as_hex()
    colordict = {'Proton': palette[0], 'Iron': palette[1],
                 'Helium': '#e5ae38', 'Oxygen': '#6d904f'}
    opts['colordict'] = colordict

    cuts = [True]*len(MC_proton_mask)
    make_performance_plots(df, cuts, **opts)
    LLHenergy_MCenergy(df, cuts, **opts)

    # # Energy resolution plots
    # fig, axarr = plt.subplots(1, 2)
    # plt.subplots_adjust(top=0.85)
    # opts['xaxis'] = 'energy'
    # for comp in ['Proton', 'Iron']:
    #     cuts = standard_mask & comp_to_cuts[comp]
    #     opts['label'] = comp
    #     opts['color'] = colordict[comp]
    #     # opts['fit'] = zfix(sim_reco['zenith'], bintype='logdist')[cuts]
    #     plot_energyres(axarr[0], sim_reco, cuts, **opts)
    # opts['xaxis'] = 'zenith'
    # for comp in ['Proton', 'Iron']:
    #     cuts = standard_mask & comp_to_cuts[comp]
    #     opts['label'] = comp
    #     opts['color'] = colordict[comp]
    #     plot_energyres(axarr[1], sim_reco, cuts, **opts)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
    #            borderaxespad=0., title='True Comp')
    # plt.tight_layout()
    # opts['outfile'] = opts['outdir'] + \
    #     '/energyres_{}.png'.format(opts['bintype'])
    # plt.savefig(opts['outfile'])
    # plt.close()

    # # Core resolution plots
    # fig2, axarr2 = plt.subplots(1, 2)
    # opts['xaxis'] = 'energy'
    # for comp in ['Proton', 'Iron']:
    #     cuts = standard_mask & comp_to_cuts[comp]
    #     opts['label'] = comp
    #     opts['color'] = colordict[comp]
    #     opts['ylim'] = [0,70]
    #     plot_coreres(axarr2[0], sim_reco, cuts, **opts)
    # opts['xaxis'] = 'zenith'
    # del opts['ylim']
    # for comp in ['Proton', 'Iron']:
    #     cuts = standard_mask & comp_to_cuts[comp]
    #     opts['label'] = comp
    #     opts['color'] = colordict[comp]
    #     plot_coreres(axarr2[1], sim_reco, cuts, **opts)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
    #            borderaxespad=0., title='True Comp')
    # plt.tight_layout()
    # opts['outfile'] = opts['outdir'] + \
    #     '/coreres_{}.png'.format(opts['bintype'])
    # plt.savefig(opts['outfile'])
    # plt.close()

    # eres_position(sim_reco, cuts, **opts)
    # LLHenergy_MCenergy(sim_reco, standard_mask, **opts)
    # LLHzenith_MCzenith(sim_reco, cuts, **opts)
    # effarea(sim_reco, standard_mask, **opts)

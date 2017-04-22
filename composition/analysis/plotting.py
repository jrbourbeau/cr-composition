#!/usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn.apionly as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve, GridSearchCV

from . import export
from .data_functions import ratio_error
from .base import cast_to_ndarray


def plot_decision_regions(X, y, classifier, resolution=0.02, scatter_fraction=0.025, ax=None):
    # setup marker generator and color map
    markers = ('s', '^', 'o', '^', 'v')
    colors = ('b', 'g', 'r', 'y', 'c')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    # x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    if ax is None:
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap, aspect='auto')
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
    else:
        ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap, aspect='auto')
        ax.set_xlim(xx1.min(), xx1.max())
        ax.set_ylim(xx2.min(), xx2.max())

    # plot class samples
    if scatter_fraction != None:
        fraction_event_selection_mask = (np.random.uniform(0, 1, len(y)) <= scatter_fraction)
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[(y == cl) & fraction_event_selection_mask, 0],
                        y=X[(y == cl) & fraction_event_selection_mask, 1],
                        alpha=0.5, c=cmap(idx),
                        marker=markers[idx], label=cl)
    plt.legend()

@export
def histogram_2D(x, y, bins, log_counts=False, make_prob=False, colorbar=True,
        logx=False, logy=False, ax=None, **opts):
    # Validate inputs
    x = cast_to_ndarray(x)
    y = cast_to_ndarray(y)
    bins = cast_to_ndarray(bins)

    h, xedges, yedges = np.histogram2d(x, y, bins=bins, normed=False)
    h = np.rot90(h)
    h = np.flipud(h)
    h = np.ma.masked_where(h == 0, h)
    if make_prob:
        ntot = np.sum(h, axis=0).astype('float')
        ntot[ntot == 0] = 1.
        h /= ntot
    if log_counts:
        h = np.log10(h)
    # extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    colormap = 'viridis'

    if ax is None:
        ax = plt.gca()

    im = ax.imshow(h, extent=extent, origin='lower',
               interpolation='none', cmap=colormap, aspect='auto')
    if logx:
        ax.set_xscale('log', nonposy='clip')
    if logy:
        ax.set_yscale('log', nonposy='clip')

    if colorbar:
        if not make_prob and not log_counts:
            cb = plt.colorbar(im, label='Counts')
        if not make_prob and log_counts:
            cb = plt.colorbar(im, label='$\log_{10}(\mathrm{Counts})$')

    return im

@export
def make_comp_frac_histogram(x, y, proton_mask, iron_mask, bins, ax):
    # charge_bins = np.linspace(0, 7, 50)
    # energy_bins = np.linspace(6.2, 9.51, 50)
#     energy_bins = np.arange(6.2, 9.51, 0.05)
    # energy_midpoints = (energy_bins[1:] + energy_bins[:-1]) / 2
    proton_hist, xedges, yedges = np.histogram2d(x[proton_mask],
                                                 y[proton_mask],
                                                 bins=bins,
                                                 normed=False)
    proton_hist = np.ma.masked_where(proton_hist == 0, proton_hist)
    iron_hist, xedges, yedges = np.histogram2d(x[iron_mask],
                                               y[iron_mask],
                                               bins=bins,
                                               normed=False)

    h = proton_hist / (proton_hist + iron_hist)
    h = np.rot90(h)
    h = np.flipud(h)
    h = np.ma.masked_where(h == 0, h)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    colormap = 'coolwarm'
    im = ax.imshow(h, extent=extent, origin='lower',
               interpolation='none', cmap=colormap,
               aspect='auto', vmin=0, vmax=1)
    # x = np.arange(6.2, 9.51, 0.1)
    x = np.arange(6.2, 8.1, 0.1)

    return im


@export
def plot_steps(edges, y, yerr=None, color='C0', lw=1, fillalpha=0.15, label=None, ax=None):

    # Ensure we're dealing with numpy.ndarray objects
    edges = cast_to_ndarray(edges)
    y = cast_to_ndarray(y)
    if yerr is not None:
        yerr = cast_to_ndarray(yerr)

    if ax is None:
        ax = plt.gca()

    ax.step(edges[:-1], y, where='post',
            marker='None', color=color, linewidth=lw,
            linestyle='-', label=label, alpha=0.8)
    ax.plot(edges[-2:], 2*[y[-1]], marker='None',
            color=color, linewidth=lw,
            linestyle='-', alpha=0.8)

    if yerr is not None:
        err_lower = y - yerr if yerr.ndim == 1 else y - yerr[0]
        err_upper = y + yerr if yerr.ndim == 1 else y + yerr[1]

        ax.fill_between(edges[:-1], y-yerr, y+yerr, step='post',
                alpha=fillalpha, color=color, linewidth=0)
        ax.fill_between(edges[-2:], 2*[(y-yerr)[-1]], 2*[(y+yerr)[-1]], step='post',
                alpha=fillalpha, color=color, linewidth=0)

    return ax


def make_verification_plot(bins, ax=None):

    if ax is None:
        ax = plt.gca()

    log_s125_bins = np.linspace(-2, 4, 100)

    gs = gridspec.GridSpec(2, 1, height_ratios=[2,1], hspace=0.0)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    # fig, ax = plt.subplots()
    log_s125_rate = {}
    for composition in comp_list:
        log_s125_rate[composition] = np.histogram(df_sim[MC_comp_mask[composition]]['log_s125'],
                              bins=log_s125_bins, weights=df_sim[MC_comp_mask[composition]]['weights'])[0]
        plotting.plot_steps(log_s125_bins, log_s125_rate[composition],
                            color=color_dict[composition], label=composition, ax=ax1)
    # Add data rate
    log_s125_rate['data'] = np.histogram(df_data['log_s125'], bins=log_s125_bins)[0]/livetime
    plotting.plot_steps(log_s125_bins, log_s125_rate['data'], color=color_dict['data'], label='data', ax=ax1)
    ax1.set_yscale("log", nonposy='clip')
    ax1.set_ylabel('Rate [Hz]')
    ax1.grid()
    ax1.legend()

    for composition in comp_list:
        plotting.plot_steps(log_s125_bins, log_s125_rate['data']/log_s125_rate[composition], log_s125_rate[composition],
                           color=color_dict[composition], ax=ax2)
    ax2.axhline(1, marker='None', ls='-.', color='k')
    ax2.set_xlabel('$\log_{10}(S_{125})$')
    ax2.grid()
    ax2.legend()

    plt.show()

# def make_verification_plot(data_df, sim_df, key, bins, label):
#
#     if not isinstance(data_df, pd.DataFrame):
#         raise TypeError('Expecting a pandas DataFrame, got {} instead'.format(type(data_df)))
#     if not isinstance(sim_df, pd.DataFrame):
#         raise TypeError('Expecting a pandas DataFrame, got {} instead'.format(type(sim_df)))
#
#     # Extract numpy arrays (and lengths) from DataFrame
#     data_array, sim_array = data_df[key].values, sim_df[key].values
#     n_data, n_sim = data_array.shape[0], sim_array.shape[0]
#
#     counts_data = np.histogram(data_array, bins)[0]
#     rate_data, rate_data_err = ratio_error(counts_data, np.sqrt(counts_data),
#                                            n_data, np.sqrt(n_data))
#
#     counts_sim = np.histogram(sim_array, bins)[0]
#     rate_sim, rate_sim_err = ratio_error(counts_sim, np.sqrt(counts_sim),
#                                          n_sim, np.sqrt(n_sim))
#
#     ratio, ratio_err = ratio_error(rate_data, rate_data_err,
#                                    rate_sim, rate_sim_err)
#
#     bin_midpoints = (bins[1:] + bins[:-1]) / 2
#
#     gs = gridspec.GridSpec(2, 1, height_ratios=[2,1], hspace=0.0)
#     ax1 = plt.subplot(gs[0])
#     ax2 = plt.subplot(gs[1], sharex=ax1)
#
#     ax1.errorbar(bin_midpoints, rate_sim, yerr=rate_sim_err, label='MC',
#                  marker='.', ms=8, ls='None', color='C0')
#     ax1.errorbar(bin_midpoints, rate_data, yerr=rate_data_err, label='Data',
#                  marker='.', ms=8, ls='None', color='C1')
#     ax1.set_yscale('log', nonposy='clip')
#     ax1.set_ylabel('Rate [Hz]')
#     ax1.grid(True)
#     ax1.legend()
#
#     ax2.errorbar(bin_midpoints, ratio, yerr=ratio_err, marker='.', ms=8, ls='None', color='C2')
#     ax2.set_ylabel('Data/MC')
#     ax2.set_xlabel(label)
#     ax2.grid(True)
#
#     plt.show()
#
#     return


from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def get_color_dict():
    color_dict = {'light': 'C0', 'heavy': 'C1', 'intermediate': 'C3',
                  'total': 'C2',
                  'PPlus': 'C0', 'He4Nucleus': 'C4', 'O16Nucleus': 'C3',
                  'Fe56Nucleus': 'C1',
                  'data': 'k'}

    return color_dict


def get_color(composition):
    color_dict = get_color_dict()
    return color_dict[composition]


def get_colormap(composition):
    cmaps = {'light': 'Blues', 'heavy': 'Oranges', 'intermediate': 'Reds',
                  'total': 'Greens',
                  'PPlus': 'Blues', 'He4Nucleus': 'Purples', 'O16Nucleus': 'Reds',
                  'Fe56Nucleus': 'Oranges',
                  'data': 'Blacks'}

    return cmaps[composition]


def histogram_2D(x, y, bins, weights=None, log_counts=False, make_prob=False,
                 colorbar=True, logx=False, logy=False, vmin=None, vmax=None,
                 cmap='viridis', ax=None, **opts):
    # Validate inputs
    x = np.asarray(x)
    y = np.asarray(y)
    bins = np.asarray(bins)
    if weights is not None:
        weights = np.asarray(weights)

    h, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=weights, normed=False)
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

    if ax is None:
        ax = plt.gca()

    im = ax.imshow(h, extent=extent, origin='lower',
                   interpolation='none', cmap=cmap, aspect='auto',
                   vmin=vmin, vmax=vmax)
    if logx:
        ax.set_xscale('log', nonposy='clip')
    if logy:
        ax.set_yscale('log', nonposy='clip')

    if colorbar:
        if not make_prob and not log_counts:
            plt.colorbar(im, label='Counts')
        if not make_prob and log_counts:
            plt.colorbar(im, label='$\log_{10}(\mathrm{Counts})$')

    return im


def make_comp_frac_histogram(x, y, proton_mask, iron_mask, bins, ax):
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


def plot_steps(edges, y, yerr=None, color='C0', lw=1, ls='-', alpha=1.0,
               fillalpha=0.2, label=None, ax=None):

    # Ensure we're dealing with numpy.ndarray objects
    edges = np.asarray(edges)
    y = np.asarray(y)
    if yerr is not None:
        yerr = np.asarray(yerr)

    if ax is None:
        ax = plt.gca()

    ax.step(edges[:-1], y, where='post',
            marker='None', color=color, linewidth=lw,
            linestyle=ls, label=label, alpha=alpha)
    ax.plot(edges[-2:], 2*[y[-1]], marker='None',
            color=color, linewidth=lw,
            linestyle=ls, alpha=alpha)

    if yerr is not None:
        err_lower = y - yerr if yerr.ndim == 1 else y - yerr[0]
        err_upper = y + yerr if yerr.ndim == 1 else y + yerr[1]

        ax.fill_between(edges[:-1], err_lower, err_upper, step='post',
                        alpha=fillalpha, color=color, linewidth=0)
        ax.fill_between(edges[-2:], 2*[(y-yerr)[-1]], 2*[(y+yerr)[-1]],
                        step='post', alpha=fillalpha, color=color, linewidth=0)

    return ax

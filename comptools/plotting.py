
from __future__ import division
import collections
from itertools import cycle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap


def get_color_dict():
    color_dict = {'light': 'C0', 'heavy': 'C1', 'intermediate': 'C3',
                  'total': 'C2',
                  'PPlus': 'C0', 'He4Nucleus': 'C4', 'O16Nucleus': 'C3',
                  'Fe56Nucleus':'C1',
                  'data': 'k'}

    return color_dict


def plot_decision_slice(xkey, ykey, data, clf, filler_feature_dict=None,
        xres=0.1, yres=0.1, xlim=None, ylim=None, colors=('C0','C1','C2','C3','C4'), ax=None):
    '''Function to plot 2D decision region of a scikit-learn classifier

    Parameters
    ----------
    xkey : str
        Key for feature on x-axis.
    ykey : str
        Key for feature on y-axis.
    data : pandas.DataFrame
        DataFrame containing the training dataset. Will use data to determine minimum and maximum values for the xkey and ykey features. The order of the columns in data must be the same as the order of features used in training the classifier clf.
    clf : fitted scikit-learn classifier or pipeline
        The fitted scikit-learn classifier for which you would like to vizulaize the decision regions
    filler_feature_dict : dict, optional
        Dictionary containing key-value pairs for the training features other than those given by xkey and ykey. Required if number of training features is larger than two.
    xres : float, optional
        The grid spacing used along the x-axis when evaluating the decision region (default is 0.1).
    yres : float, optional
        The grid spacing used along the y-axis when evaluating the decision region (default is 0.1).
    xlim : tuple, int, optional
        If specified, will be used to set the x-axis limit.
    ylim : tuple, int, optional
        If specified, will be used to set the y-axis limit.
    colors: str, optional
        Comma separated list of colors. (default is 'C0,C1,C2,C3,C4')
    ax : matplotlib.axes
        If specified, will plot decision region on ax. Otherwise will create an ax instance.

    Returns
    -------
    matplotlib.axes
        Matplotlib axes with the classifier decision region added.

    '''
    # Validate input types
    if not isinstance(data, pd.DataFrame):
        raise ValueError('data must be a pandas DataFrame')
    if not all([key in data.columns for key in [xkey, ykey]]):
        raise ValueError('Both xkey and ykey must be in data.columns')
    if not isinstance(filler_feature_dict, dict):
        raise ValueError('filler_feature_dict must be a dictionary')

    n_features = len(data.columns)
    training_features = data.columns
    # Check to see that all the specified featues are consistant
    compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
    if n_features > 2:
        if not filler_feature_dict:
            raise ValueError('filler_feature_dict must be provided if using more than 2 training features')
        if not compare(training_features, list(filler_feature_dict.keys()) + [xkey, ykey]):
            raise ValueError('The xkey, ykey, and filler feature keys are not the same as data.columns')
    else:
        if not compare(training_features, [xkey, ykey]):
            raise ValueError('Both xkey and ykey must be in data.columns')

    # Extract the minimum and maximum values of the x-y decision region features
    x_min = data[xkey].min()
    x_max = data[xkey].max()
    y_min = data[ykey].min()
    y_max = data[ykey].max()
    # Construct x-y meshgrid for the specified features
    x_array = np.arange(x_min, x_max, xres)
    y_array = np.arange(y_min, y_max, yres)
    xx1, xx2 = np.meshgrid(x_array, y_array)
    # X should have a row for each x-y point in the meshgrid (will be used in pipeline.predict later)
    X = np.array([xx1.ravel(), xx2.ravel()]).T
    # Now we need to include the filler values for the other training features
    # Construct a DataFrame from X
    df_temp = pd.DataFrame({xkey: X[:, 0], ykey: X[:, 1]}, columns=[xkey, ykey])
    # If using more than two training features, add a new column for each other the other non-plotted training features
    if n_features > 2:
        for key in filler_feature_dict:
            df_temp[key] = filler_feature_dict[key]
    # Reorder the columns of df_temp to match those used in training clf
    df_temp = df_temp[training_features]
    X_predict = df_temp.values

    Z = clf.predict(X_predict)
    Z = Z.reshape(xx1.shape)

    if ax is None:
        ax = plt.gca()

    cmap = ListedColormap(colors)
    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap, aspect='auto')

    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    return ax


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


def histogram_2D(x, y, bins, weights=None, log_counts=False, make_prob=False,
                 colorbar=True, logx=False, logy=False, vmin=None, vmax=None,
                 cmap='viridis', ax=None, **opts):
    # Validate inputs
    x = np.asarray(x)
    y = np.asarray(y)
    bins = np.asarray(bins)
    if weights is not None: weights = np.asarray(weights)

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
            cb = plt.colorbar(im, label='Counts')
        if not make_prob and log_counts:
            cb = plt.colorbar(im, label='$\log_{10}(\mathrm{Counts})$')

    return im


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

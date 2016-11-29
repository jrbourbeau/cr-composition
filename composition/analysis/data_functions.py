#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy import stats
from . import export

@export
def ratio_error(num, num_err, den, den_err):
    ratio = num/den
    ratio_err = ratio * np.sqrt((num_err / num)**2 + (den_err / den)**2)
    return ratio, ratio_err

def get_bin_mids(bins, infvalue=None):
    abins = np.asarray(bins)
    if infvalue != None:
        abins[abins == infvalue] *= np.inf
    steps = (abins[1:] - abins[:-1])
    mids = abins[:-1] + steps / 2.
    if abs(steps[0]) == np.inf:
        mids[0] = abins[1] - steps[1] / 2.
    if abs(steps[-1]) == np.inf:
        mids[-1] = abins[-2] + steps[-2] / 2.
    return mids

def get_medians(x, y, bins):
    lower_error = lambda x: np.percentile(x, 16)
    upper_error = lambda x: np.percentile(x, 84)

    bin_medians, bin_edges, binnumber = stats.binned_statistic(
        x, y, statistic='median', bins=bins)
    err_up, err_up_edges, err_up_binnum = stats.binned_statistic(
        x, y, statistic=upper_error, bins=bins)
    err_down, err_down_edges, err_down_binnum = stats.binned_statistic(
        x, y, statistic=lower_error, bins=bins)
    error = [bin_medians - err_down, err_up - bin_medians]
    # bin_centers = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2.
    bin_centers = get_bin_mids(bin_edges)

    return bin_centers, bin_medians, error

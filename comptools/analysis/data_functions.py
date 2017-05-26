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

@export
def product_error(term1, term1_err, term2, term2_err):
    product = term1 * term2
    product_err = product * np.sqrt((term1_err / term1)**2 + (term2_err / term2)**2)
    return product, product_err

@export
def averaging_error(values, errors):
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    if not isinstance(errors, np.ndarray):
        errors = np.array(errors)
    sum_value = values.sum()
    sum_error = np.sum([np.sqrt(error**2) for error in errors])

    return sum_value/len(values), sum_error/len(values)

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


def get_avg_std(x, y, bins):
    lower_error = lambda x: np.percentile(x, 16)
    upper_error = lambda x: np.percentile(x, 84)

    averages, bin_edges, bin_edges = stats.binned_statistic(x, y,
        statistic='mean', bins=bins)
    standard_devs, bin_edges, bin_edges = stats.binned_statistic(x, y,
        statistic=np.std, bins=bins)
    err_up, err_up_edges, err_up_binnum = stats.binned_statistic(
        x, y, statistic=upper_error, bins=bins)
    err_down, err_down_edges, err_down_binnum = stats.binned_statistic(
        x, y, statistic=lower_error, bins=bins)

    return averages, standard_devs, bin_edges
    # return averages, err_up-err_down, bin_edges


def get_cumprob_sigma(values):

    bins = np.linspace(values.min(), values.max(), 200)
    bin_midpoints = (bins[1:] + bins[:-1]) / 2
    binned_counts = np.histogram(values, bins=bins)[0]
    cumulative_prob = np.cumsum(binned_counts)/binned_counts.sum()
    sigma_index = np.where(cumulative_prob < 0.68)[0].max()
    sigma_containment = bin_midpoints[sigma_index]/1.51

    return sigma_containment

def get_resolution(x, y, bins):
    binned_statistic, bin_edges, binnumber = stats.binned_statistic(x, y,
            statistic=get_cumprob_sigma, bins=bins)

    return binned_statistic

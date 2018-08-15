
import os
from itertools import product
import numpy as np
import pandas as pd
import socket

from .composition_encoding import get_comp_list
from .base import get_energybins, get_paths, check_output_dir
from .data_functions import ratio_error


def unfolded_counts_dist(unfolding_df, iteration=-1, num_groups=4):
    """
    Convert unfolded distributions DataFrame from PyUnfold counts arrays
    to a dictionary containing a counts array for each composition.

    Parameters
    ----------
    unfolding_df : pandas.DataFrame
        Unfolding DataFrame returned from PyUnfold.
    iteration : int, optional
        Specific unfolding iteration to retrieve unfolded counts
        (default is -1, the last iteration).
    num_groups : int, optional
        Number of composition groups (default is 4).

    Returns
    -------
    counts : dict
        Dictionary with composition-counts key-value pairs.
    counts_sys_err : dict
        Dictionary with composition-systematic error key-value pairs.
    counts_stat_err : dict
        Dictionary with composition-statistical error key-value pairs.
    """
    comp_list = get_comp_list(num_groups=num_groups)

    df_iter = unfolding_df.iloc[iteration]

    counts, counts_sys_err, counts_stat_err = {}, {}, {}
    for idx, composition in enumerate(comp_list):
        counts[composition] = df_iter['unfolded'][idx::num_groups]
        counts_sys_err[composition] = df_iter['sys_err'][idx::num_groups]
        counts_stat_err[composition] = df_iter['stat_err'][idx::num_groups]

    counts['total'] = np.sum([counts[composition] for composition in comp_list], axis=0)
    counts_sys_err['total'] = np.sqrt(np.sum([counts_sys_err[composition]**2 for composition in comp_list], axis=0))
    counts_stat_err['total'] = np.sqrt(np.sum([counts_stat_err[composition]**2 for composition in comp_list], axis=0))

    return counts, counts_sys_err, counts_stat_err


def column_normalize(res, res_err, efficiencies, efficiencies_err):
    res_col_sum = res.sum(axis=0)
    res_col_sum_err = np.array([np.sqrt(np.sum(res_err[:, i]**2))
                                for i in range(res_err.shape[1])])

    normalizations, normalizations_err = ratio_error(
                                            res_col_sum, res_col_sum_err,
                                            efficiencies, efficiencies_err,
                                            nan_to_num=True)

    res_normalized, res_normalized_err = ratio_error(
                                            res, res_err,
                                            normalizations, normalizations_err,
                                            nan_to_num=True)

    res_normalized = np.nan_to_num(res_normalized)
    res_normalized_err = np.nan_to_num(res_normalized_err)

    # Test that the columns of res_normalized equal efficiencies
    np.testing.assert_allclose(res_normalized.sum(axis=0), efficiencies)

    return res_normalized, res_normalized_err


def response_hist(true_energy, reco_energy, true_target, pred_target,
                  energy_bins=None):
    """Computes energy-composition response matrix

    Parameters
    ----------
    true_energy : array_like
        Array of true (MC) energies.
    reco_energy : array_like
        Array of reconstructed energies.
    true_target : array_like
        Array of true compositions that are encoded to numerical values.
    pred_target : array_like
        Array of predicted compositions that are encoded to numerical values.
    energy_bins : array_like, optional
        Energy bins to be used for constructing response matrix (default is
        to use energy bins from comptools.get_energybins() function).

    Returns
    -------
    res : numpy.ndarray
        Response matrix.
    res_err : numpy.ndarray
        Uncerainty of the response matrix.
    """

    # Check that the input array shapes
    inputs = [true_energy, reco_energy, true_target, pred_target]
    assert len(set(map(np.ndim, inputs))) == 1
    assert len(set(map(np.shape, inputs))) == 1

    if energy_bins is None:
        energy_bins = get_energybins().energy_bins
    num_ebins = len(energy_bins) - 1
    num_groups = len(np.unique([true_target, pred_target]))

    true_ebin_indices = np.digitize(true_energy, energy_bins) - 1
    reco_ebin_indices = np.digitize(reco_energy, energy_bins) - 1

    res = np.zeros((num_ebins * num_groups, num_ebins * num_groups))
    bin_iter = product(range(num_ebins), range(num_ebins),
                       range(num_groups), range(num_groups))
    for true_ebin, reco_ebin, true_target_bin, pred_target_bin in bin_iter:
        # Get mask for events in true/reco energy and true/reco composition bin
        mask = np.logical_and.reduce((true_ebin_indices == true_ebin,
                                      reco_ebin_indices == reco_ebin,
                                      true_target == true_target_bin,
                                      pred_target == pred_target_bin))
        res[num_groups * reco_ebin + pred_target_bin,
            num_groups * true_ebin + true_target_bin] = mask.sum()
    # Calculate statistical error on response matrix
    res_err = np.sqrt(res)

    return res, res_err


def response_matrix(true_energy, reco_energy, true_target, pred_target,
                    efficiencies, efficiencies_err, energy_bins=None):
    """Computes normalized energy-composition response matrix

    Parameters
    ----------
    true_energy : array_like
        Array of true (MC) energies.
    reco_energy : array_like
        Array of reconstructed energies.
    true_target : array_like
        Array of true compositions that are encoded to numerical values.
    pred_target : array_like
        Array of predicted compositions that are encoded to numerical values.
    efficiencies : array_like
        Detection efficiencies (should be in a PyUnfold-compatable form).
    efficiencies_err : array_like
        Detection efficiencies uncertainties (should be in a
        PyUnfold-compatable form).
    energy_bins : array_like, optional
        Energy bins to be used for constructing response matrix (default is
        to use energy bins from comptools.get_energybins() function).

    Returns
    -------
    res_normalized : numpy.ndarray
        Normalized response matrix.
    res_normalized_err : numpy.ndarray
        Uncerainty of the normalized response matrix.
    """
    res, res_err = response_hist(true_energy=true_energy,
                                 reco_energy=reco_energy,
                                 true_target=true_target,
                                 pred_target=pred_target,
                                 energy_bins=energy_bins)

    # Normalize response matrix column-wise (i.e. $P(E|C)$)
    res_normalized, res_normalized_err = column_normalize(res=res,
                                                          res_err=res_err,
                                                          efficiencies=efficiencies,
                                                          efficiencies_err=efficiencies_err)

    return res_normalized, res_normalized_err

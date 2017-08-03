
import os
import numpy as np
import pandas as pd
import healpy as hp
from scipy.stats import ks_2samp

from ..base import get_paths
from .anisotropy import get_map, get_proj_relint


def get_proj_RI_red_chi2(file_0, file_1, smooth=0., n_bins=36, decmax=-55):
    '''Caluclates the reduced chi-squared test statistic
    '''

    kwargs_relint = {'smooth': smooth, 'scale': None, 'decmax': decmax}

    # Get 2D relative intensity maps
    relint_0 = get_map(files=file_0, name='relint', **kwargs_relint)
    relint_1 = get_map(files=file_1, name='relint', **kwargs_relint)

    # Get 2D relative intensity error maps
    relerr_0 = get_map(files=file_0, name='relerr', **kwargs_relint)
    relerr_1 = get_map(files=file_1, name='relerr', **kwargs_relint)

    # Get projected relative intensity vs. right ascension
    ri_0, ri_err_0, ra, ra_err = get_proj_relint(relint_0, relerr_0,
                                        n_bins=n_bins, decmax=decmax)
    ri_1, ri_err_1, ra, ra_err = get_proj_relint(relint_1, relerr_1,
                                        n_bins=n_bins, decmax=decmax)

    # Calculate the reduced chi-squared between the two projected RIs
    chi2 = np.sum((ri_0 - ri_1)**2 / (ri_err_0**2 + ri_err_1**2))
    red_chi2 = chi2 / ri_0.shape[0]

    return red_chi2


def get_sig_ks_pval(file_0, file_1, smooth=0., decmax=-55):

    kwargs_relint = {'smooth': smooth, 'scale': None, 'decmax': decmax}

    # Get 2D Li-Ma significance maps
    sig_0 = get_map(files=file_0, name='sig', **kwargs_relint)
    sig_1 = get_map(files=file_1, name='sig', **kwargs_relint)

    # Construct masks
    is_good_mask_0 = (sig_0 != hp.UNSEEN) & ~np.isnan(sig_0)
    is_good_mask_1 = (sig_1 != hp.UNSEEN) & ~np.isnan(sig_1)

    # Calculate ks test statistic and corresponding p-value
    ks_statistic, pval = ks_2samp(sig_0[is_good_mask_0], sig_1[is_good_mask_1])

    # Calculate ~area between cumulative distributions
    bins = np.linspace(0, 5, 50)
    counts_0 = np.histogram(np.abs(sig_0[is_good_mask_0]), bins=bins)[0]
    counts_1 = np.histogram(np.abs(sig_1[is_good_mask_1]), bins=bins)[0]

    sig_cumsum_diff_area = np.abs(np.cumsum(counts_0) - np.cumsum(counts_1)).sum()

    return ks_statistic, pval, sig_cumsum_diff_area


def get_test_stats_file(config='IC86.2012', low_energy=False, smooth=0., n_bins=36, decmax=-55):
    '''Returns test statistics file path for a given config
    '''

    if not isinstance(config, (str, list, tuple, np.ndarray)):
        raise TypeError('config must be either a string or array-like')
    if isinstance(config, str):
        config = [config]

    paths = get_paths()
    df_dir = os.path.join(paths.comp_data_dir, 'anisotropy_random_trials')

    df_basename = 'teststat'
    for c in config:
        year = c.split('.')[-1]
        df_basename += '-{}'.format(year)
    df_basename += '_smooth-{}'.format(int(smooth))
    df_basename += '_RAbins-{}'.format(int(n_bins))
    df_basename += '_decmax-{}'.format(int(decmax))
    if low_energy:
        df_basename += '_lowenergy'
    df_basename += '.hdf'

    df_file = os.path.join(df_dir, df_basename)

    return df_file

    print('Loading {}...'.format(df_file))
    with pd.HDFStore(df_file, mode='r') as store:
        df = store.select('dataframe')

    return df


def load_test_stats(config='IC86.2012', low_energy=False, smooth=0., n_bins=36, decmax=-55):
    '''Loads saved DataFrame with test statistics for a given config
    '''

    df_file = get_test_stats_file(config=config, low_energy=low_energy,
                    smooth=smooth, n_bins=n_bins, decmax=decmax)
    print('Loading {}...'.format(df_file))
    with pd.HDFStore(df_file, mode='r') as store:
        df = store.select('dataframe')

    return df

#!/usr/bin/env python

import os
import argparse
import numpy as np
import healpy as hp
from scipy.stats import ks_2samp
import pandas as pd
import pyprind
import dask
from dask import delayed, multiprocessing
from dask.diagnostics import ProgressBar

import comptools as comp
import comptools.anisotropy.anisotropy as anisotropy


def calc_proj_RI_red_chi2(file_0, file_1, smooth, n_bins):

    kwargs_relint = {'smooth': smooth, 'scale': None, 'decmax': -55}

    # Get 2D relative intensity maps
    relint_0 = anisotropy.get_map(files=file_0, name='relint', **kwargs_relint)
    relint_1 = anisotropy.get_map(files=file_1, name='relint', **kwargs_relint)

    # Get 2D relative intensity error maps
    relerr_0 = anisotropy.get_map(files=file_0, name='relerr', **kwargs_relint)
    relerr_1 = anisotropy.get_map(files=file_1, name='relerr', **kwargs_relint)

    # Get projected relative intensity vs. right ascension
    ri_0, ri_err_0, ra, ra_err = anisotropy.get_proj_relint(relint_0, relerr_0,
                                                            n_bins=n_bins)
    ri_1, ri_err_1, ra, ra_err = anisotropy.get_proj_relint(relint_1, relerr_1,
                                                            n_bins=n_bins)

    # Calculate the reduced chi-squared between the two projected RIs
    chi2 = np.sum((ri_0-ri_1)**2/(ri_err_0**2+ri_err_1**2))
    red_chi2 = chi2 / ri_0.shape[0]
    # print('\nred_chi2 = {}'.format(red_chi2))
    # print('ri_err_0 = {}'.format(ri_err_0))
    # print('ri_err_1 = {}'.format(ri_err_1))
    # print('n_dof = {}'.format(ri_0.shape[0]))

    return red_chi2


def calc_sig_ks_pval(file_0, file_1, smooth):

    kwargs_relint = {'smooth': smooth, 'scale': None, 'decmax': -55}

    # Get 2D Li-Ma significance maps
    sig_0 = anisotropy.get_map(files=file_0, name='sig', **kwargs_relint)
    sig_1 = anisotropy.get_map(files=file_1, name='sig', **kwargs_relint)

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


def process(file_0, file_1, smooth, n_bins):

    # Calculate test statistics
    red_chi2 = calc_proj_RI_red_chi2(file_0, file_1, smooth, n_bins)
    sig_ks_dist, sig_pval, sig_cumsum_diff_area = calc_sig_ks_pval(file_0, file_1, smooth)

    test_stats = {'proj_RI_red_chi2': red_chi2, 'sig_ks_dist': sig_ks_dist,
                  'sig_pval': sig_pval, 'sig_cumsum_diff_area': sig_cumsum_diff_area}

    return test_stats


def get_outfile(configs, low_energy, test, smooth, n_bins):

    if test:
        return 'teststat_test.hdf'
    else:
        outfile_basename = 'teststat'
        for config in configs:
            year = config.split('.')[-1]
            outfile_basename += '-{}'.format(year)
        outfile_basename += '_smooth-{}'.format(smooth)
        outfile_basename += '_RAbins-{}'.format(n_bins)
        if low_energy:
            outfile_basename += '_lowenergy'
        outfile_basename += '.hdf'

        outdir = os.path.join(comp.paths.comp_data_dir, 'anisotropy_random_trials')

        return os.path.join(outdir, outfile_basename)


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description='Extracts and saves desired information from simulation/data .i3 files')
    p.add_argument('-c', '--config', dest='config', nargs='*',
                   choices=comp.datafunctions.get_data_configs(),
                   help='Detector configuration')
    p.add_argument('--n_trials', dest='n_trials', type=int,
                   default=1000,
                   help='Number of trials to process')
    p.add_argument('--low_energy', dest='low_energy',
                   default=False, action='store_true',
                   help='Only use events with energy < 10**6.75 GeV')
    p.add_argument('--n_bins', dest='n_bins', type=int,
                   default=72,
                   help='Number of right-ascension bins in projected RI')
    p.add_argument('--smooth', dest='smooth', type=float,
                   default=20,
                   help='Smoothing radius to apply to skymaps')
    p.add_argument('--n_jobs', dest='n_jobs', type=int,
                   default=20,
                   help='Number processes to run in parallel')
    p.add_argument('--test', dest='test',
                   default=False, action='store_true',
                   help='Only use events with energy < 10**6.75 GeV')
    args = p.parse_args()

    if args.test:
        args.n_trials = 2

    # Build up dask graph to compute
    test_stats_trials = []
    for trial_num in range(args.n_trials):
        sample_0_files = []
        sample_1_files = []
        for config in args.config:
            map_dir = os.path.join(comp.paths.comp_data_dir, config + '_data',
                                   'anisotropy/random_trials')
            sample_0_file = os.path.join(map_dir,
                                'random_split_0_trial-{}.fits'.format(trial_num))
            sample_0_files.append(sample_0_file)

            sample_1_file = os.path.join(map_dir,
                                'random_split_1_trial-{}.fits'.format(trial_num))
            sample_1_files.append(sample_1_file)

        test_stats_trials.append(delayed(process)(sample_0_files, sample_1_files,
                                                  args.smooth, args.n_bins))

    columns = ['proj_RI_red_chi2', 'sig_ks_dist', 'sig_pval', 'sig_cumsum_diff_area']
    dataframe = delayed(pd.DataFrame)(test_stats_trials, columns=columns)
    print('Calculating test statistics for {} random trials'.format(args.n_trials))
    with ProgressBar():
        if args.test or args.n_jobs == 1:
            dataframe = dataframe.compute(get=dask.get)
        else:
            dataframe = dataframe.compute(get=multiprocessing.get,
                                          num_workers=args.n_jobs)
            # dataframe = dataframe.compute(num_workers=20)

    # Save test statistics for each trial to a pandas.DataFrame
    outfile = get_outfile(args.config, args.low_energy, args.test,
                          args.smooth, args.n_bins)
    comp.check_output_dir(outfile)
    with pd.HDFStore(outfile) as output_store:
        output_store.put('dataframe', dataframe, format='table')

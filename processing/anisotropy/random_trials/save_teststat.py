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
import comptools.anisotropy.teststatistic as ts


@delayed()
def process(file_0, file_1, smooth, n_bins, decmax):

    # Calculate test statistics
    red_chi2 = ts.get_proj_RI_red_chi2(file_0, file_1, smooth, n_bins, decmax)
    sig_ks_dist, sig_pval, sig_cumsum_diff_area = ts.get_sig_ks_pval(file_0,
                                                    file_1, smooth, decmax)

    test_stats = {'proj_RI_red_chi2': red_chi2, 'sig_ks_dist': sig_ks_dist,
                  'sig_pval': sig_pval,
                  'sig_cumsum_diff_area': sig_cumsum_diff_area}

    return test_stats


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
                   default=0.,
                   help='Smoothing radius to apply to skymaps')
    p.add_argument('--decmax', dest='decmax', type=float,
                   default=-55,
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

        test_stats_trials.append(process(sample_0_files, sample_1_files,
                                                  args.smooth, args.n_bins,
                                                  args.decmax))

    columns = ['proj_RI_red_chi2', 'sig_ks_dist', 'sig_pval', 'sig_cumsum_diff_area']
    dataframe = delayed(pd.DataFrame)(test_stats_trials, columns=columns)
    print('Calculating test statistics for {} random trials'.format(args.n_trials))
    with ProgressBar():
        if args.test or args.n_jobs == 1:
            dataframe = dataframe.compute(get=dask.get)
        else:
            dataframe = dataframe.compute(get=multiprocessing.get,
                                          num_workers=args.n_jobs)

    # Save test statistics for each trial to a pandas.DataFrame
    outfile = ts.get_test_stats_file(args.config, args.low_energy, args.smooth,
                                     args.n_bins, args.decmax)
    comp.check_output_dir(outfile)
    with pd.HDFStore(outfile) as output_store:
        output_store.put('dataframe', dataframe)

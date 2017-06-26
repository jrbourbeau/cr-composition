#!/usr/bin/env python

import os
import glob
import argparse
import numpy as np
import healpy as hp
from scipy.stats import ks_2samp
import pandas as pd
import multiprocessing as mp
import pyprind

import comptools as comp
import comptools.anisotropy.anisotropy as anisotropy


def calc_red_chi2(ri_0, ri_1, ri_err_0, ri_err_1):
    chi2 = np.sum((ri_0-ri_1)**2/(ri_err_0**2+ri_err_1**2))
    red_chi2 = chi2 / ri_0.shape[0]

    return red_chi2


def process(file_0, file_1):

    kwargs_relint = {'smooth': 20, 'scale': None, 'decmax': -55}

    print('Processing:')
    print('\t{}'.format(file_0))
    print('\t{}'.format(file_1))
    relint_0 = anisotropy.get_map(files=file_0, name='relint', **kwargs_relint)
    relint_1 = anisotropy.get_map(files=file_1, name='relint', **kwargs_relint)

    relerr_0 = anisotropy.get_map(files=file_0, name='relerr', **kwargs_relint)
    relerr_1 = anisotropy.get_map(files=file_1, name='relerr', **kwargs_relint)

    ri_0, ri_err_0, ra, ra_err = anisotropy.get_proj_relint(relint_0, relerr_0, n_bins=24)
    ri_1, ri_err_1, ra, ra_err = anisotropy.get_proj_relint(relint_1, relerr_1, n_bins=24)

    print('ri_0 = {}'.format(ri_0))
    print('ri_1 = {}'.format(ri_1))
    ks_statistic, pval = ks_2samp(ri_0, ri_1)
    chi2 = calc_red_chi2(ri_0, ri_1, ri_err_0, ri_err_1)

    return pval, chi2


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description='Extracts and saves desired information from simulation/data .i3 files')
    p.add_argument('-c', '--config', dest='config',
                   default='IC86.2012',
                   choices=['IC79', 'IC86.2012', 'IC86.2013', 'IC86.2014', 'IC86.2015'],
                   help='Detector configuration')
    p.add_argument('--low_energy', dest='low_energy',
                   default=False, action='store_true',
                   help='Only use events with energy < 10**6.75 GeV')
    p.add_argument('--test', dest='test',
                   default=False, action='store_true',
                   help='Only use events with energy < 10**6.75 GeV')
    args = p.parse_args()

    map_dir = os.path.join(comp.paths.comp_data_dir, args.config + '_data',
                           'anisotropy/random_trials')
    print('map_dir = {}'.format(map_dir))
    sample_0_file_pattern = os.path.join(map_dir, 'random_split_0_trial-*.fits')
    sample_1_file_pattern = os.path.join(map_dir, 'random_split_1_trial-*.fits')
    infiles_sample_0 = sorted(glob.glob(sample_0_file_pattern))
    infiles_sample_1 = sorted(glob.glob(sample_1_file_pattern))
    if args.test:
        infiles_sample_0 = infiles_sample_0[:2]
        infiles_sample_1 = infiles_sample_1[:2]
    zipped_files = zip(infiles_sample_0, infiles_sample_1)

    # Set up multiprocessing pool to parallelize TS calculation
    n_processes = 1 if args.test else 20
    pool = mp.Pool(processes=n_processes)
    results = [pool.apply_async(process, args=files) for files in zipped_files]
    bar = pyprind.ProgBar(len(results), title='Calculate test statistic')
    output = []
    for p in results:
        output.append(p.get())
        bar.update()
    print(bar)

    if args.low_energy:
        outfile_basename = 'teststat_dataframe_lowenergy.hdf'
    else:
        outfile_basename = 'teststat_dataframe.hdf'

    if args.test:
        outfile = 'teststat_dataframe_test.hdf'
    else:
        outfile = os.path.join(map_dir, outfile_basename)
    comp.check_output_dir(outfile)
    with pd.HDFStore(outfile) as output_store:
        dataframe = pd.DataFrame(output, columns=['pval', 'chi2'])
        output_store.put('dataframe', dataframe, format='table')

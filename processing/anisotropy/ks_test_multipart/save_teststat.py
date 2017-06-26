#!/usr/bin/env python

import os
import argparse
import numpy as np
import healpy as hp
from scipy.stats import ks_2samp
import pandas as pd

import comptools as comp
import comptools.anisotropy.anisotropy as anisotropy


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description='Extracts and saves desired information from simulation/data .i3 files')
    p.add_argument('--infiles_sample_0', dest='infiles_sample_0', nargs='*',
                   help='Input reference map files')
    p.add_argument('--infiles_sample_1', dest='infiles_sample_1', nargs='*',
                   help='Input reference map files')
    p.add_argument('--outfile', dest='outfile',
                   help='Output DataFrame file')
    p.add_argument('--overwrite', dest='overwrite',
                   default=False, action='store_true',
                   help='Option to overwrite reference map file, '
                        'if it alreadu exists')
    args = p.parse_args()

    if args.infiles_sample_0 is None or args.infiles_sample_1 is None:
        raise ValueError('Input files must be specified')
    elif len(args.infiles_sample_0) != len(args.infiles_sample_1):
        raise ValueError('Both samples of input files must be the same length')

    if args.outfile is None:
        raise ValueError('Outfile must be specified')
    else:
        comp.check_output_dir(args.outfile)

    data_dict = {'ks_statistic': [], 'pval': []}
    # Read in all the input maps
    kwargs_relint = {'smooth': 20, 'scale': None, 'decmax': -55}
    for file_0, file_1 in zip(args.infiles_sample_0, args.infiles_sample_1):

        relint_0 = anisotropy.get_map(files=file_0, name='relint', **kwargs_relint)
        relint_1 = anisotropy.get_map(files=file_1, name='relint', **kwargs_relint)

        relerr_0 = anisotropy.get_map(files=file_0, name='relerr', **kwargs_relint)
        relerr_1 = anisotropy.get_map(files=file_1, name='relerr', **kwargs_relint)

        ri_0, ri_err_0, ra, ra_err = anisotropy.get_proj_relint(relint_0, relerr_0, n_bins=100)
        ri_1, ri_err_1, ra, ra_err = anisotropy.get_proj_relint(relint_1, relerr_1, n_bins=100)

        print('Comparing:')
        print('ri_0 = {}'.format(ri_0))
        print('ri_1 = {}\n'.format(ri_1))
        ks_statistic, pval = ks_2samp(ri_0, ri_1)
        print('ks_statistic = {}'.format(ks_statistic))
        print('pval = {}\n\n'.format(pval))
        data_dict['ks_statistic'].append(ks_statistic)
        data_dict['pval'].append(pval)

        data_dict['chi2'] = np.sum((ri_0-ri_1)**2/(ri_err_0**2+ri_err_1**2))

    with pd.HDFStore(args.outfile) as output_store:
        dataframe = pd.DataFrame(data_dict)
        output_store.put('dataframe', dataframe, format='table', data_columns=True)

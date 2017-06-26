#!/usr/bin/env python

from __future__ import division
import os
import sys
import argparse
import numpy as np
import pandas as pd
import healpy as hp
from sklearn.model_selection import train_test_split

import comptools as comp
import comptools.anisotropy.anisotropy as anisotropy


def get_random_times(store, n_rows, n_events, n_resamples=20):

    random_indices = np.random.choice(np.arange(n_rows, dtype=int),
                                      size=n_events*n_resamples,
                                      replace=False)
    times_selector = store.select_as_coordinates('dataframe',
                            where='index=random_indices')
    times_df = store.select('dataframe', where=times_selector,
                            columns=['start_time_mjd'])
    times_array = times_df.start_time_mjd.values
    times = times_array.reshape(n_events, n_resamples)

    return times


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
    p.add_argument('--n_side', dest='n_side', type=int,
                   default=64,
                   help='Number of times to split the DataFrame')
    p.add_argument('--chunksize', dest='chunksize', type=int,
                   default=1000,
                   help='Number of times to split the DataFrame')
    p.add_argument('--outfile_sample_0', dest='outfile_sample_0',
                   help='Output reference map file')
    p.add_argument('--outfile_sample_1', dest='outfile_sample_1',
                   help='Output reference map file')
    p.add_argument('--overwrite', dest='overwrite',
                   default=False, action='store_true',
                   help='Option to overwrite reference map file, '
                        'if it alreadu exists')

    args = p.parse_args()

    if args.outfile_sample_0 is None or args.outfile_sample_1 is None:
        raise ValueError('Expecting two output files to be specified')
    else:
        for outfile in [args.outfile_sample_0, args.outfile_sample_1]:
            comp.check_output_dir(outfile)


    npix = hp.nside2npix(args.n_side)
    data_maps = np.zeros((2, npix))
    ref_maps = np.zeros((2, npix))
    local_maps = np.zeros((2, npix))

    # Load DataFrame for config
    df_file = os.path.join(comp.paths.comp_data_dir, args.config + '_data',
                           'anisotropy_dataframe.hdf')
    with pd.HDFStore(df_file, mode='r') as store:
        n_rows = store.get_storer('dataframe').nrows
        for df in store.select('dataframe', chunksize=args.chunksize):
        # for df in store.select('dataframe', chunksize=args.chunksize, stop=10000):

            df.reset_index(drop=True, inplace=True)

            times = get_random_times(store, n_rows, args.chunksize, n_resamples=20)

            # If specified, remove high-energy events
            if args.low_energy:
                low_energy_mask = df['lap_log_energy'] <= 6.75
                df = df.loc[low_energy_mask, :].reset_index(drop=True)
                times = times[low_energy_mask]

            train_df, test_df = train_test_split(df, test_size=0.4)
            for split_idx, split_df in enumerate([train_df, test_df]):
                split_times = times[split_df.index]
                split_df.reset_index(drop=True, inplace=True)
                data, ref, local = anisotropy.make_skymaps(split_df,
                                            split_times, n_side=args.n_side)
                data_maps[split_idx] += data
                ref_maps[split_idx] += ref
                local_maps[split_idx] += local


    maps_sample_0 = (data_maps[0], ref_maps[0], local_maps[0])
    hp.write_map(args.outfile_sample_0, maps_sample_0, coord='C')

    maps_sample_1 = (data_maps[1], ref_maps[1], local_maps[1])
    hp.write_map(args.outfile_sample_1, maps_sample_1, coord='C')

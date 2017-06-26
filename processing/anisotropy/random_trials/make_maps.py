#!/usr/bin/env python

from __future__ import division
import os
import sys
import argparse
import numpy as np
import pandas as pd
import healpy as hp
from sklearn.model_selection import train_test_split
# from memory_profiler import profile

import comptools as comp
import comptools.anisotropy.anisotropy as anisotropy


# @profile
def get_random_times(store, split_indices, n_events, n_resamples=20):

    random_indices = np.random.choice(split_indices,
                                      size=n_events*n_resamples,
                                      replace=False)
    times_selector = store.select_as_coordinates('dataframe',
                            where=['index=random_indices'])
    times_df = store.select('dataframe', where=times_selector,
                            columns=['start_time_mjd'])
    times_array = times_df.start_time_mjd.values
    times = times_array.reshape(n_events, n_resamples)

    return times


# @profile
def get_batch_start_stop_rows(n_rows, n_batches, batch_idx):

    batch_rows = np.array_split(np.arange(n_rows, dtype=int), n_batches)[batch_idx]
    if batch_rows.size == 0:
        raise ValueError('Array with indices is empty')
    start_row = batch_rows.min()
    # The "stop" parameter when reading in a hdf file in pandas will not include
    # the row with index specified by stop. To make inclusive, need to increment
    # by one.
    stop_row = batch_rows.max() + 1

    return start_row, stop_row


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
    p.add_argument('--trial_idx', dest='trial_idx', type=int,
                   default=0,
                   help='Number indicating which random trial')
    p.add_argument('--n_batches', dest='n_batches', type=int,
                   default=50,
                   help='Number batches running in parallel')
    p.add_argument('--batch_idx', dest='batch_idx', type=int,
                   default=0,
                   help='Number labeling with this particular batch ' + \
                        '(e.g. batch 0, batch 1, batch 2, ...)')
    p.add_argument('--chunksize', dest='chunksize', type=int,
                   default=500,
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

    # Load DataFrame for config
    df_file = os.path.join(comp.paths.comp_data_dir, args.config + '_data',
                           'anisotropy_dataframe.hdf')
    with pd.HDFStore(df_file, mode='r') as store:
        n_rows = store.get_storer('dataframe').nrows
        start_row, stop_row = get_batch_start_stop_rows(n_rows,
                    args.n_batches, args.batch_idx)
        splits = train_test_split(np.arange(n_rows), test_size=0.4,
                                  random_state=args.trial_idx)
        np.random.seed(args.trial_idx)
        for split_idx, split_indices in enumerate(splits):

            # print('On split {}...'.format(split_idx))
            # Set up random seed for this batch
            batch_seed = np.random.randint(1e7, size=args.n_batches)[args.batch_idx]
            # print('Using batch seed {}'.format(batch_seed))
            np.random.seed(batch_seed)

            npix = hp.nside2npix(args.n_side)
            data_map = np.zeros(npix)
            ref_map = np.zeros(npix)
            local_map = np.zeros(npix)

            # i = 0
            for df in store.select('dataframe', chunksize=args.chunksize,
                                   start=start_row, stop=stop_row):
                # print('On chunk {}...'.format(i))
                # Remove rows not in split_indices
                split_mask = df.index.isin(split_indices)
                df = df.loc[split_mask, :].reset_index(drop=True)
                times = get_random_times(store, split_indices,
                                         n_events=df.shape[0],
                                         n_resamples=20)

                # If specified, remove high-energy events
                if args.low_energy:
                    low_energy_mask = df['lap_log_energy'] <= 6.75
                    df = df.loc[low_energy_mask, :].reset_index(drop=True)
                    times = times[low_energy_mask]

                data, ref, local = anisotropy.make_skymaps(df, times,
                                                           n_side=args.n_side)
                data_map  += data
                ref_map   += ref
                local_map += local

                # i += 1
                # if i > 10:
                #     break

            maps = (data_map, ref_map, local_map)
            if split_idx == 0:
                outfile = args.outfile_sample_0
            else:
                outfile = args.outfile_sample_1
            hp.write_map(outfile, maps, coord='C')

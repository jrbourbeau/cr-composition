#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
import pandas as pd
import healpy as hp

import comptools as comp
import comptools.anisotropy.anisotropy as anisotropy


def get_nrows(df_file):
    ''' Returns the number of rows in the DataFrame located at df_file

    Parameters
    ----------
    df_file : str
        Path to file containing DataFrame

    Returns
    -------
    int
        Number of rows in the DataFrame

    '''
    with pd.HDFStore(df_file, mode='r') as store:
        nrows = store.get_storer('dataframe').nrows
    return nrows

def get_random_times(df_file, n_events, n_resamples=20, nrows=None):
    with pd.HDFStore(df_file, mode='r') as store:
        if nrows is None:
            nrows = store.get_storer('dataframe').nrows
        random_indices = np.random.choice(np.arange(nrows, dtype=int),
                                          size=n_events*n_resamples,
                                          replace=False)
        times_selector = store.select_as_coordinates('dataframe',
                                where='index=random_indices')
        times_df = store.select('dataframe', where=times_selector,
                                columns=['start_time_mjd'])
        times_array = times_df.start_time_mjd.values
        times = times_array.reshape(n_events, n_resamples)

    return times


def get_dataframe_batch(df_file, n_splits, split_idx, nrows=None):
    '''Returns specified section of DataFrame

    Splits a DataFrame in to n_splits batches and returns the batch with
    index corresponding to split_idx

    '''
    with pd.HDFStore(df_file, mode='r') as store:
        if nrows is None:
            nrows = store.get_storer('dataframe').nrows
        # Split dataframe into parts
        splits = np.array_split(np.arange(nrows, dtype=int), n_splits)
        split_indices = splits[split_idx]
        rows_selector = store.select_as_coordinates('dataframe',
                                where='index=split_indices')
        # Select the DataFrame rows corresponding to split_indices
        data_df = store.select('dataframe', where=rows_selector)

    return data_df.reset_index(drop=True)


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description='Extracts and saves desired information from simulation/data .i3 files')
    p.add_argument('-c', '--config', dest='config',
                   default='IC86.2012',
                   choices=comp.datafunctions.get_data_configs(),
                   help='Detector configuration')
    p.add_argument('--composition', dest='composition',
                   default='all', choices=['light', 'heavy', 'all'],
                   help='Whether to make individual skymaps for each composition')
    p.add_argument('--low_energy', dest='low_energy',
                   default=False, action='store_true',
                   help='Only use events with energy < 10**7.0 GeV')
    p.add_argument('--n_side', dest='n_side', type=int,
                   default=64,
                   help='Number of times to split the DataFrame')
    p.add_argument('--n_splits', dest='n_splits', type=int,
                   default=200,
                   help='Number of times to split the DataFrame')
    p.add_argument('--split_idx', dest='split_idx', type=int,
                   default=0,
                   help='Number of times to split the DataFrame')
    p.add_argument('--outfile', dest='outfile',
                   help='Output reference map file')
    p.add_argument('--overwrite', dest='overwrite',
                   default=False, action='store_true',
                   help='Option to overwrite reference map file, '
                        'if it alreadu exists')

    args = p.parse_args()

    if args.outfile is None:
        raise ValueError('Outfile must be specified')
    else:
        comp.check_output_dir(args.outfile)

    # Load DataFrame for config
    df_file = os.path.join(comp.paths.comp_data_dir, args.config + '_data',
                           'anisotropy_dataframe.hdf')
    nrows = get_nrows(df_file)
    data_df = get_dataframe_batch(df_file, args.n_splits, args.split_idx,
                                  nrows=nrows)
    times = get_random_times(df_file, data_df.shape[0], n_resamples=20,
                             nrows=nrows)

    mask = np.ones(data_df.shape[0], dtype=bool)
    if args.composition in ['light', 'heavy']:
        mask[data_df['pred_comp'] != args.composition] = False

    # # Ensure that effective area has plateaued
    # mask[data_df['lap_log_energy'] < 6.4] = False
    mask[data_df['lap_log_energy'] < 6.0] = False

    # If specified, remove high-energy events
    if args.low_energy:
        mask[data_df['lap_log_energy'] > 7.0] = False
        # mask[data_df['lap_log_energy'] > 6.75] = False

    data_df = data_df.loc[mask, :].reset_index(drop=True)
    times = times[mask]

    maps = anisotropy.make_skymaps(data_df, times, n_side=args.n_side)
    hp.write_map(args.outfile, maps, coord='C')

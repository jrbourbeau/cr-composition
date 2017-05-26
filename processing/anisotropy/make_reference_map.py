#!/usr/bin/env python

import os
import argparse
import numpy as np
import pandas as pd
import healpy as hp

import comptools as comp
import comptools.analysis.anisotropy as anisotropy


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description='Extracts and saves desired information from simulation/data .i3 files')
    p.add_argument('-c', '--config', dest='config',
                   default='IC86.2012',
                   choices=['IC79', 'IC86.2012', 'IC86.2013'],
                   help='Detector configuration')
    p.add_argument('--composition', dest='composition',
                   default='all',
                   choices=['light', 'heavy', 'all'],
                   help='Whether to make individual skymaps for each composition')
    p.add_argument('--n_side', dest='n_side', type=int,
                   default=64,
                   help='Number of times to split the DataFrame')
    p.add_argument('--n_splits', dest='n_splits', type=int,
                   default=1000,
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
    comp.checkdir(args.outfile)

    # Setup global path names
    mypaths = comp.get_paths()

    # Load full DataFrame for config
    df_file = os.path.join(mypaths.comp_data_dir, args.config + '_data',
                           'anisotropy_dataframe.hdf')
    with pd.HDFStore(df_file) as store:
        data_df = store['dataframe']

    if args.composition != 'all':
        comp_mask = data_df['pred_comp'] == args.composition
        data_df = data_df.loc[comp_mask, :]
    # Extract pandas Series of all times
    times = data_df.start_time_mjd.values
    # Split dataframe into parts
    splits = np.array_split(data_df, args.n_splits)
    # Get specific part
    data_df = splits[args.split_idx]
    maps = anisotropy.make_skymaps(data_df, times, n_side=args.n_side,
                                   verbose=True)
    hp.write_map(args.outfile, maps, coord='C')

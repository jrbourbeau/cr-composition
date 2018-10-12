#!/usr/bin/env python

import os
import argparse
import h5py
import numpy as np
import pandas as pd
import shutil

import comptools as comp
from comptools.composition_encoding import (composition_group_labels,
                                            encode_composition_groups)


def extract_vector_series(store, key, sim):
    values, index = [], []
    grouped = store[key].groupby(['Run', 'Event', 'SubEvent'])
    for name, group in grouped:
        values.append(group['item'].values)
        index.append('{}_{}_{}_{}'.format(sim, *name))
        # For data
        # index.append('{}_{}_{}_{}'.format(config, *name))
    series = pd.Series(data=values, index=index)
    return series


def make_hist(tank_x, tank_y, tank_charge):

    bins = np.linspace(-1000, 1000, 25, dtype=float)
    hist, _, _ = np.histogram2d(tank_x, tank_y,
                                bins=[bins, bins],
                                weights=tank_charge)
    return hist


def extract_charge_df(input_file):
    with pd.HDFStore(input_file, mode='r') as store:
        sim_num = int(os.path.basename(input_file).split('_')[1])
        tank_x = extract_vector_series(store, key='tank_x', sim=sim_num)
        tank_y = extract_vector_series(store, key='tank_y', sim=sim_num)
        tank_charge = extract_vector_series(store, key='tank_charge', sim=sim_num)

    df = pd.DataFrame({'tank_x': tank_x,
                       'tank_y': tank_y,
                       'tank_charge': tank_charge})
    return df

def save_hdf_file(df, outfile):
    num_events = df.shape[0]
    event_ids, hists = [], []
    for event_id, row in df.iterrows():
        hist = make_hist(row['tank_x'], row['tank_y'], row['tank_charge'])
        hists.append(hist)
        event_ids.append(event_id)

    with h5py.File(outfile, 'w') as f:
        event_id_dset = f.create_dataset('event_id', data=np.asarray(event_ids))
        hist_dset = f.create_dataset('charge_dist', data=np.asarray(hists))

    return


def process_i3_hdf(input_file, outfile):
    df = extract_charge_df(input_file)
    save_hdf_file(df, outfile)

    return


if __name__ == "__main__":

    description = ('Converts input hdf5 files (from save_hdf.py) to a '
                   'well-formatted pandas dataframe object')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config',
                        dest='config',
                        default='IC86.2012',
                        help='Detector configuration')
    parser.add_argument('--type',
                        dest='type',
                        choices=['data', 'sim'],
                        default='sim',
                        help='Option to process simulation or data')
    parser.add_argument('-i', '--input',
                        dest='input',
                        help='Path to input hdf5 file')
    parser.add_argument('-o', '--output',
                        dest='output',
                        help='Path to output hdf5 file')
    args = parser.parse_args()

    # Validate input config
    if (args.type == 'sim' and
            args.config not in comp.simfunctions.get_sim_configs()):
        raise ValueError(
            'Invalid simulation config {} entered'.format(args.config))
    elif (args.type == 'data' and
            args.config not in comp.datafunctions.get_data_configs()):
        raise ValueError('Invalid data config {} entered'.format(args.config))

    # Check if running on condor. If so, write to a local directory on the
    # worker node and copy after output file is written.
    comp.check_output_dir(args.output)
    if os.getenv('_CONDOR_SCRATCH_DIR'):
        on_condor = True
        local_outdir = os.getenv('_CONDOR_SCRATCH_DIR')
        outfile = os.path.join(local_outdir, os.path.basename(args.output))
    else:
        on_condor = False
        outfile = args.output

    process_i3_hdf(input_file=args.input, outfile=outfile)

    # If on condor, transfer from worker machine to desired destination
    if on_condor:
        comp.check_output_dir(args.output)
        shutil.move(outfile, args.output)

#!/usr/bin/env python

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd

import composition as comp


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Converts an input hdf5 file and converts to it a well-formatted output dataframe')
    parser.add_argument('--type', dest='type',
                        choices=['data', 'sim'],
                        default='sim',
                        help='Option to process simulation or data')
    parser.add_argument('-c', '--config', dest='config',
                        default='IC79',
                        help='Detector configuration')
    parser.add_argument('--output', dest='output',
                        help='(Optional) Path to output dataframe hdf5 file')
    parser.add_argument('--overwrite', dest='overwrite',
                        default=False, action='store_true',
                        help='Overwrite existing merged files')
    args = parser.parse_args()

    paths = comp.Paths()
    output = args.output if args.output else '{}/{}_{}/{}_dataframe.hdf5'.format(paths.comp_data_dir, args.config, args.type, args.type)
    # If output file already exists and you want to overwrite,
    # delete existing output file
    if args.overwrite and os.path.exists(output):
        os.remove(output)

    # Get input hdf5 files to merge
    files = glob.glob(
        '{}/{}_{}/dataframe_files/dataframe_*.hdf5'.format(paths.comp_data_dir, args.config, args.type))
    dataframes = []
    for f in files:
        print('Merging {}'.format(f))
        with pd.HDFStore(f) as store:
            dataframes.append(store['dataframe'])

    print('Merging {} DataFrames...'.format(len(dataframes)))
    merged_dataframe = pd.concat(dataframes, ignore_index=True, copy=False)

    with pd.HDFStore(output) as output_store:
        output_store['dataframe'] = merged_dataframe

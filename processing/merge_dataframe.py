#!/usr/bin/env python

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd

import comptools


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

    if args.output:
        output = args.output
    else:
        output = '{}/{}_{}/{}_dataframe.hdf5'.format(
                comptools.paths.comp_data_dir, args.config,
                args.type, args.type)
    # If output file already exists and you want to overwrite,
    # delete existing output file
    if args.overwrite and os.path.exists(output):
        os.remove(output)

    # Get input hdf5 files to merge
    file_pattern = '{}/{}_{}/dataframe_files/dataframe_*.hdf5'.format(
                            comptools.paths.comp_data_dir,
                            args.config, args.type)
    files = glob.glob(file_pattern)
    files = sorted(files)
    comptools.check_output_dir(output)
    with pd.HDFStore(output, 'w') as output_store:
        for f in files:
            with pd.HDFStore(f) as input_store:
                input_df = input_store['dataframe']

                # print('Appending {}...'.format(f))
                if args.type == 'sim':
                    output_store.append('dataframe', input_df, format='table',
                                        data_columns=True, min_itemsize={'MC_comp':15})
                else:
                    output_store.append('dataframe', input_df, format='table',
                                        data_columns=True)

    # merged_dataframe = pd.concat(dataframes, ignore_index=True, copy=False)
    #
    #     # output_store['dataframe'] = merged_dataframe
    #     output_store.put('dataframe', merged_dataframe,
    #                      format='table', data_columns=True)

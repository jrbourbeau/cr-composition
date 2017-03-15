#!/usr/bin/env python

import os
import argparse
import pandas as pd


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Converts an input hdf5 file and converts to it a well-formatted output dataframe')
    parser.add_argument('--infiles', dest='infiles', nargs='+',
                        help='Input dataframe files to be merged')
    parser.add_argument('--outfile', dest='outfile',
                        help='Path to output dataframe hdf5 file')
    parser.add_argument('--overwrite', dest='overwrite',
                        default=False, action='store_true',
                        help='Overwrite existing merged files')
    args = parser.parse_args()

    # If output file already exists and you want to overwrite,
    # delete existing output file
    if args.overwrite and os.path.exists(args.outfile):
        os.remove(args.outfile)

    # Get input hdf5 files to merge
    dataframes = []
    for file_ in args.infiles:
        dataframes.append(pd.read_csv(file_))

    print('Merging {} DataFrames...'.format(len(dataframes)))
    merged_dataframe = pd.concat(dataframes, ignore_index=True)
    merged_dataframe.to_csv(args.outfile)

    # Remove unmerged dataframe files
    for file_ in args.infiles:
        os.remove(file_)

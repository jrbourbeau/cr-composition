#!/usr/bin/env python
#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v1/icetray-start
#METAPROJECT /data/user/jbourbeau/metaprojects/icerec/trunk/build

import argparse
import glob
import os
import sys
import numpy as np

import comptools

if __name__ == "__main__":

    # Set up global path names
    paths = comptools.get_paths()

    p = argparse.ArgumentParser(description='Merges simulation hdf5 files')
    p.add_argument('-f', '--files', dest='files', nargs='*',
                   help='List of files to merge')
    p.add_argument('-o', '--outfile', dest='outfile',
                   help='Output merged hdf5 file name')
    p.add_argument('--overwrite', dest='overwrite',
                   default=False, action='store_true',
                   help='Overwrite existing merged files')
    p.add_argument('--remove', dest='remove',
                   default=False, action='store_true',
                   help='Remove unmerged hdf5 files')
    args = p.parse_args()

    if os.path.isfile(args.outfile) and not args.overwrite:
        print('Outfile {} already exists. Skipping...'.format(args.outfile))
        sys.exit()
    if os.path.isfile(args.outfile) and args.overwrite:
        print('Outfile {} already exists. Overwriting...'.format(args.outfile))
        os.remove(args.outfile)

    files = ' '.join(args.files)
    hdf = '{}/build/hdfwriter/resources'.format(paths.metaproject)
    ex = 'python {}/scripts/merge.py -o {} {}'.format(hdf, args.outfile, files)
    os.system(ex)

    # Remove un-merged hdf files
    if args.remove:
        for f in args.files:
            print('Removing {}...'.format(f))
            os.remove(f)

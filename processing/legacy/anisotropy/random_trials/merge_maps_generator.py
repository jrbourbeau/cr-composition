#!/usr/bin/env python

import os
import argparse
import numpy as np
import healpy as hp

import comptools as comp


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description='Extracts and saves desired information from simulation/data .i3 files')
    p.add_argument('--infiles', dest='infiles', nargs='*',
                   help='Input reference map files')
    p.add_argument('--outfile', dest='outfile',
                   help='Output reference map file')
    p.add_argument('--overwrite', dest='overwrite',
                   default=False, action='store_true',
                   help='Option to overwrite reference map file, '
                        'if it alreadu exists')
    args = p.parse_args()

    if args.infiles is None:
        raise ValueError('Input files must be specified')

    if args.outfile is None:
        raise ValueError('Outfile must be specified')
    else:
        comp.check_output_dir(args.outfile)

    # # Generator to yield all input maps for a give iterable of infiles
    # def generate_maps(infiles):
    #     for f in infiles:
    #         maps = hp.read_map(f, range(3), verbose=False)
    #         yield maps

    # Merge input maps
    merged_maps = np.sum([hp.read_map(f, range(3), verbose=False) for f in args.infiles], axis=0)
    # merged_maps = np.sum(list(generate_maps(args.infiles)), axis=0)
    # merged_maps = np.sum([maps for maps in gen_maps(args.infiles)], axis=0)

    # Write merged maps to file
    hp.write_map(args.outfile, merged_maps, coord='C')

    print('Merged maps successfully saved, deleting unmerged maps')
    for f in args.infiles:
        os.remove(f)

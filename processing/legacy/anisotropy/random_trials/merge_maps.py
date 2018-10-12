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

    # Read in all the input maps
    data_maps = []
    ref_maps = []
    local_maps = []
    for f in args.infiles:
        data_map, ref_map, local_map = hp.read_map(f, range(3), verbose=False)
        data_maps.append(data_map)
        ref_maps.append(ref_map)
        local_maps.append(local_map)

    # Merge maps
    merged_data = np.sum(data_maps, axis=0)
    merged_ref = np.sum(ref_maps, axis=0)
    merged_local = np.sum(local_maps, axis=0)
    hp.write_map(args.outfile, (merged_data, merged_ref, merged_local),
                 coord='C')

    print('Merged maps successfully saved, deleting unmerged maps')
    for f in args.infiles:
        os.remove(f)

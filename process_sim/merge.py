#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v1/icetray-start
#METAPROJECT /data/user/jbourbeau/metaprojects/icerec/trunk/build

import argparse
import glob
import os
import numpy as np

import composition as comp

if __name__ == "__main__":

    # Set up global path names
    mypaths = comp.Paths()
    default_sim_list = ['7006', '7579', '7241', '7263', '7791',
                        '7242', '7262', '7851', '7007', '7784']

    p = argparse.ArgumentParser(description='Merges simulation hdf5 files')
    p.add_argument('-s', '--sim', dest='sim',
                   choices=default_sim_list,
                   help='Simulation to run over')
    p.add_argument('--overwrite', dest='overwrite',
                   default=False, action='store_true',
                   help='Overwrite existing merged files')
    p.add_argument('--remove', dest='remove',
                   default=False, action='store_true',
                   help='Remove unmerged hdf5 files')
    args = p.parse_args()

    # Make comprehensive list of all sim subfiles
    master_list = glob.glob(
        '{}/*_sim/files/sim_{}_part*.hdf5'.format(mypaths.comp_data_dir,
            args.sim))
    master_list.sort()

    # Reduce list to set of all leading filenames (exclude parts)
    params = ['_'.join(f.split('_')[:-1]) for f in master_list]
    params = np.unique(params)

    for file_start in params:

        outfile = '{}.hdf5'.format(file_start)
        if os.path.isfile(outfile) and not args.overwrite:
            print('Outfile {} already exists. Skipping...'.format(outfile))
            continue
        if os.path.isfile(outfile) and args.overwrite:
            print('Outfile {} already exists. Overwriting...'.format(outfile))
            os.remove(outfile)

        file_list = glob.glob(file_start + '_part*-*.hdf5')
        file_list.sort()

        files = ' '.join(file_list)
        hdf = '{}/build/hdfwriter/resources'.format(mypaths.metaproject)
        ex = 'python {}/scripts/merge.py -o {} {}'.format(hdf, outfile, files)
        os.system(ex)

        # Remove un-merged hdf files
        if args.remove:
            for f in file_list:
                print('Removing {}...'.format(f))
                os.remove(f)

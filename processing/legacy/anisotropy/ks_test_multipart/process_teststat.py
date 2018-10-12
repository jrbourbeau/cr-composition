#!/usr/bin/env python

import os
import glob
import argparse
import numpy as np
import pandas as pd
import pycondor

import comptools as comp


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description='Extracts and saves desired information from simulation/data .i3 files')
    p.add_argument('-c', '--config', dest='config',
                   default='IC86.2012',
                   choices=['IC79', 'IC86.2012', 'IC86.2013', 'IC86.2014', 'IC86.2015'],
                   help='Detector configuration')
    p.add_argument('--low_energy', dest='low_energy',
                   default=False, action='store_true',
                   help='Only use events with energy < 10**6.75 GeV')
    args = p.parse_args()

    # Define output directories
    error = comp.paths.condor_data_dir + '/ks_test_{}/error'.format(args.config)
    output = comp.paths.condor_data_dir + '/ks_test_{}/output'.format(args.config)
    log = comp.paths.condor_scratch_dir + '/ks_test_{}/log'.format(args.config)
    submit = comp.paths.condor_scratch_dir + '/ks_test_{}/submit'.format(args.config)

    # Define path to executables
    save_teststat_ex = os.path.join(comp.paths.project_home,
                                'processing/anisotropy/ks_test_multipart',
                                'save_teststat.py')

    # Create Job for saving ks-test p-values for each trial
    save_teststat_name = 'save_teststat_{}'.format(args.config)
    if args.low_energy:
        save_teststat_name += '_lowenergy'
    save_teststat_job = pycondor.Job(save_teststat_name, save_teststat_ex,
                                  error=error, output=output,
                                  log=log, submit=submit,
                                  verbose=1)

    map_dir = os.path.join(comp.paths.comp_data_dir, args.config + '_data',
                           'anisotropy/random_splits')
    sample_0_file_pattern = os.path.join(map_dir, 'random_split_0_trail-*.fits')
    sample_1_file_pattern = os.path.join(map_dir, 'random_split_1_trail-*.fits')
    infiles_sample_0 = sorted(glob.glob(sample_0_file_pattern))
    infiles_sample_1 = sorted(glob.glob(sample_1_file_pattern))
    infiles_sample_0_str = ' '.join(infiles_sample_0)
    infiles_sample_1_str = ' '.join(infiles_sample_1)

    save_teststat_args = []
    save_teststat_args.append('--infiles_sample_0 {}'.format(infiles_sample_0_str))
    save_teststat_args.append('--infiles_sample_1 {}'.format(infiles_sample_1_str))

    if args.low_energy:
        outfile_basename = 'teststat_dataframe_lowenergy.hdf'
    else:
        outfile_basename = 'teststat_dataframe.hdf'
    outfile = os.path.join(map_dir, outfile_basename)
    save_teststat_args.append('--outfile {}'.format(outfile))

    save_teststat_arg = ' '.join(save_teststat_args)
    save_teststat_job.add_arg(save_teststat_arg)
    save_teststat_job.build_submit()

#!/usr/bin/env python

import os
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
    p.add_argument('--n_side', dest='n_side', type=int,
                   default=64,
                   help='Number of times to split the DataFrame')
    p.add_argument('--chunksize', dest='chunksize', type=int,
                   default=10000,
                   help='Number of lines used when reading in DataFrame')
    p.add_argument('--ks_trials', dest='ks_trials', type=int,
                   default=1000,
                   help='Number of random maps to generate')
    p.add_argument('--overwrite', dest='overwrite',
                   default=False, action='store_true',
                   help='Option to overwrite reference map file, '
                        'if it alreadu exists')
    args = p.parse_args()

    # Define output directories
    error = comp.paths.condor_data_dir + '/error'
    output = comp.paths.condor_data_dir + '/output'
    log = comp.paths.condor_scratch_dir + '/log'
    submit = comp.paths.condor_scratch_dir + '/submit'

    # Define path to executables
    make_maps_ex = os.path.join(comp.paths.project_home,
                                'processing/anisotropy/ks_test',
                                'make_maps.py')
    save_pvals_ex = os.path.join(comp.paths.project_home,
                                 'processing/anisotropy/ks_test',
                                 'save_pvals.py')

    make_maps_name = 'make_maps_{}_{}-trials'.format(args.config, args.ks_trials)
    if args.low_energy: make_maps_name += '_lowenergy'
    make_maps_job = pycondor.Job(make_maps_name, make_maps_ex,
                                 error=error, output=output,
                                 log=log, submit=submit,
                                 verbose=1)

    save_pvals_name = 'save_pvals_{}_{}-trials'.format(args.config, args.ks_trials)
    if args.low_energy: save_pvals_name += '_lowenergy'
    save_pvals_job = pycondor.Job(save_pvals_name, save_pvals_ex,
                                  error=error, output=output,
                                  log=log, submit=submit,
                                  verbose=1)

    # Ensure that make_maps_job completes before save_pvals_job begins
    save_pvals_job.add_parent(make_maps_job)

    save_pvals_infiles_0 = []
    save_pvals_infiles_1 = []

    for trial_index in np.arange(args.ks_trials):

        outfile_sample_1 = os.path.join(comp.paths.comp_data_dir,
                                        args.config + '_data', 'anisotropy',
                                        'random_splits',
                                        'random_split_1_trial_{}.fits'.format(trial_index))
        outfile_sample_0 = os.path.join(comp.paths.comp_data_dir,
                                        args.config + '_data', 'anisotropy',
                                        'random_splits',
                                        'random_split_0_trial_{}.fits'.format(trial_index))

        make_maps_arg = '-c {} --n_side {} --chunksize {} ' \
                        '--outfile_sample_0 {} ' \
                        '--outfile_sample_1 {}'.format(
                                args.config, args.n_side, args.chunksize,
                                outfile_sample_0, outfile_sample_1)
        if args.low_energy:
            make_maps_arg += ' --low_energy'

        make_maps_job.add_arg(make_maps_arg)
        # Add this outfile to the list of infiles for save_pvals_job
        save_pvals_infiles_0.append(outfile_sample_0)
        save_pvals_infiles_1.append(outfile_sample_1)

    infiles_sample_0_str = ' '.join(save_pvals_infiles_0)
    infiles_sample_1_str = ' '.join(save_pvals_infiles_1)
    # Assemble merged output file path
    outdir = os.path.join(comp.paths.comp_data_dir, args.config + '_data',
                          'anisotropy', 'random_splits')
    if args.low_energy:
        outfile_basename = 'ks_test_dataframe_lowenergy.hdf'
    else:
        outfile_basename = 'ks_test_dataframe.hdf'

    outfile = os.path.join(outdir, outfile_basename)

    save_pvals_arg = '--infiles_sample_0 {} --infiles_sample_1 {} ' \
                     '--outfile {}'.format(infiles_sample_0_str, infiles_sample_1_str, outfile)
    save_pvals_job.add_arg(save_pvals_arg)

    # Create Dagman instance
    dag_name = 'anisotropy_kstest_{}'.format(args.config)

    dagman = pycondor.Dagman(dag_name, submit=submit, verbose=1)
    dagman.add_job(make_maps_job)
    dagman.add_job(save_pvals_job)
    dagman.build_submit(fancyname=True)

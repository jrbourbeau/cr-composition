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
                   default=1000,
                   help='Number of lines used when reading in DataFrame')
    p.add_argument('--n_batches', dest='n_batches', type=int,
                   default=50,
                   help='Number batches running in parallel for each ks-test trial')
    p.add_argument('--ks_trials', dest='ks_trials', type=int,
                   default=100,
                   help='Number of random maps to generate')
    p.add_argument('--overwrite', dest='overwrite',
                   default=False, action='store_true',
                   help='Option to overwrite reference map file, '
                        'if it alreadu exists')
    p.add_argument('--test', dest='test',
                   default=False, action='store_true',
                   help='Option to run small test version')
    args = p.parse_args()

    if args.test:
        args.ks_trials = 20
        args.n_batches = 10000
        args.chunksize = 100

    # Define output directories
    error = comp.paths.condor_data_dir + '/ks_test_{}/error'.format(args.config)
    output = comp.paths.condor_data_dir + '/ks_test_{}/output'.format(args.config)
    log = comp.paths.condor_scratch_dir + '/ks_test_{}/log'.format(args.config)
    submit = comp.paths.condor_scratch_dir + '/ks_test_{}/submit'.format(args.config)

    # Define path to executables
    make_maps_ex = os.path.join(comp.paths.project_home,
                                'processing/anisotropy/ks_test_multipart',
                                'make_maps.py')
    merge_maps_ex = os.path.join(comp.paths.project_home,
                                 'processing/anisotropy/ks_test_multipart',
                                 'merge_maps.py')
    save_pvals_ex = os.path.join(comp.paths.project_home,
                                 'processing/anisotropy/ks_test_multipart',
                                 'save_pvals.py')

    # Create Dagman instance
    dag_name = 'anisotropy_kstest_{}'.format(args.config)
    if args.test:
        dag_name += '_test'
    dagman = pycondor.Dagman(dag_name, submit=submit, verbose=1)

    # Create Job for saving ks-test p-values for each trial
    save_pvals_name = 'save_pvals_{}'.format(args.config)
    if args.low_energy:
        save_pvals_name += '_lowenergy'
    save_pvals_job = pycondor.Job(save_pvals_name, save_pvals_ex,
                                  error=error, output=output,
                                  log=log, submit=submit,
                                  verbose=1)

    save_pvals_infiles_0 = []
    save_pvals_infiles_1 = []

    dagman.add_job(save_pvals_job)

    outdir = os.path.join(comp.paths.comp_data_dir, args.config + '_data',
                          'anisotropy', 'random_splits')
    if args.test:
        outdir = os.path.join(outdir, 'test')
    for trial_num in range(args.ks_trials):
        # Create map_maps jobs for this ks_trial
        make_maps_name = 'make_maps_{}_trial-{}'.format(args.config, trial_num)
        if args.low_energy:
            make_maps_name += '_lowenergy'
        make_maps_job = pycondor.Job(make_maps_name, make_maps_ex,
                                     error=error, output=output,
                                     log=log, submit=submit,
                                     verbose=1)
        dagman.add_job(make_maps_job)

        merge_maps_infiles_0 = []
        merge_maps_infiles_1 = []
        for batch_idx in range(args.n_batches):
            if args.test and batch_idx > 2:
                break

            outfile_sample_1 = os.path.join(outdir,
                    'random_split_1_trial-{}_batch-{}.fits'.format(trial_num, batch_idx))
            outfile_sample_0 = os.path.join(outdir,
                    'random_split_0_trial-{}_batch-{}.fits'.format(trial_num, batch_idx))

            make_maps_arg_list = []
            make_maps_arg_list.append('--config {}'.format(args.config))
            make_maps_arg_list.append('--n_side {}'.format(args.n_side))
            make_maps_arg_list.append('--chunksize {}'.format(args.chunksize))
            make_maps_arg_list.append('--n_batches {}'.format(args.n_batches))
            make_maps_arg_list.append('--batch_idx {}'.format(batch_idx))
            make_maps_arg_list.append('--outfile_sample_0 {}'.format(outfile_sample_0))
            make_maps_arg_list.append('--outfile_sample_1 {}'.format(outfile_sample_1))
            make_maps_arg = ' '.join(make_maps_arg_list)
            if args.low_energy:
                make_maps_arg += ' --low_energy'

            make_maps_job.add_arg(make_maps_arg)

            # Add this outfile to the list of infiles for merge_maps_job
            merge_maps_infiles_0.append(outfile_sample_0)
            merge_maps_infiles_1.append(outfile_sample_1)

        for sample_idx, input_file_list in enumerate([merge_maps_infiles_0,
                                                      merge_maps_infiles_1]):
            merge_maps_name = 'merge_maps_{}_trial-{}_split-{}'.format(args.config, trial_num, sample_idx)
            if args.low_energy:
                merge_maps_name += '_lowenergy'
            merge_maps_job = pycondor.Job(merge_maps_name, merge_maps_ex,
                                          error=error, output=output,
                                          log=log, submit=submit,
                                          verbose=1)

            # Ensure that make_maps_job completes before merge_maps_job begins
            make_maps_job.add_child(merge_maps_job)
            merge_maps_job.add_child(save_pvals_job)
            dagman.add_job(merge_maps_job)

            merge_infiles_str = ' '.join(input_file_list)
            # Assemble merged output file path
            merge_outfile = os.path.join(outdir, 'random_split_{}_trial-{}.fits'.format(sample_idx, trial_num))

            merge_maps_arg = '--infiles {} --outfile {}'.format(merge_infiles_str, merge_outfile)
            merge_maps_job.add_arg(merge_maps_arg)

            if sample_idx == 0:
                save_pvals_infiles_0.append(merge_outfile)
            else:
                save_pvals_infiles_1.append(merge_outfile)

    save_pvals_infiles_0_str = ' '.join(save_pvals_infiles_0)
    save_pvals_infiles_1_str = ' '.join(save_pvals_infiles_1)
    if args.low_energy:
        outfile_basename = 'ks_test_dataframe_lowenergy.hdf'
    else:
        outfile_basename = 'ks_test_dataframe.hdf'
    outfile = os.path.join(outdir, outfile_basename)
    save_pvals_arg = '--infiles_sample_0 {} --infiles_sample_1 {} ' \
                     '--outfile {}'.format(save_pvals_infiles_0_str, save_pvals_infiles_1_str, outfile)
    save_pvals_job.add_arg(save_pvals_arg)

    dagman.build_submit(fancyname=True)

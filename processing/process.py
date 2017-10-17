#!/usr/bin/env python

import os
import glob
import argparse
import numpy as np
import pyprind

import comptools
import pycondor


def add_sim_jobs(dagman, save_hdf5_ex, merge_hdf5_ex, save_df_ex, **args):

    for sim in args['sim']:
        # Create a save and merge CondorJobs for each simulation
        save_hdf5_name = 'save_hdf5_sim_{}'.format(sim)
        save_hdf5_job = pycondor.Job(save_hdf5_name, save_hdf5_ex,
                                     error=error, output=output,
                                     log=log, submit=submit,
                                     verbose=1)
        merge_hdf5_name = 'merge_hdf5_sim_{}'.format(sim)
        merge_hdf5_job = pycondor.Job(merge_hdf5_name, merge_hdf5_ex,
                                      error=error, output=output,
                                      log=log, submit=submit,
                                      verbose=1)
        # Ensure that save_hdf5_job completes before merge_hdf5_job
        merge_hdf5_job.add_parent(save_hdf5_job)

        save_df_name = 'save_df_sim_{}'.format(sim)
        save_df_job = pycondor.Job(save_df_name, save_df_ex,
                                   error=error, output=output,
                                   log=log, submit=submit,
                                   request_memory='3GB',
                                   verbose=1)
        # Ensure that merge_hdf5_job completes before save_df_job
        save_df_job.add_parent(merge_hdf5_job)

        # Get config and simulation files
        config = comptools.simfunctions.sim_to_config(sim)
        gcd, files = comptools.simfunctions.get_level3_sim_files(sim)

        # Set up output directory (also, make sure directory exists)
        outdir = os.path.join(comptools.paths.comp_data_dir,
                              config+'_sim/hdf5_files')

        # Split file list into smaller batches for submission
        if args['test']:
            args['n'] = 50
        batches = [files[i:i + args['n']] for i in range(0, len(files), args['n'])]
        if args['test']:
            batches = batches[:20]

        merger_input = ''
        for files in batches:
            # Name output hdf5 file
            start_index = files[0].find('Run') + 3
            end_index = files[0].find('.i3.gz')
            start = files[0][start_index:end_index]
            end = files[-1][start_index:end_index]
            out = '{}/sim_{}_part{}-{}.hdf5'.format(outdir, sim, start, end)

            # Don't forget to insert GCD file at beginning of FileNameList
            files.insert(0, gcd)
            files_str = ' '.join(files)

            arg = '--type sim --files {} -o {}'.format(files_str, out)
            # arg = '--files {} -s {} -o {}'.format(files_str, sim, out)
            save_hdf5_job.add_arg(arg)

            # Append out to merger_input
            merger_input += out + ' '

        # Add job for this sim to the dagmanager
        dagman.add_job(save_hdf5_job)

        # Finish constructing merge sim job
        merger_output = '{}/sim_{}.hdf5'.format(outdir, sim)
        merge_arg = '--files {} -o {}'.format(merger_input, merger_output)
        if args['remove']:
            merge_arg += ' --remove'
        if args['overwrite']:
            merge_arg += ' --overwrite'
        merge_hdf5_job.add_arg(merge_arg)
        # Add merge job for this sim to the dagmanager
        dagman.add_job(merge_hdf5_job)

        df_outfile = '{}/{}_sim/dataframe_files/dataframe_{}.hdf5'.format(
            comptools.paths.comp_data_dir, config, sim)
        df_arg = '--input {} --output {} --type sim'.format(merger_output, df_outfile)
        if args['overwrite']:
            df_arg += ' --overwrite'
        save_df_job.add_arg(df_arg)
        # save_df_job.add_arg(df_arg, name='{}'.format(sim), retry=15)
        # Add save save_df to dagmanager
        dagman.add_job(save_df_job)

    return dagman


def add_data_jobs(dagman, save_hdf5_ex, merge_hdf5_ex, save_df_ex, **args):

    config = args['config']

    # Set up output directory (also, make sure directory exists)
    outdir = os.path.join(comptools.paths.comp_data_dir,
                          '{}_data/hdf5_files'.format(config))

    # Create a save and merge CondorJobs
    save_hdf5_name = 'save_hdf5_data_{}'.format(config)
    save_hdf5_job = pycondor.Job(save_hdf5_name, save_hdf5_ex,
                                 error=error, output=output,
                                 log=log, submit=submit,
                                 verbose=1)
    merge_hdf5_name = 'merge_hdf5_data_{}'.format(config)
    merge_hdf5_job = pycondor.Job(merge_hdf5_name, merge_hdf5_ex,
                                  error=error, output=output,
                                  log=log, submit=submit,
                                  verbose=1)
    # Ensure that save_hdf5_job completes before merge_hdf5_job
    merge_hdf5_job.add_parent(save_hdf5_job)

    save_df_name = 'save_df_data_{}'.format(config)
    save_df_job = pycondor.Job(save_df_name, save_df_ex,
                               error=error, output=output,
                               log=log, submit=submit,
                               verbose=1)
    # Ensure that merge_hdf5_job completes before save_df_job
    save_df_job.add_parent(merge_hdf5_job)
    run_list = comptools.datafunctions.get_run_list(config)
    if args['test']:
        run_list = run_list[:2]
    bar = pyprind.ProgBar(len(run_list),
        title='Adding {} data jobs'.format(config))
    for run in run_list:

        # Get files associated with this run
        gcd, run_files = comptools.datafunctions.get_level3_run_i3_files(config=config, run=run)

        batches = (run_files[i:i + args['n']] for i in range(0, len(run_files), args['n']))
        merged_output = '{}/data_{}.hdf5'.format(outdir, run)
        merged_input = ''
        for idx, files in enumerate(batches):
            # Name output hdf5 file
            out = '{}/data_{}_part_{:02d}.hdf5'.format(outdir, run, idx)

            merged_input += out + ' '
            # Don't forget to insert GCD file at beginning of FileNameList
            files.insert(0, gcd)
            files_str = ' '.join(files)

            save_arg = '--type data --files {} -o {}'.format(files_str, out)
            save_hdf5_job.add_arg(save_arg)

        merge_arg = '--files {} -o {}'.format(merged_input, merged_output)
        if args['remove']:
            merge_arg += ' --remove'
        if args['overwrite']:
            merge_arg += ' --overwrite'
        merge_hdf5_job.add_arg(merge_arg)

        # Add save save_df to dagmanager
        df_outfile = '{}/{}_data/dataframe_files/dataframe_run_{}.hdf5'.format(
                            comptools.paths.comp_data_dir, config, run)
        df_arg = '--input {} --output {} --type data'.format(merged_output, df_outfile)
        if args['overwrite']:
            df_arg += ' --overwrite'
        save_df_job.add_arg(df_arg, retry=15)

        bar.update()
    print(bar)

    # Add save job to the dagmanager
    dagman.add_job(save_hdf5_job)
    # Add merge job to the dagman
    dagman.add_job(merge_hdf5_job)

    dagman.add_job(save_df_job)

    return dagman


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description='Extracts and saves desired information from simulation/data .i3 files')
    p.add_argument('--type', dest='type',
                   choices=['data', 'sim'],
                   default='sim',
                   help='Option to process simulation or data')
    p.add_argument('-d', '--date', dest='date',
                   help='Date to run over (mmyyyy)')
    p.add_argument('-c', '--config', dest='config',
                   choices=comptools.datafunctions.get_data_configs(),
                   help='Detector configuration')
    p.add_argument('-s', '--sim', dest='sim', nargs='*', type=int,
                   help='Simulation to run over')
    p.add_argument('-n', '--n', dest='n', type=int,
                   #    default=200,
                   help='Number of files to run per batch')
    p.add_argument('--test', dest='test', action='store_true',
                   default=False,
                   help='Option for running test off cluster')
    p.add_argument('--maxjobs', dest='maxjobs', type=int,
                   default=3000,
                   help='Maximum number of jobs to run at a given time.')
    p.add_argument('--overwrite', dest='overwrite',
                   default=False, action='store_true',
                   help='Overwrite existing merged files')
    p.add_argument('--remove', dest='remove',
                   default=False, action='store_true',
                   help='Remove unmerged hdf5 files')
    args = p.parse_args()

    if not args.n:
        if args.type == 'sim':
            args.n = 1000
        else:
            args.n = 50

    if args.type == 'sim' and not args.sim:
        args.sim = comptools.simfunctions.config_to_sim(args.config)

    # Define output directories
    error = os.path.join(comptools.paths.condor_data_dir, '/error')
    output = os.path.join(comptools.paths.condor_data_dir, '/output')
    log = os.path.join(comptools.paths.condor_scratch_dir, '/log')
    submit = os.path.join(comptools.paths.condor_scratch_dir, '/submit')

    # Create Dagman to manage processing workflow
    name = 'processing_{}_{}'.format(args.type, args.config)
    dagman = pycondor.Dagman(name, submit=submit, verbose=1)

    # Define path to executables
    save_hdf5_ex = os.path.join(comptools.paths.project_root, 'processing',
                                'save_hdf5.py')
    merge_hdf5_ex = os.path.join(comptools.paths.project_root, 'processing',
                                 'merge_hdf5.py')
    save_df_ex = os.path.join(comptools.paths.project_root, 'processing',
                              'save_dataframe.py')

    if args.type == 'sim':
        dagman = add_sim_jobs(dagman, save_hdf5_ex, merge_hdf5_ex, save_df_ex,
                              **vars(args))
    else:
        dagman = add_data_jobs(dagman, save_hdf5_ex, merge_hdf5_ex, save_df_ex,
                               **vars(args))

    # Add dataframe merger job
    merge_df_ex = os.path.join(comptools.paths.project_root, 'processing',
                               'merge_dataframe.py')
    merge_df_name = 'merge_df_{}'.format(args.type)
    if args.test:
        merge_request_memory = '1GB'
    elif args.type == 'sim':
        merge_request_memory = '5GB'
    else:
        merge_request_memory = '5GB'
    merge_df_job = pycondor.Job(merge_df_name, merge_df_ex,
                                error=error, output=output,
                                log=log, submit=submit,
                                request_memory=merge_request_memory,
                                verbose=1)
    merge_df_arg = '--type {} --config {}'.format(args.type, args.config)
    if args.overwrite:
        merge_df_arg += ' --overwrite'
    merge_df_job.add_arg(merge_df_arg)

    dagman.add_job(merge_df_job)

    # Ensure that all dataframes are made before merging
    for job in dagman:
        if 'save_df' in job.name:
            merge_df_job.add_parent(job)
        else:
            continue

    # Add job for training and saving energy reconstruction model
    if args.type == 'sim':
        energy_reco_ex = os.path.join(comptools.paths.project_root, 'models',
                                      'save_energy_reco_model.py')
        energy_reco_job = pycondor.Job('energy_reco', energy_reco_ex,
                                       error=error, output=output,
                                       log=log, submit=submit,
                                       verbose=1)

        energy_reco_job.add_arg('--config {}'.format(args.config))
        dagman.add_job(energy_reco_job)

    # Build and submit dagman
    dagman.build_submit(maxjobs=args.maxjobs, fancyname=True)

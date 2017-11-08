#!/usr/bin/env python

import os
import glob
import argparse
import numpy as np
import pyprind
import pycondor

import comptools as comp


def add_sim_jobs(dagman, save_hdf5_ex, save_df_ex, **args):

    save_df_name = 'save_df_sim_{}'.format(args['config'])
    save_df_job = pycondor.Job(save_df_name, save_df_ex,
                               error=error, output=output,
                               log=log, submit=submit,
                               request_memory='3GB',
                               verbose=1)
    # Add save_df_job to dagman
    dagman.add_job(save_df_job)

    save_df_input_files = []
    for sim in args['sim']:
        # Create a save and merge pycondor Job for each simulation set
        save_hdf5_name = 'save_hdf5_sim_{}'.format(sim)
        save_hdf5_job = pycondor.Job(save_hdf5_name, save_hdf5_ex,
                                     error=error, output=output,
                                     log=log, submit=submit,
                                     verbose=1)
        # Ensure that save_hdf5_job completes before save_df_job
        save_df_job.add_parent(save_hdf5_job)

        # Get config and simulation files
        config = comp.simfunctions.sim_to_config(sim)
        gcd, i3_files = comp.simfunctions.get_level3_sim_files(sim)
        # Set up output directory
        outdir = os.path.join(comp.paths.comp_data_dir, config,
                              'i3_hdf_sim')
        # Split file list into smaller batches for submission
        if args['test']:
            args['n'] = 10
            n_batches = 2
        else:
            n_batches = None

        for files in comp.file_batches(i3_files, args['n'], n_batches):
            # Name output hdf5 file
            start_index = files[0].find('Run') + 3
            end_index = files[0].find('.i3.gz')
            start = files[0][start_index:end_index]
            end = files[-1][start_index:end_index]
            out = '{}/sim_{}_part{}-{}.hdf5'.format(outdir, sim, start, end)
            comp.check_output_dir(out)

            # Don't forget to insert GCD file at beginning of FileNameList
            files.insert(0, gcd)
            files_str = ' '.join(files)

            arg = '--type sim --files {} -o {}'.format(files_str, out)
            save_hdf5_job.add_arg(arg, retry=3)

            save_df_input_files.append(out)
        # Add job for this sim to the dagmanager
        dagman.add_job(save_hdf5_job)

    df_outfile = os.path.join(comp.paths.comp_data_dir, args['config'],
                              'sim_dataframe.hdf5')
    df_input_files_str = ' '.join(save_df_input_files)
    df_arg = '--input {} --output {} --type sim --config {}'.format(
        df_input_files_str, df_outfile, args['config'])
    save_df_job.add_arg(df_arg)

    return dagman


def add_data_jobs(dagman, save_hdf5_ex, save_df_ex, **args):

    config = args['config']

    # Set up output directory (also, make sure directory exists)
    outdir = os.path.join(comp.paths.comp_data_dir, config,
                          'i3_hdf_data')

    # Create a save and merge CondorJobs
    save_hdf5_name = 'save_hdf5_data_{}'.format(config)
    save_hdf5_job = pycondor.Job(save_hdf5_name, save_hdf5_ex,
                                 error=error, output=output,
                                 log=log, submit=submit,
                                 verbose=1)
    # Add save_hdf5_job to dagman
    dagman.add_job(save_hdf5_job)

    save_df_name = 'save_df_data_{}'.format(config)
    save_df_job = pycondor.Job(save_df_name, save_df_ex,
                               error=error, output=output,
                               log=log, submit=submit,
                               request_memory='5GB',
                               verbose=1)
    # Ensure that save_df_job completes before save_df_job
    save_df_job.add_parent(save_hdf5_job)
    # Add save_df_job to dagman
    dagman.add_job(save_df_job)

    run_list = comp.datafunctions.get_run_list(config)
    if args['test']:
        run_list = run_list[:2]
        n_batches = 2
    else:
        n_batches = None

    save_df_input_files = []
    bar = pyprind.ProgBar(len(run_list),
                          title='Adding {} data jobs'.format(config))
    for run in run_list:
        # Get files associated with this run
        gcd, run_files = comp.datafunctions.get_level3_run_i3_files(config=config, run=run)
        for idx, files in enumerate(comp.file_batches(run_files, args['n'], n_batches)):
            # Name output hdf5 file
            out = '{}/data_{}_part_{:02d}.hdf5'.format(outdir, run, idx)
            # Don't forget to insert GCD file at beginning of FileNameList
            files.insert(0, gcd)
            files_str = ' '.join(files)
            save_arg = '--type data --files {} -o {}'.format(files_str, out)
            save_hdf5_job.add_arg(save_arg, retry=3)
            save_df_input_files.append(out)
        bar.update()
    print(bar)

    # Add save save_df to dagman
    df_outfile = os.path.join(comp.paths.comp_data_dir, config,
                              'data_dataframe.hdf5')
    df_input_files_str = ' '.join(save_df_input_files)
    df_arg = '--input {} --output {} --type data --config {}'.format(
        df_input_files_str, df_outfile, config)
    save_df_job.add_arg(df_arg)

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
                   default='IC86.2012',
                   help='Detector configuration')
    p.add_argument('-s', '--sim', dest='sim', nargs='*', type=int,
                   help='Simulation to run over')
    p.add_argument('-n', '--n', dest='n', type=int,
                   help='Number of files to run per batch')
    p.add_argument('--test', dest='test', action='store_true',
                   default=False,
                   help='Option for running test off cluster')
    p.add_argument('--overwrite', dest='overwrite', action='store_true',
                   default=False,
                   help='Option for overwriting existing files.')
    p.add_argument('--maxjobs', dest='maxjobs', type=int,
                   default=3000,
                   help='Maximum number of jobs to run at a given time.')
    args = p.parse_args()

    # Validate input config
    if (args.type == 'sim' and
            args.config not in comp.simfunctions.get_sim_configs()):
        raise ValueError(
            'Invalid simulation config {} entered'.format(args.config))
    elif (args.type == 'data' and
            args.config not in comp.datafunctions.get_data_configs()):
        raise ValueError('Invalid data config {} entered'.format(args.config))

    if not args.n:
        if args.type == 'sim':
            args.n = 1000
        else:
            args.n = 50

    if args.type == 'sim' and not args.sim:
        args.sim = comp.simfunctions.config_to_sim(args.config)

    # Define output directories
    error = os.path.join(comp.paths.condor_data_dir, 'error')
    output = os.path.join(comp.paths.condor_data_dir, 'output')
    log = os.path.join(comp.paths.condor_scratch_dir, 'log')
    submit = os.path.join(comp.paths.condor_scratch_dir, 'submit')

    # Create Dagman to manage processing workflow
    name = 'processing_{}_{}'.format(args.config, args.type)
    dagman = pycondor.Dagman(name, submit=submit, verbose=1)

    # Define path to executables
    save_hdf5_ex = os.path.join(comp.paths.project_root, 'processing',
                                'save_hdf5.py')
    save_df_ex = os.path.join(comp.paths.project_root, 'processing',
                              'save_dataframe.py')

    if args.type == 'sim':
        dagman = add_sim_jobs(dagman, save_hdf5_ex, save_df_ex, **vars(args))
    else:
        dagman = add_data_jobs(dagman, save_hdf5_ex, save_df_ex, **vars(args))

    # # Add job for training and saving energy reconstruction model
    # if args.type == 'sim':
    #     energy_reco_ex = os.path.join(comp.paths.project_root, 'models',
    #                                   'save_energy_reco_model.py')
    #     energy_reco_job = pycondor.Job('energy_reco', energy_reco_ex,
    #                                    error=error, output=output,
    #                                    log=log, submit=submit,
    #                                    verbose=1)
    #
    #     energy_reco_job.add_arg('--config {}'.format(args.config))
    #     energy_reco_job.add_parent(merge_df_job)
    #     dagman.add_job(energy_reco_job)

    # Build and submit dagman
    dagman.build_submit(maxjobs=args.maxjobs, fancyname=True)

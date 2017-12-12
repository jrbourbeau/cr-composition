#!/usr/bin/env python
"""
Script to process data and simulation .i3 files and save pandas.DataFrame
objects used for this analysis.

Example usages:
    python process.py  --config IC86.2012
    python process.py  --config IC86.2012 --sim
    python process.py  --config IC86.2012 --data
"""

import os
from itertools import chain
import argparse
import pyprind
import pycondor

import comptools as comp
from comptools import ComputingEnvironemtError


if os.getenv('I3_BUILD') is None:
    raise ComputingEnvironemtError(
            'Did not detect an active icecube software environment. '
            'Make sure source the env-shell.sh script in your '
            'icecube metaproject build directory before running '
            'process.py')
if 'cvmfs' not in os.getenv('ROOTSYS'):
    raise ComputingEnvironemtError(
            'CVMFS ROOT must be used for i3 file processing')


def gen_sim_jobs(save_hdf5_ex, save_df_ex, config, sims, n=1000, test=False):
    """Yields pycondor Jobs for simulation processing

    Parameters
    ----------
    save_hdf5_ex : str
        Path to icetray script.
    save_df_ex : str
        Path to script to save dataframe.
    config : str
        Detector configuration.
    sims : array_like
        Iterable of detector configurations.
    n : int, optional
        Batch size (default is 1000).
    test : bool, optional
        Option to run in testing mode (default is False).

    Yields
    ------
    job : pycondor.Job
        Simulation Job to be included in processing Dagman.
    """
    save_df_name = 'save_df_sim_{}'.format(config.replace('.', '-'))
    save_df_job = pycondor.Job(name=save_df_name,
                               executable=save_df_ex,
                               error=error,
                               output=output,
                               log=log,
                               submit=submit,
                               request_memory='3GB' if test else None,
                               verbose=1)

    save_df_input_files = []
    for sim in sims:
        # Create a save and merge pycondor Job for each simulation set
        save_hdf5_name = 'save_hdf5_sim_{}'.format(sim)
        save_hdf5_job = pycondor.Job(name=save_hdf5_name,
                                     executable=save_hdf5_ex,
                                     error=error,
                                     output=output,
                                     log=log,
                                     submit=submit,
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
        if test:
            n = 10
            n_batches = 2
        else:
            n_batches = None

        for files in comp.file_batches(i3_files, n, n_batches):
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

        yield save_hdf5_job

    df_outfile = os.path.join(comp.paths.comp_data_dir, config,
                              'sim_dataframe.hdf5')
    df_input_files_str = ' '.join(save_df_input_files)
    df_arg = '--input {} --output {} --type sim --config {}'.format(
        df_input_files_str, df_outfile, config)
    save_df_job.add_arg(df_arg)

    yield save_df_job


def gen_data_jobs(save_hdf5_ex, save_df_ex, config, n=50, test=False):
    """Yields pycondor Jobs for data processing

    Parameters
    ----------
    save_hdf5_ex : str
        Path to icetray script.
    save_df_ex : str
        Path to script to save dataframe.
    config : str
        Detector configuration.
    n : int, optional
        Batch size (default is 50).
    test : bool, optional
        Option to run in testing mode (default is False).

    Yields
    ------
    job : pycondor.Job
        Data Job to be included in processing Dagman.
    """
    # Set up output directory (also, make sure directory exists)
    outdir = os.path.join(comp.paths.comp_data_dir, config,
                          'i3_hdf_data')

    # Create a save and merge CondorJobs
    save_hdf5_name = 'save_hdf5_data_{}'.format(config.replace('.', '-'))
    save_hdf5_job = pycondor.Job(name=save_hdf5_name,
                                 executable=save_hdf5_ex,
                                 error=error,
                                 output=output,
                                 log=log,
                                 submit=submit,
                                 verbose=1)

    save_df_name = 'save_df_data_{}'.format(config.replace('.', '-'))
    save_df_job = pycondor.Job(name=save_df_name,
                               executable=save_df_ex,
                               error=error,
                               output=output,
                               log=log,
                               submit=submit,
                               request_memory='5GB' if test else None,
                               verbose=1)
    # Ensure that save_df_job completes before save_df_job
    save_df_job.add_parent(save_hdf5_job)

    run_list = comp.datafunctions.get_run_list(config)
    if test:
        run_list = run_list[:2]
        n_batches = 2
    else:
        n_batches = None

    save_df_input_files = []
    bar = pyprind.ProgBar(len(run_list),
                          title='Adding {} data jobs'.format(config))
    for run in run_list:
        # Get files associated with this run
        gcd, run_files = comp.datafunctions.get_level3_run_i3_files(
                                                        config=config, run=run)
        data_file_batches = comp.file_batches(run_files, n, n_batches)
        for idx, files in enumerate(data_file_batches):
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
    yield save_hdf5_job

    df_outfile = os.path.join(comp.paths.comp_data_dir, config,
                              'data_dataframe.hdf5')
    df_input_files_str = ' '.join(save_df_input_files)
    df_arg = '--input {} --output {} --type data --config {}'.format(
        df_input_files_str, df_outfile, config)
    save_df_job.add_arg(df_arg)

    yield save_df_job


if __name__ == "__main__":

    description = 'Processes simulation and data .i3 files'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--type', dest='type',
                        choices=['data', 'sim'],
                        help=('Option to restring process to simulation only '
                              'or data only'))
    parser.add_argument('-d', '--date', dest='date',
                        help='Date to run over (mmyyyy)')
    parser.add_argument('-c', '--config', dest='config',
                        default='IC86.2012',
                        help='Detector configuration')
    parser.add_argument('-s', '--sim', dest='sim',
                        nargs='*', type=int,
                        help='Simulation to run over')
    parser.add_argument('--n_sim', dest='n_sim',
                        type=int, default=1000,
                        help=('Number of files to run per batch for '
                              'simulation processing'))
    parser.add_argument('--n_data', dest='n_data',
                        type=int, default=50,
                        help=('Number of files to run per batch for '
                              'data processing'))
    parser.add_argument('--test', dest='test',
                        action='store_true',
                        default=False,
                        help='Option for running test off cluster')
    parser.add_argument('--overwrite', dest='overwrite',
                        action='store_true',
                        default=False,
                        help='Option for overwriting existing files.')
    parser.add_argument('--maxjobs', dest='maxjobs',
                        type=int, default=3000,
                        help='Maximum number of jobs to run at a given time.')
    args = parser.parse_args()

    # Validate user inputs
    if args.type not in ['sim', 'data', None]:
        raise ValueError("Invalid processing type entered. Must be either "
                         "'sim', 'data', or None.")
    process_types = ['sim', 'data'] if args.type is None else args.type

    sim_configs = comp.simfunctions.get_sim_configs()
    data_configs = comp.datafunctions.get_data_configs()
    if 'sim' in process_types and args.config not in sim_configs:
        raise ValueError('Invalid sim config {} entered'.format(args.config))
    if 'data' in process_types and args.config not in data_configs:
        raise ValueError('Invalid data config {} entered'.format(args.config))

    if 'sim' in process_types and not args.sim:
        args.sim = comp.simfunctions.config_to_sim(args.config)

    # Define pycondor Job/Dagman directories
    error = os.path.join(comp.paths.condor_data_dir, 'error')
    output = os.path.join(comp.paths.condor_data_dir, 'output')
    log = os.path.join(comp.paths.condor_scratch_dir, 'log')
    submit = os.path.join(comp.paths.condor_scratch_dir, 'submit')

    # Create Dagman to manage processing workflow
    name = 'processing_{}'.format(args.config.replace('.', '-'))
    dag = pycondor.Dagman(name, submit=submit, verbose=1)

    # Define path to executables used in processing
    processing_dir = os.path.join(comp.paths.project_root, 'processing')
    save_hdf5_ex = os.path.join(processing_dir, 'save_hdf5.py')
    save_df_ex = os.path.join(processing_dir, 'save_dataframe.py')

    # Add Jobs to processing Dagman
    jobs = []
    if 'sim' in process_types:
        sim_gen = gen_sim_jobs(save_hdf5_ex, save_df_ex,
                               config=args.config,
                               sims=args.sim,
                               n=args.n_sim,
                               test=args.test)
        jobs.append(sim_gen)
    if 'data' in process_types:
        data_gen = gen_data_jobs(save_hdf5_ex, save_df_ex,
                                 config=args.config,
                                 n=args.n_data,
                                 test=args.test)
        jobs.append(data_gen)
    for job in chain.from_iterable(jobs):
        dag.add_job(job)

    # Build and submit processing dagman
    dag.build_submit(maxjobs=args.maxjobs, fancyname=True)

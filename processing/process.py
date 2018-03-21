#!/usr/bin/env python

import os
from itertools import chain, islice
import argparse
import pycondor

import comptools as comp
from comptools import ComputingEnvironemtError


try:
    import icecube
except ImportError:
    raise ImportError(
            'Did not detect an active icecube software environment. '
            'Make sure source the env-shell.sh script in your '
            'icecube metaproject build directory before running '
            'process.py')

if 'cvmfs' not in os.getenv('ROOTSYS'):
    raise ComputingEnvironemtError('CVMFS ROOT must be used for i3 file processing')


def gen_sim_jobs(save_hdf5_ex, save_df_ex, config, sims, n=1000, testing=False):
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
    testing : bool, optional
        Option to run in testing mode (default is False).

    Yields
    ------
    job : pycondor.Job
        Simulation Job to be included in processing Dagman.
    """
    outdir = os.path.join(comp.paths.comp_data_dir,
                          config,
                          'i3_hdf_sim')

    save_df_name = 'save_df_sim_{}'.format(config.replace('.', '-'))
    save_df_job = pycondor.Job(name=save_df_name,
                               executable=save_df_ex,
                               error=error,
                               output=output,
                               log=log,
                               submit=submit,
                               request_memory='3GB' if testing else None)

    save_df_input_files = []
    for sim in sims:
        # Create a save and merge pycondor Job for each simulation set
        save_hdf5_name = 'save_hdf5_sim_{}'.format(sim)
        save_hdf5_job = pycondor.Job(name=save_hdf5_name,
                                     executable=save_hdf5_ex,
                                     error=error,
                                     output=output,
                                     log=log,
                                     submit=submit)
        # Ensure that save_hdf5_job completes before save_df_job
        save_df_job.add_parent(save_hdf5_job)

        config = comp.simfunctions.sim_to_config(sim)
        # Split file list into smaller batches for submission
        if testing:
            n = 10
            n_batches = 2
        else:
            n_batches = None
        gcd = comp.level3_sim_GCD_file(sim)
        for files in comp.level3_sim_file_batches(sim, size=n, max_batches=n_batches):
            # Name output hdf5 file
            start_index = files[0].find('Run') + 3
            end_index = files[0].find('.i3.gz')
            start = files[0][start_index:end_index]
            end = files[-1][start_index:end_index]
            out = os.path.join(outdir,
                               'sim_{}_part{}-{}.hdf5'.format(outdir, sim, start, end))
            comp.check_output_dir(out)

            # Don't forget to insert GCD file at beginning of FileNameList
            files.insert(0, gcd)
            files_str = ' '.join(files)

            arg = '--type sim --files {} -o {}'.format(files_str, out)
            save_hdf5_job.add_arg(arg, retry=3)
            save_df_input_files.append(out)

        yield save_hdf5_job

    df_outfile = os.path.join(comp.paths.comp_data_dir,
                              config,
                              'sim_dataframe.hdf5')
    df_input_files_str = ' '.join(save_df_input_files)
    df_arg = '--input {} --output {} --type sim --config {}'.format(df_input_files_str,
                                                                    df_outfile,
                                                                    config)
    save_df_job.add_arg(df_arg)

    yield save_df_job


def gen_data_jobs(save_hdf5_ex, save_df_ex, config, n=50, testing=False):
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
    testing : bool, optional
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
                               request_memory='5GB' if testing else None,
                               verbose=1)
    # Ensure that save_df_job completes before save_df_job
    save_df_job.add_parent(save_hdf5_job)

    run_gen = comp.datafunctions.run_generator(config)
    if testing:
        run_gen = islice(run_gen, 2)
        n_batches = 2
    else:
        n_batches = None

    save_df_input_files = []
    for run in run_gen:
        # Get files associated with this run
        gcd = comp.level3_data_GCD_file(config, run)
        data_file_batches = comp.level3_data_file_batches(config, run, size=n,
                                                          max_batches=n_batches)
        for idx, files in enumerate(data_file_batches):
            # Name output hdf5 file
            out = '{}/data_{}_part_{:02d}.hdf5'.format(outdir, run, idx)
            # Don't forget to insert GCD file at beginning of FileNameList
            files.insert(0, gcd)
            files_str = ' '.join(files)
            save_arg = '--type data --files {} -o {}'.format(files_str, out)
            save_hdf5_job.add_arg(save_arg, retry=3)
            save_df_input_files.append(out)
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
    parser.add_argument('--type',
                        dest='type',
                        choices=['data', 'sim'],
                        default=None,
                        help=('Option to restrict processing to simulation '
                              'or data files only. By default both will be '
                              'included.'))
    parser.add_argument('-d', '--date',
                        dest='date',
                        help='Date to run over (mmyyyy)')
    parser.add_argument('-c', '--config',
                        dest='config',
                        default='IC86.2012',
                        help='Detector configuration')
    parser.add_argument('-s', '--sim',
                        dest='sim',
                        nargs='*',
                        type=int,
                        help='Simulation to run over')
    parser.add_argument('--n_sim',
                        dest='n_sim',
                        type=int,
                        default=1000,
                        help='Number of files to run per batch for simulation processing')
    parser.add_argument('--n_data',
                        dest='n_data',
                        type=int,
                        default=50,
                        help='Number of files to run per batch for data processing')
    parser.add_argument('--testing',
                        dest='testing',
                        action='store_true',
                        default=False,
                        help='Run processing on a small subset of files (useful for debugging)')
    parser.add_argument('--overwrite',
                        dest='overwrite',
                        action='store_true',
                        default=False,
                        help='Option for overwriting existing files.')
    parser.add_argument('--maxjobs',
                        dest='maxjobs',
                        type=int,
                        default=3000,
                        help='Maximum number of jobs to run at a given time.')
    args = parser.parse_args()

    if args.type is None:
        process_types = ('sim', 'data')
    else:
        process_types = (args.type, )

    # Want to make sure a valid config is entered for the given process_types
    sim_configs = comp.simfunctions.get_sim_configs()
    data_configs = comp.datafunctions.get_data_configs()
    for process_type, valid_configs in zip(('sim', 'data'),
                                           (sim_configs, data_configs)):
        if process_type in process_types and args.config not in valid_configs:
            raise ValueError('Invalid {} config {} entered'.format(process_type, args.config))

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
                               testing=args.testing)
        jobs.append(sim_gen)
    if 'data' in process_types:
        data_gen = gen_data_jobs(save_hdf5_ex, save_df_ex,
                                 config=args.config,
                                 n=args.n_data,
                                 testing=args.testing)
        jobs.append(data_gen)

    for job in chain.from_iterable(jobs):
        dag.add_job(job)

    # Build and submit processing dagman
    dag.build_submit(maxjobs=args.maxjobs, fancyname=True)

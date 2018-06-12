#!/usr/bin/env python

import os
from itertools import chain, islice, product
import argparse
import pycondor

import comptools as comp
from comptools import ComputingEnvironemtError

# try:
#     import icecube
# except ImportError:
#     raise ImportError(
#             'Did not detect an active icecube software environment. '
#             'Make sure source the env-shell.sh script in your '
#             'icecube metaproject build directory before running '
#             'process.py')
#
# if 'cvmfs' not in os.getenv('ROOTSYS'):
#     raise ComputingEnvironemtError('CVMFS ROOT must be used for i3 file processing')

# Define path to executables used in processing
HERE = os.path.abspath(os.path.dirname(__file__))
SAVE_HDF5_EX = os.path.join(HERE, 'save_hdf5.py')
SAVE_DF_EX = os.path.join(HERE, 'save_dataframe.py')
SAVE_CHARGE_DIST_EX = os.path.join(HERE, 'save_images.py')
SAVE_EFFICIENCIES_EX = os.path.join(HERE, 'save_efficiencies.py')
WRAPPER_EX = os.path.join(HERE, 'wrapper.sh')

# Define pycondor Job/Dagman directories
PYCONDOR_ERROR = os.path.join(comp.paths.condor_data_dir, 'error')
PYCONDOR_OUTPUT = os.path.join(comp.paths.condor_data_dir, 'output')
PYCONDOR_LOG = os.path.join(comp.paths.condor_scratch_dir, 'log')
PYCONDOR_SUBMIT = os.path.join(comp.paths.condor_scratch_dir, 'submit')


def gen_sim_jobs(config, sims, n=1000, testing=False):
    """Yields pycondor Jobs for simulation processing

    Parameters
    ----------
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
    i3_hdf_outdir = os.path.join(comp.paths.comp_data_dir,
                                 config,
                                 'sim',
                                 'testing' if testing else '',
                                 'i3_hdf')
    df_hdf_outdir = os.path.join(comp.paths.comp_data_dir,
                                 config,
                                 'sim',
                                 'testing' if testing else '',
                                 'processed_hdf')
    charge_dist_outdir = os.path.join(comp.paths.comp_data_dir,
                                      config,
                                      'sim',
                                      'testing' if testing else '',
                                      'charge_dist_hdf')

    for sim in sims:
        # Create a save and merge pycondor Job for each simulation set
        save_hdf5_name = 'save_hdf5_sim_{}'.format(sim)
        save_hdf5_job = pycondor.Job(name=save_hdf5_name,
                                     executable=SAVE_HDF5_EX,
                                     error=PYCONDOR_ERROR,
                                     output=PYCONDOR_OUTPUT,
                                     log=PYCONDOR_LOG,
                                     submit=PYCONDOR_SUBMIT)
        save_df_name = 'save_df_sim_{}'.format(config.replace('.', '-'))
        save_df_job = pycondor.Job(name=save_df_name,
                                   executable=SAVE_DF_EX,
                                   error=PYCONDOR_ERROR,
                                   output=PYCONDOR_OUTPUT,
                                   log=PYCONDOR_LOG,
                                   submit=PYCONDOR_SUBMIT,
                                   request_memory='3GB' if testing else None)
        save_charge_dist_name = 'save_charge_dist_sim_{}'.format(config.replace('.', '-'))
        save_charge_dist_job = pycondor.Job(name=save_charge_dist_name,
                                            executable=SAVE_CHARGE_DIST_EX,
                                            error=PYCONDOR_ERROR,
                                            output=PYCONDOR_OUTPUT,
                                            log=PYCONDOR_LOG,
                                            submit=PYCONDOR_SUBMIT,
                                            request_memory='3GB' if testing else None)
        # Ensure that save_hdf5_job completes before save_df_job and save_charge_dist_job start
        save_df_job.add_parent(save_hdf5_job)
        save_charge_dist_job.add_parent(save_hdf5_job)

        config = comp.simfunctions.sim_to_config(sim)
        # Split file list into smaller batches for submission
        if testing:
            n = 10
            n_batches = 5
        else:
            n_batches = None
        gcd = comp.level3_sim_GCD_file(sim)
        for files in comp.level3_sim_file_batches(sim, size=n, max_batches=n_batches):
            # Name output hdf5 file
            start_index = files[0].find('Run') + 3
            end_index = files[0].find('.i3.gz')
            start = files[0][start_index:end_index]
            end = files[-1][start_index:end_index]
            outfile_basename = 'sim_{}_part{}-{}.hdf'.format(sim, start, end)
            i3_hdf_outfile = os.path.join(i3_hdf_outdir, outfile_basename)
            comp.check_output_dir(i3_hdf_outfile)

            # Don't forget to insert GCD file at beginning of FileNameList
            files.insert(0, gcd)
            files_str = ' '.join(files)

            arg = '--type sim --sim {} --files {} -o {}'.format(sim, files_str, i3_hdf_outfile)
            save_hdf5_job.add_arg(arg, retry=3)

            df_outfile = os.path.join(df_hdf_outdir, outfile_basename)
            df_arg = '--input {} --output {} --type sim --config {}'.format(i3_hdf_outfile,
                                                                            df_outfile,
                                                                            config)
            save_df_job.add_arg(df_arg)

            df_outfile = os.path.join(charge_dist_outdir, outfile_basename)
            df_arg = '--input {} --output {} --type sim --config {}'.format(i3_hdf_outfile,
                                                                            df_outfile,
                                                                            config)
            save_charge_dist_job.add_arg(df_arg)

        yield save_hdf5_job
        yield save_df_job
        yield save_charge_dist_job

    # # Job for calculating detection efficiencies based on simulation
    # efficiencies_job = get_efficiencies_jobs(config=config)
    # efficiencies_job.add_parent(save_df_job)
    #
    # yield efficiencies_job


def gen_data_jobs(config, n=50, testing=False):
    """Yields pycondor Jobs for data processing

    Parameters
    ----------
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
    i3_hdf_outdir = os.path.join(comp.paths.comp_data_dir,
                                 config,
                                 'data',
                                 'testing' if testing else '',
                                 'i3_hdf')
    df_hdf_outdir = os.path.join(comp.paths.comp_data_dir,
                                 config,
                                 'data',
                                 'testing' if testing else '',
                                 'processed_hdf')

    # Create a save and merge CondorJobs
    save_hdf5_name = 'save_hdf5_data_{}'.format(config.replace('.', '-'))
    save_hdf5_job = pycondor.Job(name=save_hdf5_name,
                                 executable=WRAPPER_EX,
                                 # executable=SAVE_HDF5_EX,
                                 error=PYCONDOR_ERROR,
                                 output=PYCONDOR_OUTPUT,
                                 log=PYCONDOR_LOG,
                                 submit=PYCONDOR_SUBMIT,
                                 getenv=False,
                                 )

    save_df_name = 'save_df_data_{}'.format(config.replace('.', '-'))
    save_df_job = pycondor.Job(name=save_df_name,
                               executable=WRAPPER_EX,
                               # executable=SAVE_DF_EX,
                               error=PYCONDOR_ERROR,
                               output=PYCONDOR_OUTPUT,
                               log=PYCONDOR_LOG,
                               submit=PYCONDOR_SUBMIT,
                               getenv=False,
                               request_memory='5GB' if testing else None)
    # Ensure that save_df_job completes before save_df_job
    save_df_job.add_parent(save_hdf5_job)

    run_gen = comp.datafunctions.run_generator(config)
    if testing:
        run_gen = islice(run_gen, 2)
        n = 2
        n_batches = 2
    else:
        n_batches = None

    # save_df_input_files = []
    for run in run_gen:
        print(run)
        # Get files associated with this run
        gcd = comp.level3_data_GCD_file(config, run)
        data_file_batches = comp.level3_data_file_batches(config, run, size=n,
                                                          max_batches=n_batches)
        for idx, files in enumerate(data_file_batches):
            # Name output hdf5 file
            # out = '{}/data_{}_part_{:02d}.hdf'.format(outdir, run, idx)
            outfile_basename = 'data_{}_part_{:02d}.hdf'.format(run, idx)
            i3_hdf_outfile = os.path.join(i3_hdf_outdir, outfile_basename)
            # Don't forget to insert GCD file at beginning of FileNameList
            files.insert(0, gcd)
            files_str = ' '.join(files)
            save_arg = '{} --type data --files {} -o {}'.format(SAVE_HDF5_EX,
                                                                files_str,
                                                                i3_hdf_outfile)
            save_hdf5_job.add_arg(save_arg, retry=3)

            df_outfile = os.path.join(df_hdf_outdir, outfile_basename)
            df_arg = '{} --input {} --output {} --type data --config {}'.format(SAVE_DF_EX,
                                                                                i3_hdf_outfile,
                                                                                df_outfile,
                                                                                config)
            save_df_job.add_arg(df_arg)

            # save_df_input_files.append(i3_hdf_outfile)
    yield save_hdf5_job

    # df_outfile = os.path.join(comp.paths.comp_data_dir,
    #                           config,
    #                           'data_dataframe.hdf')
    # df_input_files_str = ' '.join(save_df_input_files)
    # df_arg = '{} --input {} --output {} --type data --config {}'.format(SAVE_DF_EX,
    #                                                                     df_input_files_str,
    #                                                                     df_outfile,
    #                                                                     config)
    # save_df_job.add_arg(df_arg)

    yield save_df_job


def get_efficiencies_jobs(config):
    """Returns pycondor Job saving simulation detection efficiencies

    Parameters
    ----------
    config : str
        Detector configuration.

    Returns
    ------
    job : pycondor.Job
        Data Job to be included in processing Dagman.
    """
    efficiencies_job = pycondor.Job(name='save_efficiencies',
                                    executable=SAVE_EFFICIENCIES_EX,
                                    error=PYCONDOR_ERROR,
                                    output=PYCONDOR_OUTPUT,
                                    log=PYCONDOR_LOG,
                                    submit=PYCONDOR_SUBMIT)
    for num_groups, sigmoid in product([2, 3, 4], ['flat', 'slant']):
        arg = '--config {} --num_groups {} --sigmoid {}'.format(config,
                                                                num_groups,
                                                                sigmoid)
        efficiencies_job.add_arg(arg)

    return efficiencies_job


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

    # Create Dagman to manage processing workflow
    name = 'processing_{}'.format(args.config.replace('.', '-'))
    dag = pycondor.Dagman(name, submit=PYCONDOR_SUBMIT, verbose=1)
    # Add Jobs to processing Dagman
    jobs = []
    if 'sim' in process_types:
        sim_gen = gen_sim_jobs(config=args.config,
                               sims=args.sim,
                               n=args.n_sim,
                               testing=args.testing)
        jobs.append(sim_gen)
    if 'data' in process_types:
        data_gen = gen_data_jobs(config=args.config,
                                 n=args.n_data,
                                 testing=args.testing)
        jobs.append(data_gen)

    for job in chain.from_iterable(jobs):
        dag.add_job(job)

    # Build and submit processing dagman
    submit_options = '-maxjobs {}'.format(args.maxjobs)
    dag.build(fancyname=True)
    # dag.build_submit(fancyname=True,
    #                  submit_options=submit_options)

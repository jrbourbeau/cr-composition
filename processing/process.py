#!/usr/bin/env python

import os
from itertools import islice
import argparse
import pycondor

import comptools as comp


# Define file paths to executables used in processing
HERE = os.path.abspath(os.path.dirname(__file__))
PROCESS_I3_EX = os.path.join(HERE, 'process_i3.py')
SAVE_DF_EX = os.path.join(HERE, 'save_dataframe.py')
WRAPPER_EX = os.path.join(comp.paths.project_root, 'wrapper.sh')

# Define pycondor Job/Dagman file directories
PYCONDOR_ERROR = os.path.join(comp.paths.condor_data_dir, 'error')
PYCONDOR_OUTPUT = os.path.join(comp.paths.condor_data_dir, 'output')
PYCONDOR_LOG = os.path.join(comp.paths.condor_scratch_dir, 'log')
PYCONDOR_SUBMIT = os.path.join(comp.paths.condor_scratch_dir, 'submit')


processing_doc = """ Builds pycondor Dagman for {type} processing

Parameters
----------
config : str, optional
    Detector configuration (default is 'IC86.2012')
batch_size : int, optional
    Number of files to process for each job (default is 1000).
test : bool, optional
    Option to run in testing mode (default is False).

Yields
------
dag : pycondor.Dagman
    Processing Dagman to be submitted to HTCondor cluster.
"""


def simulation_processing_dag(config='IC86.2012', batch_size=1000, test=False,
                              snow_lambda=None):
    base_dir = os.path.join(comp.paths.comp_data_dir,
                            config,
                            'sim',
                            'test' if test else '')
    if snow_lambda is None:
        i3_hdf_outdir = os.path.join(base_dir, 'i3_hdf', 'nominal')
        df_hdf_outdir = os.path.join(base_dir, 'processed_hdf', 'nominal')
    else:
        # snow_lambda_str = str(snow_lambda).replace('.', '-')
        i3_hdf_outdir = os.path.join(base_dir, 'i3_hdf',
                                     'snow_lambda_{}'.format(snow_lambda))
        df_hdf_outdir = os.path.join(base_dir, 'processed_hdf',
                                     'snow_lambda_{}'.format(snow_lambda))

    # Create data processing Jobs / Dagman
    dag_name = 'sim_processing_{}'.format(args.config.replace('.', '-'))
    dag = pycondor.Dagman(dag_name,
                          submit=PYCONDOR_SUBMIT,
                          verbose=1)

    sims = comp.simfunctions.config_to_sim(config)
    for sim in sims:
        process_i3_job = pycondor.Job(name='process_i3_{}'.format(sim),
                                      executable=WRAPPER_EX,
                                      error=PYCONDOR_ERROR,
                                      output=PYCONDOR_OUTPUT,
                                      log=PYCONDOR_LOG,
                                      submit=PYCONDOR_SUBMIT,
                                      getenv=False,
                                      dag=dag)
        save_df_job = pycondor.Job(name='save_dataframe_{}'.format(sim),
                                   executable=WRAPPER_EX,
                                   error=PYCONDOR_ERROR,
                                   output=PYCONDOR_OUTPUT,
                                   log=PYCONDOR_LOG,
                                   submit=PYCONDOR_SUBMIT,
                                   getenv=False,
                                   dag=dag)

        # Ensure that save_df_job doesn't begin until process_i3_job completes
        save_df_job.add_parent(process_i3_job)

        # Split file list into smaller batches for submission
        if test:
            batch_size = 2
            max_batches = 2
        else:
            max_batches = None

        sim_file_batches = comp.level3_sim_file_batches(sim,
                                                        size=batch_size,
                                                        max_batches=max_batches)
        gcd = comp.level3_sim_GCD_file(sim)
        for idx, files in enumerate(sim_file_batches):
            # Set up process_i3_job arguments
            outfile_basename = 'sim_{}_part_{:02d}.hdf'.format(sim, idx)
            process_i3_outfile = os.path.join(i3_hdf_outdir, outfile_basename)
            # Don't forget to insert GCD file at beginning of FileNameList
            files.insert(0, gcd)
            files_str = ' '.join(files)
            process_i3_arg_template = '{ex} --type sim --files {i3_files} --outfile {outfile}'
            process_i3_arg = process_i3_arg_template.format(ex=PROCESS_I3_EX,
                                                            i3_files=files_str,
                                                            outfile=process_i3_outfile)
            if snow_lambda is not None: 
                process_i3_arg += ' --snow_lambda {}'.format(snow_lambda)
            process_i3_job.add_arg(process_i3_arg, retry=3)
            # Set up save_df_job arguments
            save_df_outfile = os.path.join(df_hdf_outdir, outfile_basename)
            save_df_arg_template = '{ex} --input {input} --output {output} --type sim --sim {sim} --config {config}'
            save_df_arg = save_df_arg_template.format(ex=SAVE_DF_EX,
                                                      input=process_i3_outfile,
                                                      output=save_df_outfile,
                                                      sim=sim,
                                                      config=config)
            save_df_job.add_arg(save_df_arg)

    return dag


simulation_processing_dag.__doc__ = processing_doc.format(type='simulation')


def data_processing_dag(config='IC86.2012', batch_size=1000, test=False):
    base_dir = os.path.join(comp.paths.comp_data_dir,
                            config,
                            'data',
                            'test' if test else '')
    i3_hdf_outdir = os.path.join(base_dir, 'i3_hdf')
    df_hdf_outdir = os.path.join(base_dir, 'processed_hdf')

    # Create data processing Jobs / Dagman
    dag_name = 'data_processing_{}'.format(args.config.replace('.', '-'))
    dag = pycondor.Dagman(dag_name,
                          submit=PYCONDOR_SUBMIT,
                          verbose=1)

    process_i3_job = pycondor.Job(name='process_i3',
                                  executable=WRAPPER_EX,
                                  error=PYCONDOR_ERROR,
                                  output=PYCONDOR_OUTPUT,
                                  log=PYCONDOR_LOG,
                                  submit=PYCONDOR_SUBMIT,
                                  getenv=False,
                                  dag=dag)

    save_df_job = pycondor.Job(name='save_dataframe',
                               executable=WRAPPER_EX,
                               error=PYCONDOR_ERROR,
                               output=PYCONDOR_OUTPUT,
                               log=PYCONDOR_LOG,
                               submit=PYCONDOR_SUBMIT,
                               getenv=False,
                               dag=dag)

    # Ensure that save_df_job doesn't begin until process_i3_job completes
    save_df_job.add_parent(process_i3_job)

    run_gen = comp.datafunctions.run_generator(config)
    if test:
        run_gen = islice(run_gen, 2)
        batch_size = 2
        max_batches = 2
    else:
        max_batches = None

    for run in run_gen:
        # Get files associated with this run
        gcd = comp.level3_data_GCD_file(config, run)
        data_file_batches = comp.level3_data_file_batches(config=config,
                                                          run=run,
                                                          size=batch_size,
                                                          max_batches=max_batches)
        # Process run files in batches
        for idx, files in enumerate(data_file_batches):
            # Set up process_i3_job arguments
            outfile_basename = 'run_{}_part_{:02d}.hdf'.format(run, idx)
            process_i3_outfile = os.path.join(i3_hdf_outdir, outfile_basename)
            # Don't forget to insert GCD file at beginning of FileNameList
            files.insert(0, gcd)
            files_str = ' '.join(files)
            process_i3_arg_template = '{ex} --type data --files {i3_files} --outfile {outfile}'
            process_i3_arg = process_i3_arg_template.format(ex=PROCESS_I3_EX,
                                                            i3_files=files_str,
                                                            outfile=process_i3_outfile)
            process_i3_job.add_arg(process_i3_arg, retry=3)
            # Set up save_df_job arguments
            save_df_outfile = os.path.join(df_hdf_outdir, outfile_basename)
            save_df_arg_template = '{ex} --input {input} --output {output} --type data --config {config}'
            save_df_arg = save_df_arg_template.format(ex=SAVE_DF_EX,
                                                      input=process_i3_outfile,
                                                      output=save_df_outfile,
                                                      config=config)
            save_df_job.add_arg(save_df_arg)

    return dag


data_processing_dag.__doc__ = processing_doc.format(type='data')


if __name__ == '__main__':

    description = 'Processes simulation and data .i3 files'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config',
                        dest='config',
                        default='IC86.2012',
                        help='Detector configuration')
    parser.add_argument('--sim',
                        dest='sim',
                        action='store_true',
                        default=False,
                        help='Option to submit simulation processing cluster jobs')
    parser.add_argument('--data',
                        dest='data',
                        action='store_true',
                        default=False,
                        help='Option to submit data processing cluster jobs')
    parser.add_argument('--batch_size_sim',
                        dest='batch_size_sim',
                        type=int,
                        default=1000,
                        help='Number of files to run per batch for simulation processing')
    parser.add_argument('--batch_size_data',
                        dest='batch_size_data',
                        type=int,
                        default=50,
                        help='Number of files to run per batch for data processing')
    parser.add_argument('--snow_lambda',
                        dest='snow_lambda',
                        type=float,
                        help='Snow lambda to use with Laputop reconstruction')
    parser.add_argument('--test',
                        dest='test',
                        action='store_true',
                        default=False,
                        help='Run processing on a small subset of files (useful for debugging)')
    args = parser.parse_args()

    dags = []

    if args.sim:
        sim_dag = simulation_processing_dag(config=args.config,
                                            batch_size=args.batch_size_sim,
                                            test=args.test,
                                            snow_lambda=args.snow_lambda)
        dags.append(sim_dag)

    if args.data:
        data_dag = data_processing_dag(config=args.config,
                                       batch_size=args.batch_size_data,
                                       test=args.test)
        dags.append(data_dag)

    # TODO: Consider adding more processing dags. E.g.
    # if args.efficiencies:
    # if args.models:
    # if args.livetimes:

    # Create Dagman to manage processing workflow
    if not dags:
        raise ValueError('No processing tasks were specified')
    elif len(dags) == 1:
        dag = dags[0]
    else:
        dag_name = 'processing_{}'.format(args.config.replace('.', '-'))
        dag = pycondor.Dagman(dag_name,
                              submit=PYCONDOR_SUBMIT,
                              verbose=1)
        for subdag in dags:
            dag.add_subdag(subdag)

    # Build and submit processing dagman
    dag.build_submit(fancyname=True, submit_options='-maxjobs 3000')

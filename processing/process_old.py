#!/usr/bin/env python

import glob
import os
import re
import argparse
import time
import getpass
from collections import defaultdict
import numpy as np

import composition as comp
import dagmanager as dm


def add_sim_jobs(dag_manager, save_hdf5_ex, merge_hdf5_ex, save_df_ex, **args):

    for sim in args['sim']:
        # Create a save and merge CondorJobs for each simulation
        save_hdf5_name = 'save_hdf5_sim_{}'.format(sim)
        save_hdf5_job = dm.CondorJob(name=save_hdf5_name,
                                    condorexecutable=save_hdf5_ex)
        merge_hdf5_name = 'merge_hdf5_sim_{}'.format(sim)
        merge_hdf5_job = dm.CondorJob(name=merge_hdf5_name,
                                     condorexecutable=merge_hdf5_ex)
        # Ensure that save_hdf5_job completes before merge_hdf5_job
        merge_hdf5_job.add_parent(save_hdf5_job)

        save_df_name = 'save_df_hdf5_sim_{}'.format(sim)
        save_df_job = dm.CondorJob(name=save_df_name,
                                    condorexecutable=save_df_ex)
        # Ensure that merge_hdf5_job completes before save_df_job
        save_df_job.add_parent(merge_hdf5_job)

        # Get config and simulation files
        config = comp.simfunctions.sim2cfg(sim)
        gcd, files = comp.simfunctions.get_level3_sim_files(sim)

        # Set up output directory (also, make sure directory exists)
        paths = comp.Paths()
        comp_data_dir = paths.comp_data_dir
        outdir = '{}/{}_sim/hdf5_files'.format(comp_data_dir, config)
        comp.checkdir(outdir + '/')
        if args['test']:
            args['n'] = 2

        # Split file list into smaller batches for submission
        batches = [files[i:i + args['n']]
                   for i in range(0, len(files), args['n'])]
        if args['test']:
            batches = batches[:2]

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

            arg = '--files {} -s {} -o {}'.format(files_str, sim, out)
            save_hdf5_job.add_arg(arg)

            # Append out to merger_input
            merger_input += out + ' '

        # Add job for this sim to the dagmanager
        dag_manager.add_job(save_hdf5_job)

        # Finish constructing merge sim job
        merger_output = '{}/sim_{}.hdf5'.format(outdir, sim)
        merge_arg = '--files {} -o {}'.format(merger_input, merger_output)
        if args['remove']:
            merge_arg += ' --remove'
        if args['overwrite']:
            merge_arg += ' --overwrite'
        merge_hdf5_job.add_arg(merge_arg)
        # Add merge job for this sim to the dagmanager
        dag_manager.add_job(merge_hdf5_job)

        df_outfile = '{}/{}_sim/dataframe_files/dataframe_{}.hdf5'.format(comp_data_dir, config, sim)
        comp.checkdir(df_outfile)
        df_arg = '--input {} --output {} --type sim'.format(merger_output, df_outfile)
        if args['overwrite']:
            df_arg += ' --overwrite'
        save_df_job.add_arg(df_arg)
        # Add save save_df to dagmanager
        dag_manager.add_job(save_df_job)

    return dag_manager


def add_data_jobs(dag_manager, save_hdf5_ex, merge_hdf5_ex, save_df_ex, **args):


    config = args['config']
    if args['date'] is not None:
        month = args['date'][:2]
        year = args['date'][2:]
        data_files = comp.datafunctions.get_level3_data_files(month=month,
                                                year=year, config=config)
    else:
        data_files = comp.datafunctions.get_level3_data_files(config=config)
    # Set up output directory (also, make sure directory exists)
    comp_data_dir = '/data/user/jbourbeau/composition'
    outdir = '{}/{}_data/hdf5_files'.format(comp_data_dir, config)
    comp.checkdir(outdir + '/')

    # Look at each run in files seperately
    run_numbers = np.unique(
        [re.findall(r"\D(\d{8})\D", f) for f in data_files])
    if args['test']:
        run_numbers = run_numbers[:2]
    for run in run_numbers:
        # Create a save and merge CondorJobs
        save_hdf5_name = 'save_hdf5_data_{}_run{}'.format(args['date'], run)
        save_hdf5_job = dm.CondorJob(name=save_hdf5_name,
                                    condorexecutable=save_hdf5_ex)
        merge_hdf5_name = 'merge_hdf5_data_{}_run{}'.format(args['date'], run)
        merge_hdf5_job = dm.CondorJob(name=merge_hdf5_name,
                                     condorexecutable=merge_hdf5_ex)
        # Ensure that save_hdf5_job completes before merge_hdf5_job
        merge_hdf5_job.add_parent(save_hdf5_job)

        save_df_name = 'save_df_data_{}_run{}'.format(args['date'], run)
        save_df_job = dm.CondorJob(name=save_df_name,
                                    condorexecutable=save_df_ex)
        # Ensure that merge_hdf5_job completes before save_df_job
        save_df_job.add_parent(merge_hdf5_job)

        # Get GCD file for this run
        gcd = glob.glob(
            '/data/ana/CosmicRay/IceTop_level3/exp/{}/GCD/Level3_{}_data_Run{}_????_GCD.i3.gz'.format(config, config, run))
        if len(gcd) != 1:
            raise('Found a number of GCD files for run {} not equal to one!'.format(run))
        gcd = gcd[0]
        # Split files for this run into smaller batches for submission
        run_files = [f for f in data_files if run in f]
        batches = [run_files[i:i + args['n']]
                   for i in range(0, len(run_files), args['n'])]
        merged_output = '{}/data_{}.hdf5'.format(outdir, run)
        merged_input = ''
        for idx, files in enumerate(batches):
            # Name output hdf5 file
            out = '{}/data_{}_part_{:02d}.hdf5'.format(outdir, run, idx)

            merged_input += out + ' '
            # Don't forget to insert GCD file at beginning of FileNameList
            files.insert(0, gcd)
            files_str = ' '.join(files)

            save_arg = '--files {} -o {}'.format(files_str, out)
            save_hdf5_job.add_arg(save_arg)

        # Add save job to the dagmanager
        dag_manager.add_job(save_hdf5_job)

        merge_arg = '--files {} -o {}'.format(merged_input, merged_output)
        if args['remove']:
            merge_arg += ' --remove'
        if args['overwrite']:
            merge_arg += ' --overwrite'
        # merge_argdict[run] = merge_arg
        merge_hdf5_job.add_arg(merge_arg)
        # Add merge job to the dagmanager
        dag_manager.add_job(merge_hdf5_job)

        # Add save save_df to dagmanager
        df_outfile = '{}/{}_data/dataframe_files/dataframe_run_{}.hdf5'.format(comp_data_dir, config, run)
        comp.checkdir(df_outfile)
        df_arg = '--input {} --output {} --type data'.format(merged_output, df_outfile)
        if args['overwrite']:
            df_arg += ' --overwrite'
        save_df_job.add_arg(df_arg)
        dag_manager.add_job(save_df_job)

    return dag_manager


if __name__ == "__main__":

    # Setup global path names
    mypaths = comp.Paths()
    comp.checkdir(mypaths.comp_data_dir)

    simoutput = comp.simfunctions.getSimOutput()
    default_sim_list = ['7006', '7579', '7241', '7263', '7791',
                        '7242', '7262', '7851', '7007', '7784']

    p = argparse.ArgumentParser(
        description='Runs save_sim.py on cluster en masse',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=simoutput)
    p.add_argument('--type', dest='type',
                   choices=['data', 'sim'],
                   default='sim',
                   help='Option to process simulation or data')
    p.add_argument('-d', '--date', dest='date',
                   help='Date to run over (mmyyyy)')
    p.add_argument('-c', '--config', dest='config',
                   default='IC79',
                   help='Detector configuration')
    p.add_argument('-s', '--sim', dest='sim', nargs='*',
                   choices=default_sim_list,
                   default=default_sim_list,
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
            args.n = 200
        else:
            args.n = 10

    # Create DagManager
    name = 'processing_{}'.format(args.type)
    dag_manager = dm.DagManager(name,
                                condor_data_dir=mypaths.condor_data_dir,
                                condor_scratch_dir=mypaths.condor_scratch_dir,
                                verbose=2)

    # Create save and merge CondorJobs
    name = 'save_hdf5_{}'.format(args.type)
    cmd = '{}/save_hdf5.py'.format(os.getcwd())
    save_hdf5_ex = dm.CondorExecutable(name, cmd)

    name = 'merge_hdf5_{}'.format(args.type)
    cmd = '{}/merge_hdf5.py'.format(os.getcwd())
    merge_hdf5_ex = dm.CondorExecutable(name, cmd)

    name = 'save_df_{}'.format(args.type)
    cmd = '{}/save_dataframe.py'.format(os.getcwd())
    save_df_ex = dm.CondorExecutable(name, cmd)

    if args.type == 'sim':
        dag_manager = add_sim_jobs(dag_manager, save_hdf5_ex, merge_hdf5_ex,
                                    save_df_ex, **vars(args))
    else:
        dag_manager = add_data_jobs(dag_manager, save_hdf5_ex, merge_hdf5_ex,
                                    save_df_ex, **vars(args))

    # Add dataframe merger job
    name = 'merge_df_{}'.format(args.type)
    cmd = '{}/merge_dataframe.py'.format(os.getcwd())
    merge_df_ex = dm.CondorExecutable(name, cmd)

    merge_df_name = 'merge_df_{}'.format(args.type)
    merge_df_job = dm.CondorJob(name=merge_df_name,
                                condorexecutable=merge_df_ex)
    merge_df_arg = '--type {} --config {}'.format(args.type, args.config)
    if args.overwrite:
        merge_df_arg += ' --overwrite'
    merge_df_job.add_arg(merge_df_arg)
    # Add merge_df_job to dag_manager
    dag_manager.add_job(merge_df_job)

    # Ensure that all dataframes are made before merging
    for job in dag_manager:
        if 'save_df' in job.name:
            merge_df_job.add_parent(job)
        else:
            continue


    dag_manager.build_submit(args.maxjobs)

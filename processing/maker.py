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


def add_sim_jobs(dag_manager, save_executable, merge_executable, **args):

    for sim in args['sim']:
        # Create a save and merge CondorJobs for each simulation
        save_name = 'save_sim_{}'.format(sim)
        save_sim_job = dm.CondorJob(name=save_name,
                                    condorexecutable=save_executable)
        merge_name = 'merge_sim_{}'.format(sim)
        merge_sim_job = dm.CondorJob(name=merge_name,
                                     condorexecutable=merge_executable)
        # Ensure that save_sim_job completes before merge_sim_job
        merge_sim_job.add_parent(save_sim_job)

        # Get config and simulation files
        config = comp.simfunctions.sim2cfg(sim)
        gcd, files = comp.simfunctions.get_level3_sim_files(sim)

        # Set up output directory (also, make sure directory exists)
        comp_data_dir = '/data/user/jbourbeau/composition'
        outdir = '{}/{}_sim/files'.format(comp_data_dir, config)
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
            save_sim_job.add_arg(arg)

            # Append out to merger_input
            merger_input += out + ' '

        # Add job for this sim to the dagmanager
        dag_manager.add_job(save_sim_job)

        # Finish constructing merge sim job
        merger_output = '{}/sim_{}.hdf5'.format(outdir, sim)
        merge_arg = '--files {} -o {}'.format(merger_input, merger_output)
        if args['remove']:
            merge_arg += ' --remove'
        if args['overwrite']:
            merge_arg += ' --overwrite'
        merge_sim_job.add_arg(merge_arg)
        # Add merge job for this sim to the dagmanager
        dag_manager.add_job(merge_sim_job)

    return dag_manager


def add_data_jobs(dag_manager, save_executable, merge_executable, **args):


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
    outdir = '{}/{}_data/files'.format(comp_data_dir, config)
    comp.checkdir(outdir + '/')
    if args['test']:
        args['n'] = 2

    # Look at each run in files seperately
    run_numbers = np.unique(
        [re.findall(r"\D(\d{8})\D", f) for f in data_files])
    for run in run_numbers:
        # Create a save and merge CondorJobs
        save_name = 'save_data_{}_run{}'.format(args['date'], run)
        save_data_job = dm.CondorJob(name=save_name,
                                    condorexecutable=save_executable)
        merge_name = 'merge_data_{}_run{}'.format(args['date'], run)
        merge_data_job = dm.CondorJob(name=merge_name,
                                     condorexecutable=merge_executable)
        # Ensure that save_data_job completes before merge_data_job
        merge_data_job.add_parent(save_data_job)

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
            save_data_job.add_arg(save_arg)
            # save_argdict[run].append(save_arg)

        # Add save job to the dagmanager
        dag_manager.add_job(save_data_job)

        merge_arg = '--files {} -o {}'.format(merged_input, merged_output)
        if args['remove']:
            merge_arg += ' --remove'
        if args['overwrite']:
            merge_arg += ' --overwrite'
        # merge_argdict[run] = merge_arg
        merge_data_job.add_arg(merge_arg)
        # Add merge job to the dagmanager
        dag_manager.add_job(merge_data_job)

    return dag_manager
    # return save_argdict, merge_argdict


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
    name = 'save_{}'.format(args.type)
    cmd = '{}/save_hdf5.py'.format(os.getcwd())
    save_executable = dm.CondorExecutable(name, cmd)

    name = 'merge_{}'.format(args.type)
    cmd = '{}/merge_hdf5.py'.format(os.getcwd())
    merge_executable = dm.CondorExecutable(name, cmd)

    if args.type == 'sim':
        dag_manager = add_sim_jobs(dag_manager, save_executable, merge_executable,
                                  **vars(args))
    else:
        dag_manager = add_data_jobs(dag_manager, save_executable, merge_executable,
                                  **vars(args))

    dag_manager.build_submit(args.maxjobs)

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


def get_sim_argdict(comp_data_dir, **args):

    argdict = defaultdict(list)
    for sim in args['sim']:
        # Get config and simulation files
        config = comp.simfunctions.sim2cfg(sim)
        gcd, files = comp.simfunctions.get_level3_data_files(sim)

        # Set up output directory (also, make sure directory exists)
        outdir = '{}/{}_sim/files'.format(comp_data_dir, config)
        comp.checkdir(outdir + '/')
        if args['test']:
            args['n'] = 2

        # Split file list into smaller batches for submission
        batches = [files[i:i + args['n']]
                   for i in range(0, len(files), args['n'])]
        if args['test']:
            batches = batches[:2]

        for files in batches:

            # Name output hdf5 file
            start_index = files[0].find('Run') + 3
            end_index = files[0].find('.i3.gz')
            start = files[0][start_index:end_index]
            end = files[-1][start_index:end_index]
            out = '{}/sim_{}_part{}-{}.hdf5'.format(outdir, sim, start, end)

            # Don't forget to insert GCD file at beginning of FileNameList
            files.insert(0, gcd)
            files_str = ' '.join(batch)

            arg = '--files {} -s {} -o {}'.format(files_str, sim, out)

            argdict[sim].append(arg)

    return argdict


def get_data_argdict(comp_data_dir, **args):

    save_argdict = defaultdict(list)
    merge_argdict = defaultdict(str)
    month = args['date'][:2]
    year = args['date'][2:]
    # Get config and simulation files
    config = args['config']
    data_files = comp.datafunctions.get_level3_data_files(month, year, config)

    # Set up output directory (also, make sure directory exists)
    outdir = '{}/{}_data/{:02d}_{}'.format(comp_data_dir, config, month, year)
    comp.checkdir(outdir + '/')
    if args['test']:
        args['n'] = 2

    # Look at each run in files seperately
    run_numbers = np.unique(
        [re.findall(r"\D(\d{8})\D", f) for f in data_files])
    for run in run_numbers:
        # Get GCD file for this run
        gcd = glob.glob(
            '/data/ana/CosmicRay/IceTop_level3/data/{}/GCD/Level3_IC79_data_Run{}_????_GCD.i3.gz'.format(config, run))
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
            save_argdict[run].append(save_arg)

        merge_arg = '--files {} -o {}'.format(merged_input, merged_output)
        if args['remove']:
            merge_arg += ' --remove'
        if args['overwrite']:
            merge_arg += ' --overwrite'
        merge_argdict[run] = merge_arg

    return save_argdict, merge_argdict


def make_submit_script(executable, jobID, script_path, condor_dir):

    comp.checkdir(script_path)
    lines = ["universe = vanilla\n",
             "getenv = true\n",
             "executable = {}\n".format(executable),
             "arguments = $(ARGS)\n",
             "log = {}/logs/{}.log\n".format(condor_dir, jobID),
             "output = /data/user/jbourbeau/composition/condor/outs/{}.out\n".format(
                 jobID),
             "error = /data/user/jbourbeau/composition/condor/errors/{}.error\n".format(
                 jobID),
             "notification = Never\n",
             "queue \n"]

    condor_script = script_path
    with open(condor_script, 'w') as f:
        f.writelines(lines)

    return


def getjobID(jobID, condor_dir):
    jobID += time.strftime('_%Y%m%d')
    othersubmits = glob.glob(
        '{}/submit_scripts/{}_??.submit'.format(condor_dir, jobID))
    jobID += '_{:02d}'.format(len(othersubmits) + 1)
    return jobID

if __name__ == "__main__":

    # Setup global path names
    mypaths = comp.Paths()
    comp.checkdir(mypaths.comp_data_dir)
    # Set up condor directory
    condor_dir = '/scratch/{}/condor_composition'.format(getpass.getuser())
    for directory in ['errors', 'logs', 'outs', 'submit_scripts']:
        comp.checkdir(condor_dir + '/' + directory + '/')
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

    cwd = os.getcwd()
    jobID = 'save_{}'.format(args.type)
    jobID = getjobID(jobID, condor_dir)
    cmd = '{}/save_hdf5.py'.format(cwd)
    if args.type == 'sim':
        sim_argdict = get_sim_argdict(mypaths.comp_data_dir, **vars(args))
    else:
        save_argdict, merge_argdict = get_data_argdict(mypaths.comp_data_dir, **vars(args))
    condor_script = '{}/submit_scripts/{}.submit'.format(condor_dir, jobID)
    make_submit_script(cmd, jobID, condor_script, condor_dir)

    merge_jobID = 'merge_{}'.format(args.type)
    merge_jobID = getjobID(merge_jobID, condor_dir)
    merge_cmd = '{}/merge_hdf5.py'.format(cwd)
    # merge_argdict = get_merge_argdict(**vars(args))
    merge_condor_script = '{}/submit_scripts/{}.submit'.format(
        condor_dir, merge_jobID)
    make_submit_script(merge_cmd, merge_jobID, merge_condor_script, condor_dir)

    # Set up dag file
    if args.type == 'sim':
        jobID = 'save_{}_merge'.format(args.type)
    else:
        jobID = 'save_{}_merge_{}'.format(args.type, args.date)
    jobID = getjobID(jobID, condor_dir)
    dag_file = '{}/submit_scripts/{}.submit'.format(condor_dir, jobID)
    comp.checkdir(dag_file)
    with open(dag_file, 'w') as dag:
        for key in save_argdict.keys():
            print('{}...'.format(key))
            parent_string = 'Parent '
            if len(save_argdict[key]) < 1:
                continue
            for i, arg in enumerate(save_argdict[key]):
                dag.write('JOB {}_{}_p{} '.format(args.type, key, i) +
                          condor_script + '\n')
                dag.write('VARS {}_{}_p{} '.format(args.type, key, i) +
                          'ARGS="' + arg + '"\n')
                parent_string += '{}_{}_p{} '.format(args.type, key, i)
            dag.write('JOB merge_{} '.format(
                key) + merge_condor_script + '\n')
            dag.write('VARS merge_{} '.format(key) +
                      'ARGS="' + str(merge_argdict[key]) + '"\n')
            child_string = 'Child merge_{}'.format(key)
            dag.write(parent_string + child_string + '\n')

    # Submit jobs
    os.system('condor_submit_dag -maxjobs {} {}'.format(args.maxjobs, dag_file))

#!/usr/bin/env python

import glob
import os
import argparse
import time
import getpass

import composition as comp

def get_argdict(comp_data_dir, **args):

    argdict = dict.fromkeys(args['sim'])
    for sim in args['sim']:

        arglist = []

        # Get config and simulation files
        config = 'IC79'
        files = glob.glob('/data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/{}/*/Level2a_*.i3.bz2'.format(sim))
        files = sorted(files)

        # Default parameters
        outdir = '{}/{}_sim/{}'.format(comp_data_dir, config, sim)
        comp.checkdir(outdir + '/')
        if args['test']:
            args['n'] = 2

        if args['test']:
            files = files[:2]

        for fi, f in enumerate(files):

            # Name output hdf5 file
            run = os.path.basename(f).split('.')[-3]
            out = '{}/Level3_IC79_{}_Run{:06d}.i3.gz'.format(outdir, sim, int(run))

            # If not overwriting, and outfile already exists, then continue onto next file
            if not args['overwrite']:
                if os.path.exists(out):
                    continue

            arg = '{} --isMC --do-inice --dataset {} --det IC79 --waveform -o {}'.format(f, sim, out)

            arglist.append(arg)

        argdict[sim] = arglist

    return argdict


def make_submit_script(executable, jobID, script_path, condor_dir):

    comp.checkdir(script_path)
    lines = ["universe = vanilla\n",
             "getenv = true\n",
             "executable = {}\n".format(executable),
             "arguments = $(ARGS)\n",
             "log = {}/logs/{}.log\n".format(condor_dir, jobID),
             "output = /data/user/jbourbeau/composition/condor/outs/{}.out\n".format(jobID),
             "error = /data/user/jbourbeau/composition/condor/errors/{}.error\n".format(jobID),
             "notification = Never\n",
            #  "request_memory = 5000\n",
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
        description='Runs level3_process.py on cluster en masse',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=simoutput)
    p.add_argument('-s', '--sim', dest='sim', nargs='*',
                   choices=default_sim_list,
                   default=default_sim_list,
                   help='Simulation to run over')
    p.add_argument('-n', '--n', dest='n', type=int,
                   default=1,
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
    args = p.parse_args()

    cwd = os.getcwd()
    jobID = 'level3_process'
    for sim in args.sim:
        jobID += '_{}'.format(sim)
    jobID = getjobID(jobID, condor_dir)
    cmd = '{}/level3_IceTop_InIce.py'.format(cwd)
    argdict = get_argdict(mypaths.comp_data_dir, **vars(args))
    condor_script = '{}/submit_scripts/{}.submit'.format(condor_dir, jobID)
    make_submit_script(cmd, jobID, condor_script, condor_dir)

    # Set up dag file
    jobID = 'level3_process_dag'
    for sim in args.sim:
        jobID += '_{}'.format(sim)
    jobID = getjobID(jobID, condor_dir)
    dag_file = '{}/submit_scripts/{}.submit'.format(condor_dir, jobID)
    comp.checkdir(dag_file)
    with open(dag_file, 'w') as dag:
        # dag.write('CONFIG {}/dagman.config\n'.format(cwd))
        for sim in argdict.keys():
            if len(argdict[sim]) < 1:
                continue
            for i, arg in enumerate(argdict[sim]):
                dag.write('JOB sim_{}_p{} '.format(sim, i) +
                          condor_script + '\n')
                dag.write('VARS sim_{}_p{} '.format(sim, i) +
                          'ARGS="' + arg + '"\n')

    # Submit jobs
    os.system('condor_submit_dag -maxjobs {} {}'.format(args.maxjobs, dag_file))

#!/usr/bin/env python

# Creates a condor submit file with executable `maker.sh`. 
# Then creates a DAG file that will pass the input file batches
# from `maker.py` to the `maker.sh` condor submit file. 
# Finally, submits the DAG file. 

import os
import time
import glob

def DAG_submit(exlist, npx4dir, jobID,
              maxjobs=4000, sublines=None, test=False, **vardict):

    # if test:
    #     print '(...running as test off the cluster...)'
    #     for a in arglist[:2]:
    #         os.system('{} {}'.format(executable, a.strip()))
    #     raise SystemExit('Completed testing run...')

    # Moderate number of submitted jobs
    njobs = len(vardict[])
    if njobs > maxjobs:
        yn = raw_input(
            'About to submit {} jobs at {} maxjobs.\
                    Are you sure? [y|n]: '.format(njobs, maxjobs))
        if yn != 'y':
            raise SystemExit('Aborting...')

    # Add date information
    if globaljobID is not None:
        globaljobID += time.strftime('_%Y%m%d')
        othersubmits = glob.glob(
                '/scratch/jbourbeau/logs/{}_??.log'.format(globaljobID))
        globaljobID += '_{:02d}'.format(len(othersubmits)+1)

    # Setup shell script
    ex_lines = [
        '#!/bin/bash',
        'date',
        'hostname',
    ]
    ex_lines += exlist
    ex_lines += ['date']
    ex_lines = [l + '\n' for l in ex_lines]
    outex = '/scratch/jbourbeau/execs/{}.sh'.format(jobID)
    with open(outex, 'w') as f:
        f.writelines(ex_lines)

    # Make file executable
    st = os.stat(outex)
    os.chmod(outex, st.st_mode | stat.S_IEXEC)

    # Build condor submission script
    lines = [
        'universe = vanilla\n',
        'getenv = true\n',
        'executable = {}\n'.format(outex),
        'arguments = ' + ' '.join('$({})'.format(x) for x in vardict.keys()) + '\n'
        'log = /scratch/jbourbeau/logs/{}.log\n'.format(globaljobID),
        'output = {}/outs/{}.out\n'.format(npx4dir, globaljobID),
        'error = {}/errors/{}.error\n'.format(npx4dir, globaljobID),
        'notification = Never\n',
        'queue \n'
    ]
    # Option for additional lines to submission script
    if sublines != None:
        for l in sublines:
            lines.insert(-1, '{}\n'.format(l))

    condor_script = '{}/submit-desc.sub'.format(npx4dir)
    with open(condor_script, 'w') as f:
        f.writelines(lines)

    # Build dag file
    dag_file = '/scratch/jbourbeau/dags/{}.submit'.format(globaljobID)
    with open(dag_file, 'w') as dag:
        for jobID, outfile, infiles in zip(jobID_list, outfile_list, infiles_list):
        # for i, arg in enumerate(arglist):
            dag.write('JOB ' + jobID + ' ' + condor_script + '\n')
            dag.write('VARS ' + jobID +' JOBID=\'' + jobID + '\'\n')
            dag.write('VARS ' + jobID +' OUTFILE=\'' + outfile + '\'\n')
            dag.write('VARS ' + jobID +' INFILES=\'' + infiles + '\'\n')

    # Submit dag file
    os.system('condor_submit_dag -maxjobs {} {}'.format(maxjobs, dag_file))

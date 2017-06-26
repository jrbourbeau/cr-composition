#!/usr/bin/env python

import os
import itertools
import argparse
import numpy as np

import composition as comp
import pycondor

if __name__ == "__main__":

    p = argparse.ArgumentParser(description='Runs sequential feature selection on the cluster')
    p.add_argument('--config', dest='config', default='IC79',
                   help='Detector configuration')
    p.add_argument('--pipeline', dest='pipeline',
                   default='xgboost',
                   help='Pipeline to use for classification')
    p.add_argument('--scoring', dest='scoring', default='accuracy',
                   help='Scoring metric to use in cross-validation')
    p.add_argument('--cv', dest='cv', type=int, default=3,
                   help='Number of folds in cross-validation')
    p.add_argument('--n_jobs', dest='n_jobs', type=int, default=1,
                   help='Number cores to run in parallel')
    p.add_argument('--overwrite', dest='overwrite',
                   default=False, action='store_true',
                   help='Overwrite existing merged files')
    args = p.parse_args()

    # Define output directories for condor jobs
    mypaths = comp.Paths()
    error = mypaths.condor_data_dir + '/error'
    output = mypaths.condor_data_dir + '/output'
    log = mypaths.condor_scratch_dir + '/log'
    submit = mypaths.condor_scratch_dir + '/submit'

    # Set up pycondor Job
    ex = os.getcwd() + '/sequential-feature-selection.py'
    job = pycondor.Job('SFS', ex, error=error, output=output, log=log,
                       submit=submit, request_memory='5GB',
                       extra_lines=['request_cpus = {}'.format(args.n_jobs)],
                       verbose=2)

    method = ['forward', 'backward']
    floating = [True, False]

    base_arg = '--config {} --pipeline {} --cv {} --scoring {} --n_jobs {}'.format(args.config, args.pipeline, args.cv, args.scoring, args.n_jobs)
    for method, floating in itertools.product(method, floating):
        arg = base_arg + ' --method {}'.format(method)
        if floating:
            arg += ' --floating'
        job.add_arg(arg)

    # Submit this Job under a Dagman, even though there aren't many arguments
    dagman = pycondor.Dagman('SFS_dagman', submit=submit, verbose=2)
    dagman.add_job(job)
    dagman.build_submit()

#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
import numpy as np
import pycondor

import comptools as comp

# Define path to executables used in processing
HERE = os.path.abspath(os.path.dirname(__file__))
JOB_EX = os.path.join(HERE, 'random-sample-anisotropy.py')

# Define pycondor Job/Dagman directories
PYCONDOR_ERROR = os.path.join(comp.paths.condor_data_dir, 'error')
PYCONDOR_OUTPUT = os.path.join(comp.paths.condor_data_dir, 'output')
PYCONDOR_LOG = os.path.join(comp.paths.condor_scratch_dir, 'log')
PYCONDOR_SUBMIT = os.path.join(comp.paths.condor_scratch_dir, 'submit')


if __name__ == '__main__':

    description = 'Extracts and saves desired information from simulation/data .i3 files'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('-c', '--config',
                   dest='config',
                   default='IC86.2012',
                   choices=comp.datafunctions.get_data_configs(),
                   help='Detector configuration')
    p.add_argument('--composition',
                   dest='composition',
                   default='total',
                   choices=['light', 'heavy', 'total'],
                   help='Whether to make individual skymaps for each composition')
    args = p.parse_args()

    job = pycondor.Job(name='random-sample-anisotropy',
                       executable=JOB_EX,
                       error=PYCONDOR_ERROR,
                       output=PYCONDOR_OUTPUT,
                       log=PYCONDOR_LOG,
                       submit=PYCONDOR_SUBMIT)

    num_trials = 10000
    num_trials_per_job = 100
    num_splits = num_trials // num_trials_per_job
    print('num_splits = {}'.format(num_splits))

    for random_states in np.array_split(np.arange(num_trials), num_splits):
        random_states_arg = ' '.join(map(str, random_states))
        print('random_states_arg = {}'.format(random_states_arg))

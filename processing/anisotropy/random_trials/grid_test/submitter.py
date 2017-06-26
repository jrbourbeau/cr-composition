#!/usr/bin/env python

import os
import pycondor

import comptools as comp


if __name__ == "__main__":

    # Define output directories
    error = os.path.join(comp.paths.condor_data_dir, 'grid_test/error')
    output = os.path.join(comp.paths.condor_data_dir, 'grid_test/output')
    log = os.path.join(comp.paths.condor_scratch_dir, 'grid_test/log')
    submit = os.path.join(comp.paths.condor_scratch_dir, 'grid_test/submit')

    # Define path to executables
    job_ex = os.path.abspath('test_script.py')

    # Extra lines for submitting to the open science grid
    extra_lines = ['Requirements = HAS_CVMFS_icecube_opensciencegrid_org',
                   'use_x509userproxy=true',
                   'should_transfer_files = YES',
                   'when_to_transfer_output = ON_EXIT']
    grid = 'gsiftp://gridftp-users.icecube.wisc.edu'

    # Create Dagman instance
    dag_name = 'test_dag'
    dagman = pycondor.Dagman(dag_name, submit=submit, verbose=1)

    job_name = 'test_job'
    job = pycondor.Job(job_name, job_ex, error=error, output=output,
                       log=log, submit=submit, extra_lines=extra_lines,
                       verbose=1)
    dagman.add_job(job)
    dagman.build_submit(fancyname=True)

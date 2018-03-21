#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
import warnings
import pycondor

import comptools as comp

warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")

scan_features = [['lap_cos_zenith', 'log_s125', 'log_dEdX'],
                 ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'avg_inice_radius'],
                 ]

if __name__ == '__main__':

    description = 'Saves trained composition classification model for later use'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config', dest='config',
                        choices=comp.simfunctions.get_sim_configs(),
                        help='Detector configuration')
    parser.add_argument('--num_groups', dest='num_groups', type=int,
                        default=4, choices=[2, 3, 4],
                        help='Number of composition groups')
    parser.add_argument('--pipeline', dest='pipeline',
                        default='xgboost',
                        help='Composition classification pipeline to use')
    parser.add_argument('--n_jobs', dest='n_jobs', type=int,
                        default=1, choices=list(range(1, 21)),
                        help='Number of jobs to run in parallel for the '
                             'gridsearch. Ignored if gridsearch=False.')
    args = parser.parse_args()

    config = args.config
    num_groups = args.num_groups
    pipeline = args.pipeline
    n_jobs = args.n_jobs

    executable = os.path.abspath('feature_scan.py')
    # Define pycondor Job/Dagman directories
    error = os.path.join(comp.paths.condor_data_dir, 'error')
    output = os.path.join(comp.paths.condor_data_dir, 'output')
    log = os.path.join(comp.paths.condor_scratch_dir, 'log')
    submit = os.path.join(comp.paths.condor_scratch_dir, 'submit')

    dag = pycondor.Dagman(name='feature_scan_{}'.format(pipeline),
                          submit=submit)
    for features in scan_features:
        feature_str = '-'.join(features)
        job = pycondor.Job(name='feature_scan_{}'.format(feature_str),
                           executable=executable,
                           submit=submit,
                           error=error,
                           output=output,
                           log=log,
                           request_cpus=n_jobs,
                           verbose=1,
                           dag=dag)
        argument = '--features {} '.format(' '.join(features))
        for arg_name in ['config', 'num_groups', 'pipeline', 'n_jobs']:
            argument += '--{} {} '.format(arg_name, getattr(args, arg_name))
        job.add_arg(argument)

    dag.build_submit()

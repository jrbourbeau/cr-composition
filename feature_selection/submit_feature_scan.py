#!/usr/bin/env python

from __future__ import division, print_function
import os
import numpy as np
import argparse
import warnings
from itertools import product
import pycondor

import comptools as comp

warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")

base_features = ['lap_cos_zenith', 'log_s125', 'log_dEdX']
scan_features = [base_features,
                 base_features + ['avg_inice_radius'],
                 ]
dom_numbers = [1, 15, 30, 45, 60]
scan_features += [base_features + ['NChannels_1_60'] + \
                 ['NChannels_{}_{}'.format(min_DOM, max_DOM)
                    for min_DOM, max_DOM in zip(dom_numbers[:-1], dom_numbers[1:])]
                 ]
scan_features += [base_features + ['NHits_1_60'] + \
                 ['NHits_{}_{}'.format(min_DOM, max_DOM)
                    for min_DOM, max_DOM in zip(dom_numbers[:-1], dom_numbers[1:])]
                 ]
min_dists = np.arange(0, 1125, 125)
scan_features += [base_features + ['IceTop_charge_beyond_{}m'.format(min_dist) for min_dist in min_dists]]

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

    dag_name = 'feature_scan_{}_num_groups-{}'.format(pipeline, num_groups)
    dag = pycondor.Dagman(name=dag_name,
                          submit=submit)
    for idx, (features, random_feature) in enumerate(product(scan_features, [True, False])):
        feature_str = '-'.join(features)
        job = pycondor.Job(name='feature_scan_num_groups-{}_{}'.format(num_groups, idx),
                           executable=executable,
                           submit=submit,
                           error=error,
                           output=output,
                           log=log,
                           request_cpus=n_jobs,
                           request_memory='3GB',
                           verbose=1,
                           dag=dag)
        argument = '--features {} '.format(' '.join(features))
        for arg_name in ['config', 'num_groups', 'pipeline', 'n_jobs']:
            argument += '--{} {} '.format(arg_name, getattr(args, arg_name))
        if random_feature:
            argument += '--random_feature '
        job.add_arg(argument)

    dag.build_submit()

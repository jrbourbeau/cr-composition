#!/usr/bin/env python

import os
import itertools
import argparse
import numpy as np

import composition as comp
import pycondor

def add_hyperparameter(dagman, param_name, param_range, dtype, validation_ex, merge_ex, **args):
    validation_job = pycondor.Job('{}_job'.format(param_name), validation_ex, error=error, output=output,
                        log=log, submit=submit, request_cpus=args['n_jobs'], verbose=1)
    dagman.add_job(validation_job)
    merge_job = pycondor.Job('merge_{}_job'.format(param_name), merge_ex, error=error, output=output,
                        log=log, submit=submit, verbose=1)
    dagman.add_job(merge_job)
    # Ensure that the merge script only runs after all the unmerged dataframes have been generated
    merge_job.add_parent(validation_job)

    # Add args to job
    base_arg = '--pipeline {} --param_name {} --param_type {} --cv {} --scoring {}'.format(args['pipeline'],
        param_name, dtype, args['cv'], args['scoring'])
    outfiles = []
    for value in param_range:
        outfile = os.path.join(args['outdir'], 'validation-{}-{}-{}-{}-cv{}.csv'.format(args['pipeline'],
            param_name, value, args['scoring'], args['cv']))
        outfiles.append(outfile)
        validation_job.add_arg(base_arg + ' --param_value {} --n_jobs {} --outfile {}'.format(value, args['n_jobs'], outfile))

    merge_infiles_str = ' '.join(outfiles)
    merge_outfile = os.path.join(args['outdir'], 'validation-{}-{}-{}-cv{}.csv'.format(args['pipeline'],
        param_name, args['scoring'], args['cv']))
    merge_job.add_arg('--infiles {} --outfile {} --overwrite'.format(merge_infiles_str, merge_outfile))

    return dagman

if __name__ == "__main__":

    p = argparse.ArgumentParser(description='Runs sequential feature selection on the cluster')
    p.add_argument('--pipeline', dest='pipeline',
                   default='GBDT',
                   choices=['xgboost', 'GBDT'],
                   help='Pipeline to use for classification')
    p.add_argument('--scoring', dest='scoring', default='accuracy',
                   help='Scoring metric to use in cross-validation')
    p.add_argument('--cv', dest='cv', type=int, default=10,
                   help='Number of folds in cross-validation')
    p.add_argument('--n_jobs', dest='n_jobs', type=int, default=1,
                   help='Number cores to run in parallel')
    p.add_argument('--outdir', dest='outdir',
                   default=os.path.join(os.getcwd(), 'data'),
                   help='Output directory')
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

    # Submit this Job under a Dagman, even though there aren't many arguments
    dagman = pycondor.Dagman('validation-dagman', submit=submit, verbose=2)

    validation_ex = os.path.join(os.getcwd(), 'validation-curves.py')
    merge_ex = os.path.join(os.getcwd(), 'merge_dataframe.py')

    # Maximum tree depth
    dagman = add_hyperparameter(dagman, 'max_depth', range(1, 11), 'int',
                    validation_ex, merge_ex, **vars(args))

    # Learning rate
    dagman = add_hyperparameter(dagman, 'learning_rate', np.arange(0.1, 1.1, 0.1), 'float',
                    validation_ex, merge_ex, **vars(args))

    # Number of estimators
    dagman = add_hyperparameter(dagman, 'n_estimators', range(10, 220, 20), 'int',
                    validation_ex, merge_ex, **vars(args))

    if args.pipeline == 'GBDT':
        #Minimum samples in a leaf
        dagman = add_hyperparameter(dagman, 'min_samples_leaf', range(1, 500, 50), 'int',
                        validation_ex, merge_ex, **vars(args))

        # Minimum samples to split
        dagman = add_hyperparameter(dagman, 'min_samples_split', range(1, 500, 50), 'int',
                        validation_ex, merge_ex, **vars(args))

        # Subsample size to use when fitting
        dagman = add_hyperparameter(dagman, 'subsample', np.arange(0.0, 1.1, 0.1), 'float',
                        validation_ex, merge_ex, **vars(args))



    dagman.build_submit()

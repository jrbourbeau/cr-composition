#!/usr/bin/env python

import os
import argparse
import numpy as np
import pandas as pd
import pycondor

import comptools as comp


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description='Extracts and saves desired information from simulation/data .i3 files')
    p.add_argument('-c', '--config', dest='config',
                   default='IC86.2012',
                   choices=['IC79', 'IC86.2012', 'IC86.2013', 'IC86.2014', 'IC86.2015'],
                   help='Detector configuration')
    p.add_argument('--composition', dest='composition',
                   default='all',
                   choices=['light', 'heavy', 'all', 'random_0', 'random_1'],
                   help='Whether to make individual skymaps for each composition')
    p.add_argument('--low_energy', dest='low_energy',
                   default=False, action='store_true',
                   help='Only use events with energy < 10**6.75 GeV')
    p.add_argument('--n_side', dest='n_side', type=int,
                   default=64,
                   help='Number of times to split the DataFrame')
    p.add_argument('--n_splits', dest='n_splits', type=int,
                   default=200,
                   help='Number of times to split the DataFrame')
    p.add_argument('--split_idx', dest='split_idx', type=int,
                   default=0,
                   help='Number of times to split the DataFrame')
    p.add_argument('--outfile', dest='outfile',
                   help='Output reference map file')
    p.add_argument('--overwrite', dest='overwrite',
                   default=False, action='store_true',
                   help='Option to overwrite reference map file, '
                        'if it alreadu exists')
    args = p.parse_args()

    # Define output directories
    error = comp.paths.condor_data_dir + '/error'
    output = comp.paths.condor_data_dir + '/output'
    log = comp.paths.condor_scratch_dir + '/log'
    submit = comp.paths.condor_scratch_dir + '/submit'

    # Define path to executables
    make_maps_ex = os.path.join(comp.paths.project_home,
                                'processing/anisotropy',
                                'make_reference_map.py')
    merge_maps_ex = os.path.join(comp.paths.project_home,
                                 'processing/anisotropy',
                                 'merge_maps.py')

    make_maps_name = 'make_maps_{}_{}'.format(args.config, args.composition)
    make_maps_job = pycondor.Job(make_maps_name, make_maps_ex,
                                 error=error, output=output,
                                 log=log, submit=submit,
                                 # request_memory='6GB',
                                 verbose=1)

    merge_maps_name = 'merge_maps_{}_{}'.format(args.config, args.composition)
    merge_maps_job = pycondor.Job(merge_maps_name, merge_maps_ex,
                                  error=error, output=output,
                                  log=log, submit=submit,
                                  verbose=1)

    # Ensure that make_maps_job completes before merge_maps_job begins
    merge_maps_job.add_parent(make_maps_job)

    merge_infiles = []
    for split_idx in np.arange(args.n_splits):
        outfile = os.path.join(comp.paths.comp_data_dir,
                               args.config + '_data', 'anisotropy',
                               'maps_part_{}_{}.fits'.format(split_idx, args.composition))
        make_maps_arg = '-c {} --n_side {} --n_splits {} --split_idx {} ' \
                        '--outfile {} --composition {}'.format(
                                args.config, args.n_side, args.n_splits,
                                split_idx, outfile, args.composition)
        if args.low_energy:
            make_maps_arg += ' --low_energy'

        make_maps_job.add_arg(make_maps_arg)
        # Add this outfile to the list of infiles for merge_maps_job
        merge_infiles.append(outfile)

    infiles_str = ' '.join(merge_infiles)
    # Assemble merged output file path
    merged_outdir = os.path.join(comp.paths.comp_data_dir,
                                 args.config + '_data', 'anisotropy')
    if args.low_energy:
        merged_basename = '{}_maps_nside_{}_{}_lowenergy.fits'.format(
                          args.config, args.n_side, args.composition)
    else:
        merged_basename = '{}_maps_nside_{}_{}.fits'.format(
                          args.config, args.n_side, args.composition)
    merged_outfile = os.path.join(merged_outdir, merged_basename)
    merge_maps_arg = '--infiles {} --outfile {}'.format(infiles_str, merged_outfile)
    merge_maps_job.add_arg(merge_maps_arg)

    # Create Dagman instance
    dag_name = 'anisotropy_maps_{}_{}'.format(args.config, args.composition)
    dagman = pycondor.Dagman(dag_name, submit=submit, verbose=1)
    dagman.add_job(make_maps_job)
    dagman.add_job(merge_maps_job)
    dagman.build_submit(fancyname=True)

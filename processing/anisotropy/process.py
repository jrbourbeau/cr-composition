#!/usr/bin/env python

import os
import argparse
import numpy as np
import pandas as pd
import pycondor
import sys

import comptools as comp


def save_anisotropy_dataframe(config, outfile):

    print('Loading data...')
    data_df = comp.load_dataframe(datatype='data', config=config, verbose=True)
    keep_columns = ['lap_zenith', 'lap_azimuth', 'start_time_mjd', 'pred_comp']

    comp_list = ['light', 'heavy']
    pipeline_str = 'GBDT'
    pipeline = comp.get_pipeline(pipeline_str)
    feature_list, feature_labels = comp.get_training_features()

    print('Loading simulation...')
    if 'IC86' in config:
        sim_config = 'IC86.2012'
    else:
        sim_config = 'IC79'
    sim_df = comp.load_dataframe(datatype='sim', config=sim_config, verbose=True, split=False)
    X_train, y_train = comp.dataframe_functions.dataframe_to_X_y(sim_df, feature_list)
    print('Training classifier...')
    pipeline = pipeline.fit(X_train, y_train)
    X_data = comp.dataframe_functions.dataframe_to_array(data_df, feature_list)
    data_pred = pd.Series(pipeline.predict(X_data), dtype=int)
    data_df['pred_comp'] = data_pred.apply(comp.dataframe_functions.label_to_comp)

    print('Saving anisotropy DataFrame for {}'.format(config))
    with pd.HDFStore(outfile, 'w') as store:
        store.put('dataframe', data_df.loc[:, keep_columns], format='table', data_columns=True)

    return


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description='Extracts and saves desired information from simulation/data .i3 files')
    p.add_argument('-c', '--config', dest='config',
                   default='IC86.2012',
                   choices=['IC79', 'IC86.2012', 'IC86.2013'],
                   help='Detector configuration')
    p.add_argument('--composition', dest='composition',
                   default='all',
                   choices=['light', 'heavy', 'all'],
                   help='Whether to make individual skymaps for each composition')
    p.add_argument('--n_side', dest='n_side', type=int,
                   default=64,
                   help='Number of times to split the DataFrame')
    p.add_argument('--n_splits', dest='n_splits', type=int,
                   default=1000,
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

    # Setup global path names
    mypaths = comp.get_paths()

    # Save stripped-down anisotropy DataFrame
    outfile = os.path.join(mypaths.comp_data_dir, args.config + '_data',
                           'anisotropy_dataframe.hdf')
    # save_anisotropy_dataframe(config=args.config, outfile=outfile)
    # sys.exit()

    # Define output directories
    error = mypaths.condor_data_dir + '/error'
    output = mypaths.condor_data_dir + '/output'
    log = mypaths.condor_scratch_dir + '/log'
    submit = mypaths.condor_scratch_dir + '/submit'

    # Define path to executables
    make_maps_ex = '{}/make_reference_map.py'.format(os.getcwd())
    merge_maps_ex = '{}/merge_maps.py'.format(os.getcwd())

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
        outfile = os.path.join(mypaths.comp_data_dir, args.config + '_data',
                               'anisotropy',
                               'maps_part_{}_{}.fits'.format(split_idx, args.composition))
        make_maps_arg = '-c {} --n_side {} --n_splits {} --split_idx {} ' \
                        '--outfile {} --composition {}'.format(args.config, args.n_side,
                                              args.n_splits,
                                              split_idx, outfile, args.composition)
        make_maps_job.add_arg(make_maps_arg)
        # Add this outfile to the list of infiles for merge_maps_job
        merge_infiles.append(outfile)

    infiles_str = ' '.join(merge_infiles)
    merged_outfile = os.path.join(mypaths.comp_data_dir, args.config + '_data',
                                  'anisotropy',
                                  '{}_maps_nside_{}_{}.fits'.format(args.config, args.n_side, args.composition))
    merge_maps_arg = '--infiles {} --outfile {}'.format(infiles_str, merged_outfile)
    merge_maps_job.add_arg(merge_maps_arg)

    # Create Dagman instance
    dag_name = 'anisotropy_maps_{}_{}'.format(args.config, args.composition)
    dagman = pycondor.Dagman(dag_name, submit=submit, verbose=1)
    dagman.add_job(make_maps_job)
    dagman.add_job(merge_maps_job)
    dagman.build_submit(fancyname=True)

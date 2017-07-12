#!/usr/bin/env python

import os
import argparse
import pandas as pd
import multiprocessing as mp

import comptools as comp


def save_anisotropy_dataframe(config, outfile):

    print('Loading data...')
    data_df = comp.load_dataframe(datatype='data', config=config, verbose=False)
    keep_columns = ['lap_zenith', 'lap_azimuth', 'start_time_mjd',
                    'pred_comp', 'lap_log_energy']

    comp_list = ['light', 'heavy']
    pipeline_str = 'GBDT'
    pipeline = comp.get_pipeline(pipeline_str)
    feature_list, feature_labels = comp.get_training_features()
    data_df.loc[:, feature_list].dropna(axis=0, how='any', inplace=True)

    print('Loading simulation...')
    if 'IC86' in config:
        sim_config = 'IC86.2012'
    else:
        sim_config = 'IC79'
    sim_df = comp.load_dataframe(datatype='sim', config=sim_config, verbose=False, split=False)
    X_train, y_train = comp.dataframe_functions.dataframe_to_X_y(sim_df, feature_list)
    print('Training classifier...')
    pipeline = pipeline.fit(X_train, y_train)
    X_data = comp.dataframe_functions.dataframe_to_array(data_df, feature_list)
    data_pred = pd.Series(pipeline.predict(X_data), dtype=int)
    data_df['pred_comp'] = data_pred.apply(comp.dataframe_functions.label_to_comp)
    # print('decision_function = {}'.format(pipeline.decision_function(X_data)))
    # data_df['score'] = pipeline.decision_function(X_data)

    print('Saving anisotropy DataFrame for {}'.format(config))
    with pd.HDFStore(outfile, 'w') as store:
        store.put('dataframe', data_df.loc[:, keep_columns], format='table')

    return


if __name__ == "__main__":

    p = argparse.ArgumentParser(description='Saves a stripped-down DataFrame '
                                            'for making the anisotropy maps')
    p.add_argument('-c', '--config', dest='config', nargs='*',
                   default=['IC86.2011' ,'IC86.2012', 'IC86.2013', 'IC86.2014', 'IC86.2015'],
                   help='Detector configuration(s)')
    p.add_argument('--outfile', dest='outfile',
                   help='Output reference map file')
    p.add_argument('--overwrite', dest='overwrite',
                   default=False, action='store_true',
                   help='Option to overwrite reference map file, '
                        'if it alreadu exists')
    args = p.parse_args()

    pool = mp.Pool(processes=len(args.config))
    results = []
    for config in args.config:

        outfile = os.path.join(comp.paths.comp_data_dir,
                               config + '_data',
                               'anisotropy_dataframe.hdf')
        comp.check_output_dir(outfile)

        results.append(pool.apply_async(save_anisotropy_dataframe, args=(config, outfile)))
    output = [p.get() for p in results]

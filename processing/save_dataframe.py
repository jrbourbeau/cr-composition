#!/usr/bin/env python

import os
import argparse
import numpy as np
import pandas as pd
import shutil

import comptools as comp
from comptools.composition_encoding import (composition_group_labels,
                                            encode_composition_groups)


def extract_vector_series(store, key, sim):
    values, index = [], []
    grouped = store[key].groupby(['Run', 'Event', 'SubEvent'])
    for name, group in grouped:
        values.append(group['item'].values)
        index.append('{}_{}_{}_{}'.format(sim, *name))
        # For data
        # index.append('{}_{}_{}_{}'.format(config, *name))
    series = pd.Series(data=values, index=index)
    return series


def extract_dataframe(input_file, config, datatype):
    with pd.HDFStore(input_file, mode='r') as store:
        series_size = store.get_storer('NStations').nrows
        # Dictionary of key: pd.Series pairs to get converted to pd.DataFrame
        series_dict = {}
        # For data stored in that can be accessed via a value column
        value_keys = ['IceTopMaxSignal',
                      'IceTopMaxSignalInEdge',
                      'IceTopMaxSignalString',
                      'IceTopNeighbourMaxSignal',
                      'NStations',
                      'StationDensity',
                      'FractionContainment_Laputop_IceTop',
                      'FractionContainment_Laputop_InIce',
                      'passed_IceTopQualityCuts',
                      'passed_InIceQualityCuts',
                      'avg_inice_radius',
                      'std_inice_radius',
                      'median_inice_radius',
                      'frac_outside_one_std_inice_radius',
                      'frac_outside_two_std_inice_radius',
                      ]

        for cut in ['MilliNCascAbove2', 'MilliQtotRatio', 'MilliRloglBelow2',
                    'NCh_CoincLaputopCleanedPulsesAbove7', 'StochRecoSucceeded']:
            value_keys += ['passed_{}'.format(cut)]

        min_dists = np.arange(0, 1125, 125)
        for min_dist in min_dists:
            value_keys += ['IceTop_charge_beyond_{}m'.format(min_dist)]

        dom_numbers = [1, 15, 30, 45, 60]
        for min_DOM, max_DOM in zip(dom_numbers[:-1], dom_numbers[1:]):
            key = '{}_{}'.format(min_DOM, max_DOM)
            value_keys += ['NChannels_'+key,
                           'NHits_'+key,
                           'InIce_charge_'+key,
                           'max_qfrac_'+key,
                           ]
        key = '1_60'
        value_keys += ['NChannels_'+key,
                       'NHits_'+key,
                       'InIce_charge_'+key,
                       'max_qfrac_'+key,
                       ]
        if datatype == 'sim':
            # Add MC containment
            value_keys += ['FractionContainment_MCPrimary_IceTop',
                           'FractionContainment_MCPrimary_InIce',
                           'angle_MCPrimary_Laputop',
                           ]
        for key in value_keys:
            series_dict[key] = store[key]['value']

        if datatype == 'sim':
            # Get MCPrimary information
            for key in ['x', 'y', 'energy', 'zenith', 'azimuth', 'type']:
                series_dict['MC_{}'.format(key)] = store['MCPrimary'][key]
            # Add simulation set number and corresponding composition
            sim_num = int(os.path.basename(input_file).split('_')[1])
            series_dict['sim'] = pd.Series([sim_num] * series_size, dtype=int)
            composition = comp.simfunctions.sim_to_comp(sim_num)
            series_dict['MC_comp'] = pd.Series([composition] * series_size, dtype=str)

        # Get timing information
        series_dict['start_time_mjd'] = store['I3EventHeader']['time_start_mjd']
        series_dict['end_time_mjd'] = store['I3EventHeader']['time_end_mjd']

        # Get Laputop information
        laputop = store['Laputop']
        laputop_params = store['LaputopParams']
        lap_keys = ['zenith', 'azimuth', 'x', 'y']
        if datatype == 'data':
            lap_keys += ['ra', 'dec']
        for key in lap_keys:
            series_dict['lap_{}'.format(key)] = laputop[key]
        lap_param_keys = ['s50', 's80', 's125', 's180', 's250', 's500',
                          'ndf', 'beta', 'rlogl']
        for key in lap_param_keys:
            series_dict['lap_{}'.format(key)] = laputop_params[key]
        series_dict['lap_energy'] = laputop_params['e_h4a']
        series_dict['lap_chi2'] = laputop_params['chi2'] / laputop_params['ndf']
        series_dict['lap_fitstatus_ok'] = store['lap_fitstatus_ok']['value'].astype(bool)

        # # Get DDDDR information
        # d4r_params = store['I3MuonEnergyLaputopParams']
        # d4r_keys = ['N', 'b', 'gamma', 'peak_energy', 'peak_sigma', 'exists', 'mean', 'median']
        # for key in d4r_keys:
        #     series_dict['d4r_{}'.format(key)] = d4r_params[key]

        series_dict['eloss_1500_standard'] = store['Stoch_Reco']['eloss_1500']
        # Get number of high energy stochastics
        # series_dict['n_he_stoch_standard'] = store['Stoch_Reco']['n_he_stoch']
        # series_dict['n_he_stoch_strong'] = store['Stoch_Reco2']['n_he_stoch']
        # series_dict['eloss_1500_strong'] = store['Stoch_Reco2']['eloss_1500']
        series_dict['mil_rlogl'] = store['MillipedeFitParams']['rlogl']
        series_dict['mil_qtot_measured'] = store['MillipedeFitParams']['qtotal']
        series_dict['mil_qtot_predicted'] = store['MillipedeFitParams']['predicted_qtotal']

        series_dict['IceTopLLHRatio'] = store['IceTopLLHRatio']['LLH_Ratio']

        # # Construct unique index from run/event/subevent info in I3EventHeader
        # runs = store['I3EventHeader']['Run']
        # events = store['I3EventHeader']['Event']
        # sub_events = store['I3EventHeader']['SubEvent']
        # if datatype == 'sim':
        #     index = ['{}_{}_{}_{}'.format(sim_num, run, event, sub_event)
        #              for run, event, sub_event in zip(runs, events, sub_events)]
        # else:
        #     index = ['{}_{}_{}_{}'.format(config, run, event, sub_event)
        #              for run, event, sub_event in zip(runs, events, sub_events)]

        # tank_x = extract_vector_series(store, key='tank_x', sim=sim_num)
        # tank_y = extract_vector_series(store, key='tank_y', sim=sim_num)
        # tank_charge = extract_vector_series(store, key='tank_charge', sim=sim_num)

    df = pd.DataFrame(series_dict)
    # df.index = index

    return df


def add_extra_columns(df, datatype='sim'):
    '''
    Function to add to new columns to DataFrame that are commonly used

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    datatype : str, optional
        Specify whether this is a DataFrame based on simulation or data
        (default is 'sim').

    Returns
    -------
    df : pandas.DataFrame
        Returns DataFrame with added columns
    '''
    if datatype == 'sim':
        df['MC_log_energy'] = np.nan_to_num(np.log10(df['MC_energy']))
        # Add composition group labels
        for num_groups in [2, 3, 4]:
            label_key = 'comp_group_{}'.format(num_groups)
            df[label_key] = composition_group_labels(df['MC_comp'],
                                                     num_groups=num_groups)
            target_key = 'comp_target_{}'.format(num_groups)
            df[target_key] = encode_composition_groups(df[label_key],
                                                       num_groups=num_groups)
    # Add log-scale columns to df
    df['lap_log_energy'] = np.nan_to_num(np.log10(df['lap_energy']))
    df['lap_cos_zenith'] = np.cos(df['lap_zenith'])
    for dist in ['50', '80', '125', '180', '250', '500']:
        df['log_s'+dist] = np.log10(df['lap_s'+dist])
    df['log_dEdX'] = np.log10(df['eloss_1500_standard'])

    return df


def process_i3_hdf(input_file, config, datatype):
    df = extract_dataframe(input_file=input_file,
                           config=config,
                           datatype=datatype)
    df = comp.io.apply_quality_cuts(df,
                                    datatype=datatype,
                                    log_energy_min=None,
                                    log_energy_max=None,
                                    verbose=False)
    df = add_extra_columns(df, datatype=datatype)

    return df


if __name__ == "__main__":

    description = ('Converts input hdf5 files (from save_hdf.py) to a '
                   'well-formatted pandas dataframe object')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config',
                        dest='config',
                        default='IC86.2012',
                        help='Detector configuration')
    parser.add_argument('--type',
                        dest='type',
                        choices=['data', 'sim'],
                        default='sim',
                        help='Option to process simulation or data')
    parser.add_argument('-i', '--input',
                        dest='input',
                        help='Path to input hdf5 file')
    parser.add_argument('-o', '--output',
                        dest='output',
                        help='Path to output hdf5 file')
    args = parser.parse_args()

    # Validate user input
    if args.type == 'sim' and args.config not in comp.simfunctions.get_sim_configs():
        raise ValueError('Invalid simulation config {} entered'.format(args.config))
    elif args.type == 'data' and args.config not in comp.datafunctions.get_data_configs():
        raise ValueError('Invalid data config {} entered'.format(args.config))

    comp.check_output_dir(args.output)

    print('\ninput:\n\t{}'.format(args.input))
    print('\noutput:\n\t{}'.format(args.output))
    with comp.localized(inputs=args.input, output=args.output) as (inputs, output):
        print('local inputs:\n{}'.format(inputs))
        print('local output:\n{}'.format(output))
        df = process_i3_hdf(input_file=inputs,
                            config=args.config,
                            datatype=args.type)
        df.to_hdf(output, key='dataframe', mode='w', format='table')

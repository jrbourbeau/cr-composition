#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
import pandas as pd

import comptools


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Converts an input hdf5 file and converts to it a well-formatted output dataframe')
    parser.add_argument('--type', dest='type',
                        choices=['data', 'sim'],
                        default='sim',
                        help='Option to process simulation or data')
    parser.add_argument('--input', dest='input',
                        help='Path to input hdf5 file')
    parser.add_argument('--output', dest='output',
                        help='Path to output hdf5 file')
    parser.add_argument('--overwrite', dest='overwrite',
                        default=False, action='store_true',
                        help='Overwrite existing merged files')
    args = parser.parse_args()

    # If output file already exists and you want to overwrite,
    # delete existing output file
    if args.overwrite and os.path.exists(args.output):
        os.remove(args.output)

    # Open input hdf5 file and get number of events
    with pd.HDFStore(args.input) as input_store:
        series_size = input_store['NStations']['value'].size
        # Dictionary of key: pd.Series pairs to get converted to pd.DataFrame
        series_dict = {}

        # For data stored in that can be accessed via a value column
        value_keys = ['IceTopMaxSignal',
                      'IceTopMaxSignalInEdge',
                      'IceTopMaxSignalString',
                      'IceTopNeighbourMaxSignal',
                      'NStations',
                    #   'IceTop_charge',
                      'StationDensity',
                      'FractionContainment_Laputop_IceTop',
                      'FractionContainment_Laputop_InIce',
                    #   'num_millipede_particles',
                      'passed_IceTopQualityCuts',
                      'passed_InIceQualityCuts',
                    #   'IceTop_charge_175m',
                    #   'refit_beta', 'refit_log_s125',
                      'avg_inice_radius', 'std_inice_radius', 'median_inice_radius',
                      'frac_outside_one_std_inice_radius',
                      'frac_outside_two_std_inice_radius']

        for cut in ['MilliNCascAbove2', 'MilliQtotRatio', 'MilliRloglBelow2', 'NCh_CoincLaputopCleanedPulsesAbove7', 'StochRecoSucceeded']:
            value_keys += ['passed_{}'.format(cut)]

        # for i in ['1_60']:
        #     value_keys += ['avg_inice_radius_'+i, 'std_inice_radius_'+i,
        #                    'qweighted_inice_radius_'+i, 'invqweighted_inice_radius_'+i]
        for i in ['1_60']:
            value_keys += ['NChannels_'+i, 'NHits_'+i, 'InIce_charge_'+i, 'max_qfrac_'+i]

        # Add MC containment
        if args.type == 'sim':
            value_keys += ['FractionContainment_MCPrimary_IceTop',
                           'FractionContainment_MCPrimary_InIce',
                           'angle_MCPrimary_Laputop']
        for key in value_keys:
            series_dict[key] = input_store[key]['value']

        # if args.type == 'sim':
        #     # Add IceTop tank charge and distance information
        #     for key in ['x', 'y', 'charge']:
        #         grouped = input_store['tanks_{}'.format(key)].groupby(['Run','Event'])
        #         event_lists = []
        #         for name, group in grouped:
        #             event_lists.append(group['item'].values)
        #         series_dict['tank_{}'.format(key)] = pd.Series(event_lists, dtype=object)
        #     # # Add IceTop tank charge and distance information
        #     # for key in ['dist', 'charge']:
        #     #     grouped = input_store['tanks_{}_Laputop'.format(key)].groupby(['Run','Event'])
        #     #     event_lists = []
        #     #     for name, group in grouped:
        #     #         event_lists.append(group['item'].values)
        #     #     series_dict['tanks_{}_Laputop'.format(key)] = pd.Series(event_lists, dtype=object)
        #     # Add InIce DOM charge and distance information
        #     for key in ['dists', 'charges']:
        #         grouped = input_store['inice_dom_{}_1_60'.format(key)].groupby(['Run','Event'])
        #         event_lists = []
        #         for name, group in grouped:
        #             event_lists.append(group['item'].values)
        #         series_dict['inice_dom_{}_1_60'.format(key)] = pd.Series(event_lists, dtype=object)

        # Get MCPrimary information
        if args.type == 'sim':
            for key in ['x', 'y', 'energy', 'zenith', 'azimuth', 'type']:
                series_dict['MC_{}'.format(key)] = input_store['MCPrimary'][key]
            # Add simulation set number and corresponding composition
            sim_num = int(os.path.splitext(args.input)[0].split('_')[-1])
            series_dict['sim'] = pd.Series([sim_num] * series_size, dtype=int)
            composition = comptools.simfunctions.sim_to_comp(sim_num)
            series_dict['MC_comp'] = pd.Series([composition] * series_size, dtype=str)
            # Add composition class
            MC_comp_class = 'light' if composition in ['PPlus', 'He4Nucleus'] else 'heavy'
            series_dict['MC_comp_class'] = pd.Series([MC_comp_class] * series_size, dtype=str)

        # Get timing information
        series_dict['start_time_mjd'] = input_store['I3EventHeader']['time_start_mjd']
        series_dict['end_time_mjd'] = input_store['I3EventHeader']['time_end_mjd']

        # Get Laputop information
        laputop = input_store['Laputop']
        laputop_params = input_store['LaputopParams']
        lap_keys = ['zenith', 'azimuth', 'x', 'y']
        if args.type == 'data':
            lap_keys += ['ra', 'dec']
        for key in lap_keys:
            series_dict['lap_{}'.format(key)] = laputop[key]
        lap_param_keys = ['s50', 's80', 's125', 's180', 's250', 's500',
                          'ndf', 'beta', 'rlogl']
        for key in lap_param_keys:
            series_dict['lap_{}'.format(key)] = laputop_params[key]
        series_dict['lap_energy'] = laputop_params['e_h4a']
        series_dict['lap_chi2'] = laputop_params['chi2'] / laputop_params['ndf']
        series_dict['lap_fitstatus_ok'] = input_store['lap_fitstatus_ok']['value'].astype(bool)

        # Get DDDDR information
        d4r_params = input_store['I3MuonEnergyLaputopParams']
        d4r_keys = ['N', 'b', 'gamma', 'peak_energy', 'peak_sigma', 'exists', 'mean', 'median']
        for key in d4r_keys:
            series_dict['d4r_{}'.format(key)] = d4r_params[key]

        # Get number of high energy stochastics
        # series_dict['n_he_stoch_standard'] = input_store[
        #     'Stoch_Reco']['n_he_stoch']
        # series_dict['n_he_stoch_strong'] = input_store[
        #     'Stoch_Reco2']['n_he_stoch']
        series_dict['eloss_1500_standard'] = input_store['Stoch_Reco']['eloss_1500']
        # series_dict['eloss_1500_strong'] = input_store[
        #     'Stoch_Reco2']['eloss_1500']
        series_dict['mil_rlogl'] = input_store['MillipedeFitParams']['rlogl']
        series_dict['mil_qtot_measured'] = input_store['MillipedeFitParams']['qtotal']
        series_dict['mil_qtot_predicted'] = input_store['MillipedeFitParams']['predicted_qtotal']

        # Construct unique index from run/event/subevent info in I3EventHeader
        runs = input_store['I3EventHeader']['Run']
        events = input_store['I3EventHeader']['Event']
        sub_events = input_store['I3EventHeader']['SubEvent']
        index = ['{}_{}_{}'.format(run, event, sub_event)
                 for run, event, sub_event in zip(runs, events, sub_events)]

        # Extract measured charge for each DOM
        grouped = input_store['NNcharges'].groupby(['Run','Event'])
        charges_list, columns = [], []
        for name, group in grouped:
            if not columns:
                col_info = zip(group['string'].values, group['om'].values,
                               group['pmt'].values)
                columns = ['{}_{}_{}'.format(*info) for info in col_info]
            charges_list.append(group['item'].values)

        df_tank_charges = pd.DataFrame(charges_list, columns=columns)

    # Open HDFStore for output hdf5 file
    comptools.check_output_dir(args.output)
    with pd.HDFStore(args.output, mode='w') as output_store:
        dataframe = pd.DataFrame(series_dict, index=index)
        # # Don't want to save data events that don't pass quality cuts
        # # because there is just too much data for that
        # if args.type == 'data':
        #     dataframe = comptools.apply_quality_cuts(dataframe, datatype='data',
        #                                              dataprocessing=True)
        # Add dataframe to output_store
        output_store.put('dataframe', dataframe, format='fixed')
        output_store.put('tank_charges', df_tank_charges, format='fixed')

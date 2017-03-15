#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
import pandas as pd

import composition as comp


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
                      'IceTop_charge',
                      'StationDensity',
                      'Laputop_IceTop_FractionContainment',
                      'Laputop_InIce_FractionContainment',
                      'num_millipede_particles',
                      'IceTopQualityCuts',
                      'InIceQualityCuts',
                      'avg_inice_radius',
                    #   'charge_inice_radius',
                    #   'chargesquared_inice_radius', 'charge_inice_radiussquared', 'hits_weighted_inice_radius',
                      'invcharge_inice_radius',
                      'max_inice_radius']

        for cut in ['MilliNCascAbove2', 'MilliQtotRatio', 'MilliRloglBelow2', 'NCh_CoincLaputopCleanedPulsesAbove7', 'StochRecoSucceeded']:
            value_keys += ['InIceQualityCuts_{}'.format(cut)]

        for i in ['1_60']:
        # for i in ['1_60', '1_45', '1_30', '1_15', '1_6', '45_60']:
            value_keys += ['NChannels_'+i, 'NHits_'+i, 'InIce_charge_'+i, 'max_qfrac_'+i]

        # Add MC containment
        if args.type == 'sim':
            value_keys.extend(['IceTop_FractionContainment',
                               'InIce_FractionContainment'])
        for key in value_keys:
            series_dict[key] = input_store[key]['value']

        # Get MCPrimary information
        if args.type == 'sim':
            for key in ['x', 'y', 'energy', 'zenith', 'azimuth', 'type']:
                series_dict['MC_{}'.format(key)] = input_store[
                    'MCPrimary'][key]
            # Add simulation set number and corresponding composition
            sim_num = os.path.splitext(args.input)[0].split('_')[-1]
            series_dict['sim'] = pd.Series([sim_num] * series_size)
            series_dict['MC_comp'] = pd.Series(
                [comp.simfunctions.sim2comp(sim_num)] * series_size)
            MC_comp_class = 'light' if comp.simfunctions.sim2comp(sim_num) in [
                'P', 'He'] else 'heavy'
            series_dict['MC_comp_class'] = pd.Series(
                [MC_comp_class] * series_size)

        # Get timing information
        series_dict['start_time_mjd'] = input_store[
            'I3EventHeader']['time_start_mjd']
        series_dict['end_time_mjd'] = input_store[
            'I3EventHeader']['time_end_mjd']

        # Get Laputop information
        laputop = input_store['Laputop']
        laputop_params = input_store['LaputopParams']
        lap_keys = ['zenith', 'x', 'y']
        for key in lap_keys:
            series_dict['lap_{}'.format(key)] = laputop[key]
        lap_param_keys = ['s50', 's80', 's125', 's180', 's250', 's500',
                          'ndf', 'beta', 'rlogl']
        for key in lap_param_keys:
            series_dict['lap_{}'.format(key)] = laputop_params[key]
        series_dict['lap_energy'] = laputop_params['e_h4a']
        series_dict['lap_chi2'] = laputop_params[
            'chi2'] / laputop_params['ndf']
        series_dict['lap_fitstatus_ok'] = input_store[
            'Laputop_fitstatus_ok']['value'].astype(bool)

        # Get LLHRatio info
        # series_dict['llhratio'] = input_store['IceTopLLHRatio']['LLH_Ratio']

        # Get number of high energy stochastics
        series_dict['n_he_stoch_standard'] = input_store[
            'Stoch_Reco']['n_he_stoch']
        series_dict['n_he_stoch_strong'] = input_store[
            'Stoch_Reco2']['n_he_stoch']
        series_dict['eloss_1500_standard'] = input_store[
            'Stoch_Reco']['eloss_1500']
        series_dict['eloss_1500_strong'] = input_store[
            'Stoch_Reco2']['eloss_1500']
        series_dict['mil_rlogl'] = input_store['MillipedeFitParams']['rlogl']
        series_dict['mil_qtot_measured'] = input_store[
            'MillipedeFitParams']['qtotal']
        series_dict['mil_qtot_predicted'] = input_store[
            'MillipedeFitParams']['predicted_qtotal']

    # Open HDFStore for output hdf5 file
    with pd.HDFStore(args.output) as output_store:
        dataframe = pd.DataFrame(series_dict)
        # Don't want to save data events that don't pass quality cuts
        # because there is just too much data for that
        if args.type == 'data':
            dataframe = comp.apply_quality_cuts(dataframe, datatype='data',
                dataprocessing=True)
        # Add dataframe to output_store
        output_store['dataframe'] = dataframe
        output_store.close()

#!/usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
import time
from collections import OrderedDict

from .paths import Paths


def load_sim(config='IT73', bintype='logdist', return_cut_dict=False, old=False):

    # Load simulation dataframe
    mypaths = Paths()
    if old:
        infile = '{}/{}_sim/sim_dataframe_old.hdf5'.format(
            mypaths.comp_data_dir, config, bintype)
    else:
        infile = '{}/{}_sim/sim_dataframe.hdf5'.format(
            mypaths.comp_data_dir, config, bintype)
    df = pd.read_hdf(infile)

    # Quality Cuts #
    # Adapted from PHYSICAL REVIEW D 88, 042004 (2013)
    cut_dict = OrderedDict()
    # IT specific cuts
    # cut_dict['ShowerLLH_reco_exists'] = df['reco_exists']
    cut_dict['lap_fitstatus_ok'] = df['lap_fitstatus_ok']
    cut_dict['MC_zenith'] = (np.cos(df['MC_zenith']) >= 0.8)
    # cut_dict['reco_zenith'] = (np.cos(df['reco_zenith']) >= 0.85)
    # cut_dict['LLHlap_zenith'] = (np.cos(df['LLHlap_zenith']) >= 0.8)
    cut_dict['lap_zenith'] = (np.cos(df['lap_zenith']) >= 0.8)
    # cut_dict['LLHLF_zenith'] = (np.cos(df['LLHLF_zenith']) >= 0.8)
    # cut_dict['ShowerLLH_zenith'] = (np.cos(np.pi - df['reco_zenith']) >= 0.8)
    cut_dict['lap_IT_containment'] = (df['Laputop_IceTop_FractionContainment'] < 0.96)
    cut_dict['MC_IT_containment'] = (df['IceTop_FractionContainment'] < 1.0)
    # cut_dict['reco_IT_containment'] = (df['reco_IT_containment'] < 1.0)
    # cut_dict['LLHlap_IT_containment'] = (df['LLHlap_IT_containment'] < 1.0)
    # cut_dict['LLHLF_IT_containment'] = (df['LLHLF_IT_containment'] < 1.0)
    cut_dict['IceTopMaxSignalInEdge'] = np.logical_not(
        df['IceTopMaxSignalInEdge'].astype(bool))
    cut_dict['IceTopMaxSignal'] = (df['IceTopMaxSignal'] >= 6)
    cut_dict['IceTopNeighbourMaxSignal'] = (
        df['IceTopNeighbourMaxSignal'] >= 4)
    cut_dict['NStations'] = (df['NStations'] >= 5)
    cut_dict['StationDensity'] = (df['StationDensity'] >= 0.2)
    # cut_dict['min_energy'] = (df['reco_energy'] > 10**6.2)
    # cut_dict['max_energy'] = (df['reco_energy'] < 10**8.0)
    cut_dict['min_energy_lap'] = (df['lap_energy'] > 10**6.2)
    cut_dict['max_energy_lap'] = (df['lap_energy'] < 10**8.0)

    # InIce specific cuts
    cut_dict['NChannels_1_60'] = (df['NChannels_1_60'] >= 8)
    # cut_dict['NChannels_1_45'] = (df['NChannels_1_45'] >= 8)
    cut_dict['NChannels_1_30'] = (df['NChannels_1_30'] >= 8)
    # cut_dict['NChannels_45_60'] = (df['NChannels_45_60'] >= 8)
    # cut_dict['NChannels_1_15'] = (df['NChannels_1_15'] >= 8)
    # cut_dict['NChannels_1_6'] = (df['NChannels_1_6'] >= 8)
    cut_dict['max_qfrac_1_60'] = (df['max_qfrac_1_60'] < 0.3)
    # cut_dict['max_qfrac_1_45'] = (df['max_qfrac_1_45'] < 0.3)
    cut_dict['max_qfrac_1_30'] = (df['max_qfrac_1_30'] < 0.3)
    # cut_dict['max_qfrac_45_60'] = (df['max_qfrac_45_60'] < 0.3)
    # cut_dict['max_qfrac_1_15'] = (df['max_qfrac_1_15'] < 0.3)
    # cut_dict['max_qfrac_1_6'] = (df['max_qfrac_1_6'] < 0.3)
    cut_dict['MC_InIce_containment'] = (df['InIce_FractionContainment'] < 1.0)
    cut_dict['lap_InIce_containment'] = (df['Laputop_InIce_FractionContainment'] < 1.0)
    # cut_dict['reco_InIce_containment'] = (df['reco_InIce_containment'] < 1.0)
    # cut_dict['LLHlap_InIce_containment'] = (df['LLHlap_InIce_containment'] < 1.0)
    # cut_dict['LLHLF_InIce_containment'] = (df['LLHLF_InIce_containment'] < 0.95)
    # cut_dict['max_charge_frac'] = (df['max_charge_frac'] < 0.3)
    cut_dict['lap_beta'] = (df['lap_beta'] < 9.5) & (df['lap_beta'] > 1.4)
    cut_dict['lap_rlogl'] = (df['lap_likelihood'] < 2)

    # Some conbined cuts
    cut_dict['lap_reco_success'] = (cut_dict['lap_fitstatus_ok']) & (cut_dict['lap_beta']) & (cut_dict['lap_rlogl'])
    # cut_dict['lap_reco_success'] = (df['lap_ndf'] >= 50) & (cut_dict['lap_fitstatus_ok']) & (cut_dict['lap_beta']) & (cut_dict['lap_rlogl'])
    # cut_dict['LLHlap_reco_exists'] = df['LLHlap_reco_exists'] & (cut_dict['lap_beta'])
    # cut_dict['LLHLF_reco_exists'] = df['LLHLF_reco_exists']
    # cut_dict['reco_exists'] = cut_dict['ShowerLLH_reco_exists'] & cut_dict['combined_reco_exists']
    # cut_dict['num_hits'] = cut_dict['NChannels_1_60'] & cut_dict['NStations']
    cut_dict['num_hits_1_60'] = cut_dict['NChannels_1_60'] & cut_dict['NStations']
    # cut_dict['num_hits_1_45'] = cut_dict['NChannels_1_45'] & cut_dict['NStations']
    cut_dict['num_hits_1_30'] = cut_dict['NChannels_1_30'] & cut_dict['NStations']
    # cut_dict['num_hits_1_15'] = cut_dict['NChannels_1_15'] & cut_dict['NStations']
    # cut_dict['num_hits_45_60'] = cut_dict['NChannels_45_60'] & cut_dict['NStations']
    # cut_dict['num_hits_1_6'] = cut_dict['NChannels_1_6'] & cut_dict['NStations']
    # cut_dict['reco_containment'] = cut_dict['reco_IT_containment'] & cut_dict['reco_InIce_containment']
    cut_dict['lap_containment'] = cut_dict['lap_IT_containment'] & cut_dict['lap_InIce_containment']
    # cut_dict['LLHlap_containment'] = cut_dict['LLHlap_IT_containment'] & cut_dict['LLHlap_InIce_containment']
    # cut_dict['LLHLF_containment'] = cut_dict['LLHLF_IT_containment'] & cut_dict['LLHLF_InIce_containment']
    # cut_dict['reco_containment'] = cut_dict[
    #     'reco_IT_containment'] & cut_dict['reco_InIce_containment']
    cut_dict['IT_signal'] = cut_dict['IceTopMaxSignalInEdge'] & cut_dict[
        'IceTopMaxSignal'] & cut_dict['IceTopNeighbourMaxSignal']
    # cut_dict['energy_range'] = cut_dict['min_energy'] & cut_dict['max_energy']
    cut_dict['energy_range_lap'] = cut_dict['min_energy_lap'] & cut_dict['max_energy_lap']

    # Add log-energy and log-charge columns to df
    df['MC_log_energy'] = np.nan_to_num(np.log10(df['MC_energy']))
    df['lap_log_energy'] = np.nan_to_num(np.log10(df['lap_energy']))
    # df['reco_log_energy'] = np.nan_to_num(np.log10(df['reco_energy']))
    df['InIce_log_charge_1_60'] = np.nan_to_num(np.log10(df['InIce_charge_1_60']))
    # df['InIce_log_charge_1_45'] = np.nan_to_num(np.log10(df['InIce_charge_1_45']))
    df['InIce_log_charge_1_30'] = np.nan_to_num(np.log10(df['InIce_charge_1_30']))
    # df['InIce_log_charge_1_15'] = np.nan_to_num(np.log10(df['InIce_charge_1_15']))
    # df['InIce_log_charge_1_6'] = np.nan_to_num(np.log10(df['InIce_charge_1_6']))
    # df['InIce_log_charge_45_60'] = np.nan_to_num(np.log10(df['InIce_charge_45_60']))
    df['log_NChannels_1_30'] = np.nan_to_num(np.log10(df['NChannels_1_30']))
    df['log_NHits_1_30'] = np.nan_to_num(np.log10(df['NHits_1_30']))
    # df['reco_cos_zenith'] = np.cos(np.pi - df['reco_zenith'])
    df['lap_cos_zenith'] = np.cos(df['lap_zenith'])
    # df['LLHlap_cos_zenith'] = np.cos(df['LLHlap_zenith'])
    # df['LLHLF_cos_zenith'] = np.cos(df['LLHLF_zenith'])
    # df['ShowerPlane_cos_zenith'] = np.cos(df['ShowerPlane_zenith'])
    df['log_s125'] = np.log10(df['lap_s125'])

    # Add ratio of features (could help improve RF classification)
    df['charge_nchannels_ratio'] = df['InIce_charge_1_30']/df['NChannels_1_30']
    df['charge_nhits_ratio'] = df['InIce_charge_1_30']/df['NHits_1_30']
    df['nchannels_nhits_ratio'] = df['NChannels_1_30']/df['NHits_1_30']
    df['stationdensity_charge_ratio'] = df['StationDensity']/df['InIce_charge_1_30']
    df['stationdensity_nchannels_ratio'] = df['StationDensity']/df['NChannels_1_30']
    df['stationdensity_nhits_ratio'] = df['StationDensity']/df['NHits_1_30']

    if return_cut_dict:
        return df, cut_dict
    else:
        selection_mask = np.array([True] * len(df))
        standard_cut_keys = ['lap_reco_success', 'lap_zenith', 'num_hits_1_30', 'IT_signal',
                     'StationDensity', 'max_qfrac_1_30', 'lap_containment', 'energy_range_lap']
        for key in standard_cut_keys:
            selection_mask *= cut_dict[key]
        # Print cut event flow
        n_total = len(df)
        cut_eff = {}
        cumulative_cut_mask = np.array([True] * n_total)
        print('Cut event flow:')
        for key in standard_cut_keys:
            cumulative_cut_mask *= cut_dict[key]
            print('{:>30}:  {:>5.3}  {:>5.3}'.format(key, np.sum(
                cut_dict[key]) / n_total, np.sum(cumulative_cut_mask) / n_total))
        print('\n')

        return df[selection_mask]

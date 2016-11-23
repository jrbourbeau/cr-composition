#!/usr/bin/env python


from __future__ import division
import numpy as np
import pandas as pd
import time
import glob
import re
from collections import OrderedDict

import composition.support_functions.paths as paths


def load_sim(config='IT73', bintype='logdist', return_cut_dict=False):

    # Load simulation dataframe
    mypaths = paths.Paths()
    infile = '{}/{}_sim/sim_dataframe.hdf5'.format(
        mypaths.comp_data_dir, config, bintype)
    df = pd.read_hdf(infile)

    # Quality Cuts #
    # Adapted from PHYSICAL REVIEW D 88, 042004 (2013)
    cut_dict = OrderedDict()
    # IT specific cuts
    cut_dict['ShowerLLH_reco_exists'] = df['reco_exists']
    cut_dict['MC_zenith'] = (np.cos(df['MC_zenith']) >= 0.8)
    cut_dict['reco_zenith'] = (np.cos(df['reco_zenith']) >= 0.8)
    # cut_dict['reco_zenith'] = (np.cos(np.pi - df['reco_zenith']) >= 0.8)
    cut_dict['IT_containment'] = (df['IceTop_FractionContainment'] < 1.0)
    cut_dict['reco_IT_containment'] = (df['reco_IT_containment'] < 1.0)
    cut_dict['LLHlap_IT_containment'] = (df['LLHlap_IT_containment'] < 1.0)
    cut_dict['IceTopMaxSignalInEdge'] = np.logical_not(
        df['IceTopMaxSignalInEdge'].astype(bool))
    cut_dict['IceTopMaxSignal'] = (df['IceTopMaxSignal'] >= 6)
    cut_dict['IceTopNeighbourMaxSignal'] = (
        df['IceTopNeighbourMaxSignal'] >= 4)
    cut_dict['NStations'] = (df['NStations'] >= 5)
    cut_dict['StationDensity'] = (df['StationDensity'] >= 0.2)
    cut_dict['min_energy'] = (df['reco_energy'] > 10**6.2)
    cut_dict['max_energy'] = (df['reco_energy'] < 10**8.0)

    # InIce specific cuts
    cut_dict['NChannels'] = (df['NChannels'] >= 8)
    cut_dict['InIce_containment'] = (df['InIce_FractionContainment'] < 1.0)
    cut_dict['reco_InIce_containment'] = (df['reco_InIce_containment'] < 1.0)
    cut_dict['LLHlap_InIce_containment'] = (
        df['LLHlap_InIce_containment'] < 1.0)
    cut_dict['max_charge_frac'] = (df['max_charge_frac'] < 0.3)
    
    # Some conbined cuts
    cut_dict['combined_reco_exists'] = df['combined_reco_exists']
    cut_dict['reco_exists'] = cut_dict[
        'ShowerLLH_reco_exists'] & cut_dict['combined_reco_exists']
    cut_dict['num_hits'] = cut_dict['NChannels'] & cut_dict['NStations']
    cut_dict['reco_containment'] = cut_dict[
        'reco_IT_containment'] & cut_dict['LLHlap_InIce_containment']
    # cut_dict['reco_containment'] = cut_dict[
    #     'reco_IT_containment'] & cut_dict['reco_InIce_containment']
    cut_dict['IT_signal'] = cut_dict['IceTopMaxSignalInEdge'] & cut_dict[
        'IceTopMaxSignal'] & cut_dict['IceTopNeighbourMaxSignal']
    cut_dict['energy_range'] = cut_dict['min_energy'] & cut_dict['max_energy']

    # Add log-energy and log-charge columns to df
    df['MC_log_energy'] = np.nan_to_num(np.log10(df['MC_energy']))
    df['reco_log_energy'] = np.nan_to_num(np.log10(df['reco_energy']))
    df['InIce_log_charge'] = np.nan_to_num(np.log10(df['InIce_charge']))
    df['reco_cos_zenith'] = np.cos(np.pi - df['reco_zenith'])
    df['ShowerPlane_cos_zenith'] = np.cos(df['ShowerPlane_zenith'])
    df['log_s125'] = np.log10(df['s125'])

    if return_cut_dict:
        return df, cut_dict
    else:
        selection_mask = np.array([True] * len(df))
        standard_cut_keys = ['reco_exists', 'reco_zenith', 'num_hits', 'IT_signal',
                             'StationDensity', 'reco_containment', 'max_charge_frac', 'energy_range']
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

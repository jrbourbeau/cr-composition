
from __future__ import print_function, division
import os
from functools import wraps, partial
from functools32 import lru_cache
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.preprocessing import LabelEncoder

from .base import get_paths
from .simfunctions import sim_to_thinned, get_sim_configs
from .datafunctions import get_data_configs


def validate_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Expecting a DataFrame, but got {}'.format(type(df)))


def validate_datatype(datatype):
    assert datatype in ['sim', 'data'], 'datatype must be either \'sim\' or \'data\''


def apply_quality_cuts(df, datatype='sim', return_cut_dict=False,
                       dataprocessing=False, verbose=True, log_energy_min=None,
                       log_energy_max=None):

    validate_dataframe(df)
    validate_datatype(datatype)

    # Quality Cuts #
    # Adapted from PHYSICAL REVIEW D 88, 042004 (2013)
    cut_dict = {}
    # MC-level cuts
    if datatype == 'sim':
        cut_dict['FractionContainment_MCPrimary_IceTop'] = (
            df['FractionContainment_MCPrimary_IceTop'] < 1.0)
        cut_dict['FractionContainment_MCPrimary_InIce'] = (
            df['FractionContainment_MCPrimary_InIce'] < 1.0)
        df['MC_log_energy'] = np.nan_to_num(np.log10(df['MC_energy']))
    # IT specific cuts
    cut_dict['passed_IceTopQualityCuts'] = df['passed_IceTopQualityCuts'].astype(bool)
    cut_dict['lap_fitstatus_ok'] = df['lap_fitstatus_ok']
    cut_dict['FractionContainment_Laputop_IceTop'] = df[
        'FractionContainment_Laputop_IceTop'] < 0.96
    cut_dict['lap_beta'] = (df['lap_beta'] < 9.5) & (df['lap_beta'] > 1.4)
    cut_dict['lap_rlogl'] = df['lap_rlogl'] < 2
    cut_dict['IceTopMaxSignalInEdge'] = ~df['IceTopMaxSignalInEdge'].astype(bool)
    cut_dict['IceTopMaxSignal'] = (df['IceTopMaxSignal'] >= 6)
    cut_dict['IceTopNeighbourMaxSignal'] = df['IceTopNeighbourMaxSignal'] >= 4
    cut_dict['NStations'] = df['NStations'] >= 5
    cut_dict['StationDensity'] = df['StationDensity'] >= 0.2

    # Set up min/max energy cuts
    cut_dict['min_energy_lap'] = np.ones(len(df['lap_energy']), dtype=bool)
    cut_dict['max_energy_lap'] = np.ones(len(df['lap_energy']), dtype=bool)
    if log_energy_min is not None:
        cut_dict['min_energy_lap'] = df['lap_energy'] > 10**log_energy_min
    if log_energy_max is not None:
        cut_dict['max_energy_lap'] = df['lap_energy'] < 10**log_energy_max

    # cut_dict['min_energy_lap'] = df['lap_energy'] > 10**6.0
    # cut_dict['max_energy_lap'] = df['lap_energy'] < 10**8.0

    # cut_dict['IceTop_charge_175m'] = np.logical_not(df['IceTop_charge_175m'].isnull())
    # cut_dict['IceTop_charge'] = np.logical_not(df['IceTop_charge'].isnull()) & cut_dict['IceTop_charge_175m']

    # InIce specific cuts
    cut_dict['passed_InIceQualityCuts'] = df['passed_InIceQualityCuts'].astype(bool)
    for cut in ['MilliNCascAbove2', 'MilliQtotRatio', 'MilliRloglBelow2',
                'NCh_CoincLaputopCleanedPulsesAbove7', 'StochRecoSucceeded']:
        cut_dict['passed_{}'.format(cut)] = df['passed_{}'.format(cut)].astype(bool)
    for i in ['1_60']:
    # for i in ['1_60', '1_45', '1_30', '1_15', '1_6', '45_60']:
        cut_dict['NChannels_' + i] = df['NChannels_' + i] >= 8
        cut_dict['max_qfrac_' + i] = df['max_qfrac_' + i] < 0.3
    cut_dict['FractionContainment_Laputop_InIce'] = df['FractionContainment_Laputop_InIce'] < 1.0

    # # Millipede specific cuts
    # cut_dict['mil_rlogl'] = df['mil_rlogl'] < 2.0
    # cut_dict['mil_qtot_ratio'] = df['mil_qtot_predicted']/df['mil_qtot_measured'] > -0.03
    # cut_dict['num_millipede_cascades'] = df['num_millipede_cascades'] >= 3

    # Some conbined cuts
    cut_dict['lap_reco_success'] = cut_dict['lap_fitstatus_ok'] & cut_dict['lap_beta'] & cut_dict['lap_rlogl']
    # cut_dict['num_hits_1_60'] = cut_dict['NChannels_1_60'] & cut_dict['NStations']
    for i in ['1_60']:
    # for i in ['1_60', '1_45', '1_30', '1_15', '1_6', '45_60']:
        cut_dict['num_hits_'+i] = cut_dict['NChannels_'+i] & cut_dict['NStations']
    cut_dict['lap_containment'] = cut_dict[
        'FractionContainment_Laputop_IceTop'] & cut_dict['FractionContainment_Laputop_InIce']
    cut_dict['IT_signal'] = cut_dict['IceTopMaxSignalInEdge'] & cut_dict[
        'IceTopMaxSignal'] & cut_dict['IceTopNeighbourMaxSignal']
    cut_dict['reco_energy_range'] = cut_dict['min_energy_lap'] & cut_dict['max_energy_lap']
    cut_dict['lap_IT_containment'] = cut_dict['FractionContainment_Laputop_IceTop'] & cut_dict['IT_signal']
    # cut_dict['mil_reco_success'] = cut_dict['mil_rlogl'] & cut_dict[
    #     'mil_qtot_ratio'] & cut_dict['num_millipede_cascades']

    if return_cut_dict:
        print('Returning without applying quality cuts')
        return df, cut_dict
    else:
        selection_mask = np.array([True] * len(df))
        if dataprocessing:
            standard_cut_keys = ['passed_IceTopQualityCuts', 'FractionContainment_Laputop_InIce',
                # 'num_hits_1_60', 'max_qfrac_1_60']
                'reco_energy_range', 'num_hits_1_60', 'max_qfrac_1_60']
        else:
            standard_cut_keys = ['passed_IceTopQualityCuts', 'FractionContainment_Laputop_InIce',
                # 'passed_InIceQualityCuts', 'num_hits_1_60']
                'passed_InIceQualityCuts', 'num_hits_1_60', 'reco_energy_range']
        for key in standard_cut_keys:
            selection_mask *= cut_dict[key]
        # Print cut event flow
        if verbose:
            n_total = len(df)
            print('Starting out with {} {} events'.format(n_total, datatype))
            cut_eff = {}
            cumulative_cut_mask = np.array([True] * n_total)
            print('{} quality cut event flow:'.format(datatype))
            for key in standard_cut_keys:
                cumulative_cut_mask *= cut_dict[key]
                print('{:>30}:  {:>5.3}  {:>5.3}'.format(key, np.sum(
                    cut_dict[key]) / n_total, np.sum(cumulative_cut_mask) / n_total))
            print('\n')

        df_cut = df[selection_mask].reset_index(drop=True)
        # df_cut = df_cut.reset_index(drop=True)

        return df_cut


def add_convenience_variables(df):
    validate_dataframe(df)

    # Add log-scale columns to df
    df['lap_log_energy'] = np.nan_to_num(np.log10(df['lap_energy']))
    # df['InIce_log_charge_1_60'] = np.nan_to_num(np.log10(df['InIce_charge_1_60']))
    for i in ['1_60']:
    # for i in ['1_60', '1_45', '1_30', '1_15', '1_6', '45_60']:
        # df['InIce_log_charge_'+i] = np.nan_to_num(np.log10(df['InIce_charge_'+i]))
        df['InIce_log_charge_'+i] = np.log10(df['InIce_charge_'+i])
        df['log_NChannels_'+i] = np.log10(df['NChannels_'+i])
        df['log_NHits_'+i] = np.log10(df['NHits_'+i])
    df['lap_cos_zenith'] = np.cos(df['lap_zenith'])
    for dist in ['50', '80', '125', '180', '250', '500']:
        df['log_s'+dist] = np.log10(df['lap_s'+dist])
    df['log_dEdX'] = np.log10(df['eloss_1500_standard'])
    df['log_d4r_peak_energy'] = np.log10(df['d4r_peak_energy'])
    df['log_d4r_peak_sigma'] = np.log10(df['d4r_peak_sigma'])

    # df['log_IceTop_charge'] = np.log10(df['IceTop_charge'])
    # df['log_IceTop_charge_175m'] = np.log10(df['IceTop_charge_175m'])
    # df['IT_charge_ratio'] = df['IceTop_charge_175m']/df['IceTop_charge']
    # df['charge_ratio'] = df['InIce_charge_1_60']/df['IceTop_charge']

    # Add ratio of features (could help improve RF classification)
    # df['charge_nchannels_ratio'] = df['InIce_charge_1_30'] / df['NChannels_1_30']
    # df['charge_nhits_ratio'] = df['InIce_charge_1_30'] / df['NHits_1_30']
    # df['nhits_nchannels_ratio'] =  df['NHits_1_30'] / df['NChannels_1_30']
    # df['stationdensity_charge_ratio'] = df[
    #     'StationDensity'] / df['InIce_charge_1_30']
    # df['stationdensity_nchannels_ratio'] = df[
    #     'StationDensity'] / df['NChannels_1_30']
    # df['stationdensity_nhits_ratio'] = df['StationDensity'] / df['NHits_1_30']

    return df


def load_dataframe(df_file=None, datatype='sim', config='IC79', test_size=0.3,
                   comp_key='MC_comp_class', columns=None, verbose=False,
                   log_energy_min=6.0, log_energy_max=8.0):
    '''Loads pandas DataFrame object with appropreiate information

    Parameters
    ----------
    df_file : path, optional
        If specified, the given path to a pandas.DataFrame will be loaded
        (default is None, so the file path will be determined from the
        datatype and config).
    datatype : {'sim', 'data'}
        Specifies whether to load a simulation or experimental data
        DataFrame.
    config : str, optional
        Detector configuration (default is 'IC79').
    test_size : float, optional
        Fraction of DataFrame to split of into a seperate testing set
        (default is 0.3). Note: if datatype is 'data', then test_size will
        be set to 0.0.
    comp_key : {'MC_comp_class', 'MC_comp'}
        Option to use the true composition, or composition classes
        (light and heavy) as the target variable (default is
        'MC_comp_class').
    verbose : bool, optional
        Option for verbose output (default is True).

    Returns
    -------
    pandas.DataFrame, tuple
        Return a single DataFrame if test_size is 0, otherwise return
        a 2-tuple of training and testing DataFrame.

    '''
    # Validate user input
    validate_datatype(datatype)
    if datatype == 'data':
        test_size = 0.0
    if datatype == 'sim' and not comp_key:
        raise ValueError('Must specify a comp_key variable for simulation data')
    if datatype == 'sim' and not config in get_sim_configs():
        raise ValueError('config must be in {}'.format(get_sim_configs()))
    if datatype == 'data' and not config in get_data_configs():
        raise ValueError('config must be in {}'.format(get_data_configs()))

    # Load simulation dataframe
    paths = get_paths()
    # If df_file is not specified, use default path
    if df_file is None:
        df_file = os.path.join(paths.comp_data_dir,
                               '{}_{}'.format(config, datatype),
                               '{}_dataframe.hdf5'.format(datatype))

    with pd.HDFStore(df_file, mode='r') as store:
        df = store.select('dataframe')

    df = (df.pipe(apply_quality_cuts, datatype, log_energy_min=log_energy_min,
                  log_energy_max=log_energy_max, verbose=verbose)
            .pipe(add_convenience_variables)
         )

    if datatype == 'sim':
        df['target'] = df[comp_key].apply(comp_to_label)
        df['is_thinned'] = df['sim'].apply(sim_to_thinned)

    # If specified, split into training and testing DataFrames
    if test_size:
        # If target variable is specified, perform a stratified split, otherwise a shuffle split
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=2)
        train_mask, test_mask = next(splitter.split(df, df['target']))

        train_df = (df.iloc[train_mask]
                      .reset_index(drop=True))
        test_df = (df.iloc[test_mask]
                     .reset_index(drop=True))

        output = train_df, test_df
    else:
        output = df

    return output


# Define convenience functions for loading simulation and data DataFrames
# By default __module__ is undefined, and __doc__ isn't meaningful for partials
load_sim = partial(load_dataframe, datatype='sim')
load_sim.__doc__ = load_dataframe.__doc__
load_sim.__module__ = load_dataframe.__module__

load_data = partial(load_dataframe, datatype='data')
load_data.__doc__ = load_dataframe.__doc__
load_data.__module__ = load_dataframe.__module__


def dataframe_to_array(df, columns, drop_null=True):

    validate_dataframe(df)
    if not isinstance(columns, (list, tuple, np.ndarray, str)):
        raise ValueError('columns must be a string or array-like')

    # Select desired columns from DataFrame
    df = df.loc[:, columns]
    # If specified, drop rows that contain a null value
    if drop_null:
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    array = df.values

    return array


def dataframe_to_X_y(df, feature_list, drop_null=True):

    validate_dataframe(df)

    X = dataframe_to_array(df, feature_list, drop_null=drop_null)
    y = dataframe_to_array(df, 'target', drop_null=drop_null)

    return X, y


comp_to_label_dict = {'light': 0, 'heavy': 1}

def comp_to_label(composition):
    try:
        return comp_to_label_dict[composition]
    except KeyError:
        raise KeyError('Incorrect composition ({}) entered'.format(composition))

def label_to_comp(label):
    label_to_comp_dict = {value: key for key, value in comp_to_label_dict.iteritems()}
    try:
        return label_to_comp_dict[label]
    except KeyError:
        raise KeyError('Incorrect label ({}) entered'.format(label))

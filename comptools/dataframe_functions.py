
from __future__ import print_function, division
from collections import OrderedDict
import os
from functools import wraps, partial
import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

from .base import get_paths
from .simfunctions import sim_to_thinned, get_sim_configs
from .datafunctions import get_data_configs
from .composition_encoding import (composition_group_labels,
                                   encode_composition_groups, comp_to_label,
                                   label_to_comp)


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
    # IT specific cuts
    cut_dict['passed_IceTopQualityCuts'] = df['passed_IceTopQualityCuts'].astype(bool)
    # cut_dict['lap_fitstatus_ok'] = df['lap_fitstatus_ok']
    # cut_dict['FractionContainment_Laputop_IceTop'] = df[
    #     'FractionContainment_Laputop_IceTop'] < 0.96
    # cut_dict['lap_beta'] = (df['lap_beta'] < 9.5) & (df['lap_beta'] > 1.4)
    # cut_dict['lap_rlogl'] = df['lap_rlogl'] < 2
    # cut_dict['IceTopMaxSignalInEdge'] = ~df['IceTopMaxSignalInEdge'].astype(bool)
    # cut_dict['IceTopMaxSignal'] = (df['IceTopMaxSignal'] >= 6)
    # cut_dict['IceTopNeighbourMaxSignal'] = df['IceTopNeighbourMaxSignal'] >= 4
    cut_dict['NStations'] = df['NStations'] >= 5
    # cut_dict['StationDensity'] = df['StationDensity'] >= 0.2

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
    cut_dict['eloss_positive'] = df['eloss_1500_standard'] > 0
    cut_dict['passed_InIceQualityCuts'] = df['passed_InIceQualityCuts'].astype(bool) & cut_dict['eloss_positive']
    # for cut in ['MilliNCascAbove2', 'MilliQtotRatio', 'MilliRloglBelow2',
    #             'NCh_CoincLaputopCleanedPulsesAbove7', 'StochRecoSucceeded']:
    #     cut_dict['passed_{}'.format(cut)] = df['passed_{}'.format(cut)].astype(bool)
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
    # cut_dict['lap_reco_success'] = cut_dict['lap_fitstatus_ok'] & cut_dict['lap_beta'] & cut_dict['lap_rlogl']
    # cut_dict['num_hits_1_60'] = cut_dict['NChannels_1_60'] & cut_dict['NStations']
    for i in ['1_60']:
    # for i in ['1_60', '1_45', '1_30', '1_15', '1_6', '45_60']:
        cut_dict['num_hits_'+i] = cut_dict['NChannels_'+i] & cut_dict['NStations']
    # cut_dict['lap_containment'] = cut_dict[
    #     'FractionContainment_Laputop_IceTop'] & cut_dict['FractionContainment_Laputop_InIce']
    # cut_dict['IT_signal'] = cut_dict['IceTopMaxSignalInEdge'] & cut_dict[
    #     'IceTopMaxSignal'] & cut_dict['IceTopNeighbourMaxSignal']

    cut_dict['reco_energy_range'] = cut_dict['min_energy_lap'] & cut_dict['max_energy_lap']
    # cut_dict['lap_IT_containment'] = cut_dict['FractionContainment_Laputop_IceTop'] & cut_dict['IT_signal']
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

        df_cut = df[selection_mask]
        # df_cut = df[selection_mask].reset_index(drop=True)

        return df_cut


def add_convenience_variables(df, datatype='sim'):
    validate_dataframe(df)

    if datatype == 'sim':
        df['MC_log_energy'] = np.nan_to_num(np.log10(df['MC_energy']))
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


def _load_basic_dataframe(df_file=None, datatype='sim', config='IC86.2012',
                          energy_key='reco_log_energy', columns=None,
                          verbose=False, log_energy_min=None,
                          log_energy_max=None):

    validate_datatype(datatype)
    # If df_file is not specified, use default path
    if df_file is None:
        paths = get_paths()
        df_file = os.path.join(paths.comp_data_dir,
                               '{}_{}'.format(config, datatype),
                               '{}_dataframe.hdf5'.format(datatype))
    if not os.path.exists(df_file):
        raise IOError('The DataFrame file {} doesn\'t exist'.format(df_file))

    # If specified, construct energy selection string
    with pd.HDFStore(df_file, mode='r') as store:
        df = store.select('dataframe', columns=columns)

    model_dict = load_trained_model('RF_energy_{}'.format(config))
    pipeline = model_dict['pipeline']
    feature_list = list(model_dict['training_features'])
    df['reco_log_energy'] = pipeline.predict(df[feature_list])

    energy_mask = np.ones_like(df[energy_key].values, dtype=bool)
    if log_energy_min is not None:
        energy_mask = energy_mask & (df[energy_key] > log_energy_min)
    if log_energy_max is not None:
        energy_mask = energy_mask & (df[energy_key] < log_energy_max)

    return df.loc[energy_mask, :]


def load_sim(df_file=None, config='IC86.2012', test_size=0.3, num_groups=2,
             columns=None, energy_key='reco_log_energy', log_energy_min=6.0,
             log_energy_max=8.0, verbose=False):
    '''Function to load processed simulation DataFrame

    Parameters
    ----------
    df_file : path, optional
        If specified, the given path to a pandas.DataFrame will be loaded
        (default is None, so the file path will be determined from the
        datatype and config).
    config : str, optional
        Detector configuration (default is 'IC86.2012').
    test_size : int, float, optional
        Fraction or number of events to be split off into a seperate testing
        set (default is 0.3). test_size will be passed to
        sklearn.model_selection.StratifiedShuffleSplit.
    energy_key : str, optional
        Energy key to apply cuts to (default is 'lap_log_energy').
    log_energy_min : int, float, optional
        Option to set a lower limit on the reconstructed log energy in GeV
        (default is 6.0).
    log_energy_max : int, float, optional
        Option to set a upper limit on the reconstructed log energy in GeV
        (default is 8.0).
    verbose : bool, optional
        Option for verbose output (default is True).

    Returns
    -------
    pandas.DataFrame, tuple of pandas.DataFrame
        Return a single DataFrame if test_size is 0, otherwise return
        a 2-tuple of training and testing DataFrame.

    '''

    if not config in get_sim_configs():
        raise ValueError('config must be in {}'.format(get_sim_configs()))
    if not isinstance(test_size, (int, float)):
        raise TypeError('test_size must be a floating-point number')

    df = _load_basic_dataframe(df_file=df_file, datatype='sim', config=config,
                               energy_key=energy_key, columns=columns,
                               log_energy_min=log_energy_min,
                               log_energy_max=log_energy_max, verbose=verbose)

    # If specified, split into training and testing DataFrames
    if test_size > 0:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                          random_state=2)
        split_gen = splitter.split(df, df['comp_target_{}'.format(num_groups)])
        train_mask, test_mask = next(split_gen)
        train_df = df.iloc[train_mask]
        test_df = df.iloc[test_mask]
        output = train_df, test_df
    else:
        output = df

    return output


def load_data(df_file=None, config='IC86.2012', columns=None,
              energy_key='reco_log_energy', log_energy_min=6.0,
              log_energy_max=8.0, verbose=False):
    '''Function to load processed data DataFrame

    Parameters
    ----------
    df_file : path, optional
        If specified, the given path to a pandas.DataFrame will be loaded
        (default is None, so the file path will be determined from the
        datatype and config).
    config : str, optional
        Detector configuration (default is 'IC86.2012').
    comp_key : {'MC_comp_class', 'MC_comp'}
        Option to use the true composition, or composition classes
        (light and heavy) as the target variable (default is
        'MC_comp_class').
    log_energy_min : int, float, optional
        Option to set a lower limit on the reconstructed log energy in GeV
        (default is 6.0).
    log_energy_max : int, float, optional
        Option to set a upper limit on the reconstructed log energy in GeV
        (default is 8.0).
    verbose : bool, optional
        Option for verbose output (default is True).

    Returns
    -------
    pandas.DataFrame
        Return a DataFrame with processed data

    '''

    if not config in get_data_configs():
        raise ValueError('config must be in {}'.format(get_data_configs()))

    df = _load_basic_dataframe(df_file=df_file, datatype='data', config=config,
                               energy_key=energy_key, columns=columns,
                               log_energy_min=log_energy_min,
                               log_energy_max=log_energy_max, verbose=verbose)

    return df


def load_tank_charges(config='IC79.2010', datatype='sim', return_dask=False):
    paths = get_paths()
    file_pattern = os.path.join(paths.comp_data_dir,
                           '{}_{}'.format(config, datatype),
                           'dataframe_files',
                           'dataframe_*.hdf5')
    tank_charges = dd.read_hdf(file_pattern, 'tank_charges')

    if return_dask:
        return tank_charges
    else:
        return tank_charges.compute()


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


def dataframe_to_X_y(df, feature_list, target='comp_target_2', drop_null=True):

    validate_dataframe(df)

    X = dataframe_to_array(df, feature_list, drop_null=drop_null)
    y = dataframe_to_array(df, target, drop_null=drop_null)

    return X, y


def load_trained_model(pipeline_str='BDT'):
    """Function to load pre-trained model to avoid re-training

    Parameters
    ----------
    pipeline_str : str, optional
        Name of model to load (default is 'BDT').

    Returns
    -------
    model_dict : dict
        Dictionary containing trained model as well as relevant metadata.

    """
    paths = get_paths()
    model_file = os.path.join(paths.project_root, 'models',
                              '{}.pkl'.format(pipeline_str))
    if not os.path.exists(model_file):
        raise IOError('There is no saved model file {}'.format(model_file))

    model_dict = joblib.load(model_file)

    return model_dict

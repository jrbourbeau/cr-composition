
from __future__ import print_function, division
import os
import glob
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
import dask.dataframe as dd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from .base import get_config_paths
from .simfunctions import get_sim_configs
from .datafunctions import get_data_configs


def validate_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Expecting a DataFrame, but got {}'.format(type(df)))


def validate_datatype(datatype):
    assert datatype in ['sim', 'data'], 'datatype must be either \'sim\' or \'data\''


def apply_quality_cuts(df, datatype='sim', return_cut_dict=False, verbose=True):

    validate_dataframe(df)
    validate_datatype(datatype)

    # Quality Cuts #
    # Adapted from PHYSICAL REVIEW D 88, 042004 (2013)
    cut_dict = {}
    # IT specific cuts
    cut_dict['passed_IceTopQualityCuts'] = df['passed_IceTopQualityCuts'].astype(bool)
    cut_dict['NStations'] = df['NStations'] >= 5

    # InIce specific cuts
    cut_dict['eloss_positive'] = df['eloss_1500_standard'] > 0
    cut_dict['passed_InIceQualityCuts'] = df['passed_InIceQualityCuts'].astype(bool) & cut_dict['eloss_positive']
    for i in ['1_60']:
        cut_dict['NChannels_' + i] = df['NChannels_' + i] >= 8
        cut_dict['max_qfrac_' + i] = df['max_qfrac_' + i] < 0.3
    cut_dict['FractionContainment_Laputop_InIce'] = df['FractionContainment_Laputop_InIce'] < 1.0

    for i in ['1_60']:
        cut_dict['num_hits_'+i] = cut_dict['NChannels_'+i] & cut_dict['NStations']

    if return_cut_dict:
        print('Returning without applying quality cuts')
        return df, cut_dict
    else:
        selection_mask = np.ones(len(df), dtype=bool)
        standard_cut_keys = ['passed_IceTopQualityCuts',
                             'passed_InIceQualityCuts',
                             'FractionContainment_Laputop_InIce',
                             'num_hits_1_60',
                             ]
        for key in standard_cut_keys:
            selection_mask *= cut_dict[key]
        # Print cut event flow
        if verbose:
            n_total = len(df)
            print('Starting out with {} {} events'.format(n_total, datatype))
            cumulative_cut_mask = np.array([True] * n_total)
            print('{} quality cut event flow:'.format(datatype))
            for key in standard_cut_keys:
                cumulative_cut_mask *= cut_dict[key]
                print('{:>30}:  {:>5.3}  {:>5.3}'.format(key, np.sum(
                    cut_dict[key]) / n_total, np.sum(cumulative_cut_mask) / n_total))
            print('\n')

        # df_cut = df[selection_mask]
        df_cut = df.loc[selection_mask, :].reset_index(drop=True)

        return df_cut


def add_convenience_variables(df, datatype='sim'):
    validate_dataframe(df)

    if datatype == 'sim':
        df['MC_log_energy'] = np.nan_to_num(np.log10(df['MC_energy']))
    # Add log-scale columns to df
    df['lap_log_energy'] = np.nan_to_num(np.log10(df['lap_energy']))
    # df['InIce_log_charge_1_60'] = np.nan_to_num(np.log10(df['InIce_charge_1_60']))
    for i in ['1_60']:
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
                          energy_reco=True, energy_cut_key='reco_log_energy',
                          log_energy_min=None, log_energy_max=None,
                          columns=None, n_jobs=1, verbose=False,
                          compute=True):

    validate_datatype(datatype)

    if df_file is not None:
        files = df_file
    else:
        paths = get_config_paths()
        file_pattern = os.path.join(paths.comp_data_dir,
                                    config,
                                    datatype,
                                    'processed_hdf',
                                    'nominal' if datatype == 'sim' else '',
                                    '*.hdf')
        files = sorted(glob.glob(file_pattern))

    ddf = dd.read_hdf(files,
                      key='dataframe',
                      mode='r',
                      columns=columns,
                      chunksize=10000)

    # Energy reconstruction
    if energy_reco:
        model_dict = load_trained_model('linearregression_energy_{}'.format(config),
                                        return_metadata=True)
        pipeline = model_dict['pipeline']
        feature_list = list(model_dict['training_features'])

        def add_reco_energy(partition):
            partition['reco_log_energy'] = pipeline.predict(partition[feature_list])
            partition['reco_energy'] = 10**partition['reco_log_energy']
            return partition
        ddf = ddf.map_partitions(add_reco_energy)

    # Energy range cut
    if log_energy_min is not None and log_energy_max is not None:
        def apply_energy_cut(partition):
            energy_mask = (partition[energy_cut_key] > log_energy_min) & (partition[energy_cut_key] < log_energy_max)
            return partition.loc[energy_mask, :]

        ddf = ddf.map_partitions(apply_energy_cut)

    if compute:
        if verbose:
            pbar = ProgressBar()
            pbar.register()
        scheduler = 'processes' if n_jobs > 1 else 'synchronous'
        df = ddf.compute(scheduler=scheduler, num_workers=n_jobs)
        df = df.reset_index(drop=True)
    else:
        df = ddf

    return df


_load_parameters_docstring = """Parameters
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
        sklearn.model_selection.ShuffleSplit.
    energy_reco : bool, optional
        Option to perform energy reconstruction for each event
        (default is True).
    energy_cut_key : str, optional
        Energy key to apply energy range cuts to (default is 'lap_log_energy').
    log_energy_min : int, float, optional
        Option to set a lower limit on the reconstructed log energy in GeV
        (default is 6.0).
    log_energy_max : int, float, optional
        Option to set a upper limit on the reconstructed log energy in GeV
        (default is 8.0).
    columns : array_like, optional
        Option to specify the columns that should be in the returned
        DataFrame(s) (default is None, all columns are returned).
    n_jobs : int, optional
        Number of chunks to load in parallel (default is 1).
    verbose : bool, optional
        Option for verbose progress bar output (default is True)."""


def load_sim(df_file=None, config='IC86.2012', test_size=0.5,
             energy_reco=True, energy_cut_key='reco_log_energy',
             log_energy_min=6.0, log_energy_max=8.0, columns=None, n_jobs=1,
             verbose=False, compute=True):

    if config not in get_sim_configs():
        raise ValueError('config must be in {}'.format(get_sim_configs()))
    if not isinstance(test_size, (int, float)):
        raise TypeError('test_size must be a floating-point number')

    df = _load_basic_dataframe(df_file=df_file,
                               datatype='sim',
                               config=config,
                               energy_reco=energy_reco,
                               energy_cut_key=energy_cut_key,
                               columns=columns,
                               log_energy_min=log_energy_min,
                               log_energy_max=log_energy_max,
                               n_jobs=n_jobs,
                               verbose=verbose,
                               compute=compute)

    # If specified, split into training and testing DataFrames
    if test_size > 0:
        output = train_test_split(df, test_size=test_size, shuffle=True,
                                  random_state=2)

    else:
        output = df

    return output

load_sim.__doc__ = """ Function to load processed simulation DataFrame

    {_load_parameters_docstring}

    Returns
    -------
    pandas.DataFrame, tuple of pandas.DataFrame
        Return a single DataFrame if test_size is 0, otherwise return
        a 2-tuple of training and testing DataFrame.
    """.format(_load_parameters_docstring=_load_parameters_docstring)


def load_data(df_file=None, config='IC86.2012', energy_reco=True,
              energy_cut_key='reco_log_energy', log_energy_min=6.0,
              log_energy_max=8.0, columns=None, n_jobs=1, verbose=False,
              compute=True, processed=True):

    if config not in get_data_configs():
        raise ValueError('config must be in {}'.format(get_data_configs()))

    if processed:
        # Load processed dataset with quality cuts already applied
        paths = get_config_paths()
        data_file = os.path.join(paths.comp_data_dir,
                                 config,
                                 'data',
                                 'data_dataframe_quality_cuts.hdf'
                                 )

        ddf = dd.read_hdf(data_file,
                          key='dataframe',
                          mode='r',
                          columns=columns,
                          chunksize=100000)
        scheduler = 'synchronous'
        if verbose:
            with ProgressBar():
                df = ddf.compute(scheduler=scheduler, num_workers=n_jobs)
        else:
            df = ddf.compute(scheduler=scheduler, num_workers=n_jobs)
    else:
        print('FYI: Loading non-processed dataset. This takes longer than '
              'loading the processed dataset...')
        df = _load_basic_dataframe(df_file=df_file,
                                   datatype='data',
                                   config=config,
                                   energy_reco=energy_reco,
                                   energy_cut_key=energy_cut_key,
                                   columns=columns,
                                   log_energy_min=log_energy_min,
                                   log_energy_max=log_energy_max,
                                   n_jobs=n_jobs,
                                   verbose=verbose,
                                   compute=compute)

    return df


load_data.__doc__ = """  Function to load processed data DataFrame

    {_load_parameters_docstring}
    processed : bool, optional
        Whether to load processed (quality + energy cuts applied) or
        pre-processed data (default is True).

    Returns
    -------
    pandas.DataFrame
        Return a DataFrame with processed data
    """.format(_load_parameters_docstring=_load_parameters_docstring)


def load_tank_charges(config='IC79.2010', datatype='sim', return_dask=False):
    paths = get_config_paths()
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


def load_trained_model(pipeline_str='BDT', config='IC86.2012',
                       return_metadata=False):
    """ Function to load pre-trained model to avoid re-training

    Parameters
    ----------
    pipeline_str : str, optional
        Name of model to load (default is 'BDT').
    config : str, optional
        Detector configuration (default is 'IC86.2012').
    return_metadata : bool, optional
        Option to return metadata associated with saved model (e.g. list of
        training features used, scikit-learn version, etc) (default is False).

    Returns
    -------
    pipeline : sklearn.Pipeline
        Trained scikit-learn pipeline.
    model_dict : dict
        Dictionary containing trained model as well as relevant metadata.
    """

    paths = get_config_paths()
    model_file = os.path.join(paths.comp_data_dir,
                              config,
                              'models',
                              '{}.pkl'.format(pipeline_str))
    if not os.path.exists(model_file):
        raise IOError('There is no saved model file {}'.format(model_file))

    model_dict = joblib.load(model_file)

    if return_metadata:
        return model_dict
    else:
        return model_dict['pipeline']

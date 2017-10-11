#!/usr/bin/env python

import os
import glob
import pandas as pd
import numpy as np


def get_data_configs():
    return ['IC79.2010', 'IC86.2011', 'IC86.2012', 'IC86.2013', 'IC86.2014',
            'IC86.2015']


def _get_data_path_prefix(config=None):
    if config is None:
        raise ValueError('Detector configuration not specified...')
    # elif config == 'IC79.2010':
    #     prefix = '/data/ana/CosmicRay/IceTop_level3/exp/v1/{}/'.format(config)
    # else:
    prefix = '/data/ana/CosmicRay/IceTop_level3/exp/{}/'.format(config)

    return prefix


def get_run_generator(config=None):

    prefix = _get_data_path_prefix(config=config)
    file_pattern = os.path.join(prefix, '????/????/Run*')
    run_gen = (os.path.basename(f).replace('Run', '') for f in glob.iglob(file_pattern))

    return run_gen


def get_run_list(config=None):
    return list(get_run_generator(config))


def get_level3_run_i3_files(config=None, run=None):

    if not config in get_data_configs():
        raise ValueError('Invalid configuration, {}'.format(config))

    prefix = _get_data_path_prefix(config=config)
    data_file_pattern = os.path.join(prefix,
        '????/????/Run{}/Level3_{}_data_Run{}_Subrun*.i3.bz2'.format(run, config, run))
    run_files = glob.glob(data_file_pattern)

    # Extract (and validate) the GCD file for this run
    GCD_file_pattern = os.path.join(prefix, '????/????/Run{}/*GCD*'.format(run, config, run))
    GCD_files = glob.glob(GCD_file_pattern)
    if len(GCD_files) != 1:
        raise ValueError('Should have found only a single GCD files for run '
                         '{} in {}, but found {}'.format(run, config, len(GCD_files)))
    else:
        GCD_file = GCD_files[0]

    return GCD_file, run_files


def get_level3_livetime_hist(config=None, month=None):

    if not isinstance(month, int):
        raise ValueError('Month must be an integer, got {}'.format(month))

    prefix = _get_data_path_prefix(config=config)
    file_pattern = os.path.join(prefix, '????/{:02d}??/Run*/*livetime.pickle'.format(month))
    file_gen = glob.iglob(file_pattern)

    bin_values = []
    for livetime_pickle in file_gen:
        df = pd.read_pickle(livetime_pickle)
        prod_hist = df['Livetime']
        bin_values.append(prod_hist.bin_values)

    bin_values = np.asarray(bin_values, dtype=int)
    summed_counts = np.sum(bin_values, axis=0)

    return summed_counts


def reco_pulses():

    IT_pulses = 'IceTopHLCSeedRTPulses'
    inice_pulses = 'SRTCoincPulses'

    return IT_pulses, inice_pulses


def null_stream(config):
    nullstream = {}
    nullstream['IT73'] = 'nullsplit'
    nullstream['IT81'] = 'in_ice'
    return nullstream[config]


def it_stream(config):

    itstream = {}
    itstream[config] = 'ice_top'

    return itstream[config]

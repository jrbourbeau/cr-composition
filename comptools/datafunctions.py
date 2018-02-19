
import os
import glob
from itertools import islice
import pandas as pd
import numpy as np

from .base import partition


def get_data_configs():
    return ['IC79.2010', 'IC86.2011', 'IC86.2012', 'IC86.2013', 'IC86.2014',
            'IC86.2015']


def _get_data_path_prefix(config=None):
    if config is None:
        raise ValueError('Detector configuration not specified...')
    prefix = '/data/ana/CosmicRay/IceTop_level3/exp/{}/'.format(config)

    return prefix


def run_generator(config=None):

    prefix = _get_data_path_prefix(config=config)
    file_pattern = os.path.join(prefix, '????/????/Run*')
    run_gen = (os.path.basename(f).replace('Run', '') for f in glob.iglob(file_pattern))

    return run_gen


def get_run_list(config=None):
    return list(run_generator(config))


def level3_data_files(config=None, run=None):

    if config not in get_data_configs():
        raise ValueError('Invalid configuration, {}'.format(config))

    prefix = _get_data_path_prefix(config=config)
    data_file_pattern = os.path.join(
                            prefix,
                            '????/????/Run{}/Level3_{}_data_Run{}_Subrun*.i3.bz2'.format(run, config, run))
    files = glob.glob(data_file_pattern)

    return files


def level3_data_file_batches(config, run, size, max_batches=None):
    """Generates level3 data file paths in batches

    Parameters
    ----------
    config : str
        Detector configuration
    run : str
        Number of data taking run
    size: int
        Number of files in each batch
    max_batches : int, optional
        Option to only yield ``max_batches`` number of file batches (default
        is to yield all batches)

    Returns
    -------
    generator
        Generator that yields batches of data files

    Examples
    --------
    Basic usage:

    >>> from comptools.datafunctions import level3_data_file_batches
    >>> list(level3_data_file_batches(config='IC86.2012', run='00122174', size=3, max_batches=2))
    [('/data/ana/CosmicRay/IceTop_level3/exp/IC86.2012/2013/0413/Run00122174/Level3_IC86.2012_data_Run00122174_Subrun00000050.i3.bz2',
      '/data/ana/CosmicRay/IceTop_level3/exp/IC86.2012/2013/0413/Run00122174/Level3_IC86.2012_data_Run00122174_Subrun00000110.i3.bz2',
      '/data/ana/CosmicRay/IceTop_level3/exp/IC86.2012/2013/0413/Run00122174/Level3_IC86.2012_data_Run00122174_Subrun00000000.i3.bz2'),
     ('/data/ana/CosmicRay/IceTop_level3/exp/IC86.2012/2013/0413/Run00122174/Level3_IC86.2012_data_Run00122174_Subrun00000330.i3.bz2',
      '/data/ana/CosmicRay/IceTop_level3/exp/IC86.2012/2013/0413/Run00122174/Level3_IC86.2012_data_Run00122174_Subrun00000150.i3.bz2',
      '/data/ana/CosmicRay/IceTop_level3/exp/IC86.2012/2013/0413/Run00122174/Level3_IC86.2012_data_Run00122174_Subrun00000240.i3.bz2')]

    """

    prefix = _get_data_path_prefix(config=config)
    data_file_pattern = os.path.join(prefix, '????', '????',
                                     'Run{}'.format(run),
                                     'Level3_{}_data_Run{}_Subrun*.i3.bz2'.format(config, run))
    files_iter = glob.iglob(data_file_pattern)

    return partition(files_iter, size=size, max_batches=max_batches)


def level3_data_GCD_file(config, run):

    if config not in get_data_configs():
        raise ValueError('Invalid configuration, {}'.format(config))

    prefix = _get_data_path_prefix(config=config)
    gcd_file_pattern = os.path.join(prefix, '????/????/Run{}/*GCD*'.format(run))
    # Don't want to make any assumptions about how many files will be found
    # when globing --> use iglob + next()
    gcd_gen = glob.iglob(gcd_file_pattern)
    try:
        gcd_file = next(gcd_gen)
    except StopIteration:
        raise ValueError('Could not find a GCD file for {} run {}'.format(config, run))
    try:
        second_gcd_file = next(gcd_gen)
        raise ValueError('Found more than one GCD file for {} run {}'.format(config, run))
    except StopIteration:
        pass

    return gcd_file


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

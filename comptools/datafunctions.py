#!/usr/bin/env python

import glob
from sklearn.utils import shuffle


def get_level3_data_files(month=None, year=None, config=None):

    if config is None:
        raise('Detector configuration not specified...')
    prefix = '/data/ana/CosmicRay/IceTop_level3/exp/v1/{}/'.format(config)

    if (month is None) and (year is None):
        files = glob.glob(prefix + '??_????/Level3_*.i3.gz')
    else:
        files = glob.glob(prefix + '{}_{}/Level3_*.i3.gz'.format(month, year))

    # Don't forget to sort files
    files = sorted(files)

    return files


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

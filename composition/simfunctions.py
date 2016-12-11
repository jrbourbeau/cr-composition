#!/usr/bin/env python

import glob
from sklearn.utils import shuffle


def getSimDict():

    # Simulation info for various detector configurations and compositions
    s = {}
    s['IT81'] = {'P': ['9166'], 'Fe': ['9165']}
    s['IT73'] = {}
    # Note - need to look into the lowE simulation. Appears to be more.
    s['IT73']['P'] = ['7351', '7006', '7579']
    s['IT73']['He'] = ['7483', '7241', '7263', '7791']
    s['IT73']['O'] = ['7486', '7242', '7262', '7851']
    s['IT73']['Fe'] = ['7394', '7007', '7784']

    return s


def getSimOutput():

    # Custom help output for argparse
    s = getSimDict()
    simOutput = '\nIceTop simulation datasets:\n'
    for config in sorted(s.keys()):
        simOutput += '  %s\n' % config
        for comp in sorted(s[config].keys()):
            simOutput += '  %4s : %s\n' % (comp, s[config][comp])

    return simOutput


def getErange(inverted=False):
    e = {}
    e['low'] = ['7351', '7483', '7486', '7394']
    # e['mid'] = ['7006', '7007']
    e['mid'] = ['9166', '9165', '7006', '7007', '7241', '7263', '7242', '7262']
    # e['high'] = ['7579', '7784']
    e['high'] = ['7579', '7791', '7851', '7784']
    if inverted:
        e = {v: k for k in e for v in e[k]}

    return e


def get_energy_range_to_sim(inverted=False):
    e = {}
    e['low'] = ['7351', '7483', '7486', '7394']
    # e['mid'] = ['7006', '7007']
    e['mid'] = ['9166', '9165', '7006', '7007', '7241', '7263', '7242', '7262']
    # e['high'] = ['7579', '7784']
    e['high'] = ['7579', '7791', '7851', '7784']
    if inverted:
        e = {v: k for k in e for v in e[k]}

    return e


def sim2cfg(sim):

    s = getSimDict()
    inverted_dict = {v: k for k in s for k2 in s[k] for v in s[k][k2]}
    return inverted_dict[sim]


# def sim2comp(sim, full=False, convert=False):
#
#     s = getSimDict()
#     converter = {'P': 'p', 'He': 'h', 'O': 'o', 'Fe': 'f'}
#     if convert:
#         inverted_dict = {v: converter[k]
#                          for k2 in s for k in s[k2] for v in s[k2][k]}
#     else:
#         inverted_dict = {v: k for k2 in s for k in s[k2] for v in s[k2][k]}
#     comp = inverted_dict[sim]
#     fullDict = {'P': 'proton', 'He': 'helium', 'O': 'oxygen', 'Fe': 'iron'}
#     if full:
#         comp = fullDict[comp]
#     return comp

def sim2comp(sim):

    sim_dict = {'7006': 'P', '7579': 'P', '7007': 'Fe', '7784': 'Fe'}
    sim_dict.update({'7241': 'He'})
    return sim_dict[sim]


def sim2comp(sim, full=False, convert=False):

    s = getSimDict()
    converter = {'P': 'p', 'He': 'h', 'O': 'o', 'Fe': 'f'}
    if convert:
        inverted_dict = {v: converter[k]
                         for k2 in s for k in s[k2] for v in s[k2][k]}
    else:
        inverted_dict = {v: k for k2 in s for k in s[k2] for v in s[k2][k]}
    comp = inverted_dict[sim]
    fullDict = {'P': 'proton', 'He': 'helium', 'O': 'oxygen', 'Fe': 'iron'}
    if full:
        comp = fullDict[comp]
    return comp


def getGCD(config):

    gcd = {}
    gcd['IT59'] = '/data/sim/sim-new/downloads/GCD_20_04_10/' + \
        'GeoCalibDetectorStatus_IC59.55040_official.i3.gz'
    gcd['IT73'] = '/data/sim/sim-new/downloads/GCD_31_08_11/' + \
        'GeoCalibDetectorStatus_IC79.55380_L2a.i3.gz'
    gcd['IT81'] = '/data/sim/sim-new/downloads/GCD/' + \
        'GeoCalibDetectorStatus_IC86.55697_V2.i3.gz'

    return gcd[config]


def getSimFiles(sim):

    config = sim2cfg(sim)
    if config == 'IT73':
        prefix = '/data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/'
        files = glob.glob(prefix + sim + '/*/Level2a_*.i3.bz2')
    if config == 'IT81':
        prefix = '/data/sim/IceTop/2011/filtered/CORSIKA-ice-top/level2/'
        files = glob.glob(prefix + sim + '/*/Level2_*.i3.bz2')

    # Remove files with stream errors
    badFiles = getStreamErrors()
    files = [f for f in files if f not in badFiles]

    return sorted(files)


def get_level3_files(sim, just_gcd=False, testing=False, training=False):

    # Get GCD file
    prefix = '/data/ana/CosmicRay/IceTop_level3/sim/IC79/'
    gcd_file = prefix + 'GCD/Level3_{}_GCD.i3.gz'.format(sim)
    if just_gcd:
        return gcd_file

    # # Ensure that you are either training or testing
    # if not testing and not training:
    #     raise('If you\'re not testing and you\'re not training, what ARE you doing?')
    # # Ensure that you are not both training and testing
    # if testing and training:
    #     raise('If you\'re not testing and you\'re not training, what ARE you doing?')

    if sim in ['7241', '7263', '7791']:
        prefix = '/data/user/jbourbeau/composition/IC79_sim/'
    files = glob.glob(prefix + sim + '/Level3_*.i3.gz')
    # Don't forget to sort files
    files = sorted(files)

    # # Split file list into training and testing batches
    # # First half --> training
    # # Second half --> testing
    # files = shuffle(files, random_state=2)
    # files_training = files[:len(files) // 2]
    # files_testing = files[len(files) // 2:]

    # if training:
    #     return gcd_file, files_training
    # else:
    #     return gcd_file, files_testing
    return gcd_file, files


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
    # itstream['IT59'] = ''
    # # itstream['IT73'] = 'top_hlc_clusters'
    # for cfg in ['IT73', 'IT81', 'IT81-II', 'IT81-III']:
    # # for cfg in ['IT81', 'IT81-II', 'IT81-III']:
    #     itstream[cfg] = 'ice_top'

    return itstream[config]


def filter_mask(config):

    filtermask = {}
    for cfg in ['IT59', 'IT73', 'IT81']:
        filtermask[cfg] = 'FilterMask'
    for cfg in ['IT81-II', 'IT81-III']:
        filtermask[cfg] = 'QFilterMask'

    return filtermask[config]

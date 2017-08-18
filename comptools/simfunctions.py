#!/usr/bin/env python

import os
import glob
import numpy as np
from sklearn.utils import shuffle

from icecube.weighting.weighting import from_simprod


def comp2mass(composition):
    mass_dict = {'P': 1,
                 'He': 2,
                 'O': 3,
                 'Fe': 4}
    # mass_dict = {'P': 0.938272013,
    #              'He': 3.727379,
    #              'O': 14.8950815346,
    #              'Fe': 52.0898090795}
    try:
        return mass_dict[composition]
    except KeyError as error:
        raise('Got a KeyError:\n\t{}'.format(error))


def get_sim_configs():
    return ['IC79.2010', 'IC86.2012']


def get_sim_dict():

    sim_dict = {}
    # Add IC79.2010 simulation sets
    IC79_sims = [7006, 7579, 7241, 7263, 7791, 7242, 7262,
                 7851, 7007, 7784]
    for sim in IC79_sims:
        sim_dict[sim] = 'IC79.2010'

    # Add IC86.2012 simulation sets
    IC86_2012_sims = [12360, 12362, 12630, 12631]
    for sim in IC86_2012_sims:
        sim_dict[sim] = 'IC86.2012'

    return sim_dict


def sim_to_config(sim):

    if not isinstance(sim, int):
        raise TypeError('sim must be an integer (the simulation set ID)')

    sim_dict = get_sim_dict()

    try:
        return sim_dict[sim]
    except KeyError:
        raise ValueError('Invalid simulation set, {}, entered'.format(sim))


def config_to_sim(config):

    if not config in get_sim_configs():
        raise ValueError('Invalid config entered')

    sim_dict = get_sim_dict()
    sim_list = []
    for sim, sim_config in sim_dict.iteritems():
        if sim_config == config: sim_list.append(sim)

    return sim_list

def sim_to_comp(sim):
    # Will utilize the weighting project found here
    # http://software.icecube.wisc.edu/documentation/projects/weighting

    # Query database to extract composition from simulation set
    generator = from_simprod(int(sim))
    assert len(generator.spectra) == 1 # Ensure that there is only one composition
    composition = generator.spectra.keys()[0].name

    return composition

def sim_to_thinned(sim):
    # IC79.2010 simulation sets
    not_thinned = [7006, 7241, 7263, 7242, 7262, 7007]
    is_thinned = [7579, 7791, 7851, 7784]
    # IC82.2012 simulation sets
    not_thinned.extend([12360, 12362, 12630, 12631])

    if sim in not_thinned:
        return False
    elif sim in is_thinned:
        return True
    else:
        raise ValueError('Invalid simulation set, {}, entered'.format(sim))



def get_level3_sim_files(sim, just_gcd=False):

    # Get GCD file
    config = sim_to_config(sim)
    # prefix = '/data/ana/CosmicRay/IceTop_level3/sim/v1/{}/'.format(config)
    prefix = '/data/ana/CosmicRay/IceTop_level3/sim/{}'.format(config)
    gcd_file = os.path.join(prefix,'GCD/Level3_{}_GCD.i3.gz'.format(sim))
    if just_gcd:
        return gcd_file

    sim_file_pattern = os.path.join(prefix, str(sim), 'Level3_*.i3.gz')
    files = glob.glob(sim_file_pattern)
    # Don't forget to sort files
    files = sorted(files)

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

    return itstream[config]


def filter_mask(config):

    filtermask = {}
    for cfg in ['IT59', 'IT73', 'IT81']:
        filtermask[cfg] = 'FilterMask'
    for cfg in ['IT81-II', 'IT81-III']:
        filtermask[cfg] = 'QFilterMask'

    return filtermask[config]

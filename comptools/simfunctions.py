
import os
import glob
from itertools import chain
import numpy as np

try:
    from icecube.weighting.weighting import from_simprod
except ImportError as e:
    pass

from .base import requires_icecube, partition


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

    if config not in get_sim_configs():
        raise ValueError('Invalid config entered')

    sim_dict = get_sim_dict()
    sim_list = []
    for sim, sim_config in sim_dict.items():
        if sim_config == config:
            sim_list.append(sim)

    return sim_list


@requires_icecube
def sim_to_comp(sim):
    # Will utilize the weighting project found here
    # http://software.icecube.wisc.edu/documentation/projects/weighting

    # Query database to extract composition from simulation set
    generator = from_simprod(int(sim))
    # Ensure that there is only one composition
    assert len(generator.spectra) == 1
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


def sim_to_energy_bins(sim):
    # IC79.2010 simulation sets
    not_thinned = [7006, 7241, 7263, 7242, 7262, 7007]
    is_thinned = [7579, 7791, 7851, 7784]
    # IC82.2012 simulation sets
    not_thinned.extend([12360, 12362, 12630, 12631])

    if sim in not_thinned:
        ebin_first = 5.0
        ebin_last = 7.9
    elif sim in is_thinned:
        ebin_first = 7.0
        ebin_last = 9.4
    else:
        raise ValueError('Invalid simulation set, {}, entered'.format(sim))

    return ebin_first, ebin_last


def _get_level3_sim_file_pattern(sim):

    config = sim_to_config(sim)
    if config == 'IC79.2010':
        config = 'IC79'
    prefix = os.path.abspath(os.path.join(os.sep, 'data', 'ana', 'CosmicRay',
                                          'IceTop_level3', 'sim', config))
    sim_file_pattern = os.path.join(prefix, str(sim), 'Level3_*.i3.gz')

    return sim_file_pattern


def level3_sim_files(sim, files_per_batch=None, max_batches=None):

    files = glob.glob(_get_level3_sim_file_pattern(sim))
    return files


def level3_sim_file_batches(sim, size, max_batches=None):
    """Generates level3 simulation file paths in batches

    Parameters
    ----------
    sim : int
        Simulation dataset (e.g. 7006, 7241)
    size: int
        Number of files in each batch
    max_batches : int, optional
        Option to only yield ``max_batches`` number of file batches (default
        is to yield all batches)

    Returns
    -------
    generator
        Generator that yields batches of simulation files

    Examples
    --------
    Basic usage:

    >>> from comptools.simfunctions import level3_sim_file_batches
    >>> list(level3_sim_file_batches(7241, size=3, max_batches=2))
    [('/data/ana/CosmicRay/IceTop_level3/sim/IC79/7241/Level3_IC79_7241_Run005347.i3.gz',
      '/data/ana/CosmicRay/IceTop_level3/sim/IC79/7241/Level3_IC79_7241_Run005393.i3.gz',
      '/data/ana/CosmicRay/IceTop_level3/sim/IC79/7241/Level3_IC79_7241_Run009678.i3.gz'),
     ('/data/ana/CosmicRay/IceTop_level3/sim/IC79/7241/Level3_IC79_7241_Run001015.i3.gz',
      '/data/ana/CosmicRay/IceTop_level3/sim/IC79/7241/Level3_IC79_7241_Run002597.i3.gz',
      '/data/ana/CosmicRay/IceTop_level3/sim/IC79/7241/Level3_IC79_7241_Run007939.i3.gz')]

    """

    files_iter = glob.iglob(_get_level3_sim_file_pattern(sim))

    return partition(files_iter, size=size, max_batches=max_batches)


def level3_sim_GCD_file(sim):

    config = sim_to_config(sim)
    if config == 'IC79.2010':
        config = 'IC79'
    prefix = '/data/ana/CosmicRay/IceTop_level3/sim/{}'.format(config)
    gcd_file = os.path.join(prefix, 'GCD', 'Level3_{}_GCD.i3.gz'.format(sim))

    return gcd_file


def get_level3_sim_files_iterator(sim_list):
    '''Function to return an iterable of simulation files

    Parameters
    ----------
    sim_list : int, array-like
        Simulation(s) sets to get i3 files for (e.g. 12360 or
        [12360, 12362, 12630, 12631]).

    Returns
    -------
    file_iter : itertools.chain
        Iterable of simulation i3 files.
    '''
    if isinstance(sim_list, int):
        sim_list = [sim_list]

    file_patterns = [_get_level3_sim_file_pattern(sim) for sim in sim_list]
    file_iter = chain.from_iterable(glob.iglob(pattern) for pattern in file_patterns)

    return file_iter


def run_to_energy_bin(run, sim):
    '''Gives the CORSIKA energy bin for a given simulation run

    Parameters
    ----------
    run : int
        Run number for a simulation set.

    Returns
    -------
    energy_bin : float
        Corresponding CORSIKA energy bin for run.
    '''
    ebin_first, ebin_last = sim_to_energy_bins(sim)
    # Taken from simulation production webpage:
    # http://simprod.icecube.wisc.edu/cgi-bin/simulation/cgi/cfg?dataset=12360
    return (ebin_first * 10 + (run - 1) % (ebin_last * 10 - ebin_first * 10 + 1)) / 10


@np.vectorize
def get_sim_thrown_radius(log_energy):
    '''Gives the thrown simulation radius for a given log_energy

    Parameters
    ----------
    log_energy : float, array-like
        Log energy (in GeV) for a CR shower.

    Returns
    -------
    thrown_radius : float, array-like
        Corresponding thrown radius.
    '''
    if log_energy <= 6:
        thrown_radius = 800.0
    elif (log_energy > 6) & (log_energy <= 7):
        thrown_radius = 1100.0
    elif (log_energy > 7) & (log_energy <= 8):
        thrown_radius = 1700.0
    elif (log_energy > 8) & (log_energy <= 9):
        thrown_radius = 2600.0
    elif (log_energy > 9) & (log_energy <= 10):
        thrown_radius = 2900.0
    else:
        raise ValueError('Invalid energy entered')

    return thrown_radius


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

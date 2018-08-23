
from collections import namedtuple
import os
from functools import wraps
from itertools import islice, count
import numpy as np
import yaml


def requires_icecube(func):
    """Decorator to wrap functions that require any icecube software
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import icecube
        except ImportError:
            message = ('The function {} requires icecube software. '
                       'Make sure the env-shell.sh script has been '
                       'run.'.format(func.__name__))
            raise ImportError(message)

        return func(*args, **kwargs)

    return wrapper


def get_config_paths():
    """ Function to return paths used in this analysis

    Specifically,

    metaproject: Path to IceCube metaproject being used
    comp_data_dir: Path to where data and simulation is stored
    condor_data_dir: Path to where HTCondor error and output files are stored
    condor_scratch_dir: Path to where HTCondor log and submit files are stored
    figures_dir: Path to where figures are saved
    project_root: Path to where cr-composition project is located

    Returns
    -------
    paths : collections.namedtuple
        Namedtuple containing relavent paths (e.g. figures_dir is where
        figures will be saved, condor_data_dir is where data/simulation will
        be saved to / loaded from, etc).
    """

    here = os.path.abspath(os.path.dirname(__file__))
    yaml_file = os.path.join(here, os.pardir, 'config.yml')

    with open(yaml_file, 'r') as f:
        config = yaml.load(f.read()) or {}

    # Create path namedtuple object
    path_names = ['metaproject',
                  'comp_data_dir',
                  'condor_data_dir',
                  'condor_scratch_dir',
                  'figures_dir',
                  'project_root',
                  'virtualenv_dir',
                  ]
    PathObject = namedtuple('PathType', path_names)

    # Create instance of PathObject with appropriate path information
    paths = PathObject(metaproject=config['paths']['metaproject'],
                       comp_data_dir=config['paths']['comp_data_dir'],
                       condor_data_dir=config['paths']['condor_data_dir'],
                       condor_scratch_dir=config['paths']['condor_scratch_dir'],
                       figures_dir=config['paths']['figures_dir'],
                       project_root=config['paths']['project_root'],
                       virtualenv_dir=config['paths']['virtualenv_dir'],
                       )

    return paths


def check_output_dir(outfile, makedirs=True):
    '''Function to check if the directory for an output file exists

    This function will check whether the output directory containing the
    outfile specified exists. If the output directory doesn't exist, then
    there is an option to create the output directory. Otherwise, this
    function will raise an IOError.

    Parameters
    ----------
    outfile : str
        Path to output file.
    makedirs : bool, optional
        Option to create the output directory containing the output file if
        it doesn't already exist (default: True)

    Returns
    -------
    None
    '''
    # Ensure that outfile is an absolute path
    outfile = os.path.abspath(outfile)
    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        if makedirs:
            print('The directory {} doesn\'t exist. Creating it...'.format(outdir))
            os.makedirs(outdir)
        else:
            raise IOError('The directory {} doesn\'t exist'.format(outdir))

    return


def partition(seq, size, max_batches=None):
    '''Generates partitions of length ``size`` from the iterable ``seq``

    Parameters
    ----------
    seq : iterable
        Iterable object to be partitioned.
    size : int
        Number of items to have in each partition.
    max_batches : int, optional
        Limit the number of partitions to yield (default is to yield all
        partitions).

    Yields
    -------
    batch : list
        Partition of ``seq`` that is (at most) ``size`` items long.

    Examples
    --------
    >>> from comptools import partition
    >>> list(partition(range(10), 3))
    [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]

    '''
    if not isinstance(max_batches, (int, type(None))):
        raise TypeError('max_batches must either be an integer or None, '
                        'got {}'.format(type(max_batches)))

    seq_iter = iter(seq)
    for num_batches in islice(count(), max_batches):
        batch = list(islice(seq_iter, size))
        if len(batch) == 0:
            return
        else:
            yield batch


class ComputingEnvironemtError(Exception):
    """
    Custom exception that should be raised when a problem related to the
    computing environment is found
    """
    pass


def get_energybins(config='IC86.2012'):
    from .binning import get_energybins as get_ebins
    return get_ebins(config=config)


LABEL_DICT = {'reco_log_energy': '$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$',
              'lap_log_energy': '$\log_{10}(E_{\mathrm{Lap}}/\mathrm{GeV})$',
              'log_s50': '$\log_{10}(S_{\mathrm{50}})$',
              'log_s80': '$\log_{10}(S_{\mathrm{80}})$',
              'log_s125': '$\log_{10}(S_{\mathrm{125}})$',
              'log_s180': '$\log_{10}(S_{\mathrm{180}})$',
              'log_s250': '$\log_{10}(S_{\mathrm{250}})$',
              'log_s500': '$\log_{10}(S_{\mathrm{500}})$',
              'lap_rlogl': '$r\log_{10}(l)$',
              'lap_beta': 'lap beta',
              'InIce_log_charge_1_60': 'InIce charge',
              'InIce_log_charge_1_45': 'InIce charge (top 75\%)',
              'InIce_charge_1_30': 'InIce charge (top 50\%)',
              'InIce_log_charge_1_30': '$\log_{10}(InIce charge (top 50))$',
              'InIce_log_charge_1_15': 'InIce charge (top 25\%)',
              'InIce_log_charge_1_6': 'InIce charge (top 10\%)',
              'reco_cos_zenith': '$\cos(\\theta_{\mathrm{reco}})$',
              'lap_cos_zenith': '$\cos(\\theta)$',
              'LLHlap_cos_zenith': '$\cos(\\theta_{\mathrm{Lap}})$',
              'LLHLF_cos_zenith': '$\cos(\\theta_{\mathrm{LLH+COG}})$',
              'lap_chi2': '$\chi^2_{\mathrm{Lap}}/\mathrm{n.d.f}$',
              'NChannels_1_60': 'NChannels',
              'NChannels_1_45': 'NChannels (top 75\%)',
              'NChannels_1_30': 'NChannels (top 50\%)',
              'NChannels_1_15': 'NChannels (top 25\%)',
              'NChannels_1_6': 'NChannels (top 10\%)',
              'log_NChannels_1_30': '$\log_{10}$(NChannels (top 50\%))',
              'StationDensity': 'StationDensity',
              'charge_nchannels_ratio': 'Charge/NChannels',
              'stationdensity_charge_ratio': 'StationDensity/Charge',
              'NHits_1_30': 'NHits',
              'log_NHits_1_30': '$\log_{10}$(NHits (top 50\%))',
              'charge_nhits_ratio': 'Charge/NHits',
              'nhits_nchannels_ratio': 'NHits/NChannels',
              'stationdensity_nchannels_ratio': 'StationDensity/NChannels',
              'stationdensity_nhits_ratio': 'StationDensity/NHits',
              'llhratio': 'llhratio',
              'n_he_stoch_standard': 'Num HE stochastics (standard)',
              'n_he_stoch_strong': 'Num HE stochastics (strong)',
              'eloss_1500_standard': 'dE/dX (standard)',
              'log_dEdX': '$\mathrm{\log_{10}(dE/dX)}$',
              'eloss_1500_strong': 'dE/dX (strong)',
              'num_millipede_particles': '$N_{\mathrm{mil}}$',
              'avg_inice_radius': '$\mathrm{\langle R_{\mu} \\rangle }$',
              'invqweighted_inice_radius_1_60': '$\mathrm{R_{\mu \ bundle}}$',
              'avg_inice_radius_1_60': '$\mathrm{R_{\mu \ bundle}}$',
              'avg_inice_radius_Laputop': '$R_{\mathrm{core, Lap}}$',
              'FractionContainment_Laputop_InIce': '$C_{\mathrm{IC}}$',
              'FractionContainment_Laputop_IceTop': '$C_{\mathrm{IT}}$',
              'max_inice_radius': '$R_{\mathrm{max}}$',
              'invcharge_inice_radius': '$R_{\mathrm{q,core}}$',
              'lap_zenith': 'zenith',
              'NStations': 'NStations',
              'IceTop_charge': 'IT charge',
              'IceTop_charge_175m': 'Signal greater 175m',
              'log_IceTop_charge_175m': '$\log_{10}(Q_{IT, 175})$',
              'IT_charge_ratio': 'IT charge ratio',
              'refit_beta': '$\mathrm{\\beta_{refit}}$',
              'log_d4r_peak_energy': '$\mathrm{\log_{10}(E_{D4R})}$',
              'log_d4r_peak_sigma': '$\mathrm{\log_{10}(\sigma E_{D4R})}$',
              'd4r_N': 'D4R N',
              'median_inice_radius': 'Median InIce',
              'IceTopLLHRatio': 'IceTopLLHRatio',
              }


def get_training_features(feature_list=None):

    # Features used in the 3-year analysis
    if feature_list is None:
        feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX']
        # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'avg_inice_radius']
        # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'log_d4r_peak_energy', 'log_d4r_peak_sigma']
        # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'median_inice_radius', 'd4r_peak_energy']
        # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'FractionContainment_Laputop_InIce']
        # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'avg_inice_radius']
    # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'max_inice_radius']
    # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'avg_inice_radius']

    dom_numbers = [1, 15, 30, 45, 60]
    for min_DOM, max_DOM in zip(dom_numbers[:-1], dom_numbers[1:]):
        key = 'NChannels_{}_{}'.format(min_DOM, max_DOM)
        label = 'NChannels {} {}'.format(min_DOM, max_DOM)
        LABEL_DICT[key] = label
    min_DOM, max_DOM = 1, 60
    key = 'NChannels_{}_{}'.format(min_DOM, max_DOM)
    label = 'NChannels {} {}'.format(min_DOM, max_DOM)
    LABEL_DICT[key] = label

    for min_DOM, max_DOM in zip(dom_numbers[:-1], dom_numbers[1:]):
        key = 'NHits_{}_{}'.format(min_DOM, max_DOM)
        label = 'NHits {} {}'.format(min_DOM, max_DOM)
        LABEL_DICT[key] = label
    min_DOM, max_DOM = 1, 60
    key = 'NHits_{}_{}'.format(min_DOM, max_DOM)
    label = 'NHits {} {}'.format(min_DOM, max_DOM)
    LABEL_DICT[key] = label

    min_dists = np.arange(0, 1125, 125)
    for min_dist in min_dists:
        key = 'IceTop_charge_beyond_{}m'.format(min_dist)
        LABEL_DICT[key] = 'IT Q > {}m'.format(min_dist)

    feature_labels = [LABEL_DICT[feature] for feature in feature_list]

    return feature_list, feature_labels

#!/usr/bin/env python

from collections import namedtuple
import os
import getpass
from functools import wraps
import numpy as np


def requires_icecube(func):
    """Decorator to wrap functions that require any icecube software
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import icecube
        except ImportError as e:
            message = ('The function {} requires icecube software. '
                       'Make sure the env-shell.sh script has been '
                       'run.'.format(func.__name__))
            raise ImportError(message)

        return func(*args, **kwargs)

    return wrapper


def get_paths(username=None):
    '''Function to return paths used in this analysis

    Specifically,

    metaproject - Path to IceCube metaproject being used
    comp_data_dir - Path to where data and simulation is stored
    condor_data_dir - Path to where HTCondor error and output files are stored
    condor_scratch_dir - Path to where HTCondor log and submit files are stored
    figures_dir - Path to where figures are saved
    project_home - Path to where cr-composition project is located
    project_root - Path to where cr-composition project is located

    Parameters
    ----------
    username : str, optional
        Username on the machine that will be used for analysis. This is used
        to construct the paths for this analysis (default is getpass.getuser()).

    Returns
    -------
    paths : collections.namedtuple
        Namedtuple containing relavent paths (e.g. figures_dir is where
        figures will be saved, condor_data_dir is where data/simulation will
        be saved to / loaded from, etc).

    '''
    if username is None:
        username = getpass.getuser()

    # Create path namedtuple object
    PathObject = namedtuple('PathType', ['metaproject', 'comp_data_dir',
        'condor_data_dir', 'condor_scratch_dir', 'figures_dir',
        'project_home', 'project_root'])

    metaproject = '/data/user/{}/metaprojects/icerec/V05-01-00'.format(username)
    comp_data_dir = '/data/user/{}/composition'.format(username)
    condor_data_dir = '/data/user/{}/composition/condor'.format(username)
    condor_scratch_dir = '/scratch/{}/composition/condor'.format(username)
    figures_dir = '/home/{}/public_html/figures/composition'.format(username)
    project_home = '/home/{}/cr-composition'.format(username)
    project_root = '/home/{}/cr-composition'.format(username)

    # Create instance of PathObject with appropriate path information
    paths = PathObject(metaproject=metaproject,
                       comp_data_dir=comp_data_dir,
                       condor_data_dir=condor_data_dir,
                       condor_scratch_dir=condor_scratch_dir,
                       figures_dir=figures_dir,
                       project_home=project_home,
                       project_root=project_root)

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
    # Ensure that outfile gives an absolute path
    outfile = os.path.abspath(outfile)
    outdir, basename = os.path.split(outfile)

    if not os.path.exists(outdir):
        if makedirs:
            print('The directory {} doesn\'t exist. '
                  'Creating it...'.format(outdir))
            os.makedirs(outdir)
        else:
            raise IOError('The directory {} doesn\'t exist'.format(outdir))

    return


def file_batches(files, n_files, n_batches=None):
    '''Generates batches of files

    Parameters
    ----------
    files : array-like
        Iterable of files.
    n_files : int
        Number of files to have in each batch.
    n_batches : int, optional
        Limit the number of batches to yield (default is to yield all batches).

    Returns
    -------
    batch : list
        Batch of files of size n_files.
    '''
    for batch_num, i in enumerate(range(0, len(files), n_files), start=1):
        if n_batches is not None and batch_num > n_batches:
            raise StopIteration
        batch = list(files[i:i+n_files])
        yield batch


class ComputingEnvironemtError(Exception):
    """
    Custom exception that should be raised when a problem related to the
    computing environment is found
    """
    pass


def get_energybins(config='IC86.2012'):
    """Function to return analysis energy bin information

    Parameters
    ----------
    config : str, optional
        Detector configuration (default is 'IC86.2012').

    Returns
    -------
    energybins : namedtuple
        Namedtuple containing analysis energy bin information.
    """
    # Create EnergyBin namedtuple
    EnergyBin = namedtuple('EnergyBins', ['energy_min', 'energy_max',
        'energy_bins', 'energy_midpoints', 'energy_bin_widths',
        'log_energy_min', 'log_energy_max', 'log_energy_bin_width',
        'log_energy_bins', 'log_energy_midpoints'])

    # Define energy range for this analysis
    if 'IC79' in config:
        log_energy_min = 6.1
        log_energy_break = 8.0
        log_energy_max = 9.0
        log_energy_bins = np.concatenate(
                        (np.arange(log_energy_min, log_energy_break-0.1, 0.1),
                         np.arange(log_energy_break, log_energy_max+0.2, 0.2)))
    elif 'IC86' in config:
        log_energy_min = 6.1
        log_energy_max = 8.0
        log_energy_bins = np.arange(log_energy_min, log_energy_max+0.1, 0.1)
    else:
        raise ValueError(
            'Invalid detector configuration entered: {}'.format(config))

    log_energy_bin_width = log_energy_bins[1:] - log_energy_bins[:-1]
    log_energy_midpoints = (log_energy_bins[1:] + log_energy_bins[:-1]) / 2

    energy_min = 10**log_energy_min
    energy_max = 10**log_energy_max
    energy_bins = 10**log_energy_bins
    energy_midpoints = 10**log_energy_midpoints
    energy_bin_widths = energy_bins[1:] - energy_bins[:-1]

    # Create instance of EnergyBins with appropriate binning
    energybins = EnergyBin(energy_min=energy_min,
                           energy_max=energy_max,
                           energy_bins=energy_bins,
                           energy_midpoints=energy_midpoints,
                           energy_bin_widths=energy_bin_widths,
                           log_energy_min=log_energy_min,
                           log_energy_max=log_energy_max,
                           log_energy_bin_width=log_energy_bin_width,
                           log_energy_bins=log_energy_bins,
                           log_energy_midpoints=log_energy_midpoints)

    return energybins


def get_training_features(feature_list=None):

    # Features used in the 3-year analysis
    if feature_list is None:
        feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX']
        # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'log_d4r_peak_energy', 'log_d4r_peak_sigma']
        # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'median_inice_radius', 'd4r_peak_energy']
        # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'FractionContainment_Laputop_InIce']
        # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'avg_inice_radius']
    # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'max_inice_radius']
    # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'avg_inice_radius']

    label_dict = {'reco_log_energy': '$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$',
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
                  'lap_cos_zenith': '$\cos(\\theta_{\mathrm{Lap}})$',
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
                  'avg_inice_radius': '$\mathrm{R_{\mu \ bundle}}$',
                  'invqweighted_inice_radius_1_60': '$\mathrm{R_{\mu \ bundle}}$',
                  'avg_inice_radius_1_60': '$\mathrm{R_{\mu \ bundle}}$',
                  'avg_inice_radius_Laputop': '$R_{\mathrm{core, Lap}}$',
                  'FractionContainment_Laputop_InIce': '$C_{\mathrm{IC}}$',
                  'Laputop_IceTop_FractionContainment': '$C_{\mathrm{IT}}$',
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
                  'median_inice_radius': 'Median InIce'
                  }

    feature_labels = [label_dict[feature] for feature in feature_list]

    return feature_list, feature_labels

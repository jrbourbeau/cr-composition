
from collections import namedtuple
import numpy as np


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
    energy_field_names = ['energy_min',
                          'energy_max',
                          'energy_bins',
                          'energy_midpoints',
                          'energy_bin_widths',
                          'log_energy_min',
                          'log_energy_max',
                          'log_energy_bin_width',
                          'log_energy_bins',
                          'log_energy_midpoints',
                          ]
    EnergyBin = namedtuple('EnergyBins', energy_field_names)

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


def get_comp_bins(num_groups=2):
    """Function to return analysis composition bin information

    Parameters
    ----------
    num_groups : int, optional
        Number of composition groups (default is 2).

    Returns
    -------
    comp_bins : numpy.ndarray
        Array containing analysis compsition bin edges.
    """
    comp_bins = np.arange(num_groups + 1)
    return comp_bins


def get_zenith_bins(zenith_bin_width=10):
    """Function to return analysis zenith bin information

    Parameters
    ----------
    num_groups : int, optional
        Number of composition groups (default is 2).

    Returns
    -------
    comp_bins : namedtuple
        Namedtuple containing analysis zenith bin information.
    """
    # Create ZenithBin namedtuple
    zenith_field_names = ['zenith_min',
                          'zenith_max',
                          'zenith_bins',
                          'zenith_midpoints',
                          'zenith_bin_widths',
                          ]
    ZenithBin = namedtuple('ZenithBins', zenith_field_names)

    # Define zenith range for this analysis
    zenith_min = 0
    zenith_max = 30
    zenith_bins = np.arange(zenith_min,
                            zenith_max + zenith_bin_width,
                            zenith_bin_width)

    zenith_bin_widths = zenith_bins[1:] - zenith_bins[:-1]
    zenith_midpoints = (zenith_bins[1:] + zenith_bins[:-1]) / 2

    # Create instance of ZenithBin with appropriate binning
    zenithbins = ZenithBin(zenith_min=zenith_min,
                           zenith_max=zenith_max,
                           zenith_bins=zenith_bins,
                           zenith_midpoints=zenith_midpoints,
                           zenith_bin_widths=zenith_bin_widths)

    return zenithbins


def get_bins(config='IC86.2012', num_groups=2, zenith_bin_width=10,
             log_energy=True, include_zenith=False, return_columns=False):
    # Energy bins
    energybins = get_energybins(config=config)
    if log_energy:
        energy_bins = energybins.log_energy_bins
    else:
        energy_bins = energybins.energy_bins

    # Composition bins
    comp_bins = get_comp_bins(num_groups=num_groups)

    bins = [energy_bins, comp_bins]
    columns = ['reco_log_energy', 'pred_comp_target']

    # Zenith bins
    if include_zenith:
        zenith_bins = get_zenith_bins(zenith_bin_width=zenith_bin_width).zenith_bins
        bins.append(zenith_bins)
        columns.append('lap_zenith')

    if return_columns:
        return bins, columns
    else:
        return bins


def bin_edges_to_midpoints(bin_edges):
    midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
    return midpoints

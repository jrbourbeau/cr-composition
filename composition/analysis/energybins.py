
from collections import namedtuple
import numpy as np

def get_energybins():

    # Create EnergyBins object to store all realted information
    EnergyBins = namedtuple('EnergyBins', ['energy_min', 'energy_max',
        'energy_bins', 'energy_midpoints', 'energy_bin_widths',
        'log_energy_min', 'log_energy_max', 'log_energy_bin_width',
        'log_energy_bins', 'log_energy_midpoints'])

    # Define energy range for this analysis
    log_energy_min = 6.3
    log_energy_max = 8.0
    energy_min = 10**6.3
    energy_max = 10**8.0
    # Define energy binning for this analysis
    log_energy_bin_width = 0.1
    log_energy_bins = np.arange(log_energy_min,
        log_energy_max+log_energy_bin_width, log_energy_bin_width)
    log_energy_midpoints = (log_energy_bins[1:] + log_energy_bins[:-1]) / 2

    energy_bins = 10**log_energy_bins
    energy_midpoints = 10**log_energy_midpoints
    energy_bin_widths = energy_bins[1:] - energy_bins[:-1]

    # Create instance of EnergyBins with appropriate binning
    energybins = EnergyBins(energy_min=energy_min,
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


from __future__ import division
import collections
import numpy as np
from scipy.optimize import curve_fit

try:
    from icecube.weighting.weighting import from_simprod
except ImportError as e:
    pass

from .base import requires_icecube
from .simfunctions import get_level3_sim_files, sim_to_comp
from .io import load_sim


def effective_area(df, log_energy_bins):

    # MC_energy = df['MC_energy'].values
    # log_MC_energy = np.log10(MC_energy)
    # # Throw out particles outside energy range
    # mask = (log_MC_energy > log_energy_bins[0]) & (log_MC_energy < log_MC_energy[-1])
    # MC_energy = MC_energy[mask]
    # log_MC_energy = log_MC_energy[mask]
    log_MC_energy = df['MC_log_energy'].values
    print(log_MC_energy)

    sim_id = df['sim'].values
    # sim_id = df[mask]['sim'].values
    num_ptype = len(np.unique(df['MC_type']))

    # Set radii for finding effective area
    energy_range_to_radii = np.array([800, 1100, 1700, 2600, 2900])
    energy_breaks = np.array([5, 6, 7, 8, 9])
    event_energy_bins = np.digitize(log_MC_energy, energy_breaks) - 1

    weights = []
    resamples = 100
    for energy_bin_idx, sim in zip(event_energy_bins, sim_id):
        if energy_bin_idx < 0 or energy_bin_idx > len(energy_range_to_radii): continue
        weight = np.pi * energy_range_to_radii[energy_bin_idx]**2
        if sim in ['7579', '7791', '7851', '7784']:
            num_thrown = 480*resamples
        else:
            if sim == '7262':
                num_thrown = (19999/20000)*1000*resamples
            elif sim == '7263':
                num_thrown = (19998/20000)*1000*resamples
            else:
                num_thrown = 1000*resamples
        if energy_bin_idx == 2:
            weight = weight/(2*num_thrown)
        else:
            weight = weight/num_thrown
        weights.append(weight)
    print(np.array(weights))

    energy_hist = np.histogram(log_MC_energy, bins=log_energy_bins, weights=weights)[0]
    error_hist = np.sqrt(np.histogram(log_MC_energy, bins=log_energy_bins,
                                weights=np.array(weights)**2)[0])

    eff_area = energy_hist/num_ptype
    eff_area_error = error_hist/num_ptype

    # eff_area  = (10**-6)*E_hist.bincontent/n_gen
    # eff_area_error = (10**-6)*np.sqrt(error_hist.bincontent)/n_gen

    return eff_area, eff_area_error


@requires_icecube
def calculate_effective_area_vs_energy(df_sim, energy_bins, verbose=True):
    '''Calculated effective area vs. energy from simulation

    Parameters
    ----------
    df_sim : pandas.DataFrame
        Simulation DataFrame returned from comptools.load_sim.
    energy_bins : array-like
        Energy bins (in GeV) that will be used for calculation.
    verbose : bool, optional
        Option for verbose output (default is True).

    Returns
    -------
    eff_area : numpy.ndarray
        Effective area for each bin in energy_bins
    eff_area_error : numpy.ndarray
        Statistical ucertainty on the effective area for each bin in
        energy_bins.
    energy_midpoints : numpy.ndarray
        Midpoints of energy_bins. Useful for plotting effective area versus
        energy.

    '''

    if verbose:
        print('Calculating effective area...')

    simlist = np.unique(df_sim['sim'])
    # # Get the number of times each composition is present
    # comp_counter = collections.Counter([sim_to_comp(sim) for sim in simlist])
    # print('comp_counter = {}'.format(comp_counter))
    for i, sim in enumerate(simlist):
        gcd_file, sim_files = get_level3_sim_files(sim)
        num_files = len(sim_files)
        if verbose:
            print('Simulation set {}: {} files'.format(sim, num_files))
        composition = sim_to_comp(sim)
        if i == 0:
            generator = num_files*from_simprod(int(sim))
        else:
            generator += num_files*from_simprod(int(sim))

    energy = df_sim['MC_energy'].values
    ptype = df_sim['MC_type'].values
    # num_ptypes = 2
    num_ptypes = np.unique(ptype).size
    cos_theta = np.cos(df_sim['MC_zenith']).values
    areas = 1.0/generator(energy, ptype, cos_theta)
    # binwidth = 2*np.pi*(1-np.cos(40*(np.pi/180)))*np.diff(energy_bins)
    binwidth = 2*np.pi*(1-np.cos(40*(np.pi/180)))*np.diff(energy_bins)*num_ptypes
    eff_area = np.histogram(energy, weights=areas, bins=energy_bins)[0]/binwidth
    eff_area_error = np.sqrt(np.histogram(energy, bins=energy_bins, weights=areas**2)[0])/binwidth

    energy_midpoints = (energy_bins[1:] + energy_bins[:-1]) / 2

    return eff_area, eff_area_error, energy_midpoints


def sigmoid_flat(energy, p0, p1, p2):
    return p0 / (1 + np.exp(-p1*np.log10(energy) + p2))


def sigmoid_slant(energy, p0, p1, p2, p3):
    return (p0 + p3*np.log10(energy)) / (1 + np.exp(-p1*np.log10(energy) + p2))


def get_effective_area_fit(config='IC86.2012', fit_func=sigmoid_slant, energy_points=None):
    '''Calculated effective area from simulation

    Parameters
    ----------
    config : str, optional
        Detector configuration (default is IC86.2012).

    Returns
    -------
    avg : float
        Effective area
    avg_err : float
        Statistical error on effective area

    '''

    df_sim = load_sim(config=config, test_size=0, verbose=False)

    log_energy_bins = np.arange(5.0, 9.51, 0.05)
    energy_bins = 10**log_energy_bins
    energy_midpoints = (energy_bins[1:] + energy_bins[:-1]) / 2

    energy_min_fit, energy_max_fit = 5.8, 8.0
    midpoints_fitmask = np.logical_and(energy_midpoints > 10**energy_min_fit,
                                       energy_midpoints < 10**energy_max_fit)


    # Calculate the effective areas
    eff_area, eff_area_error, _ = calculate_effective_area_vs_energy(
                                df_sim, energy_bins, verbose=False)
    eff_area_light, eff_area_error_light, _ = calculate_effective_area_vs_energy(
                                df_sim[df_sim.MC_comp_class == 'light'],
                                energy_bins, verbose=False)
    eff_area_heavy, eff_area_error_heavy, _ = calculate_effective_area_vs_energy(
                                df_sim[df_sim.MC_comp_class == 'heavy'],
                                energy_bins, verbose=False)

    if fit_func.__name__ == 'sigmoid_flat':
        p0 = [1.5e5, 8.0, 50.0]
    elif fit_func.__name__ == 'sigmoid_slant':
        p0 = [1.4e5, 8.5, 50.0, 800]
    popt_light, pcov_light = curve_fit(fit_func,
                                energy_midpoints[midpoints_fitmask],
                                eff_area_light[midpoints_fitmask], p0=p0,
                                sigma=eff_area_error_light[midpoints_fitmask])
    perr_light = np.sqrt(np.diag(pcov_light))

    popt_heavy, pcov_heavy = curve_fit(fit_func,
                                energy_midpoints[midpoints_fitmask],
                                eff_area_heavy[midpoints_fitmask], p0=p0,
                                sigma=eff_area_error_heavy[midpoints_fitmask])
    perr_heavy = np.sqrt(np.diag(pcov_heavy))
    popt_avg = (popt_light + popt_heavy) / 2

    return fit_func(energy_points, *popt_avg)


def get_sigmoid_params(df_sim, energy_bins):

    eff_area, eff_area_error, energy_midpoints = get_effective_area(df_sim, energy_bins, verbose=True)

    p_init = [1.4e5, 8.0, 50.0]
    print(energy_midpoints)
    eff_area_init = sigmoid(energy_midpoints, *p_init)
    print(eff_area)
    print(eff_area_init)

    # Fit with error bars
    # popt, pcov = optimize.curve_fit(sigmoid, np.log10(energy_midpoints), eff_area)
    popt, pcov = optimize.curve_fit(sigmoid, energy_midpoints, eff_area, sigma=eff_area_error)
    # popt, pcov = optimize.curve_fit(sigmoid, energy_midpoints, eff_area, p0=p_init, sigma=eff_area_error)
    eff_area_fit = sigmoid(energy_midpoints, *popt)
    print(eff_area_fit)
    chi = np.sum(((eff_area_fit - eff_area)/eff_area_error) ** 2) / len(energy_midpoints)
    print('ppot = {}'.format(popt))
    print('chi2 = {}'.format(chi))

    return popt, pcov


from __future__ import division
import numpy as np
from .base import DataSet
from .base import get_energybins
from .data_functions import ratio_error


def get_num_particles(train, test, pipeline, comp_list, log_energy_bins=get_energybins().log_energy_bins):
    '''Calculates the number of particles identified in each energy bin
    for each composition in comp_list. In addition, the statisitcal error for the
    number of events is calculated.'''

    assert isinstance(train, DataSet), 'train dataset must be a DataSet'
    assert isinstance(test, DataSet), 'test dataset must be a DataSet'
    assert train.y is not None, 'train must have true y values'
    assert test.log_energy is not None, 'teset must have log_energ values'

    pipeline.fit(train.X, train.y)
    test_predictions = pipeline.predict(test.X)

    # Get number of identified comp in each energy bin
    num_particles, num_particles_err = {}, {}
    for composition in comp_list:
        comp_mask = train.le.inverse_transform(test_predictions) == composition
        num_particles[composition] = np.histogram(test.log_energy[comp_mask],
            bins=log_energy_bins)[0]
        num_particles_err[composition] = np.sqrt(num_particles[composition])

    num_particles['total'] = np.histogram(test.log_energy, bins=log_energy_bins)[0]
    num_particles_err['total'] = np.sqrt(num_particles['total'])

    return num_particles, num_particles_err


def get_flux(counts, counts_err, energybins=get_energybins().energy_bins,
             eff_area=156390.673059, livetime=27114012.0, livetime_err=1,
             solid_angle=1., scalingindex=2.7):
    # Calculate energ bin widths and midpoints
    energy_bin_widths = energybins[1:] - energybins[:-1]
    energy_midpoints = (energybins[1:] + energybins[:-1]) / 2
    # Calculate dN/dE
    y = counts/energy_bin_widths
    y_err = counts_err/energy_bin_widths
    # Add effective area
    eff_area = np.array([eff_area]*len(y))
    eff_area_error = 0.01 * eff_area
    y, y_err = ratio_error(y, y_err, eff_area, eff_area_error)
    # Add solid angle
    y = y / solid_angle
    y_err = y_err / solid_angle
    # Add time duration
    flux, flux_err = ratio_error(y, y_err, livetime, livetime_err)
    # Add energy scaling
    scaled_flux = energy_midpoints**scalingindex * flux
    scaled_flux_err = energy_midpoints**scalingindex * flux_err

    return scaled_flux, scaled_flux_err

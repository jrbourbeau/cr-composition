
from __future__ import division
import numpy as np
import pandas as pd
from .base import DataSet
from .base import get_energybins
from .data_functions import ratio_error
from ..composition_encoding import composition_group_labels, get_comp_list
from icecube.weighting.weighting import from_simprod, PDGCode, ParticleType
from icecube.weighting.fluxes import GaisserH3a, GaisserH4a, Hoerandel5, Hoerandel_IT, CompiledFlux


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


def get_flux(counts, counts_err=None, energybins=get_energybins().energy_bins,
             eff_area=156390.673059, eff_area_err=None, livetime=27114012.0,
             livetime_err=1, solid_angle=1., scalingindex=2.7):
    # Calculate energ bin widths and midpoints
    energy_bin_widths = energybins[1:] - energybins[:-1]
    energy_midpoints = (energybins[1:] + energybins[:-1]) / 2
    # Calculate dN/dE
    y = counts/energy_bin_widths
    if counts_err is None:
        counts_err = np.sqrt(counts)
    y_err = counts_err/energy_bin_widths
    # Add effective area
    if isinstance(eff_area, (int, float)):
        eff_area = np.array([eff_area]*len(y))
    else:
        eff_area = np.asarray(eff_area)

    if eff_area_err is None:
        y = y / eff_area
        y_err = y_err / eff_area
    else:
        y, y_err = ratio_error(y, y_err, eff_area, eff_area_err)
    # Add solid angle
    y = y / solid_angle
    y_err = y_err / solid_angle
    # Add time duration
    flux, flux_err = ratio_error(y, y_err, livetime, livetime_err)
    # Add energy scaling
    scaled_flux = energy_midpoints**scalingindex * flux
    scaled_flux_err = energy_midpoints**scalingindex * flux_err

    return scaled_flux, scaled_flux_err


def get_model_flux(model='H3a', energy=None, num_groups=2):

    comp_list = get_comp_list(num_groups=num_groups)

    model_names = ['H3a', 'H4a', 'Polygonato']
    assert model in model_names
    flux_models = [GaisserH3a(), GaisserH4a(), Hoerandel5()]
    model_dict = dict(zip(model_names, flux_models))

    flux = model_dict[model]

    if energy is None:
        energy = get_energybins()

    p = PDGCode().values
    pdg_codes = np.array([2212, 1000020040, 1000080160, 1000260560])
    particle_names = [p[pdg_code].name for pdg_code in pdg_codes]

    group_names = np.array(composition_group_labels(particle_names,
                                                    num_groups=num_groups))

    comp_to_pdg_list = {composition: pdg_codes[group_names == composition]
                        for composition in comp_list}

    # Replace O16Nucleus with N14Nucleus + Al27Nucleus
    for composition, pdg_list in comp_to_pdg_list.iteritems():
        if 1000080160 in pdg_list:
            pdg_list = pdg_list[pdg_list != 1000080160]
            comp_to_pdg_list[composition] = np.append(pdg_list, [1000070140, 1000130270])
        else:
            continue

    flux_df = pd.DataFrame()
    for composition in comp_list:
        comp_flux = []
        for energy_mid in energy:
            flux_energy_mid = flux(energy_mid, comp_to_pdg_list[composition]).sum()
            comp_flux.append(flux_energy_mid)
        flux_df['flux_{}'.format(composition)] = comp_flux

    flux_df['flux_total'] = flux_df.sum(axis=1)

    return flux_df


from __future__ import division
import numpy as np
import pandas as pd

try:
    from icecube.weighting.weighting import PDGCode
    from icecube.weighting.fluxes import GaisserH3a, GaisserH4a, Hoerandel5
except ImportError as e:
    pass

from .base import get_energybins, requires_icecube
from .data_functions import ratio_error
from .composition_encoding import composition_group_labels, get_comp_list



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


@requires_icecube
def model_flux(model='H3a', energy=None, num_groups=2):

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

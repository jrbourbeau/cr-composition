
from __future__ import division
import numpy as np
from .base import DataSet
from .base import get_energybins
from .data_functions import ratio_error


def get_frac_correct(train, test, pipeline, comp_list, log_energy_bins=get_energybins().log_energy_bins):
    '''Calculates the fraction of correctly identified samples in each energy bin
    for each composition in comp_list. In addition, the statisitcal error for the
    fraction correctly identified is calculated.'''

    assert isinstance(train, DataSet), 'train dataset must be a DataSet'
    assert isinstance(test, DataSet), 'test dataset must be a DataSet'
    assert train.y is not None, 'train must have true y values'
    assert test.y is not None, 'test must have true y values'

    pipeline.fit(train.X, train.y)
    test_predictions = pipeline.predict(test.X)
    correctly_identified_mask = (test_predictions == test.y)

    # Construct MC composition masks
    MC_comp_mask = {}
    for composition in comp_list:
        MC_comp_mask[composition] = (test.le.inverse_transform(test.y) == composition)
    MC_comp_mask['total'] = np.array([True]*len(test))

    reco_frac, reco_frac_err = {}, {}
    for composition in comp_list+['total']:
        comp_mask = MC_comp_mask[composition]
        # Get number of MC comp in each reco energy bin
        num_MC_energy = np.histogram(test.log_energy[comp_mask],
                                     bins=log_energy_bins)[0]
        num_MC_energy_err = np.sqrt(num_MC_energy)

        # Get number of correctly identified comp in each reco energy bin
        num_reco_energy = np.histogram(test.log_energy[comp_mask & correctly_identified_mask],
                                       bins=log_energy_bins)[0]
        num_reco_energy_err = np.sqrt(num_reco_energy)

        # Calculate correctly identified fractions as a function of MC energy
        reco_frac[composition], reco_frac_err[composition] = ratio_error(
            num_reco_energy, num_reco_energy_err,
            num_MC_energy, num_MC_energy_err)

    return reco_frac, reco_frac_err


from __future__ import division
import pytest
import numpy as np
from comptools.io import load_sim, load_data


@pytest.mark.needs_data
def test_load_sim_test_size():
    test_size = 0.4
    df_train, df_test = load_sim(test_size=test_size, energy_reco=False,
                                 log_energy_min=None, log_energy_max=None)

    n_train = len(df_train)
    n_test = len(df_test)

    np.testing.assert_allclose(n_test / (n_test + n_train), test_size,
                               rtol=1e-2)


@pytest.mark.needs_data
def test_load_sim_log_energy_min():
    log_energy_min = 7.5
    df = load_sim(test_size=0, energy_reco=False,
                  energy_cut_key='MC_log_energy',
                  log_energy_min=log_energy_min, log_energy_max=None)

    np.testing.assert_allclose(log_energy_min, df['MC_log_energy'].min(),
                               rtol=1e-2)


@pytest.mark.needs_data
def test_load_sim_log_energy_max():
    log_energy_max = 7.5
    df = load_sim(test_size=0, energy_reco=False,
                  energy_cut_key='MC_log_energy',
                  log_energy_min=None, log_energy_max=log_energy_max)

    np.testing.assert_allclose(log_energy_max, df['MC_log_energy'].max(),
                               rtol=1e-2)


@pytest.mark.needs_data
@pytest.mark.parametrize('energy_reco', [True, False])
def test_load_sim_energy_reco(energy_reco):
    df = load_sim(test_size=0, energy_reco=energy_reco,
                  log_energy_min=None, log_energy_max=None)

    assert ('reco_log_energy' in df.columns) == energy_reco

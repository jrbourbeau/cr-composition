
import os
import numpy as np
import pandas as pd

from .base import get_config_paths, get_energybins
from .simfunctions import get_sim_configs
from .composition_encoding import get_comp_list


def get_efficiencies_file(config, num_groups, sigmoid):

    paths = get_config_paths()
    efficiencies_file = os.path.join(paths.comp_data_dir,
                                     config,
                                     'efficiencies',
                                     'efficiency_fit_num_groups_{}_sigmoid-{}.hdf'.format(num_groups, sigmoid))

    return efficiencies_file


def get_detector_efficiencies(config, num_groups, sigmoid='slant',
                              pyunfold_format=False):
    """Loads fitted detection efficiencies vs. energy values

    Parameters
    ----------
    config : str
        Detector configuration.
    num_groups : int
        Number of composition groups.
    sigmoid : {'slant', 'flat'}
        Whether to use a flant or slanted sigmoid function to fit the
        detection efficiency vs. energy.
    pyunfold_format : bool, optional
        Whether or not to return formatted efficiencies for use in the
        PyUnfold response matrix (default is False).

    Returns
    -------
    df_eff : pandas.DataFrame
        Returned if pyunfold_format is False. Pandas DataFrame containing
        the detection efficiencies and errors for each composition.
    efficiencies, efficiencies_err : numpy.ndarray
        Returned if pyunfold_format is True. PyUnfold formatted detection
        efficiencies and efficiencies errors.
    """

    valid_configs = get_sim_configs()
    if config not in valid_configs:
        raise ValueError('Invalid detector configuration {}. '
                         'Must be in {}'.format(config, valid_configs))
    if not sigmoid in ['slant', 'flat']:
        raise ValueError('sigmoid must be either "slant" or "flat".')

    efficiencies_file = get_efficiencies_file(config, num_groups, sigmoid)
    if not os.path.exists(efficiencies_file):
        raise IOError('Detector efficiencies file, {}, doesn\'t '
                      'exist...'.format(efficiencies_file))
    df_eff = pd.read_hdf(efficiencies_file)
    if pyunfold_format:
        # Format for PyUnfold response matrix use
        energybins = get_energybins()
        comp_list = get_comp_list(num_groups=num_groups)
        efficiencies = np.empty(num_groups * len(energybins.energy_midpoints))
        efficiencies_err = np.empty(num_groups * len(energybins.energy_midpoints))
        for idx, composition in enumerate(comp_list):
            efficiencies[idx::num_groups] = df_eff['eff_median_{}'.format(composition)]
            efficiencies_err[idx::num_groups] = df_eff['eff_err_low_{}'.format(composition)]

        return efficiencies, efficiencies_err
    else:
        return df_eff

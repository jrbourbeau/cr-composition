
import os
import pandas as pd
import datetime

from .base import get_config_paths


def get_livetime_file():

    paths = get_config_paths()
    livetime_file = os.path.join(paths.comp_data_dir, 'data_livetimes.csv')

    return livetime_file


def get_detector_livetime(config=None, months=None):

    if config is None:
        raise ValueError('Detector configuration must be specified')

    livetime_file = get_livetime_file()
    try:
        livetime_df = pd.read_csv(livetime_file, index_col=0)
    except IOError:
        raise IOError('Livetime DataFrame file, {}, doesn\'t '
                      'exist...'.format(livetime_file))

    try:
        if months is None:
                livetime = livetime_df.loc[config]['livetime(s)']
                livetime_err = livetime_df.loc[config]['livetime_err(s)']
        else:
            livetime, livetime_err = 0., 0.
            for month in months:
                month_str = datetime.date(2000, month, 1).strftime('%B')
                livetime += livetime_df.loc[config]['{}_livetime(s)'.format(month_str)]
                livetime_err += livetime_df.loc[config]['{}_livetime_err(s)'.format(month_str)]
    except KeyError:
        raise KeyError('Detector configuration {} doesn\'t exist in '
                       'the livetime DataFrame'.format(config))

    return livetime, livetime_err

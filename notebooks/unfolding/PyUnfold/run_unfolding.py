#!/usr/bin/env python

import os
import pandas as pd
from dask import delayed, multiprocessing

from Unfold import Unfold


if __name__ == '__main__':

    # priors_list = ['h3a']
    priors_list = ['h3a', 'antih3a', 'h4a', 'Hoerandel5', 'antiHoerandel5']
    # priors_list = ['h3a', 'antih3a', 'h4a', 'Hoerandel5', 'antiHoerandel5', 'uniform', 'alllight', 'allheavy']

    # Load formatted DataFrame
    formatted_df_outfile = os.path.join('/data/user/jbourbeau/composition',
                    'unfolding', 'unfolding-dataframe-PyUnfold-formatted.csv')
    formatted_df = pd.read_csv(formatted_df_outfile, index_col='log_energy_bin_idx')

    effects = formatted_df['counts'].values
    effects_err = formatted_df['counts_err'].values

    for prior in priors_list:
        f = 'unfolded_output_{}.root'.format(prior)
        if os.path.exists(f):
            print('Removing unfolded_output_{}.root...'.format(prior))
            os.remove(f)

        priors = formatted_df['{}_priors'.format(prior)].values

        Unfold(config_name='config_{}.cfg'.format(prior), return_dists=False,
               EffDist=effects, effects_err=effects_err,
               priors=priors, plot_local=False, df_outfile=f.replace('.root', '.hdf'))

        # os.system('python Unfold.py -c config_{}.cfg --fluxmodel {}'.format(prior, prior))

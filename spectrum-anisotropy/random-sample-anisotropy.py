#!/usr/bin/env python

from __future__ import division, print_function
import argparse
import numpy as np
import pandas as pd
import healpy as hp
import dask
from scipy import stats
from scipy.special import erfcinv

import comptools as comp
import sky_anisotropy as sa


def calculate_local_sigma(df, nside=64, bins=None, random_state=None):

    if bins is None:
        raise ValueError('bins cannot be None')

    if random_state is None:
        ra = df.loc[:, 'lap_ra'].values
    else:
        ra = df.loc[:, 'lap_ra'].sample(frac=1.0, random_state=random_state).values
    dec = df.loc[:, 'lap_dec'].values

    theta, phi = comp.equatorial_to_healpy(ra, dec)
    pix_array = hp.ang2pix(nside, theta, phi)
    df['pix'] = pix_array

    npix = hp.nside2npix(nside)
    map_pix = np.arange(npix)
    theta, phi = hp.pix2ang(nside, map_pix)
    ra, dec = sa.healpy_to_equatorial(theta, phi)

    dec_max_deg = -65
    size = np.deg2rad(5)
    on_region = 'disc'
    off_region = 'theta_band'

    has_data = dec < np.deg2rad(dec_max_deg)
    if off_region == 'theta_band':
        has_data = has_data & (dec > np.deg2rad(-90) + size)

    pix_disc = map_pix[has_data]

    data = df.loc[:, ['reco_log_energy', 'pred_comp_target']].values
    pix = df.loc[:, 'pix'].values

    binned_skymaps = sa.binned_skymaps(data=data,
                                       pix=pix,
                                       bins=bins,
                                       nside=nside)

    with dask.config.set(scheduler='sync', num_workers=1):
        results = sa.on_off_chi_squared(binned_maps=binned_skymaps,
                                        pix_center=pix_disc,
                                        on_region=on_region,
                                        size=size,
                                        off_region=off_region,
                                        nside=nside,
                                        hist_func=unfolding_func,
                                        )

    dof = 13
    pval = stats.chi2.sf(results['chi2'], dof)
    sig = erfcinv(2 * pval) * np.sqrt(2)

    return sig.max()


if __name__ == '__main__':

    description = 'Extracts and saves desired information from simulation/data .i3 files'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('-c', '--config',
                   dest='config',
                   default='IC86.2012',
                   choices=comp.datafunctions.get_data_configs(),
                   help='Detector configuration')
    p.add_argument('--composition',
                   dest='composition',
                   default='total',
                   choices=['light', 'heavy', 'total'],
                   help='Whether to make individual skymaps for each composition')
    p.add_argument('--random_states',
                   dest='random_states',
                   type=int,
                   nargs='*',
                   help='Random states to use.')
    p.add_argument('--outfile',
                   dest='outfile',
                   help='Output file path')
    args = p.parse_args()

    if args.outfile is None:
        raise ValueError('Outfile must be specified')
    else:
        comp.check_output_dir(args.outfile)

    config = args.config
    num_groups = 2
    comp_list = comp.get_comp_list(num_groups=num_groups)
    analysis_bins = comp.get_bins(config=config, num_groups=num_groups)
    energybins = comp.get_energybins(config)
    num_ebins = len(energybins.log_energy_midpoints)
    nside = 64

    feature_list, feature_labels = comp.get_training_features()

    energy_pipeline_name = 'linearregression_energy_{}'.format(config)
    energy_pipeline = comp.load_trained_model(energy_pipeline_name)

    # pipeline_str = 'SGD_comp_{}_{}-groups'.format(config, num_groups)
    comp_pipeline_name = 'xgboost_comp_{}_{}-groups'.format(config, num_groups)
    comp_pipeline = comp.load_trained_model(comp_pipeline_name)

    df_data = pd.read_hdf('data_dataframe.hdf', 'dataframe', mode='r')

    print('Running energy reconstruction...')
    X_data = df_data[feature_list].values
    df_data['reco_log_energy'] = energy_pipeline.predict(X_data)
    df_data['reco_energy'] = 10**df_data['reco_log_energy']
    print('Running composition classifications...')
    df_data['pred_comp_target'] = comp_pipeline.predict(X_data)

    def unfolding_func(counts, composition='total'):
        counts_err = np.sqrt(counts)

        counts_total = counts.sum(axis=1)
        counts_err_total = np.sqrt(np.sum(counts_err**2, axis=1))

        unfolding_energy_range_mask = np.logical_and(energybins.log_energy_midpoints >= 6.4,
                                                     energybins.log_energy_midpoints <= 7.8)

        return counts_total[unfolding_energy_range_mask], counts_err_total[unfolding_energy_range_mask]

    sig_max = [calculate_local_sigma(df=df_data, nside=nside, bins=analysis_bins, random_state=i)
               for i in args.random_states]
    sig_max = np.array(sig_max)
    np.save(args.outfile, sig_max)

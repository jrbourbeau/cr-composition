#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse

import comptools as comp


if __name__ == '__main__':

    description = 'Saves processed data with quality cuts applied'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config',
                        dest='config',
                        default='IC86.2012',
                        choices=comp.simfunctions.get_sim_configs(),
                        help='Detector configuration')
    parser.add_argument('--n_jobs',
                        dest='n_jobs',
                        type=int,
                        default=10,
                        help='Number of jobs to run in parallel')
    args = parser.parse_args()

    config = args.config
    energybins = comp.get_energybins(config=config)
    log_energy_min = energybins.log_energy_min
    log_energy_max = energybins.log_energy_max

    print('Loading full pre-processed dataset for {} into memory...'.format(config))
    df_data = comp.load_data(config=config,
                             processed=False,
                             energy_reco=True,
                             energy_cut_key='reco_log_energy',
                             log_energy_min=log_energy_min,
                             log_energy_max=log_energy_max,
                             n_jobs=args.n_jobs,
                             verbose=True)

    outfile = os.path.join(comp.paths.comp_data_dir,
                           config,
                           'data',
                           'data_dataframe_quality_cuts.hdf'
                           )
    comp.check_output_dir(outfile)
    print('Saving processed dataset to {}...'.format(outfile))
    df_data.to_hdf(outfile, 'dataframe', format='table')

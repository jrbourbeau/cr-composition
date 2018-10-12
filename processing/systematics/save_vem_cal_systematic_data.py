#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
import dask.array as da
from dask.diagnostics import ProgressBar

import comptools as comp


def add_reco_energy(partition, pipeline, feature_list):
    partition['reco_log_energy'] = pipeline.predict(partition[feature_list])
    partition['reco_energy'] = 10**partition['reco_log_energy']
    return partition


def apply_energy_cut(partition, log_energy_min, log_energy_max):
    energy_mask = (partition['reco_log_energy'] > log_energy_min) & (partition['reco_log_energy'] < log_energy_max)
    return partition.loc[energy_mask, :]


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
    n_jobs = args.n_jobs
    energybins = comp.get_energybins(config=config)
    log_energy_min = energybins.log_energy_min
    log_energy_max = energybins.log_energy_max
    feature_list, feature_labels = comp.get_training_features()

    print('Loading full non-processed dataset for {} into memory...'.format(config))
    ddf = comp.load_sim(config=config,
                        # processed=False,
                        test_size=0,
                        energy_reco=False,
                        log_energy_min=None,
                        log_energy_max=None,
                        compute=False)

    # ddf = comp.load_data(config=config,
    #                      processed=False,
    #                      energy_reco=False,
    #                      log_energy_min=None,
    #                      log_energy_max=None,
    #                      compute=False)

    # Energy reconstruction model
    energy_pipeline = comp.load_trained_model('linearregression_energy_{}'.format(config),
                                              return_metadata=False)

    for shift_type in ['up', 'down']:
        print('Processing VEM calibration {} shifted dataset...'.format(shift_type))
        s125_scaling_factor = 1.03 if shift_type == 'up' else 0.97

        # Process data:
        #     - Shift S125 value to account for VEM calibration systematic uncertainty
        #     - Energy reconstruction
        #     - Energy range cut
        ddf_systematic = (ddf.assign(lap_s125=s125_scaling_factor * ddf.lap_s125)
                             .assign(log_s125 = lambda x: da.log10(x.lap_s125))
                             .map_partitions(add_reco_energy, energy_pipeline, feature_list)
                             .map_partitions(apply_energy_cut, log_energy_min, log_energy_max)
                          )

        outfile = os.path.join(comp.paths.comp_data_dir,
                               config,
                            #    'data',
                               'sim',
                               'sim_dataframe_vem_cal_{}.hdf'.format(shift_type)
                               )
        comp.check_output_dir(outfile)
        with ProgressBar():
            # Want to compute before saving to disk so we can reset_index
            df_systematic = ddf_systematic.compute(scheduler='processes',
                                                   num_workers=n_jobs)
            df_systematic = df_systematic.reset_index(drop=True)
        print('Saving processed VEM calibration systematic dataset to {}...'.format(outfile))
        df_systematic.to_hdf(outfile,
                             key='dataframe',
                             format='table',
                             mode='w')

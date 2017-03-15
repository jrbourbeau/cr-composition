#!/usr/bin/env python
# !/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v1/icetray-start
# METAPROJECT /data/user/jbourbeau/metaprojects/icerec/trunk/build

import numpy as np
import pandas as pd
import time
import glob
import argparse
import os
from collections import defaultdict

# import pyprind
# from icecube.weighting.weighting import from_simprod
# from icecube.weighting.fluxes import GaisserH3a, GaisserH4a

import composition as comp


if __name__ == "__main__":
    # Setup global path names
    mypaths = comp.Paths()
    comp.checkdir(mypaths.comp_data_dir)

    p = argparse.ArgumentParser(
        description='Runs extra modules over a given fileList')
    p.add_argument('--type', dest='type',
                   choices=['data', 'sim'],
                   default='sim',
                   help='Option to process simulation or data')
    p.add_argument('-c', '--config', dest='config',
                   default='IC79',
                   help='Detector configuration')
    p.add_argument('-o', '--outfile', dest='outfile',
                   help='Output file')
    args = p.parse_args()

    dataframe_dict = defaultdict(list)

    # Get simulation information
    t_sim = time.time()
    print('Loading simulation information...')
    file_list = sorted(glob.glob(mypaths.comp_data_dir +
                                 '/{}_{}/files/{}_*.hdf5'.format(args.config, args.type, args.type)))
    file_list = [f for f in file_list if 'part' not in f]
    # if args.type == 'sim':
    #     high_energy_sim = ['7579', '7791', '7851', '7784']
    #     file_list = [f for f in file_list if os.path.splitext(f)[0].split('_')[-1] in high_energy_sim]
    #     print('file_list = {}'.format(file_list))
    value_keys = ['IceTopMaxSignal',
                  'IceTopMaxSignalInEdge',
                  'IceTopMaxSignalString',
                  'IceTopNeighbourMaxSignal',
                  'NChannels_1_60', 'NHits_1_60', 'InIce_charge_1_60', 'max_qfrac_1_60',
                  'NChannels_1_30', 'NHits_1_30', 'InIce_charge_1_30', 'max_qfrac_1_30',
                  #   'InIce_charge_1_45', 'NChannels_1_45', 'max_qfrac_1_45',
                  #   'InIce_charge_1_30', 'NChannels_1_30', 'max_qfrac_1_30',
                  #   'InIce_charge_1_15', 'NChannels_1_15', 'max_qfrac_1_15',
                  #   'InIce_charge_45_60', 'NChannels_45_60', 'max_qfrac_45_60',
                  #   'InIce_charge_1_6', 'NChannels_1_6', 'max_qfrac_1_6',
                  'NStations',
                  'StationDensity',
                  'Laputop_IceTop_FractionContainment',
                  'Laputop_InIce_FractionContainment']
    # Add MC containment
    if args.type == 'sim':
        value_keys += ['IceTop_FractionContainment',
                       'InIce_FractionContainment']

    if args.type == 'data':
        file_list = file_list[:20]

    # bar = pyprind.ProgBar(len(file_list), bar_char='#',
    #                       monitor=True, title='Save DataFrame', width=60)
    for i, f in enumerate(file_list):
        print('\tWorking on {} [file {} of {}]'.format(f, i+1, len(file_list)))
        hdf5_dict = {}
        store = pd.HDFStore(f)
        for key in value_keys:
            hdf5_dict[key] = store.select(key).value
        # Get MCPrimary information
        if args.type == 'sim':
            for key in ['x', 'y', 'energy', 'zenith', 'azimuth', 'type']:
                hdf5_dict['MC_{}'.format(key)] = store.select('MCPrimary')[key]
            # Add simulation set number and corresponding composition
            sim_num = os.path.splitext(f)[0].split('_')[-1]
            hdf5_dict['sim'] = np.array(
                [sim_num] * len(store.select('MCPrimary')))
            hdf5_dict['MC_comp'] = np.array(
                [comp.simfunctions.sim2comp(sim_num)] * len(store.select('MCPrimary')))
            hdf5_dict['MC_comp_class'] = np.array(
                ['light' if comp.simfunctions.sim2comp(sim_num) in ['P', 'He'] else 'heavy'] * len(store.select('MCPrimary')))
        # Get timing information
        hdf5_dict['start_time_mjd'] = store.select('I3EventHeader')[
            'time_start_mjd']
        hdf5_dict['end_time_mjd'] = store.select('I3EventHeader')[
            'time_end_mjd']
        # Get Laputop data
        Laputop_particle = store.select('Laputop')
        for dist in ['50', '80', '125', '180', '250', '500']:
            hdf5_dict['lap_s{}'.format(dist)] = store.select(
                'LaputopParams')['s{}'.format(dist)]
        hdf5_dict['lap_chi2'] = store.select('LaputopParams')[
            'chi2'] / store.select('LaputopParams')['ndf']
        hdf5_dict['lap_ndf'] = store.select('LaputopParams')['ndf']
        hdf5_dict['lap_beta'] = store.select('LaputopParams')['beta']
        hdf5_dict['lap_x'] = store.select('Laputop')['x']
        hdf5_dict['lap_y'] = store.select('Laputop')['y']
        hdf5_dict['lap_zenith'] = store.select('Laputop')['zenith']
        hdf5_dict['lap_energy'] = store.select('LaputopParams')['e_h4a']
        hdf5_dict['lap_likelihood'] = store.select('LaputopParams')['rlogl']
        hdf5_dict['lap_fitstatus_ok'] = store.select(
            'Laputop_fitstatus_ok').value.astype(bool)
        store.close()
        for key in hdf5_dict.keys():
            dataframe_dict[key] += hdf5_dict[key].tolist()

        hdf5_dict.clear()
        # bar.update(force_flush=True)

    # # Calculate simulation event weights
    # print('\nCalculating simulation event weights...\n')
    # simlist = np.unique(dataframe_dict['sim'])
    # num_files_dict = {'7006': 30000, '7007': 30000,
    #                   '7579': 12000, '7784': 12000}
    # for i, sim in enumerate(simlist):
    #     if i == 0:
    #         generator = num_files_dict[sim] * from_simprod(int(sim))
    #     else:
    #         generator += num_files_dict[sim] * from_simprod(int(sim))
    # flux = GaisserH3a()
    # dataframe_dict['weights_H3a'] = flux(dataframe_dict['MC_energy'], dataframe_dict[
    #                         'MC_type']) / generator(dataframe_dict['MC_energy'], dataframe_dict['MC_type'])
    # flux = GaisserH4a()
    # dataframe_dict['weights_H4a'] = flux(dataframe_dict['MC_energy'], dataframe_dict['MC_type']) / \
    #     generator(dataframe_dict['MC_energy'], dataframe_dict['MC_type'])
    # dataframe_dict['areas'] = 1.0 / generator(dataframe_dict['MC_energy'], dataframe_dict['MC_type'])
    print('Time taken: {}'.format(time.time() - t_sim))
    print('Time per file: {}\n'.format((time.time() - t_sim) / len(file_list)))

    # # Get ShowerLLH reconstruction information
    # t_LLH = time.time()
    # print('Loading ShowerLLH reconstructions...')
    # file_list = sorted(glob.glob(mypaths.llh_dir +
    #                              '/IT73_sim/files/SimLLH_????_logdist.hdf5'))
    # for f in file_list:
    #     print('\tWorking on {}'.format(f))
    #     LLH_dict = {}
    #     store = pd.HDFStore(f)
    #     # Get most-likely composition
    #     LLH_particle = store.select('ShowerLLH')
    #     LLH_dict['reco_exists'] = LLH_particle.exists.astype(bool)
    #     # Get ML energy
    #     LLH_dict['reco_energy'] = LLH_particle.energy
    #     # Get ML core position
    #     LLH_dict['reco_x'] = LLH_particle.x
    #     LLH_dict['reco_y'] = LLH_particle.y
    #     # Get ML core radius
    #     LLH_dict['reco_radius'] = np.sqrt(
    #         LLH_dict['reco_x']**2 + LLH_dict['reco_y']**2)
    #     # Get ML zenith
    #     LLH_dict['reco_zenith'] = LLH_particle.zenith
    #     # Get ShowerLLH containment information
    #     LLH_dict['reco_IT_containment'] = store.select(
    #         'ShowerLLH_IceTop_containment').value
    #     LLH_dict['reco_InIce_containment'] = store.select(
    #         'ShowerLLH_InIce_containment').value
    #     # Get ShowerLLH+lap hybrid containment information
    #     LLHLF_particle = store.select('LLHLF_particle')
    #     LLH_dict['LLHLF_zenith'] = LLHLF_particle.zenith
    #     LLH_dict['LLHLF_IT_containment'] = store.select(
    #                                         'LLHLF_IceTop_containment').value
    #     LLH_dict['LLHLF_InIce_containment'] = store.select(
    #                                         'LLHLF_InIce_containment').value
    #     LLH_dict['LLHLF_reco_exists'] = LLHLF_particle.exists.astype(bool)
    #     # Get ShowerLLH+lap hybrid containment information
    #     LLHlap_particle = store.select('LLHlap_particle')
    #     LLH_dict['LLHlap_zenith'] = LLHlap_particle.zenith
    #     LLH_dict['LLHlap_IT_containment'] = store.select(
    #                                         'LLHlap_IceTop_containment').value
    #     LLH_dict['LLHlap_InIce_containment'] = store.select(
    #                                         'LLHlap_InIce_containment').value
    #     LLH_dict['LLHlap_reco_exists'] = LLHlap_particle.exists.astype(bool)
    #
    #     store.close()
    #
    #     for key in LLH_dict.keys():
    #         dataframe_dict[key] += LLH_dict[key].tolist()
    #
    # print('Time taken: {}'.format(time.time() - t_LLH))
    # print('Time per file: {}'.format((time.time() - t_LLH) / 4))

    # Convert value lists to arrays (faster than using np.append?)
    for key in dataframe_dict.keys():
        dataframe_dict[key] = np.asarray(dataframe_dict[key])

    df = pd.DataFrame.from_dict(dataframe_dict)
    df.to_hdf('{}/{}_{}/{}_dataframe.hdf5'.format(mypaths.comp_data_dir, args.config, args.type, args.type),
              'dataframe', mode='w')
    # print(bar)

#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v1/icetray-start
#METAPROJECT /data/user/jbourbeau/metaprojects/icerec/V05-00-00/build

import numpy as np
import pandas as pd
import time
import glob
import argparse
import os
from collections import defaultdict

from icecube.weighting.weighting import from_simprod
from icecube.weighting.fluxes import GaisserH3a, GaisserH4a

import composition as comp


if __name__ == "__main__":
    # Setup global path names
    mypaths = comp.Paths()
    comp.checkdir(mypaths.comp_data_dir)

    p = argparse.ArgumentParser(
        description='Runs extra modules over a given fileList')
    p.add_argument('-o', '--outfile', dest='outfile',
                   help='Output file')
    args = p.parse_args()

    dataframe_dict = defaultdict(list)

    # Get simulation information
    t_sim = time.time()
    print('Loading simulation information...')
    file_list = sorted(glob.glob(mypaths.comp_data_dir +
                                 '/IT73_sim/files/sim_????.hdf5'))
    value_keys = ['IceTopMaxSignal',
                  'IceTopMaxSignalInEdge',
                  'IceTopMaxSignalString',
                  'IceTopNeighbourMaxSignal',
                  'InIce_charge_1_60',
                  'NChannels_1_60',
                  'InIce_charge_1_45',
                  'NChannels_1_45',
                  'InIce_charge_1_30',
                  'NChannels_1_30',
                  'InIce_charge_1_15',
                  'NChannels_1_15',
                  'NStations',
                  'StationDensity',
                  'IceTop_FractionContainment',
                  'InIce_FractionContainment']
    for f in file_list:
        print('\tWorking on {}'.format(f))
        sim_dict = {}
        store = pd.HDFStore(f)
        for key in value_keys:
            sim_dict[key] = store.select(key).value
        # Get MCPrimary information
        for key in ['x', 'y', 'energy', 'zenith', 'azimuth', 'type']:
            sim_dict['MC_{}'.format(key)] = store.select('MCPrimary')[key]
        # Get s125
        sim_dict['s125'] = store.select('LaputopParams')['s125']
        # Get ShowerPlane zenith reconstruction
        sim_dict['ShowerPlane_zenith'] = store.select('ShowerPlane').zenith
        # Add simulation set number and corresponding composition
        sim_num = os.path.splitext(f)[0].split('_')[-1]
        sim_dict['sim'] = np.array([sim_num] * len(store.select('MCPrimary')))
        sim_dict['MC_comp'] = np.array(
            [comp.simfunctions.sim2comp(sim_num)] * len(store.select('MCPrimary')))
        # Get Laputop reduced chi-squared
        sim_dict['lap_chi2'] = store.select('LaputopParams')[
            'chi2'] / store.select('LaputopParams')['ndf']
        sim_dict['lap_x'] = store.select('Laputop')['x']
        sim_dict['lap_y'] = store.select('Laputop')['y']
        store.close()
        for key in sim_dict.keys():
            dataframe_dict[key] += sim_dict[key].tolist()

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
    print('Time per file: {}\n'.format((time.time() - t_sim) / 4))

    # Get ShowerLLH reconstruction information
    t_LLH = time.time()
    print('Loading ShowerLLH reconstructions...')
    file_list = sorted(glob.glob(mypaths.llh_dir +
                                 '/IT73_sim/files/SimLLH_????_logdist.hdf5'))
    for f in file_list:
        print('\tWorking on {}'.format(f))
        LLH_dict = {}
        store = pd.HDFStore(f)
        # Get most-likely composition
        LLH_particle = store.select('ShowerLLH')
        LLH_dict['reco_exists'] = LLH_particle.exists.astype(bool)
        # Get ML energy
        LLH_dict['reco_energy'] = LLH_particle.energy
        # Get ML core position
        LLH_dict['reco_x'] = LLH_particle.x
        LLH_dict['reco_y'] = LLH_particle.y
        # Get ML core radius
        LLH_dict['reco_radius'] = np.sqrt(
            LLH_dict['reco_x']**2 + LLH_dict['reco_y']**2)
        # Get ML zenith
        LLH_dict['reco_zenith'] = LLH_particle.zenith
        # Get ShowerLLH containment information
        LLH_dict['reco_IT_containment'] = store.select(
            'ShowerLLH_IceTop_containment').value
        LLH_dict['reco_InIce_containment'] = store.select(
            'ShowerLLH_InIce_containment').value
        # Get ShowerLLH+lap hybrid containment information
        LLH_dict[
            'LLHlap_IT_containment'] = store.select('LLHlap_IceTop_containment').value
        LLH_dict[
            'LLHlap_InIce_containment'] = store.select('LLHlap_InIce_containment').value
        LLH_dict['combined_reco_exists'] = store.select(
            'LLHlap_InIce_containment').exists.astype(bool)

        store.close()

        for key in LLH_dict.keys():
            dataframe_dict[key] += LLH_dict[key].tolist()

    print('Time taken: {}'.format(time.time() - t_LLH))
    print('Time per file: {}'.format((time.time() - t_LLH) / 4))

    # Convert value lists to arrays (faster than using np.append?)
    for key in dataframe_dict.keys():
        dataframe_dict[key] = np.asarray(dataframe_dict[key])

    df = pd.DataFrame.from_dict(dataframe_dict)
    df.to_hdf('{}/IT73_sim/sim_dataframe.hdf5'.format(mypaths.comp_data_dir),
              'dataframe', mode='w')

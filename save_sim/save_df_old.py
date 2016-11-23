#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v1/icetray-start
#METAPROJECT /data/user/jbourbeau/metaprojects/icerec/V05-00-00/build

from __future__ import division
import numpy as np
import pandas as pd
import time
import glob
import argparse
import os
from collections import defaultdict

import composition.support_functions.simfunctions as simfunctions
import composition.support_functions.paths as paths
from composition.support_functions.checkdir import checkdir
# from ShowerLLH_scripts.analysis.zfix import zfix


if __name__ == "__main__":
    # Setup global path names
    mypaths = paths.Paths()
    checkdir(mypaths.comp_data_dir)

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
                  'InIce_charge',
                  'NChannels',
                  'max_charge_frac',
                  'NStations',
                  'StationDensity',
                  'IceTop_FractionContainment',
                  'InIce_FractionContainment',
                  'LineFit_InIce_FractionContainment']
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
            [simfunctions.sim2comp(sim_num)] * len(store.select('MCPrimary')))
        store.close()
        for key in sim_dict.keys():
            dataframe_dict[key] += sim_dict[key].tolist()
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
        proton_maxLLH = store.select('ShowerLLHParams_proton').maxLLH
        iron_maxLLH = store.select('ShowerLLHParams_iron').maxLLH
        LLH_array = np.array([proton_maxLLH, iron_maxLLH]).T
        maxLLH_index = np.argmax(LLH_array, axis=1)
        showerLLH_proton = store.select('ShowerLLH_proton')
        showerLLH_iron = store.select('ShowerLLH_iron')
        LLH_dict['reco_exists'] = showerLLH_proton.exists.astype(bool)
        # Get ML energy
        energy_choices = [showerLLH_proton.energy.values, showerLLH_iron.energy.values]
        LLH_dict['reco_energy'] = np.choose(maxLLH_index, energy_choices)
        # Get ML core position
        x_choices = [showerLLH_proton.x, showerLLH_iron.x]
        LLH_dict['reco_x'] = np.choose(maxLLH_index, x_choices)
        y_choices = [showerLLH_proton.y, showerLLH_iron.y]
        LLH_dict['reco_y'] = np.choose(maxLLH_index, y_choices)
        # Get ML core radius
        r_choices = [np.sqrt(showerLLH_proton.x**2 + showerLLH_proton.y**2),
                     np.sqrt(showerLLH_iron.x**2 + showerLLH_iron.y**2)]
        LLH_dict['reco_radius'] = np.choose(maxLLH_index, r_choices)
        # Get ML zenith
        zenith_choices = [showerLLH_proton.zenith, showerLLH_iron.zenith]
        LLH_dict['reco_zenith'] = np.choose(maxLLH_index, zenith_choices)
        # Get ShowerLLH containment information
        IT_containment_choices = [store.select('ShowerLLH_IceTop_containment_proton').value,
                                  store.select('ShowerLLH_IceTop_containment_iron').value]
        LLH_dict['reco_IT_containment'] = np.choose(
            maxLLH_index, IT_containment_choices)
        InIce_containment_choices = [store.select('ShowerLLH_InIce_containment_proton').value,
                                     store.select('ShowerLLH_InIce_containment_iron').value]
        LLH_dict['reco_InIce_containment'] = np.choose(
            maxLLH_index, InIce_containment_choices)
        # Get ShowerLLH+lap hybrid containment information
        IT_containment_choices = [store.select('LLH-lap_IceTop_containment_proton').value,
                                  store.select('LLH-lap_IceTop_containment_iron').value]
        LLH_dict['reco_IT_containment'] = np.choose(
            maxLLH_index, IT_containment_choices)
        InIce_containment_choices = [store.select('LLH-lap_InIce_containment_proton').value,
                                     store.select('LLH-lap_InIce_containment_iron').value]
        LLH_dict['reco_InIce_containment'] = np.choose(
            maxLLH_index, InIce_containment_choices)

        # LLH_dict['reco_energy'] = 10**(np.log10(LLH_dict['reco_energy'])-zfix(np.pi-LLH_dict['reco_zenith']))

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

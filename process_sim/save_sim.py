#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v1/icetray-start
#METAPROJECT /data/user/jbourbeau/metaprojects/icerec/trunk/build

import time
import argparse
import os

from icecube import dataio, toprec, dataclasses, icetray, phys_services
from icecube.frame_object_diff.segments import uncompress
from I3Tray import *
from icecube.tableio import I3TableWriter
from icecube.hdfwriter import I3HDFTableService
from icecube.icetop_Level3_scripts.functions import count_stations

import composition as comp
import composition.i3modules as i3modules


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description='Runs extra modules over a given fileList')
    p.add_argument('-f', '--files', dest='files', nargs='*',
                   help='Files to run over')
    p.add_argument('-s', '--sim', dest='sim',
                   help='Simulation dataset')
    p.add_argument('-o', '--outfile', dest='outfile',
                   help='Output file')
    args = p.parse_args()

    # Starting parameters
    IT_pulses, inice_pulses = comp.simfunctions.reco_pulses()

    # Keys to write to frame
    keys = []
    keys += ['I3EventHeader']
    keys += ['ShowerPlane']
    keys += ['ShowerCOG']
    keys += ['MCPrimary']
    keys += ['IceTopMaxSignal', 'IceTopMaxSignalString',
             'IceTopMaxSignalInEdge', 'IceTopNeighbourMaxSignal',
             'StationDensity', 'NStations']
    keys += ['NChannels_1_60', 'InIce_charge_1_60', 'max_qfrac_1_60']
    # keys += ['NChannels_1_45', 'InIce_charge_1_45', 'max_qfrac_1_45']
    keys += ['NChannels_1_30', 'InIce_charge_1_30', 'max_qfrac_1_30']
    # keys += ['NChannels_1_15', 'InIce_charge_1_15', 'max_qfrac_1_15']
    # keys += ['NChannels_1_6', 'InIce_charge_1_6', 'max_qfrac_1_6']
    keys += ['NChannels_45_60', 'InIce_charge_45_60', 'max_qfrac_45_60']
    keys += ['InIce_FractionContainment', 'IceTop_FractionContainment']
    keys += ['Laputop_InIce_FractionContainment',
             'Laputop_IceTop_FractionContainment']
    keys += ['Laputop', 'LaputopParams']
    keys += ['Laputop_fitstatus_ok']

    t0 = time.time()

    # Construct list of non-truncated files to process
    # icetray.set_log_level(icetray.I3LogLevel.LOG_DEBUG)
    good_file_list = []
    for test_file in args.files:
        try:
            test_tray = I3Tray()
            test_tray.context['I3FileStager'] = dataio.get_stagers(
                staging_directory=os.environ['_CONDOR_SCRATCH_DIR'])
            test_tray.Add('I3Reader', FileName=test_file)
            test_tray.Add(uncompress, 'uncompress')
            test_tray.Execute()
            test_tray.Finish()
            good_file_list.append(test_file)
        except:
            print('file {} is truncated'.format(test_file))
            pass
    del test_tray

    tray = I3Tray()
    tray.context['I3FileStager'] = dataio.get_stagers(
        staging_directory=os.environ['_CONDOR_SCRATCH_DIR'])
    # icetray.logging.log_dedug('good_file_list = {}'.format(good_file_list))
    tray.Add('I3Reader', FileNameList=good_file_list)
    # Uncompress Level3 diff files
    tray.Add(uncompress, 'uncompress')
    hdf = I3HDFTableService(args.outfile)

    # Filter out non-coincident P frames
    tray.Add(lambda frame: frame['IceTopInIce_StandardFilter'].value)

    def get_nstations(frame):
        nstation = 0
        if IT_pulses in frame:
            nstation = count_stations(
                dataclasses.I3RecoPulseSeriesMap.from_frame(frame, IT_pulses))
        frame.Put('NStations', icetray.I3Int(nstation))

    tray.Add(get_nstations)

    # Add total inice charge to frame
    tray.Add(i3modules.AddInIceCharge, inice_pulses=inice_pulses,
             min_DOM=1, max_DOM=60)
    # tray.Add(i3modules.AddInIceCharge, inice_pulses=inice_pulses,
    #          min_DOM=1, max_DOM=45)
    tray.Add(i3modules.AddInIceCharge, inice_pulses=inice_pulses,
             min_DOM=1, max_DOM=30)
    # tray.Add(i3modules.AddInIceCharge, inice_pulses=inice_pulses,
    #          min_DOM=1, max_DOM=15)
    # tray.Add(i3modules.AddInIceCharge, inice_pulses=inice_pulses,
    #          min_DOM=1, max_DOM=6)
    tray.Add(i3modules.AddInIceCharge, inice_pulses=inice_pulses,
             min_DOM=45, max_DOM=60)

    # Add containment to frame
    tray.Add(i3modules.AddMCContainment)
    tray.Add(i3modules.AddInIceRecoContainment)

    # Add Laputop fit status to frame
    def lap_fitstatus_ok(frame):
        status_ok = False
        if 'Laputop' in frame:
            lap_particle = frame['Laputop']
            if (lap_particle.fit_status == dataclasses.I3Particle.OK):
                status_ok = True

        frame.Put('Laputop_fitstatus_ok', icetray.I3Bool(status_ok))

    tray.Add(lap_fitstatus_ok)

    #====================================================================
    # Finish

    tray.Add(I3TableWriter, tableservice=hdf, keys=keys,
             SubEventStreams=['ice_top'])

    tray.Execute()
    tray.Finish()

    print('Time taken: {}'.format(time.time() - t0))

#!/usr/bin/env python
#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v2/icetray-start
#METAPROJECT /data/user/jbourbeau/metaprojects/icerec/trunk/build

import time
import argparse
import os
import socket

from icecube import dataio, toprec, dataclasses, icetray, phys_services, stochastics, millipede
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
    p.add_argument('--type', dest='type',
                   choices=['data', 'sim'],
                   default='sim',
                   help='Option to process simulation or data')
    p.add_argument('-s', '--sim', dest='sim',
                   help='Simulation dataset')
    p.add_argument('-o', '--outfile', dest='outfile',
                   help='Output file')
    args = p.parse_args()

    # Starting parameters
    IT_pulses, inice_pulses = comp.datafunctions.reco_pulses()

    # Keys to write to frame
    keys = []
    if args.type == 'sim':
        keys += ['MCPrimary']
        # keys += ['MC_x', 'MC_y', 'MC_azimuth', 'MC_zenith', 'MC_energy', 'MC_type']
        keys += ['InIce_FractionContainment', 'IceTop_FractionContainment']

    keys += ['I3EventHeader']
    keys += ['IceTopMaxSignal', 'IceTopMaxSignalString',
             'IceTopMaxSignalInEdge', 'IceTopNeighbourMaxSignal',
             'StationDensity', 'NStations', 'IceTop_charge']
    for i in ['1_60', '1_30']:
        keys += ['avg_inice_radius_'+i, 'max_inice_radius_'+i]
    for i in ['1_60']:
        keys += ['NChannels_'+i, 'NHits_'+i, 'InIce_charge_'+i, 'max_qfrac_'+i]
    keys += ['Laputop_InIce_FractionContainment',
             'Laputop_IceTop_FractionContainment']
    keys += ['Laputop', 'LaputopParams']
    keys += ['Laputop_fitstatus_ok']

    keys += ['IceTopQualityCuts', 'InIceQualityCuts']
    for cut in ['MilliNCascAbove2', 'MilliQtotRatio', 'MilliRloglBelow2', 'NCh_CoincLaputopCleanedPulsesAbove7', 'StochRecoSucceeded']:
        keys += ['InIceQualityCuts_{}'.format(cut)]
    keys += ['Stoch_Reco', 'Stoch_Reco2', 'MillipedeFitParams', 'num_millipede_particles']

    # keys += ['avg_inice_radius', 'max_inice_radius']
    # for track in ['Laputop', 'CoincMuonReco_LineFit']:
    #     keys += ['avg_inice_radius_{}'.format(track),
    #         'invcharge_inice_radius_{}'.format(track),
    #         'max_inice_radius_{}'.format(track),
    #         'sum_earliest_tr_dist_{}'.format(track)]
    keys += ['angle_MC_Laputop']

    keys += ['tank_dists']
    keys += ['tank_charges']
    keys += ['IceTop_charge_175m']
    keys += ['refit_beta']

    t0 = time.time()

    # Construct list of non-truncated files to process
    # icetray.set_log_level(icetray.I3LogLevel.LOG_DEBUG)
    good_file_list = []
    for test_file in args.files:
        try:
            test_tray = I3Tray()
            if 'cobalt' not in socket.gethostname():
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
    if 'cobalt' not in socket.gethostname():
        tray.context['I3FileStager'] = dataio.get_stagers(
            staging_directory=os.environ['_CONDOR_SCRATCH_DIR'])
    tray.Add('I3Reader', FileNameList=good_file_list)
    # Uncompress Level3 diff files
    tray.Add(uncompress, 'uncompress')
    hdf = I3HDFTableService(args.outfile)

    # if args.type == 'data':
    # Filter out all events that don't pass standard IceTop cuts
    # tray.Add(lambda frame: frame['IT73AnalysisIceTopQualityCuts']['IceTop_StandardFilter'])
    tray.Add(lambda frame: all(frame['IT73AnalysisIceTopQualityCuts'].values()))
    # Filter out non-coincident P frames
    tray.Add(lambda frame: inice_pulses in frame)

    def add_quality_cuts_to_frame(frame):
        if 'IT73AnalysisIceTopQualityCuts' in frame:
            passed = all(frame['IT73AnalysisIceTopQualityCuts'].values())
            frame.Put('IceTopQualityCuts', icetray.I3Bool(passed))
        if 'IT73AnalysisInIceQualityCuts' in frame:
            passed = all(frame['IT73AnalysisInIceQualityCuts'].values())
            frame.Put('InIceQualityCuts', icetray.I3Bool(passed))
            # Add individual InIce quality cuts to frame
            for key, value in frame['IT73AnalysisInIceQualityCuts']:
                frame.Put('InIceQualityCuts_{}'.format(key), icetray.I3Bool(value))

    tray.Add(add_quality_cuts_to_frame)

    def get_nstations(frame):
        nstation = 0
        if IT_pulses in frame:
            nstation = count_stations(
                dataclasses.I3RecoPulseSeriesMap.from_frame(frame, IT_pulses))
        frame.Put('NStations', icetray.I3Int(nstation))

    tray.Add(get_nstations)

    # tray.Add(i3modules.AddIceTopCharge, icetop_pulses=IT_pulses)

    # Add total inice charge to frame
    tray.Add(i3modules.AddInIceCharge, inice_pulses=inice_pulses,
             min_DOM=1, max_DOM=60)
    # tray.Add(i3modules.AddInIceCharge, inice_pulses=inice_pulses,
    #          min_DOM=1, max_DOM=45)
    # tray.Add(i3modules.AddInIceCharge, inice_pulses=inice_pulses,
    #          min_DOM=1, max_DOM=30)
    # tray.Add(i3modules.AddInIceCharge, inice_pulses=inice_pulses,
    #          min_DOM=1, max_DOM=15)
    # tray.Add(i3modules.AddInIceCharge, inice_pulses=inice_pulses,
    #          min_DOM=1, max_DOM=6)
    # tray.Add(i3modules.AddInIceCharge, inice_pulses=inice_pulses,
    #          min_DOM=45, max_DOM=60)

    # Add muon radius
    # tray.Add(i3modules.AddMuonRadius, track='CoincMuonReco_LineFit', pulses=inice_pulses)
    tray.Add(i3modules.AddMuonRadius, track='Laputop', pulses='CoincLaputopCleanedPulses',
        min_DOM=1, max_DOM=60)
    tray.Add(i3modules.AddMuonRadius, track='Laputop', pulses='CoincLaputopCleanedPulses',
        min_DOM=1, max_DOM=30)

    # Add containment to frame
    tray.Add(i3modules.AddInIceRecoContainment)
    if args.type == 'sim':
        tray.Add(i3modules.AddMCContainment)
        # tray.Add(i3modules.addMCprimarykeys)

    # Add Laputop fit status to frame
    def lap_fitstatus_ok(frame):
        status_ok = False
        if 'Laputop' in frame:
            lap_particle = frame['Laputop']
            if (lap_particle.fit_status == dataclasses.I3Particle.OK):
                status_ok = True

        frame.Put('Laputop_fitstatus_ok', icetray.I3Bool(status_ok))

    tray.Add(lap_fitstatus_ok)

    # Add num_millipede_cascades to frame
    tray.Add(i3modules.add_num_mil_particles)

    tray.Add(i3modules.add_MC_Laputop_angle)


    # tray.Add(i3modules.AddIceTopChargeDistance, track='Laputop', pulses=IT_pulses)
    tray.Add(i3modules.AddIceTopChargeDistance, track='Laputop',
        pulses=['IceTopLaputopSeededSelectedHLC', 'IceTopLaputopSeededSelectedSLC'])

    #====================================================================
    # Finish

    tray.Add(I3TableWriter, tableservice=hdf, keys=keys,
             SubEventStreams=['ice_top'])

    tray.Execute()
    tray.Finish()

    print('Time taken: {}'.format(time.time() - t0))

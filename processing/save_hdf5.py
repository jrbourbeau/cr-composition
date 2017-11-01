#!/usr/bin/env python

import time
import argparse
import os
import socket

from icecube import (dataio, tableio, astro, toprec, dataclasses, icetray,
                     phys_services, stochastics, millipede, ddddr)
from icecube.frame_object_diff.segments import uncompress
from I3Tray import *
from icecube.tableio import I3TableWriter
from icecube.hdfwriter import I3HDFTableService
from icecube.icetop_Level3_scripts.functions import count_stations

import comptools
import comptools.icetray_software as icetray_software


def get_good_file_list(files):
    '''Checks that input i3 files aren't corrupted

    Parameters
    ----------
    files : array-like
        Iterable of i3 file paths to check.

    Returns
    -------
    good_file_list : list
        List of i3 files (from input files) that were able to be
        succeessfully loaded.

    '''
    good_file_list = []
    for i3file in files:
        try:
            test_tray = I3Tray()
            if 'cobalt' not in socket.gethostname():
                test_tray.context['I3FileStager'] = dataio.get_stagers(
                    staging_directory=os.environ['_CONDOR_SCRATCH_DIR'])
            test_tray.Add('I3Reader', FileName=i3file)
            test_tray.Add(uncompress, 'uncompress')
            test_tray.Execute()
            test_tray.Finish()
            good_file_list.append(i3file)
        except:
            icetray.logging.log_warn('File {} is truncated'.format(i3file))
            pass
        del test_tray

    return good_file_list


def check_keys(frame, *keys):
    '''Function to check if all keys are in frame

    Parameters
    ----------
    frame : I3Frame
        I3Frame
    keys:
        Series of keys to look for in frame

    Returns
    -------
    boolean
        Whether or not all the keys in keys are in frame

    '''
    return all([key in frame for key in keys])


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description='Runs extra modules over a given fileList')
    p.add_argument('-f', '--files', dest='files', nargs='*',
                   help='Files to run over')
    p.add_argument('--type', dest='type',
                   choices=['data', 'sim'],
                   default='sim',
                   help='Option to process simulation or data')
    p.add_argument('-o', '--outfile', dest='outfile',
                   help='Output file')
    args = p.parse_args()

    # Starting parameters
    IT_pulses, inice_pulses = comptools.datafunctions.reco_pulses()
    # Keys to write to frame
    keys = []
    if args.type == 'sim':
        keys += ['MCPrimary']
        keys += ['FractionContainment_MCPrimary_IceTop',
                 'FractionContainment_MCPrimary_InIce']
        keys += ['tanks_charge_Laputop', 'tanks_dist_Laputop']
        # keys += ['tanks_x', 'tanks_y', 'tanks_charge']
        # keys += ['inice_dom_dists_1_60', 'inice_dom_charges_1_60']

    # Keys read directly from level3 processed i3 files
    keys += ['I3EventHeader']
    keys += ['IceTopMaxSignal', 'IceTopMaxSignalString',
             'IceTopMaxSignalInEdge', 'IceTopNeighbourMaxSignal',
             'StationDensity']
    keys += ['Laputop', 'LaputopParams']
    keys += ['Stoch_Reco', 'Stoch_Reco2', 'MillipedeFitParams']
    keys += ['I3MuonEnergyLaputopParams']

    # Keys that are added to the frame
    keys += ['NStations']
    keys += ['avg_inice_radius', 'std_inice_radius', 'median_inice_radius',
             'frac_outside_one_std_inice_radius',
             'frac_outside_two_std_inice_radius']
    # for i in ['1_60']:
    #     keys += ['avg_inice_radius_'+i, 'std_inice_radius_'+i,
    #              'qweighted_inice_radius_'+i, 'invqweighted_inice_radius_'+i]
    for i in ['1_60']:
        keys += ['NChannels_'+i, 'NHits_'+i, 'InIce_charge_'+i, 'max_qfrac_'+i]
    keys += ['FractionContainment_Laputop_IceTop',
             'FractionContainment_Laputop_InIce']
    keys += ['lap_fitstatus_ok']
    keys += ['passed_IceTopQualityCuts', 'passed_InIceQualityCuts']
    for cut in ['MilliNCascAbove2', 'MilliQtotRatio', 'MilliRloglBelow2', 'NCh_CoincLaputopCleanedPulsesAbove7', 'StochRecoSucceeded']:
        keys += ['passed_{}'.format(cut)]
    keys += ['angle_MCPrimary_Laputop']
    # keys += ['tank_charge_dist_Laputop']
    # keys += ['IceTop_charge_175m']

    # keys += ['refit_beta', 'refit_log_s125']
    # keys += ['NNcharges']

    t0 = time.time()

    icetray.set_log_level(icetray.I3LogLevel.LOG_WARN)

    # Construct list of non-truncated files to process
    good_file_list = get_good_file_list(args.files)

    tray = I3Tray()
    # If not running on cobalt (i.e. running a cluster job), add a file stager
    if 'cobalt' not in socket.gethostname():
        tray.context['I3FileStager'] = dataio.get_stagers(
            staging_directory=os.environ['_CONDOR_SCRATCH_DIR'])
    tray.Add('I3Reader', FileNameList=good_file_list)
    # Uncompress Level3 diff files
    tray.Add(uncompress, 'uncompress')

    if args.type == 'data':
        # Filter out all events that don't pass standard IceTop cuts
        tray.Add(lambda frame: all(frame['IT73AnalysisIceTopQualityCuts'].values()))
        # Filter out non-coincident P frames
        tray.Add(lambda frame: inice_pulses in frame)

    tray.Add(icetray_software.add_IceTop_quality_cuts,
             If=lambda frame: 'IT73AnalysisIceTopQualityCuts' in frame)

    tray.Add(icetray_software.add_InIce_quality_cuts,
             If=lambda frame: 'IT73AnalysisInIceQualityCuts' in frame)

    tray.Add(icetray_software.add_nstations, pulses=IT_pulses,
             If=lambda frame: IT_pulses in frame)

    # Add total inice charge to frame
    tray.Add(icetray_software.AddInIceCharge,
             pulses=inice_pulses, min_DOM=1, max_DOM=60,
             If=lambda frame: 'I3Geometry' in frame and inice_pulses in frame)

    # Add InIce muon radius to frame
    tray.Add(icetray_software.AddInIceMuonRadius,
             track='Laputop', pulses='CoincLaputopCleanedPulses', min_DOM=1, max_DOM=60,
             If=lambda frame: check_keys(frame, 'I3Geometry', 'Laputop', 'CoincLaputopCleanedPulses') )

    # Add fraction containment to frame
    tray.Add(icetray_software.add_fraction_containment, track='Laputop',
             If=lambda frame: check_keys(frame, 'I3Geometry', 'Laputop') )
    # if args.type == 'sim':
    tray.Add(icetray_software.add_fraction_containment, track='MCPrimary',
             If=lambda frame: check_keys(frame, 'I3Geometry', 'MCPrimary') )

    # Add Laputop fitstatus ok boolean to frame
    tray.Add(icetray_software.lap_fitstatus_ok,
             If=lambda frame: 'Laputop' in frame)

    # Add opening angle between Laputop and MCPrimary for angular resolution calculation
    tray.Add(icetray_software.add_opening_angle,
             particle1='MCPrimary', particle2='Laputop',
             key='angle_MCPrimary_Laputop',
             If=lambda frame: 'MCPrimary' in frame and 'Laputop' in frame)

    pulses=['IceTopLaputopSeededSelectedHLC', 'IceTopLaputopSeededSelectedSLC']
    # tray.Add(i3modules.add_icetop_charge, pulses=pulses)
    tray.Add(icetray_software.add_IceTop_tankXYcharge, pulses=pulses,
             If=lambda frame: check_keys(frame, 'I3Geometry', *pulses))
    # tray.Add(icetray_software.AddIceTopNNCharges, pulses=pulses,
    #          If=lambda frame: check_keys(frame, 'I3Geometry', *pulses))
    # tray.Add(icetray_software.AddIceTopChargeDistance, track='Laputop', pulses=pulses,
    #          If=lambda frame: check_keys(frame, 'I3Geometry', 'Laputop', *pulses))

    #====================================================================
    # Finish
    comptools.check_output_dir(args.outfile)

    hdf = I3HDFTableService(args.outfile)
    keys = {key: tableio.default for key in keys}
    if args.type == 'data':
        keys['Laputop'] = [dataclasses.converters.I3ParticleConverter(),
                           astro.converters.I3AstroConverter()]

    tray.Add(I3TableWriter, tableservice=hdf, keys=keys, SubEventStreams=['ice_top'])

    tray.Execute()
    tray.Finish()

    print('Time taken: {}'.format(time.time() - t0))

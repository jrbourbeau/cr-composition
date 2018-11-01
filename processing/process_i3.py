#!/usr/bin/env python

import time
import argparse
import os
import socket
import math
import numpy as np

from icecube import (dataio, tableio, astro, toprec, dataclasses, icetray,
                     phys_services, stochastics, millipede, ddddr)
from icecube.frame_object_diff.segments import uncompress
from I3Tray import *
from icecube.tableio import I3TableWriter
from icecube.hdfwriter import I3HDFTableService
from icecube.icetop_Level3_scripts.functions import count_stations


from icecube import icetop_Level3_scripts, stochastics, dataclasses, millipede, photonics_service, ddddr, STTools
from icecube.icetop_Level3_scripts.segments import EnergylossReco


import comptools as comp
import icetray_software


def validate_i3_files(files):
    """ Checks that input i3 files aren't corrupted

    Parameters
    ----------
    files : array-like
        Iterable of i3 file paths to check.

    Returns
    -------
    good_file_list : list
        List of i3 files (from input files) that were able to be
        succeessfully loaded.
    """
    if isinstance(files, str):
        files = [files]

    good_file_list = []
    for i3file in files:
        try:
            test_tray = I3Tray()
            test_tray.Add('I3Reader', FileName=i3file)
            test_tray.Add(uncompress, 'uncompress')
            test_tray.Execute()
            test_tray.Finish()
            good_file_list.append(i3file)
        except RuntimeError:
            icetray.logging.log_warn('File {} is truncated'.format(i3file))
        finally:
            del test_tray

    return good_file_list


def check_keys(frame, *keys):
    """ Function to check if all keys are in frame

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
    """
    return all([key in frame for key in keys])


def delete_keys(frame, keys):
    """ Deletes existing keys in an I3Frame

    Parameters
    ----------
    frame : I3Frame
        I3Frame
    keys:
        Iterable of keys to delete
    """
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        if key in frame:
            frame.Delete(key)


if __name__ == '__main__':

    description='Runs extra modules over a given fileList'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-f', '--files',
                        dest='files',
                        nargs='*',
                        help='Files to run over')
    parser.add_argument('--type', 
                        dest='type',
                        choices=['data', 'sim'],
                        default='sim',
                        help='Option to process simulation or data')
    parser.add_argument('--sim',
                        dest='sim',
                        help='Simulation dataset')
    parser.add_argument('--snow_lambda', 
                        dest='snow_lambda',
                        type=float,
                        help='Snow lambda to use with Laputop reconstruction')
    parser.add_argument('--dom_eff', 
                        dest='dom_eff',
                        type=float,
                        help='DOM efficiency to use with Millipede reconstruction')
    parser.add_argument('-o', '--outfile',
                        dest='outfile',
                        help='Output file')
    args = parser.parse_args()

    # Starting parameters
    IT_pulses, inice_pulses = comp.datafunctions.reco_pulses()
    # Keys to write to frame
    keys = []
    if args.type == 'sim':
        keys += ['MCPrimary']
        keys += ['FractionContainment_MCPrimary_IceTop',
                 'FractionContainment_MCPrimary_InIce']
        keys += ['tanks_charge_Laputop', 'tanks_dist_Laputop']

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

    dom_numbers = [1, 15, 30, 45, 60]
    for min_DOM, max_DOM in zip(dom_numbers[:-1], dom_numbers[1:]):
        key = '{}_{}'.format(min_DOM, max_DOM)
        keys += ['NChannels_'+key,
                 'NHits_'+key,
                 'InIce_charge_'+key,
                 'max_qfrac_'+key,
                 ]
    key = '1_60'
    keys += ['NChannels_'+key,
             'NHits_'+key,
             'InIce_charge_'+key,
             'max_qfrac_'+key,
             ]
    keys += ['FractionContainment_Laputop_IceTop',
             'FractionContainment_Laputop_InIce']
    keys += ['lap_fitstatus_ok']
    keys += ['passed_IceTopQualityCuts', 'passed_InIceQualityCuts']
    keys += ['angle_MCPrimary_Laputop']

    t0 = time.time()

    icetray.set_log_level(icetray.I3LogLevel.LOG_WARN)

    comp.check_output_dir(args.outfile)
    with comp.localized(inputs=args.files, output=args.outfile) as (inputs, output):
        # Construct list of non-truncated files to process
        good_file_list = validate_i3_files(inputs)

        tray = I3Tray()
        tray.Add('I3Reader', FileNameList=good_file_list)
        # Uncompress Level3 diff files
        tray.Add(uncompress, 'uncompress')

        if args.snow_lambda is not None:
            # Re-run Laputop reconstruction with specified snow correction lambda value
            tray = icetray_software.rerun_reconstructions_snow_lambda(tray, 
                                                                      snow_lambda=args.snow_lambda)
        

        if args.dom_eff is not None:
            delete_keys = ['Millipede',
                           'MillipedeFitParams',
                           'Stoch_Reco',
                           'Stoch_Reco2',
                           'Millipede_dEdX',
                           'I3MuonEnergyLaputopParams',
                           'I3MuonEnergyLaputopCascadeParams',
                           'IT73AnalysisInIceQualityCuts',
                           ]
            tray.Add('Delete', keys=delete_keys)

            from icecube.icetop_Level3_scripts import icetop_globals
            # from icecube.icetop_Level3_scripts.segments import muonReconstructions
            from icecube.icetop_Level3_scripts.modules import MakeQualityCuts
            name = 'reco'
            spline_dir="/data/sim/sim-new/downloads/spline-tables/"
            inice_clean_coinc_pulses = icetop_globals.inice_clean_coinc_pulses
            tray.AddSegment(EnergylossReco,
                            name+'_ElossReco',
                            InIcePulses=inice_clean_coinc_pulses,
                            dom_eff=args.dom_eff,
                            splinedir=spline_dir,
                            IceTopTrack='Laputop',
                            If=lambda fr: "NCh_"+inice_clean_coinc_pulses in fr and fr['NCh_' + inice_clean_coinc_pulses].value
                            )

            # Collect in IT73AnalysisInIceQualityCuts
            CutOrder = ["NCh_"+inice_clean_coinc_pulses,
                        "MilliRlogl",
                        "MilliQtot",
                        "MilliNCasc",
                        "StochReco"]
            CutsToEvaluate={"NCh_"+inice_clean_coinc_pulses:(lambda fr: fr["NCh_"+inice_clean_coinc_pulses].value),
                            "MilliRlogl":(lambda fr: "MillipedeFitParams" in fr and math.log10(fr["MillipedeFitParams"].rlogl)<2),
                            "MilliQtot": (lambda fr: "MillipedeFitParams" in fr and math.log10(fr["MillipedeFitParams"].predicted_qtotal/fr["MillipedeFitParams"].qtotal)>-0.03),
                            "MilliNCasc": (lambda fr: "Millipede_dEdX" in fr and len([part for part in fr["Millipede_dEdX"] if part.energy > 0]) >= 3),
                            "StochReco": (lambda fr: "Stoch_Reco" in fr and fr["Stoch_Reco"].status == dataclasses.I3Particle.OK)}
            CutsNames={"NCh_"+inice_clean_coinc_pulses:"NCh_"+inice_clean_coinc_pulses+"Above7",
                       "MilliRlogl":"MilliRloglBelow2",
                       "MilliQtot":"MilliQtotRatio",
                       "MilliNCasc":"MilliNCascAbove2",
                       "StochReco":"StochRecoSucceeded"}
            tray.AddModule(MakeQualityCuts,
                           name+'_DoInIceCuts',
                           RemoveEvents=False,
                           CutOrder=CutOrder,
                           CutsToEvaluate=CutsToEvaluate,
                           CutsNames=CutsNames,
                           CollectBools="IT73AnalysisInIceQualityCuts"
                           )  


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
        for min_DOM, max_DOM in zip(dom_numbers[:-1], dom_numbers[1:]):
            tray.Add(icetray_software.AddInIceCharge,
                     pulses=inice_pulses,
                     min_DOM=min_DOM,
                     max_DOM=max_DOM,
                     If=lambda frame: 'I3Geometry' in frame and inice_pulses in frame)
        tray.Add(icetray_software.AddInIceCharge,
                 pulses=inice_pulses,
                 min_DOM=1,
                 max_DOM=60,
                 If=lambda frame: 'I3Geometry' in frame and inice_pulses in frame)

        # Add InIce muon radius to frame
        tray.Add(icetray_software.AddInIceMuonRadius,
                 track='Laputop',
                 pulses='CoincLaputopCleanedPulses',
                 min_DOM=1,
                 max_DOM=60,
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

        #====================================================================
        # Finish

        hdf = I3HDFTableService(output)
        keys = {key: tableio.default for key in keys}
        if args.type == 'data':
            keys['Laputop'] = [dataclasses.converters.I3ParticleConverter(),
                               astro.converters.I3AstroConverter()]

        tray.Add(I3TableWriter,
                 tableservice=hdf,
                 keys=keys,
                 SubEventStreams=['ice_top'])

        tray.Execute()
        tray.Finish()

    print('Time taken: {}'.format(time.time() - t0))

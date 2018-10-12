# -*- coding: utf-8 -*-
#
## copyright  (C) 2018
# The Icecube Collaboration
#
# $Id$
#
# @version $Revision$
# @date $LastChangedDate$
# @author Hershal Pandya <hershal@udel.edu> Last changed by: $LastChangedBy$
#
#------------------------------------------
# ICECUBE THINGS
#------------------------------------------
import sys
import os
import argparse
import numpy as np

from icecube import icetray, dataclasses, dataio, recclasses, shield
from icecube.frame_object_diff.segments import uncompress
from I3Tray import I3Tray
from icecube.tableio import I3TableWriter
from icecube.hdfwriter import I3HDFTableService

from i3module import IceTop_LLHRatio
from globals import (rotate_to_shower_cs, logEnergyBins, cosZenBins,
                     logChargeBins, logDBins, logTBins, pulses1, pulses2,
                     pulses3, reco_track2, reco_track1)
from icetop_l3_dataprep import Generate_Input_IceTop_LLHRatio

def print_length(frame, key):
    if key in frame:
        getpulse=dataclasses.I3RecoPulseSeriesMap.from_frame(frame,key)
        print('len({}) = {}'.format(key, len(getpulse)))
    return

def print_length2(frame, key):
    if key in frame:
        print('len({}) = {}'.format(key, len(frame[key])))

def merge_excluded_tanks_lists(frame, MergedListName=None,
                               ListofExcludedTanksLists=None):

    if ListofExcludedTanksLists is None:
        ListofExcludedTanksLists = []

    excluded_tanks = dataclasses.TankKey.I3VectorTankKey()
    for tag in ListofExcludedTanksLists:
        if tag in frame:
            tanks = frame[tag]
            for tank in tanks:
                if tank not in excluded_tanks:
                    excluded_tanks.append(tank)

    if MergedListName in frame:
         frame.Delete(MergedListName)

    frame.Put(MergedListName, excluded_tanks)
    return

def print_key(frame, key):
    if key in frame:
        print('{} = {}'.format(key, frame[key]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output',
                        dest='outfile',
                        default='output',
                        help='outfile only created in case of CalcLLHR mode')
    parser.add_argument('--PDF_File',
                        dest='pdf_file',
                        default='hist.hd5',
                        help='output PDF HDF5 only created in case of GeneratePDF mode')
    parser.add_argument('--RunMode',
                        dest='RunMode',
                        default='GeneratePDF',
                        choices=['GeneratePDF', 'CalcLLHR'],
                        help='GeneratePDF/CalcLLHR')
    parser.add_argument('--inputs',
                        dest='inputs',
                        nargs='*',
                        help='Input files to run over')
    args = parser.parse_args()

    icetray.logging.console()
    icetray.set_log_level(icetray.I3LogLevel.LOG_INFO)

    tray = I3Tray()
    tray.Add('I3Reader', filenamelist=args.inputs)
    # Uncompress Level3 diff files
    tray.Add(uncompress, 'uncompress')

    icetop_excluded_tanks_lists = [
                                   # 'TankPulseMergerExcludedSLCTanks',
                                   'IceTopHLCSeedRTExcludedTanks',
                                   ]

    tray.Add(merge_excluded_tanks_lists, 'mergethem',
             MergedListName='IceTopExcludedTanksAll',
             ListofExcludedTanksLists=icetop_excluded_tanks_lists)

    tray.Add(Generate_Input_IceTop_LLHRatio, 'create inputs for next module',
             HLCTankPulsesName='IceTopHLCSeedRTPulses',
             SLCTankPulsesName='IceTopLaputopSeededSelectedSLC',
             ExcludedTanksListName='IceTopExcludedTanksAll',
             RecoName='Laputop',
             ExcludedFalseCharge=1e-3,
             ExcludedFalseTime=1e-4,
             UnhitFalseCharge=1e-3,
             UnhitFalseTime=1e-4,
             SubtractCurvatureBool=False,
             Hits_I3VectorShieldHitRecord='ITLLHR_Hits',
             Unhits_I3VectorShieldHitRecord='ITLLHR_Unhits',
             Excluded_I3VectorShieldHitRecord='ITLLHR_Excluded')

    #tray.Add(print_length2,key='IceTopHLCSeedRTExcludedTanks')
    #tray.Add(print_length2,key='TankPulseMergerExcludedSLCTanks')
    tray.Add(print_length2, key='IceTopExcludedTanksAll')
    tray.Add(print_key, key='IceTopExcludedTanksAll')
    tray.Add(print_key, key='I3EventHeader')
    #tray.Add(print_length,key='IceTopHLCSeedRTPulses')
    #tray.Add(print_length,key='IceTopLaputopSeededSelectedSLC')
    tray.Add(print_length2, key='ITLLHR_Hits')
    tray.Add(print_length2, key='ITLLHR_Unhits')
    tray.Add(print_length2, key='ITLLHR_Excluded')

    if args.RunMode == 'GeneratePDF':
        tray.Add(IceTop_LLHRatio, 'make_hist',
                 Hits_I3VectorShieldHitRecord='ITLLHR_Hits',
                 Unhits_I3VectorShieldHitRecord='ITLLHR_Unhits',
                 Excluded_I3VectorShieldHitRecord='ITLLHR_Excluded',
                 AngularReco_I3Particle='Laputop',
                 # EnergyReco_I3Particle=reco_track1,
                 LaputopParamsName='LaputopParams',
                 OutputFileName='hist2.hdf',
                 BinEdges5D=[logEnergyBins,cosZenBins,logChargeBins,logTBins,logDBins],
                 DistinctRegionsBinEdges3D = [[logChargeBins,logTBins,logDBins]],
                 RunMode='GeneratePDF')
    else:
        tray.Add(IceTop_LLHRatio, 'calc_llhr',
                 Hits_I3VectorShieldHitRecord=pulses1,
                 Unhits_I3VectorShieldHitRecord=pulses2+'_llhr',
                 Excluded_I3VectorShieldHitRecord=pulses3+'_llhr',
                 AngularReco_I3Particle=reco_track2,
                 EnergyReco_I3Particle=reco_track1,
                 SigPDFInputFileName='hist2.hd5',
                 BkgPDFInputFileName='hist.hd5',
                 RunMode='CalcLLHR',
                 SubtractEventFromPDF='Sig')

        tray.Add("I3Writer", "EventWriter",
                 Filename=os.path.join(args.outfile, ".i3.zst"),
                 streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics],
                 DropOrphanStreams=[icetray.I3Frame.DAQ],
                 )


    tray.Execute()
    tray.Finish()

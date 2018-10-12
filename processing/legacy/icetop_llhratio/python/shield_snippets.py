
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
from icecube import shield 

    tray.AddModule(IceTopPulseMerger, 'IceTopPulseMerger',
        output=pulses_merged,
        input=[icetop_globals.icetop_hlc_vem_pulses, correctedSLC],
        If = field_exists(icetop_globals.icetop_hlc_vem_pulses))

def IceTopPulseMerger(frame, output='', input=[]):
    ''' Merges inputs in output. Used to merge SLC and HLC IT pulses '''
    if output not in frame.keys():
        pulses = []
        for i in input:
            if not i in frame:
                continue
            ps = frame[i]
            pulses.append(i)
        frame[output] = dataclasses.I3RecoPulseSeriesMapUnion(frame, pulses)
        #pulses_output = dataclasses.I3RecoPulseSeriesMapUnion(frame, pulses)
        #frame[output] =  pulses_output.apply(frame)

    tray.AddModule("I3ShieldDataCollector",
        InputRecoPulses=veto_globals.pulses_tank_merged,
        InputTrack=fit_type,
        OutputName=hit_output_flat,
        BadDomList = icetop_globals.icetop_bad_doms,
        ReportUnhitDOMs=True,
        ReportCharge = True,
        useCurvatureApproximation=False,
        If = field_exists(veto_globals.pulses_tank_merged))

    #if you are using a curvature to subtract out of times. can be done in next iteration.
#    hit_output_curved = "Shield_" + veto_globals.pulses_tank_merged +'_'+ fit_type
#    unhit_output_curved = hit_output_curved +"_UnHit"
#    print(hit_output_curved)
#    tray.AddModule("I3ShieldDataCollector",
#        InputRecoPulses=veto_globals.pulses_tank_merged,
#        InputTrack=fit_type,
#        OutputName=hit_output_curved,
#        BadDomList = icetop_globals.icetop_bad_doms,
#        ReportUnhitDOMs=True,
#        ReportCharge = True,
#        useCurvatureApproximation= True,
#        coefficients = veto_globals.coefficients,
#        If = field_exists(veto_globals.pulses_tank_merged))

def assemble_excluded_doms_list(frame, geometry = 'I3Geometry',icetop_excluded_tanks_lists=[], out = veto_globals.icetop_excluded_doms):
    if out in frame:
         log_warn('%s already in frame.being deleted'%out)
         frame.Delete(out)
    geo = frame[geometry]
    mergedExcludedList=dataclasses.I3VectorOMKey()
    for container in icetop_excluded_tanks_lists:
        icetop_excluded_tanks = frame[container]
        for tank in icetop_excluded_tanks:
            for omkey in geo.stationgeo[tank.string][tank.tank].omkey_list:
                if omkey not in mergedExcludedList:
                    mergedExcludedList.append(omkey)
        frame.Put(out,mergedExcludedList)


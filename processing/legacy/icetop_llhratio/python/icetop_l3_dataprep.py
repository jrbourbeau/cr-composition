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
from icecube import phys_services, dataclasses, icetray, recclasses
import numpy as np
from icecube.icetray.i3logging import log_fatal,log_warn
from icecube.dataclasses import I3Constants
from globals import to_shower_cs

class Generate_Input_IceTop_LLHRatio(icetray.I3ConditionalModule):
    """
    """

    def __init__(self,ctx):
        icetray.I3ConditionalModule.__init__(self, ctx)

        #inputs
        self.AddParameter('HLCTankPulsesName',
                          'Input HLCTankPulses','IceTopHLCSeedRTPulses')
        self.AddParameter('SLCTankPulsesName',
                          'Input SLCTankPulses','LaputopSeededSelectedSLC')
        self.AddParameter('ExcludedTanksListName',
                          'Input ExcludedTanksList','IceTopExcludedTanks')
        self.AddParameter('RecoName',
                          'I3Particle with Shower core + direction',
                          'Laputop')
        self.AddParameter('ExcludedFalseCharge',
                          'FalseCharge Assigned to Excluded Tanks',
                          1e-3)
        self.AddParameter('ExcludedFalseTime',
                          'FalseTime Assigned to Excluded Tanks',
                          1e-4)
        self.AddParameter('UnhitFalseCharge',
                          'FalseCharge Assigned to Unhit Tanks',
                          1e-3)
        self.AddParameter('UnhitFalseTime',
                          'FalseTime Assigned to Unhit Tanks',
                          1e-4)

        #optional inputs
        self.AddParameter('SubtractCurvatureBool',
                          'If true, provide curvature correction table.',
                          False)
        self.AddParameter('SubtractCurvatureFile',
                          'Curvature correction table.',
                          None)
        self.AddParameter('LaputopParamsName',
                          'LaputopParams from which logS125 is to be drawn only accepted if EnergyReco_I3Particle not provided',
                          None)

        #outputs
        self.AddParameter('Hits_I3VectorShieldHitRecord',
                          'Output Hits I3VectorShieldHitRecord',None)
        self.AddParameter('Unhits_I3VectorShieldHitRecord',
                          'Output UnHits I3VectorShieldHitRecord',None)
        self.AddParameter('Excluded_I3VectorShieldHitRecord',
                          'Output Excluded I3VectorShieldHitRecord',None)

        return

    def Configure(self):
        
        #input parameters
        self.hlc_name = self.GetParameter('HLCTankPulsesName')
        self.slc_name = self.GetParameter('SLCTankPulsesName')
        self.excluded_name = self.GetParameter('ExcludedTanksListName')
        self.reco_name = self.GetParameter('RecoName')
        self.exc_fake_q = self.GetParameter('ExcludedFalseCharge')
        self.exc_fake_t = self.GetParameter('ExcludedFalseTime')
        self.unhit_fake_q = self.GetParameter('UnhitFalseCharge')
        self.unhit_fake_t = self.GetParameter('UnhitFalseTime')

        #output parameters
        self.HitsName = self.GetParameter('Hits_I3VectorShieldHitRecord')
        self.UnhitsName = self.GetParameter('Unhits_I3VectorShieldHitRecord')
        self.ExcludedName = self.GetParameter('Excluded_I3VectorShieldHitRecord')

        #optional input parameters
        self.subtract_curvature = self.GetParameter('SubtractCurvatureBool')
        if self.subtract_curvature:
            self.curvature_file = self.GetParameter('SubtractCurvatureFile')
            self.LaputopParamsName = self.GetParameter('LaputopParamsName')

        return

    def Geometry(self,frame):
        self.geometry = frame['I3Geometry']
        self.PushFrame(frame)

    def Physics(self,frame):
      
        #initiate output vectors
        hits = recclasses.I3VectorShieldHitRecord() 
        unhits = recclasses.I3VectorShieldHitRecord() 
        excluded = recclasses.I3VectorShieldHitRecord()
        not_unhit=[] # will keep track of hit+excluded doms 

        #load reconstruction
        axis=frame[self.reco_name]
        rotation = to_shower_cs(axis)
        origin = np.array([[axis.pos.x], [axis.pos.y], [axis.pos.z]])
        
        # populate hits vector
        for tankpulses in [self.slc_name, self.hlc_name]:
            pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,tankpulses)

            for omkey,pulse_items in pulses.iteritems():
                if not omkey in self.geometry.omgeo:
                    log_fatal("OM {om} not in geometry!".format(om=k))

                if omkey[1]>64 or omkey[1]<61:
                    log_fatal("OM {om} not an IceTop DOM!".format(om=k))

                if len(pulse_items)!=1:
                    log_fatal('only one pulse per omkey should be present. something might be fishy here.')
                
                not_unhit.append(omkey)

                for pulse in pulse_items:
                    #calculate the tank's location/ time in shower frame ref
                    # time is w.r.t. planar shower front arrival time at tank location

                    position = self.geometry.omgeo[omkey].position
                    det_cs_position = np.array([[position.x],
                                      [position.y],
                                      [position.z]])
                    shower_cs_position = rotation*(det_cs_position - origin)
                    shower_cs_radius = np.sqrt(shower_cs_position[0]**2 + shower_cs_position[1]**2)
                    time = pulse.time - float(axis.time - shower_cs_position[2]/ I3Constants.c)

                    #store all the pulse information in I3ShieldHitRecord and add it to hits vector
                    new_pulse = recclasses.I3ShieldHitRecord()

                    new_pulse.DOMkey = omkey
                    new_pulse.charge = pulse.charge
                    new_pulse.time_residual = time
                    new_pulse.distance = np.float(shower_cs_radius)
                    
                    hits.append(new_pulse)

        # populate excluded vector
        tanklist = frame[self.excluded_name]
        for tank in tanklist:
            not_unhit.append(tank.default_omkey)
            new_pulse = recclasses.I3ShieldHitRecord()
            # new_pulse.DOMkey = tank
            new_pulse.DOMkey = tank.default_omkey
            new_pulse.charge = self.exc_fake_q
            new_pulse.time_residual = self.exc_fake_t 

            position = self.geometry.omgeo[new_pulse.DOMkey].position

            det_cs_position = np.array([[position.x],
                              [position.y],
                              [position.z]])
            shower_cs_position = rotation*(det_cs_position - origin)
            shower_cs_radius = np.sqrt(shower_cs_position[0]**2 + shower_cs_position[1]**2)

            new_pulse.distance = np.float(shower_cs_radius)
            excluded.append(new_pulse)

        #populate unhits vector
        stations = frame['I3Geometry'].stationgeo
        for station in stations:
            for tank in station.data():
                is_unhit=0
                for dom in tank.omkey_list:
                    if dom not in not_unhit:
                        is_unhit+=1
                if is_unhit==2:
                    new_pulse=recclasses.I3ShieldHitRecord()
                    new_pulse.DOMkey = dom # pick the last one in the loop above
                    new_pulse.charge = self.unhit_fake_q 
                    new_pulse.time_residual = self.unhit_fake_t

                    position = self.geometry.omgeo[new_pulse.DOMkey].position

                    det_cs_position = np.array([[position.x],
                                      [position.y],
                                      [position.z]])
                    shower_cs_position = rotation*(det_cs_position - origin)
                    shower_cs_radius = np.sqrt(shower_cs_position[0]**2 + shower_cs_position[1]**2)

                    new_pulse.distance = np.float(shower_cs_radius)
                    unhits.append(new_pulse)
         
        #put the results to frame
        frame.Put(self.HitsName, hits)
        frame.Put(self.UnhitsName, unhits)
        frame.Put(self.ExcludedName, excluded)
        self.PushFrame(frame)

        return


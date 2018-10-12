
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
import numpy as np
logEnergyBins = np.linspace(3,8,26)
logEnergyBins=np.array([logEnergyBins[i] for i in range(len(logEnergyBins)) if i%2==0],dtype=float)

cosZenBin0 = 0.86
cosZenBins = np.linspace(cosZenBin0, 1.0+ np.finfo(float).eps , (1-cosZenBin0)/0.01+1)
cosZenBins=np.array([cosZenBins[i] for i in range(len(cosZenBins)) if i%2==0],dtype=float)

logChargeBins = np.linspace(-3,4,71)
deltaCharge = 0.1
unhitCharge = logChargeBins[0]-0.5*deltaCharge
logChargeBins = np.hstack([unhitCharge-0.5*deltaCharge, logChargeBins])
excludedCharge = logChargeBins[0]-0.5*deltaCharge
logChargeBins = np.hstack([excludedCharge-0.5*deltaCharge, logChargeBins])

deltaT = 0.1        
nBins = 5.0/deltaT  
tBinsUp = np.linspace(0,5,nBins+1)
tBinsDown = -1.0*tBinsUp
tBinsDown.sort()   
logTBins = np.hstack([tBinsDown[0:-1],tBinsUp])
unhitTime = logTBins[0]-0.5*deltaT
logTBins = np.hstack([unhitTime-0.5*deltaT, logTBins])
excludedTime = logTBins[0]-0.5*deltaT
logTBins = np.hstack([excludedTime-0.5*deltaT, logTBins])

logDBins = np.linspace(0,3.5,36) 

pulses1='Shield_HLCSLCTimeCorrectedTankMerged_SplineMPEfast_SRT_Split_InIcePulses_singleHits'
pulses2='Shield_HLCSLCTimeCorrectedTankMerged_SplineMPEfast_SRT_Split_InIcePulses_singleHits_UnHit'
pulses3='IceTopExcludedTanks'
reco_track2='SplineMPEfast_SRT_Split_InIcePulses'
reco_track1='MuEx_mie_SplineMPEfast_SRT_Split_InIcePulses'

def rotate_to_shower_cs(x,y,z,phi,theta,core_x,core_y,core_z):
    """ 
    Rotate to shower CS takes a fit (assumes is set) and returns a rotation matrix.
    Requires np.
    """
    # counter-clockwise (pi + phi) rotation
    d_phi = np.matrix([ [ -np.cos(phi), -np.sin(phi), 0], 
                           [  np.sin(phi), -np.cos(phi), 0], 
                           [  0,                 0,                1] ])
    # clock-wise (pi - theta) rotation
    d_theta = np.matrix([ [  -np.cos(theta), 0, -np.sin(theta)],
                             [  0,                  1,  0,                ],  
                             [  np.sin(theta), 0,  -np.cos(theta)] ])

    rotation=d_theta*d_phi

    origin = np.array([[core_x], [core_y], [core_z]])
    det_cs_position = np.array([[x],
                                  [y],
                                  [z]])
    shower_cs_position = rotation*(det_cs_position - origin)
    shower_cs_radius = np.sqrt(shower_cs_position[0]**2 + shower_cs_position[1]**2)
    
    return np.float(shower_cs_radius)

def to_shower_cs(fit):
    """ 
    Rotate to shower CS takes a fit (assumes fit.dir is set) and returns a rotation matrix.
    Requires numpy.
    """
    import numpy
    from math import cos, sin 
    # counter-clockwise (pi + phi) rotation
    d_phi = numpy.matrix([ [ -cos(fit.dir.phi), -sin(fit.dir.phi), 0], 
                           [  sin(fit.dir.phi), -cos(fit.dir.phi), 0], 
                           [  0,                 0,                1] ])
    # clock-wise (pi - theta) rotation
    d_theta = numpy.matrix([ [  -cos(fit.dir.theta), 0, -sin(fit.dir.theta)],
                             [  0,                  1,  0,                ],  
                             [  sin(fit.dir.theta), 0,  -cos(fit.dir.theta)] ])
    return d_theta*d_phi

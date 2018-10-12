
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
decimals=2

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

total_hist_range=[[-3.1999999,4.0],[-5.1999999999999993,5.0],[0.0,3.5]]
hits_hist_range=[[-3.0,4.0],[-5.0,5.0],[0.0,3.5]]
unhits_hist_range=[[-3.1,-3.0],[-5.1,-5.0],[0.0,3.5]]
excluded_hist_range=[[-3.1999999,-3.1],[-5.1999999999999993,-5.1],[0.0,3.5]]

excluded_hist_binedges = [ [-3.2, -3.1], [-5.2, -5.1], logDBins]
unhits_hist_binedges = [ [-3.1, -3.0], [-5.1, -5.0], logDBins]
hits_hist_binedges = [ logChargeBins[2:], logTBins[2:], logDBins]

binedges= [logChargeBins, logTBins, logDBins]
#for i in range(len(binedges)):
#    binedges[i] = np.round(binedges[i],decimals=2)

print hits_hist_range
print unhits_hist_range
print excluded_hist_range 
distinct_regions_binedges = [hits_hist_binedges, unhits_hist_binedges, excluded_hist_binedges]

for region_edges in distinct_regions_binedges:
    region_range = [ [i[0],i[-1]] for i in region_edges]
    print region_range

for i in range(len(distinct_regions_binedges)):
    for j in range(len(distinct_regions_binedges[0])):
        if i<len(distinct_regions_binedges)-1:
            next_one=i+1
        else:
            next_one=0
        intersection=np.intersect1d(np.round(distinct_regions_binedges[i][j],decimals=2),np.round(distinct_regions_binedges[next_one][j],decimals=2))
        if len(intersection)>1:
            if len(intersection)!=len(binedges[j]):
                print 'comparing distinct regions %i and %i'%(i,next_one)
                print 'they have more than 1 elements in binedges intersecting'
                print 'the regions need to be exclusive, i.e. distinct'
                print 'only way they have more than 1 element intersecting is if all elements are same in both edges'
                raise Exception('Inconsistency found')
import sys
sys.exit()

print binedges
combine_edges=[]
for i in range(len(distinct_regions_binedges)):
    for j in range(len(distinct_regions_binedges[i])):
        if i==0: 
            combine_edges.append(distinct_regions_binedges[i][j])
        else:
            combine_edges[j]=np.unique(np.sort(np.concatenate((combine_edges[j],distinct_regions_binedges[i][j]))))
print combine_edges

for i in range(len(binedges)):
    are_equal=(np.round(binedges[i],decimals=2)==np.round(combine_edges[i],decimals=2)).all()
    if not are_equal:
        print 'DistinctRegionsBinEdges do not add up to binedges for this dimension'
        print combine_edges[i], binedges[i]


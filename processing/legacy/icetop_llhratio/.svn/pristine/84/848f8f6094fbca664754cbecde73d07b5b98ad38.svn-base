
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
import copy

def value_in_nborhood_new(heatmap,g_index,nborhd_size=1):
    '''
    return value of N-d histogram in the nborhood of
    g_index. nborhood defined by number of bins around g_index
    nborhd_size
    '''

    slices=[]
    for dim in range(len(g_index)):
        slices.append(slice(g_index[dim]-nborhd_size,g_index[dim]+nborhd_size))

    # if a slice value runs outside the range of the array 
    #it just takes until the end of the array 
    total_value_in_nborhood=np.sum(heatmap[slices])

    # nbins from center bin to farthest bin in nborhd + center bin
    nborhd_area = (2.*nborhd_size+1.)**len(g_index)
    #in older code, nborhd_area is only limited to area inside the edges of the histogram
    #is wrong because edges of histogram are arbitary
    #nborhd_area ought to be 2*nborhd_size + 1 regardless of whether this
    #area extends outside of the edges (by edges, I mean boundary not binedges)
    #this can cause mismatch between older and newer llh ratio values
    
    return total_value_in_nborhood, nborhd_area

def extrapolate_heatmap(heatmap,extrapolation_mask):
    '''
    heatmap = N-dim PDF
    extrapolation_mask = N-dim mask pointing to 
                        locations in heatmap where extrapolations are required

    In this method, one creates a square with assumption
    That your xbinsize = 1 unit and ybinsize = 1 unit.
    
    This square is generated around your point of interest
    which is given by the g_index.
    
    This square is expanded in 2 units of length each time i.e. lenght = 3,5,7...
    
    And when a populated bin is hit, it stops and returns the value.
    
    Value = sum of counts in the square
    nborhd_area = nbins in the square
    
    Generating denstiy = Value / nborhd_area is left on to the user.
    '''
    old_norm = np.sum(heatmap)

    # find indices in N-dim array where extrapolations need to be carried out
    extrapolation_indices = np.argwhere(extrapolation_mask)

    longer_side=np.amax(np.shape(heatmap))
    # cannot modify old_heatmap in loop
    # extrapolations need to be made using old_heatmap
    new_heatmap = copy.deepcopy(heatmap)

    for index in extrapolation_indices:
        index=tuple(index)
        for nborhd_size in range(1,longer_side,1):
            value,area = value_in_nborhood_new(heatmap,index,nborhd_size=nborhd_size)
            if value>0:
                density = np.float(value)/np.float(area)
                new_heatmap[index] = density
                break
        if new_heatmap[index]==0:
            print 'nborhood size max reached. extrapolated value still zero. assigning nan.'
            new_heatmap[index]=np.nan

    new_norm = np.sum(new_heatmap)
    new_heatmap = old_norm * new_heatmap / new_norm

    return new_heatmap

def log_likelihood(heatmap,event_hist):
    """
    heatmap = N-dim PDF
    event_hist = N-dim histogram made out of one event
    """
    assert(np.shape(heatmap)==np.shape(event_hist))

    # create a N-dim mask for points where extrapolations are required
    extrapolation_mask = (heatmap==0)&(event_hist>0)
    n_extrapolations_reqd = len(extrapolation_mask[extrapolation_mask])

    # extrapolate if necessary
    if n_extrapolations_reqd > 0:
        heatmap = extrapolate_heatmap(heatmap, extrapolation_mask)
    
    product_ = event_hist * heatmap
    product = product_[event_hist!=0]
    llh = np.sum(np.log10(product))

    return llh,n_extrapolations_reqd,heatmap,product_

def log_likelihood_ratio(heatmap1,heatmap2,event_hist):
    """
    give heatmaps with proper normalization
    no re-normalization being done in here.
    heatmap1 = N-dim PDF for hypothesis 1
    heatmap2 = N-dim PDF for hypothesis 2
    event_hist = N-dim histogram made out of one event
    """
    assert(np.shape(heatmap1)==np.shape(heatmap2))
    assert(np.shape(heatmap1)==np.shape(event_hist))

    #calculate log_likelihoods
    #toz = tanks on zero = n_extrapolations_reqd
    llh1,toz_1,heatmap1,logprod1 = log_likelihood(heatmap1, event_hist)
    llh2,toz_2,heatmap2,logprod2 = log_likelihood(heatmap2, event_hist)

    return llh1-llh2,toz_1,toz_2,heatmap1, heatmap2, llh1, llh2, logprod1, logprod2

def get_slice_vector(edges, sub_range):
    slice_vector=[]
    for dim in range(len(edges)):
        tedges = edges[dim]
        trange = sub_range[dim]
        slice_start=np.where(np.absolute(tedges-trange[0])<1e-4)[0][0]
        slice_end=np.where(np.absolute(tedges-trange[1])<1e-4)[0][0]
        slice_vector.append(slice(slice_start,slice_end))
    return slice_vector

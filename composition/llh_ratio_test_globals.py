import cPickle as p
import numpy as np

zenith_bins=[[1.0,0.95],[0.95,0.9],[0.9,0.85],[0.85,0.8],[0.8,0.75],[0.75,0.7]]
s125_bins = np.array(zip(np.arange(-0.5,2.1,0.1),np.arange(-0.4,2.2,0.1)))
#s125_bins = np.array(zip(np.arange(-0.5,1.1,0.1),np.arange(-0.4,1.2,0.1)))

global3drange = [[0,4],[-3,4],[-2,4]] 
#global3dbins = [80,70,120] # ORIGINAL HIST BINS
global3dbins = [80,70,60]

globalhistbins = {}
globalhistrange = {}

key = 'q_r'
globalhistbins[key]=[80,70]
globalhistrange[key] = [[0,4],[-3,4]]

key = 'q_t' 
#globalhistbins[key]=[120,70] #ORIGNIAL HIST BINS
globalhistbins[key]=[60,70]
globalhistrange[key] = [[-2,4],[-3,4]]

key = 't_r' 
#globalhistbins[key]=[80,120] #ORIGNIAL HIST BINS
globalhistbins[key]=[80,60]
globalhistrange[key] = [[0,4],[-2,4]]

def correct_slc_time(mean_slc_charge,median_time_diff,slc_time,slc_charge):
    if slc_charge<=0:
        #do nothing
        return slc_time

    xx = np.log10(slc_charge)
    if np.isnan(xx):
        #do nothing
        return slc_time

    yy = np.absolute(xx - mean_slc_charge)
    select = yy==np.amin(yy)
    correction = median_time_diff[select]

    correction = np.float(correction)
    corrected_slc_time=slc_time + correction

    return corrected_slc_time


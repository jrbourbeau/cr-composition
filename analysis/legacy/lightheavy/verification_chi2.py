#!/usr/bin/env python

import sys, os, simFunctions
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as n
from pylab import *
from support_functions import cut_maker, get_fig_dir, hits_llh_cutter
sys.path.append(os.path.expandvars("$HOME"))
import dashi
dashi.visual()
plt.style.use('mystyle')
fig_dir = get_fig_dir()

phi_0    = 2.95*10**-6
livetime = 1150511./10.

def flux(E):
    return phi_0*E**(-2.7)*(1+(E/3.)**100)**(-0.4/100.)*10**-10
    
if __name__ == "__main__":
    cuts   = {'standard':1, 'laputop_it':1}#, 's125':10**(0.1)}
    sets   = ['burn_sample', '12360', '12362']
    labels = ['October 2012 Data', '2012 Protons', '2012 Iron']
    colors = ['k', 'red', 'blue']

    xlabel = '$log_{10}(\chi^2$/NDF)'
    fname  = 'chi2_verif_new.png'

    bins   = np.arange(-2,2,0.05)
    key = 'chi2_ndf_ldf'
    #key = 'StationDensity'

    left = 0.15
    width = 0.8
    ax0 = plt.axes([left, 0.35, width, 0.6])
    ax1 = plt.axes([left, 0.14, width, 0.19])

    plt.sca(ax0)
    s_min = -0.25 
    s_max = 1

    for i, set in enumerate(sets):
        print(set)
        f = cut_maker(set,cuts)
        if 'burn' in set:
            mask = np.greater_equal(f['mjd_time'], 56201)&np.less(f['mjd_time'], 56231)&np.less(f['s125'], 10**s_max)&np.greater(f['s125'], 10**s_min)
            data_hist = dashi.factory.hist1d(np.log10(f[key][mask]), bins, weights = [1/livetime]*len(f['chi2_ndf_ldf'][mask]))
            data_hist.line(label = labels[i], color = colors[i])
        else:
            mask = np.less(f['s125'], 10**s_max)&np.greater(f['s125'], 10**s_min)
            l3_hist = dashi.factory.hist1d(np.log10(f[key][mask]), bins, weights = flux(f['primary_E'][mask]*10**-6)*f['weights'][mask])
            l3_hist.line(label = labels[i], color = colors[i])
            ax1.scatter(l3_hist.bincenters, data_hist.bincontent/l3_hist.bincontent, color = colors[i], marker = '^')
       
    ax1.set_xlim([-2,2])
    ax1.set_ylim([0,7])
    ax1.axhline(y=1, color = 'k', linestyle='--')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Data/MC')

    ax0.xaxis.set_visible(False)
    ax0.set_ylabel('Rate (Hz)')
    ax0.set_ylim([10**-8, 1])
    ax0.set_yscale('log')
    ax0.legend(loc='upper right', fontsize = 12)
    #plt.title('%0.2f < %s < %0.2f' % (s_min, xlabel, s_max))
    plt.savefig(fig_dir + fname, transparent=True)
    plt.close()

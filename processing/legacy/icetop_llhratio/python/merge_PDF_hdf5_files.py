
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
# IPython log file

from optparse import OptionParser
import numpy as np
import tables

parser=OptionParser()
parser.add_option('--output',dest='output',help='output hdf5 file',type='str',default=None)
(options,inputfilelist)=parser.parse_args()

file1=inputfilelist[0]
f1 = tables.open_file(file1)
new_hist = f1.root.hist[:]
n_events = f1.root.n_events[:][0]

for file2 in inputfilelist[1:]: 
    f2 = tables.open_file(file2)
    equal_edges = (f1.root.binedges_0[:]==f2.root.binedges_0[:]).all()
    equal_labels = (f1.root.labels[:]==f2.root.labels[:]).all()
    if not equal_edges or not equal_labels:
        raise Exception('edges/labels dont match between files %s %s'%(file1,file2))

    new_hist += f2.root.hist[:]
    n_events += f2.root.n_events[:][0]
    f2.close()

f=tables.open_file(options.output,'w')
f.create_carray('/', 'hist', obj=new_hist,filters=tables.Filters(complib='blosc:lz4hc', complevel=1))

for i in range(5):
    f.create_carray('/', 'binedges_%i'%i, 
                    obj=eval('f1.root.binedges_%i[:]'%i),
                    filters=tables.Filters(complib='blosc:lz4hc', 
                    complevel=1))

for i in range(1):
    for j in range(3):
        f.create_carray('/', 'region_%i_binedges_%i'%(i,j), 
                        obj=eval('f1.root.region_%i_binedges_%i[:]'%(i,j)),
                        filters=tables.Filters(complib='blosc:lz4hc', 
                        complevel=1))

f.create_carray('/', 'labels', obj=f1.root.labels[:],filters=tables.Filters(complib='blosc:lz4hc', complevel=1))
f.create_carray('/', 'n_events', obj=[n_events],filters=tables.Filters(complib='blosc:lz4hc', complevel=1))
f.close()

f1.close()

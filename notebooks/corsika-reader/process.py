#!/usr/bin/env python

from root_numpy import root2array, tree2array

filename = 'ldf_proton_1PeV.root'
r = root2array(filename, treename='ldftree', branches='r')
e_em = root2array(filename, treename='ldftree', branches='e_em')

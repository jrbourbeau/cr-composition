#!/usr/bin/env python

import numpy as np

# Class to define power law and further functionality
class PowerLaw:
    def __init__(self, name="ExamplePowerLaw", Nevents=1e5, Index=2.7, Xlim=[1e12,1e15]):
        self.name = name
        self.Nevents = Nevents
        self.idx = Index
        self.xlo = Xlim[0]
        self.xhi = Xlim[1]

    # Get the number of events in a x-range
    def getN(self,xlo,xhi):
        if (self.idx == 1):
            numer = np.log(xhi)-np.log(xlo)
            denom = np.log(self.xhi)-np.log(self.xlo)
        else:    
            g = 1-self.idx
            numer = xhi**g-xlo**g
            denom = self.xhi**g-self.xlo**g
        return numer/denom

    # Fill a frequency spectrum
    # Can either draw from distribution randomly or
    # keep analytical form of dist scaled by Nevents
    def Fill(self,X=None,method="rand"):
        N = np.zeros(len(X)-1)
        if (method=="rand"):
            rand_E = self.Random()
            N, bin_edges = np.histogram(rand_E,X)
        elif (method=="analytic"):
            for i in xrange(0,len(N)):
                N[i] = self.Nevents*self.getN(X[i],X[i+1])
                N[i] = np.int(N[i])
        return N

    # Draw a random number or array of random
    # numbers distributed via the parent pl dist
    def Random(self,N=None):
        if (N is None):
            N = self.Nevents
        g = 1-self.idx
        y = np.random.rand(N)
        pl_rand = (self.xhi**g-self.xlo**g)*y+self.xlo**g
        pl_rand = pl_rand**(1/g)
        return pl_rand

    def Print(self):
        print ""
        print ""
        print "Power Law %s data:"%self.name
        print "\tSpectral (Flux) Index:\t%.01f"%self.idx
        print "\tNevents: \t\t%.0e"%self.Nevents
        print "\tEnergy Limits: \t\t%.01e %.01e"%(self.xlo,self.xhi)
        print ""
        print ""

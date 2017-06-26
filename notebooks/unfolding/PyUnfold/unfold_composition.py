#!/usr/bin/env python

from Unfold import Unfold
from Utils import *

# grab the effects distribution
# with errors and axis properties
# as numpy arrays here
# set the RecoDist object here to pass to unfolder
RecoDist = DataDist("Reco",data=data,error=error,axis=cause,edges=cause_edges,
                     xlabel="Reco Var",ylabel="Freq.",units="Arb Units")
# do the unfolding
CauseDist = Unfold(config_name="config.cfg",return_dists=True,EffDist=RecoDist)
# grab the unfolded distribution and errors
unf_dist = CauseDist.getData()
unf_err = CauseDist.getError()

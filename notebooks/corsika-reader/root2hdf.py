#!/usr/bin/env python

from icecube import icetray,tableio
from icecube.hdfwriter import I3HDFTableService
from icecube.rootwriter import I3ROOTTableService

from icecube.tableio import I3TableTranscriber

outservice = I3HDFTableService('ldf_proton_1PeV.hdf')
inservice = I3ROOTTableService('ldf_proton_1PeV.root', 'r')

scribe = I3TableTranscriber(inservice,outservice)

scribe.Execute()

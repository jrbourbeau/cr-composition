#!/usr/bin/env python

import os
import sys
from glob import glob
from collections import defaultdict
import argparse
import time
import socket
import pandas as pd
import pyprind

from I3Tray import *
from icecube import icetray, dataio, dataclasses, phys_services, corsika_reader


# Taken from simulatino production webpage
# E.g. https://grid.icecube.wisc.edu/simulation/dataset/12360 detector
# simulation is based on CORSIKA dataset 10410
SIM_TO_CORSIKA = {}
SIM_TO_CORSIKA[12360] = '/data/sim/IceTop/2009/generated/CORSIKA-ice-top/10410/*/DAT*.bz2'


class PrimaryInfo(icetray.I3ConditionalModule):

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter('outfile', 'Output HDF file', None)
        self.AddOutBox('OutBox')

    def Configure(self):
        outfile = self.GetParameter('outfile')
        if outfile is None:
            raise ValueError('outfile must not be None')
        self.outfile = outfile

        self.data = defaultdict(list)
        pass

    def Geometry(self, frame):
        self.PushFrame(frame)

    def DAQ(self, frame):
        primary = dataclasses.get_most_energetic_primary(frame['I3MCTree'])
        self.data['energy'].append(primary.energy)
        self.data['zenith'].append(primary.dir.zenith)
        self.PushFrame(frame)

    def Finish(self):
        df = pd.DataFrame(self.data)
        print(df.head())
        df.to_hdf(self.outfile, 'dataframe', mode='w', format='table')
        return


class ProgressBar(icetray.I3ConditionalModule):

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter('length', 'Number of files', None)
        self.AddOutBox('OutBox')

    def Configure(self):
        length = self.GetParameter('length')
        if length is None:
            raise ValueError('length must not be None')
        self.length = length
        self.bar = pyprind.ProgBar(length, stream=sys.stdout)

    def DAQ(self, frame):
        self.bar.update()
        self.PushFrame(frame)

    def Finish(self):
        print(self.bar)
        return


if __name__ == "__main__":

    description = 'Extracts primary particle information from CORSIKA simulation'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-s', '--sim',
                        dest='sim',
                        nargs='*',
                        type=int,
                        help='Simulation to run over')
    parser.add_argument('-o', '--outfile',
                        dest='outfile',
                        help='')
    parser.add_argument('--test',
                        dest='test',
                        action='store_true',
                        default=False,
                        help='Run processing on a small subset of files (useful for debugging)')
    args = parser.parse_args()

    if args.sim is None:
        sims = [12360]
    else:
        sims = args.sim

    for sim in sims:

        files = glob(SIM_TO_CORSIKA[sim])
        if args.test:
            files = files[:10]
        print('Reading in {} CORSIKA files for simulation dataset {}'.format(len(files), sim))

        time_start = time.time()

        tray = I3Tray()
        # If not running on cobalt (i.e. running a cluster job), add a file stager
        if 'cobalt' not in socket.gethostname():
            tray.context['I3FileStager'] = dataio.get_stagers(
                staging_directory=os.environ['_CONDOR_SCRATCH_DIR'])
        tray.context['I3RandomService'] = phys_services.I3GSLRandomService(42)

        tray.AddModule('I3CORSIKAReader', FilenameList=files,
                       NEvents=1)

        tray.AddModule(PrimaryInfo, outfile=args.outfile)
        # tray.AddModule(ProgressBar, length=len(files))
        tray.Execute()

        time_stop = time.time()
        print('Took {:0.2f} seconds'.format(time_stop - time_start))

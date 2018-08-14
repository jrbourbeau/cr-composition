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

import comptools as comp

# Taken from simulation production webpage
# E.g. https://grid.icecube.wisc.edu/simulation/dataset/12360 detector
# simulation is based on CORSIKA dataset 10410
SIM_TO_CORSIKA = {}
SIM_TO_CORSIKA[12360] = '/data/sim/IceTop/2009/generated/CORSIKA-ice-top/10410/*/DAT*.bz2'
SIM_TO_CORSIKA[12362] = '/data/sim/IceTop/2009/generated/CORSIKA-ice-top/10889/*/DAT*.bz2'
SIM_TO_CORSIKA[12630] = '/data/sim/IceTop/2009/generated/CORSIKA-ice-top/11663/*/DAT*.bz2'
SIM_TO_CORSIKA[12631] = '/data/sim/IceTop/2009/generated/CORSIKA-ice-top/12605/*/DAT*.bz2'


class PrimaryInfo(icetray.I3ConditionalModule):

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter('outfile', 'Output HDF file', None)
        self.AddParameter('composition', 'Shower composition', None)
        self.AddOutBox('OutBox')

    def Configure(self):
        outfile = self.GetParameter('outfile')
        if outfile is None:
            raise ValueError('outfile must not be None')
        self.outfile = outfile

        composition = self.GetParameter('composition')
        if composition is None:
            raise ValueError('composition must not be None')
        self.composition = composition

        self.data = defaultdict(list)
        pass

    def Geometry(self, frame):
        self.PushFrame(frame)

    def DAQ(self, frame):
        primary = dataclasses.get_most_energetic_primary(frame['I3MCTree'])
        # frame['energy'] = primary.energy
        # frame['zenith'] = primary.dir.zenith
        self.data['energy'].append(primary.energy)
        self.data['zenith'].append(primary.dir.zenith)
        self.PushFrame(frame)

    def Finish(self):
        df = pd.DataFrame(self.data)
        df['composition'] = self.composition
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
                        type=int,
                        required=True,
                        help='Simulation dataset to run over')
    parser.add_argument('--inputs',
                        dest='inputs',
                        nargs='*',
                        help='Input CORSIKA files to run over')
    parser.add_argument('-o', '--outfile',
                        dest='outfile',
                        help='Output file path')
    parser.add_argument('--test',
                        dest='test',
                        action='store_true',
                        default=False,
                        help='Run processing on a small subset of files '
                             '(useful for debugging)')
    parser.add_argument('--verbose',
                        dest='verbose',
                        action='store_true',
                        default=False,
                        help='Option to display a progress bar')
    args = parser.parse_args()

    sim = args.sim

    if args.inputs is None:
        try:
            file_pattern = SIM_TO_CORSIKA[sim]
            files = glob(file_pattern)
        except KeyError as e:
            raise ValueError('Invalid simulation entered: {}. '
                             'Must be in {}'.format(sim, SIM_TO_CORSIKA.keys()))
    else:
        files = args.inputs

    if args.test:
        files = files[:2]
    print('Reading in {} CORSIKA files for simulation dataset {}'.format(len(files), sim))

    time_start = time.time()

    tray = I3Tray()
    # If not running on cobalt (i.e. running a cluster job), add a file stager
    if 'cobalt' not in socket.gethostname():
        tray.context['I3FileStager'] = dataio.get_stagers(
            staging_directory=os.environ['_CONDOR_SCRATCH_DIR'])
    tray.context['I3RandomService'] = phys_services.I3GSLRandomService(42)

    tray.AddModule('I3CORSIKAReader',
                   FilenameList=files,
                   NEvents=1)
    corsika_comp = comp.simfunctions.sim_to_comp(sim)
    composition = comp.composition_encoding.encode_composition_groups(corsika_comp,
                                                                      num_groups=4)
    tray.AddModule(PrimaryInfo,
                   outfile=args.outfile,
                   composition=composition)
    if args.verbose:
        tray.AddModule(ProgressBar, length=len(files))
    tray.Execute()

    time_stop = time.time()
    print('Took {:0.2f} seconds'.format(time_stop - time_start))

#!/usr/bin/env python

import time
import argparse
import os
import pandas as pd
from collections import defaultdict

from icecube import dataio, dataclasses, icetray, phys_services
from icecube.frame_object_diff.segments import uncompress
from I3Tray import *

import comptools as comp


if __name__ == "__main__":

    # Setup global path names
    comp.check_output_dir(comp.paths.comp_data_dir)

    p = argparse.ArgumentParser(
        description='Saves tank xy coordinates for plotting purposes')
    p.add_argument('-o', '--outfile', dest='outfile',
                   default=os.path.join(comp.paths.comp_data_dir, 'tankcoordinates.hdf'),
                   help='Output file')
    args = p.parse_args()

    t0 = time.time()

    file_list = ['/data/ana/CosmicRay/IceTop_level3/sim/IC79/GCD/Level3_7006_GCD.i3.gz']

    tray = I3Tray()

    tray.Add('I3Reader', FileNameList=file_list)
    # Uncompress Level3 diff files
    tray.Add(uncompress, 'uncompress')

    class GetTankCoordinates(icetray.I3Module):

        def __init__(self, context):
            icetray.I3Module.__init__(self, context)
            self.AddParameter('outfile', 'Output file for tank coordinates',
                              args.outfile)
            self.AddOutBox('OutBox')

        def Configure(self):
            self.outfile = self.GetParameter('outfile')
            self.coordinates = {}
            pass

        def Geometry(self, frame):
            self.geometry = frame['I3Geometry']
            self.geomap = self.geometry.omgeo

            for omkey, omgeo in self.geomap:
                # Only interested in saving IceTop OM charges
                if omgeo.omtype.name != 'IceTop':
                    continue
                x, y, z = omgeo.position
                key = 'IT_{}_{}_{}'.format(omkey.string, omkey.om, omkey.pmt)
                self.coordinates[key] = [x, y, z]

            self.PushFrame(frame)

        def Finish(self):
            # print(self.coordinates)
            dataframe = pd.DataFrame.from_dict(self.coordinates, orient='index')
            dataframe.columns = ['x', 'y', 'z']
            print(dataframe)
            dataframe.to_hdf(self.outfile, 'dataframe')
            return

    tray.Add(GetTankCoordinates)

    tray.Execute()
    tray.Finish()

    print('Time taken: {}'.format(time.time() - t0))

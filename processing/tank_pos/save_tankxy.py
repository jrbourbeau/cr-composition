#!/usr/bin/env python

import time
import argparse
import os
import pandas as pd
from collections import defaultdict

from icecube import dataio, dataclasses, icetray, phys_services
from icecube.frame_object_diff.segments import uncompress
from I3Tray import *

import composition as comp
import composition.i3modules as i3modules
# from composition.llh_ratio_i3_module import IceTop_LLH_Ratio


if __name__ == "__main__":

    # Setup global path names
    mypaths = comp.Paths()
    comp.checkdir(mypaths.comp_data_dir)

    p = argparse.ArgumentParser(
        description='Saves tank xy coordinates for plotting purposes')
    p.add_argument('-o', '--outfile', dest='outfile',
                   default=os.path.join(mypaths.comp_data_dir, 'tankcoordinates.csv'),
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
            self.AddParameter('outfile', 'Output file for tank coordinates', args.outfile)
            self.AddOutBox('OutBox')

        def Configure(self):
            self.outfile = self.GetParameter('outfile')
            self.coordinates = defaultdict(list)
            pass

        def Geometry(self, frame):
            self.geometry = frame['I3Geometry']
            self.stationgeo = self.geometry.stationgeo

            for tank1, tank2 in self.stationgeo.itervalues():
                self.coordinates['x'].append(tank1.position.x)
                self.coordinates['y'].append(tank1.position.y)
                self.coordinates['x'].append(tank2.position.x)
                self.coordinates['y'].append(tank2.position.y)
            self.PushFrame(frame)

        def Finish(self):
            dataframe = pd.DataFrame.from_dict(self.coordinates)
            print(dataframe.head())
            dataframe.to_csv(self.outfile)
            return

    tray.Add(GetTankCoordinates)

    tray.Execute()
    tray.Finish()

    print('Time taken: {}'.format(time.time() - t0))

#!/usr/bin/env python

import os
import glob
from collections import defaultdict
import pandas as pd
import ROOT

if __name__ == '__main__':

    file_pattern = os.path.join(os.getcwd(), 'rootfiles/capitals/*.root')
    for file_path in glob.iglob(file_pattern):
        data_dict = defaultdict(list)
        root_file = ROOT.TFile(file_path)
        for entry in root_file.Get('tinyTree'):
            data_dict['s125'].append(entry.s125)
            data_dict['lap_zenith'].append(entry.zenith)
            data_dict['lap_x'].append(entry.x)
            data_dict['lap_y'].append(entry.y)
            data_dict['lap_z'].append(entry.z)
            data_dict['eloss_1500'].append(entry.eloss_1500)
            data_dict['lap_beta'].append(entry.beta)
            data_dict['nchannels'].append(entry.nch)
            data_dict['MC_weight'].append(entry.mc_weight)

        dataframe = pd.DataFrame(data_dict)
        capital_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dataframe_file = os.path.join(os.getcwd(), 'dataframes', capital_name+'.hdf')
        print('Saving {} DataFrame...'.format(capital_name))
        dataframe.to_hdf(output_dataframe_file, 'dataframe', mode='w')

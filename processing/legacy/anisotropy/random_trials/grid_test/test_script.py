#!/usr/bin/env python

import os
import pandas as pd
import pycondor

import comptools as comp


if __name__ == "__main__":

    df_file = os.path.join(comp.paths.condor_data_dir, 'IC86.2014_data',
                           'anisotropy_dataframe.hdf')

    with pd.HDFStore(df_file, mode='r') as store:
        for df in store.select('dataframe', chunksize=10, start=0, stop=100):
            print(df)

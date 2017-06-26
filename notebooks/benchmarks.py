
import numpy as np

import comptools as comp
import comptools.analysis.anisotropy as anisotropy

def cut_df():
    data_df = comp.load_dataframe(datatype='data', config='IC86.2012')
    keep_columns = ['start_time_mjd', 'lap_zenith', 'lap_azimuth']
    data_df.drop([col for col in data_df.columns if col not in keep_columns],
                 inplace=True, axis=1)
    return data_df

def ref_process():
    # Load full DataFrame for config
    data_df = comp.load_dataframe(datatype='data', config='IC86.2012',
                                  verbose=False)
    keep_columns = ['start_time_mjd', 'lap_zenith', 'lap_azimuth']
    data_df.drop([col for col in data_df.columns if col not in keep_columns],
                 inplace=True, axis=1)
    # Extract pandas Series of all times
    times = data_df.start_time_mjd.values
    # Split dataframe into parts
    splits = np.array_split(data_df, 1000)
    # Get specific part
    data_df = splits[0]
    reference_map = anisotropy.get_reference_skymap(data_df, times, verbose=True)

    return

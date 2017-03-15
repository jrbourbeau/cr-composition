## Data processing

*There is a single script, `process.py`, that will submit the entire processing chain to the HTCondor cluster.*

Steps in the processing chain:

1. Get full list of all .i3 files to be processed.
2. Divide this list of file into batches for parallel processing.
3. Next, `save_hdf5.py` is run on each batch of files.`save_hdf5.py` is an icetray script that will write all desired data from input i3 files stored in `/data/ana` to output hdf5 files.
4. hdf5 files are then appropriately (e.g. `run123_p01.hdf5` and `run123_p02.hdf5` will be merged to `run123.hdf5`) with `merge_hdf5.py`.
5. Next, `save_dataframe.py` is used to convert hdf5 files containing a lot of information to hdf5 containing a simple pandas DataFrame object. **NOTE**: This step is handled differently for data data and simulation.  
6. merge dataframes.

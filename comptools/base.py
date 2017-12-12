#!/usr/bin/env python

from collections import namedtuple
import os
import getpass


def get_paths(username=None):
    '''Function to return paths used in this analysis

    Specifically,

    metaproject - Path to IceCube metaproject being used
    comp_data_dir - Path to where data and simulation is stored
    condor_data_dir - Path to where HTCondor error and output files are stored
    condor_scratch_dir - Path to where HTCondor log and submit files are stored
    figures_dir - Path to where figures are saved
    project_home - Path to where cr-composition project is located
    project_root - Path to where cr-composition project is located

    Parameters
    ----------
    username : str, optional
        Username on the machine that will be used for analysis. This is used
        to construct the paths for this analysis (default is getpass.getuser()).

    Returns
    -------
    paths : collections.namedtuple
        Namedtuple containing relavent paths (e.g. figures_dir is where
        figures will be saved, condor_data_dir is where data/simulation will
        be saved to / loaded from, etc).

    '''
    if username is None:
        username = getpass.getuser()

    # Create path namedtuple object
    PathObject = namedtuple('PathType', ['metaproject', 'comp_data_dir',
        'condor_data_dir', 'condor_scratch_dir', 'figures_dir',
        'project_home', 'project_root'])

    metaproject = '/data/user/{}/metaprojects/icerec/V05-01-00'.format(username)
    comp_data_dir = '/data/user/{}/composition'.format(username)
    condor_data_dir = '/data/user/{}/composition/condor'.format(username)
    condor_scratch_dir = '/scratch/{}/composition/condor'.format(username)
    figures_dir = '/home/{}/public_html/figures/composition'.format(username)
    project_home = '/home/{}/cr-composition'.format(username)
    project_root = '/home/{}/cr-composition'.format(username)

    # Create instance of PathObject with appropriate path information
    paths = PathObject(metaproject=metaproject,
                       comp_data_dir=comp_data_dir,
                       condor_data_dir=condor_data_dir,
                       condor_scratch_dir=condor_scratch_dir,
                       figures_dir=figures_dir,
                       project_home=project_home,
                       project_root=project_root)

    return paths


def check_output_dir(outfile, makedirs=True):
    '''Function to check if the directory for an output file exists

    This function will check whether the output directory containing the
    outfile specified exists. If the output directory doesn't exist, then
    there is an option to create the output directory. Otherwise, this
    function will raise an IOError.

    Parameters
    ----------
    outfile : str
        Path to output file.
    makedirs : bool, optional
        Option to create the output directory containing the output file if
        it doesn't already exist (default: True)

    Returns
    -------
    None
    '''
    # Ensure that outfile gives an absolute path
    outfile = os.path.abspath(outfile)
    outdir, basename = os.path.split(outfile)

    if not os.path.exists(outdir):
        if makedirs:
            print('The directory {} doesn\'t exist. '
                  'Creating it...'.format(outdir))
            os.makedirs(outdir)
        else:
            raise IOError('The directory {} doesn\'t exist'.format(outdir))

    return


def file_batches(files, n_files, n_batches=None):
    '''Generates batches of files

    Parameters
    ----------
    files : array-like
        Iterable of files.
    n_files : int
        Number of files to have in each batch.
    n_batches : int, optional
        Limit the number of batches to yield (default is to yield all batches).

    Returns
    -------
    batch : list
        Batch of files of size n_files.
    '''
    for batch_num, i in enumerate(range(0, len(files), n_files), start=1):
        if n_batches is not None and batch_num > n_batches:
            raise StopIteration
        batch = list(files[i:i+n_files])
        yield batch


class ComputingEnvironemtError(Exception):
    """
    Custom exception that should be raised when a problem related to the
    computing environment is found
    """
    pass

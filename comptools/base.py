#!/usr/bin/env python

from collections import namedtuple
import os


def get_paths():

    # Create path namedtuple object
    PathObject = namedtuple('PathType', ['metaproject', 'comp_data_dir',
        'condor_data_dir', 'condor_scratch_dir', 'figures_dir',
        'project_home'])

    metaproject = '/data/user/jbourbeau/metaprojects/icerec/V05-01-00'
    comp_data_dir = '/data/user/jbourbeau/composition'
    condor_data_dir = '/data/user/jbourbeau/composition/condor'
    condor_scratch_dir = '/scratch/jbourbeau/composition/condor'
    figures_dir = '/home/jbourbeau/public_html/figures/composition'
    project_home = '/home/jbourbeau/cr-composition'

    # Create instance of PathObject with appropriate path information
    paths = PathObject(metaproject=metaproject,
                       comp_data_dir=comp_data_dir,
                       condor_data_dir=condor_data_dir,
                       condor_scratch_dir=condor_scratch_dir,
                       figures_dir=figures_dir,
                       project_home=project_home)

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

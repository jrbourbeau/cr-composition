#!/usr/bin/env python

from collections import namedtuple
import os


# class Paths(object):
#
#     def __init__(self):
#         self.metaproject = "/data/user/jbourbeau/metaprojects/icerec/V05-00-00"
#         self.comp_data_dir = "/data/user/jbourbeau/composition"
#         self.condor_data_dir = "/data/user/jbourbeau/composition/condor"
#         self.condor_scratch_dir = "/scratch/jbourbeau/composition/condor"
#
#
def get_paths():
    # Create path namedtuple object
    PathObject = namedtuple('PathType', ['metaproject', 'comp_data_dir',
        'condor_data_dir', 'condor_scratch_dir'])

    metaproject = "/data/user/jbourbeau/metaprojects/icerec/V05-00-00"
    comp_data_dir = "/data/user/jbourbeau/composition"
    condor_data_dir = "/data/user/jbourbeau/composition/condor"
    condor_scratch_dir = "/scratch/jbourbeau/composition/condor"
    # Create instance of PathObject with appropriate path information
    paths = PathObject(metaproject=metaproject,
                       comp_data_dir=comp_data_dir,
                       condor_data_dir=condor_data_dir,
                       condor_scratch_dir=condor_scratch_dir)

    return paths

# composition/__init__.py

from __future__ import division, print_function

__version__ = '0.0.1'

# # Let users know if they're missing any hard dependencies
# hard_dependencies = ('icecube', 'I3Tray')
# missing_dependencies = []
# for dependency in hard_dependencies:
#     try:
#         __import__(dependency)
#     except ImportError as e:
#         missing_dependencies.append(dependency)
#
# if missing_dependencies:
#     raise ImportError(
#         'Missing required dependencies {0}'.format(missing_dependencies))
# del hard_dependencies, dependency, missing_dependencies

from .base import get_paths
from .checkdir import checkdir
from .dataframe_functions import apply_quality_cuts, load_dataframe, dataframe_to_X_y, comp_to_label, label_to_comp
from . import simfunctions
from . import datafunctions
from .analysis import *
from .livetime import get_livetime_file, get_detector_livetime
# from .serialize import serialize_SFS, deserialize_SFS
# from .PyUnfold import *
# from .RootReader import get1d

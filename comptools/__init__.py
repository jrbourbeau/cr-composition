
from __future__ import division, print_function

__version__ = '0.0.1'

from .base import get_paths, check_output_dir
from . import simfunctions
from . import datafunctions
from .dataframe_functions import (load_data, load_sim, apply_quality_cuts,
                                  dataframe_to_X_y)
from .composition_encoding import get_comp_list, comp_to_label, label_to_comp
from .analysis import *
from . import anisotropy
from .livetime import get_livetime_file, get_detector_livetime

try:
    import icecube
    _has_icecube = True
except ImportError as e:
    _has_icecube = False
    print('Couldn\'t find IceCube software. Importing comptools without it.')

if _has_icecube:
    from . import icetray_software

paths = get_paths()

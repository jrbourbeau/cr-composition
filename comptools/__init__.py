
from __future__ import division, print_function

__version__ = '0.0.1'

try:
    import icecube
    _has_icecube = True
except ImportError as e:
    _has_icecube = False
    print('Couldn\'t import IceCube software. Importing comptools without it.')

import os
from .base import (get_paths, check_output_dir, file_batches,
                   ComputingEnvironemtError, get_energybins,
                   get_training_features)
from . import simfunctions
from . import datafunctions
from .io import (load_data, load_sim, apply_quality_cuts,
                 dataframe_to_X_y)
from .composition_encoding import (get_comp_list, comp_to_label, label_to_comp,
                                   decode_composition_groups)
from .livetime import get_livetime_file, get_detector_livetime
from .plotting import get_color_dict
from .pipelines import get_pipeline, load_trained_model
from .model_selection import get_CV_frac_correct, cross_validate_comp
from .spectrumfunctions import get_flux, model_flux


# if _has_icecube and 'py2-v3' in os.getenv('ROOTSYS'):
#     from . import icetray_software

paths = get_paths()
color_dict = get_color_dict()
energybins = get_energybins()

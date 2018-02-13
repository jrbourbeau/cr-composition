
__version__ = '0.0.1'

import os
from .base import (get_paths, check_output_dir, file_batches,
                   ComputingEnvironemtError, get_energybins,
                   get_training_features)
from . import simfunctions
from . import datafunctions
from .io import (load_data, load_sim, apply_quality_cuts,
                 dataframe_to_X_y, load_trained_model)
from .composition_encoding import (get_comp_list, comp_to_label, label_to_comp,
                                   decode_composition_groups)
from .livetime import get_livetime_file, get_detector_livetime
from .plotting import get_color_dict, plot_steps, get_colormap, get_color
from .pipelines import get_pipeline
from .model_selection import get_CV_frac_correct, cross_validate_comp
from .spectrumfunctions import (get_flux, model_flux, counts_to_flux,
                                broken_power_law_flux)
from .data_functions import ratio_error
from .unfolding import unfolded_counts_dist

paths = get_paths()
color_dict = get_color_dict()

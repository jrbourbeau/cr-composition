# composition/__init__.py

from __future__ import division, print_function

# Let users know if they're missing any of our hard dependencies
hard_dependencies = ('numpy', 'matplotlib', 'pandas',
                     'sklearn', 'icecube', 'I3Tray')
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError(
        'Missing required dependencies {0}'.format(missing_dependencies))
del hard_dependencies, dependency, missing_dependencies

from .paths import Paths
from .checkdir import checkdir
from .load_sim import load_sim
from . import simfunctions
from .analysis import * 

#!/usr/bin/env python

from __future__ import print_function
import os
import stat
import comptools as comp

here = os.path.abspath(os.path.dirname(__file__))
wrapper_path = os.path.join(here, 'wrapper.sh')
wrapper_virtualenv_path = os.path.join(here, 'wrapper_virtualenv.sh')

wrapper = """#!/bin/bash -e

eval `/cvmfs/icecube.opensciencegrid.org/py2-v3/setup.sh`

{icecube_env_script} \\
{wrapper_virtualenv_path} \\
python $*
"""

virtualenv_wrapper = """#!/bin/sh

source {virtualenv_activate}

$@
"""

icecube_env_script = os.path.join(comp.paths.metaproject,
                                  'build',
                                  'env-shell.sh')
virtualenv_activate = os.path.join(comp.paths.virtualenv_dir,
                                   'bin',
                                   'activate')

print('Writing wrapper script {}...'.format(wrapper_path))
with open(wrapper_path, 'w') as f:
    lines = wrapper.format(icecube_env_script=icecube_env_script,
                           wrapper_virtualenv_path=wrapper_virtualenv_path)
    f.write(lines)

print('Writing wrapper script {}...'.format(wrapper_virtualenv_path))
with open(wrapper_virtualenv_path, 'w') as f:
    lines = virtualenv_wrapper.format(virtualenv_activate=virtualenv_activate)
    f.write(lines)


def make_executable(path):
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


make_executable(wrapper_path)
make_executable(wrapper_virtualenv_path)

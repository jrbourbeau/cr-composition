.. _installation:

:github_url: https://github.com/jrbourbeau/cr-composition

********************
Installation & setup
********************

This page walks through the initial setup and installation steps to configure the computing environment used for this analysis. It is assumed that you are currently logged onto one of the IceCube cobalt testbed machines.


Cloning the analysis repository
-------------------------------

The code for this analysis is hosted on `GitHub <https://github.com/jrbourbeau/cr-composition>`_. A copy of the analysis repository can be cloned to your local machine using ``git clone``

.. code-block:: bash

    git clone https://github.com/jrbourbeau/cr-composition.git

This will create a ``cr-composition/`` directory containing the analysis code which you can then ``cd`` into.

.. code-block:: bash

    cd cr-composition


Setting up the CVMFS toolset
----------------------------

To configure the system level dependencies we'll use the ``py2-v3`` toolset in the `IceCube CVMFS <http://software.icecube.wisc.edu/documentation/info/cvmfs.html>`_

.. code-block:: bash

    eval $(/cvmfs/icecube.opensciencegrid.org/py2-v3/setup.sh)


This will set up many system-wide tools for us to use (e.g. Python 2.7.13).


Creating a virtual environment
------------------------------

Next we'll set up a virtual Python environment using ``virtualenv`` to install our Python dependencies.

.. code-block:: bash

    virtualenv --prompt="(cr-composition) " /path/to/virtualenv
    source /path/to/virtualenv/bin/activate

Here, ``/path/to/virtualenv`` should be replaced with wherever you'd like your virtual environment to be installed. To activate the virtual environment, use

.. code-block:: bash

    source /path/to/virtualenv/bin/activate


Installing Python dependencies
------------------------------

The Python dependencies for this analysis are listed in ``requirements.txt``. They, along with the ``comptools`` Python pacakge, can be installed using ``pip``

.. code-block:: bash

    pip install -e .


Building the IceCube software
-----------------------------

We'll make use of the `icerec metaproject <http://code.icecube.wisc.edu/projects/icecube/browser/IceCube/meta-projects/icerec/releases/V05-01-05>`_ in conjunction with the `weighting <http://code.icecube.wisc.edu/projects/icecube/browser/IceCube/projects/weighting/releases/V00-02-01>`_ and `corsika-reader <http://code.icecube.wisc.edu/projects/icecube/browser/IceCube/projects/corsika-reader/releases/V00-01-02>`_ projects. The specific version used are listed below.

============== =========
Project name   Version
============== =========
icerec         V05-01-05
weighting      V00-02-01
corsika-reader V00-01-02
============== =========

The source code for these projects can be retrieved from the IceCube SVN repository using the following commands

.. code-block:: bash

    mkdir /path/to/icecube/software
    cd /path/to/icecube/software
    svn co http://code.icecube.wisc.edu/svn/meta-projects/icerec/releases/V05-01-05 src
    svn co http://code.icecube.wisc.edu/svn/projects/weighting/releases/V00-02-01 src/weighting
    svn co http://code.icecube.wisc.edu/svn/projects/corsika-reader/releases/V00-01-02 src/corsika-reader


Above ``/path/to/icecube/software`` should be replaced with wherever you’d like the IceCube software to be installed. The source code can be compiled into the ``build/`` directory via

.. code-block:: bash

    mkdir build
    cd build
    cmake -DCMAKE_CXX_STANDARD=11 ../src
    make
